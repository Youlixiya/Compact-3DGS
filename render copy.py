#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from random import randint
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, TriplaneGaussianModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mask_background, args):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    mask_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask_renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(mask_render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    mask_index = args.mask_index
    # instance_embedding = gaussians.instance_embeddings[mask_index]
    # clip_embedding = gaussians.clip_embeddings[mask_index]
    # encoder_embeddings = torch.stack([instance_embedding, clip_embedding]).unsqueeze(0)
    # triplane = gaussians.triplane_tokens[mask_index]
        # triplane = pc.triplane_upsample(triplane_lowres)
        # masks_precomp = pc.triplane_encoder(xyz)
    # masks_precomp = triplane_sample(triplane.embeddings, xyz)
    # rendered_feature = render(viewpoint_cam, gaussians, pipe, feature_bg_color, render_feature=True, encoder_hidden_states=encoder_embeddings)['render_feature']

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        
        rendered_mask = render(view, gaussians, pipeline, mask_background, render_feature=True, triplane_index=mask_index)['render_feature']
        rendered_mask = (rendered_mask > 0.6).float()
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendered_mask, os.path.join(mask_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = TriplaneGaussianModel(dataset, rvq=False)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        params_path = os.path.join(dataset.model_path,
                            "point_cloud",
                            "iteration_" + str(30000),
                            f"mask_triplane_{2000}.pt")
        gaussians.load_feature_params(params_path)
        
        gaussians.precompute()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        mask_bg_color = [0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        mask_background = torch.tensor(mask_bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, mask_background, args)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, mask_background, args)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--mask_index", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)