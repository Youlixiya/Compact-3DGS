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

import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, TriplaneGaussianModel
from scene.camera_scene import CamScene
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.extract_masks import MaskDataset
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def get_cosine_similarities(vectors, query_vector):
    return F.cosine_similarity(query_vector.unsqueeze(0), vectors, dim=1)

def training(args, dataset, opt, pipe, saving_iterations):
    cur_iter = 0
    # tb_writer = prepare_output_and_logger(dataset)
    gaussians = TriplaneGaussianModel(dataset)
    gaussians.load_ply(args.gs_source)
    # print(gaussians.instance_feature_dim)
    img_name = os.listdir(os.path.join(args.colmap_dir, args.images))[0]
    h, w = cv2.imread(os.path.join(args.colmap_dir, args.images, img_name)).shape[:2]
    scene = CamScene(args.colmap_dir, h=h, w=w, eval=True)
    
    # scene = CamScene(args.colmap_dir, h=-1, w=-1, images=args.images)

    feature_bg_color = torch.tensor([0], dtype=torch.float32, device="cuda")
    # bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    view_num = len(scene.cameras)
    loss_for_log = 0.0
    downscale = '' if args.images == 'images' else args.images.split('_')[-1]
    downscale = downscale if downscale == '' else f'_{downscale}'
    mask_dataset = MaskDataset(args.colmap_dir, scene.cameras.copy(), mask_dir=f'masks{downscale}')
    instance_num = len(mask_dataset.instance_colors)
    print(f'instance num: {instance_num}')
    gaussians.set_instance_embeddings(len(mask_dataset.instance_colors))
    gaussians.set_triplane_tokens(len(mask_dataset.instance_colors))
    gaussians.set_clip_embeddings(mask_dataset.clip_embeddings)
    gaussians.set_instance_colors(mask_dataset.instance_colors)
    gaussians.feature_training_setup(opt)
    progress_bar = tqdm(range(cur_iter, opt.mask_iterations), desc="Training Feature GS progress")
    while cur_iter < opt.mask_iterations:
        cur_iter += 1
        iter_start.record()
        index = randint(0, len(mask_dataset)-1)
        viewpoint_stack = scene.cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(index)
        # mask_index = randint(1, instance_num - 1)
        instance_masks = mask_dataset.instance_masks[index].reshape(h * w)
        unique_instance_index = torch.unique(instance_masks)
        loss = None
        for instance_index in unique_instance_index:
            rendered_feature = render(viewpoint_cam, gaussians, pipe, feature_bg_color, render_feature=True, triplane_index=instance_index)['render_feature']
            h, w = rendered_feature.shape[1:]
            instance_mask = (instance_masks == instance_index).float()
            if loss is None:    
                loss = F.binary_cross_entropy(rendered_feature.reshape(-1), instance_mask.reshape(-1))
            else:
                loss += F.binary_cross_entropy(rendered_feature.reshape(-1), instance_mask.reshape(-1))
        loss = loss / len(unique_instance_index)
        loss.backward()
        # instance_embedding = gaussians.instance_embeddings[[mask_index]]
        # clip_embedding = gaussians.clip_embeddings[mask_index]
        # encoder_embeddings = instance_embedding.unsqueeze(0)
        # rendered_feature = render(viewpoint_cam, gaussians, pipe, feature_bg_color, render_feature=True, encoder_hidden_states=encoder_embeddings)['render_feature']
        

        iter_end.record()
        with torch.no_grad():
            # Progress bar
            loss_for_log = loss.item()
            # loss_for_log = total_loss.item()
            # if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{loss_for_log:.{7}f}"})
            progress_bar.update(1)
            if cur_iter + 1 == opt.mask_iterations:
                progress_bar.close()
            if (cur_iter + 1 in saving_iterations):
                print("\n[ITER {}] Saving Feature Gaussians".format(cur_iter + 1))
                save_path = os.path.abspath(os.path.join(args.gs_source, os.pardir))
                extra = 'mask'
                gaussians.save_feature_params(save_path, cur_iter + 1, extra)

            # Optimizer step
            if cur_iter + 1 < opt.mask_iterations:
                gaussians.optimizer.step()
                # scaler.step(gaussians.optimizer)
                gaussians.optimizer.zero_grad(set_to_none = True)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2_000])
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=True)  #
    args = parser.parse_args(sys.argv[1:])
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.save_iterations)

    # All done
    print("\nTraining complete.")
