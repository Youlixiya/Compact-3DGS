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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.clip_encoder import OpenCLIPNetworkConfig, OpenCLIPNetwork
from utils.general_utils import safe_state
import numpy as np
from tqdm import tqdm
from PIL import Image
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
                

def project(dataset):
    gaussians = GaussianModel(dataset)
    scene = Scene(dataset, gaussians, load_iteration=30000)
    registry = sam_model_registry[dataset.sam_model_type]
    sam = registry(checkpoint=dataset.sam_model_ckpt)
    sam = sam.to('cuda')
    sam_mask_generator = SamAutomaticMaskGenerator(model=sam)
    clip = OpenCLIPNetwork(OpenCLIPNetworkConfig())
    viewpoint_stack = scene.getTrainCameras().copy()
    language_embeddings = torch.zeros((gaussians._opacity.shape[0], 512), device='cuda')
    for i in tqdm(range(len(viewpoint_stack))):
        cur_cam = viewpoint_stack[i]
        image_np = (cur_cam.original_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # image_pil = Image.fromarray(image_np)
        masks = sam_mask_generator.generate(image_np)
        masks = [{'segmentation': m['segmentation'], 'bbox': m['bbox']} for m in masks] # already as bool
        masks = sorted(masks, key=lambda x: x['segmentation'].sum())
        for mask in masks:
            weights = torch.zeros_like(gaussians._opacity)
            weights_cnt = torch.zeros_like(gaussians._opacity, dtype=torch.int32)
            segmentation = mask['segmentation']
            image_box = image_np.copy()
            image_box[segmentation] = np.array([0, 0,  0], dtype=np.uint8)
            x,y,w,h = np.int32(mask['bbox'])
            seg_img = Image.fromarray(image_box[y:y+h, x:x+w, ...])
            clip_embedding = clip.encode_image(seg_img)
            segmentation = torch.FloatTensor(segmentation)[None, ...].cuda()
            gaussians.apply_weights(cur_cam, weights, weights_cnt, segmentation)
            weights /= weights_cnt + 1e-7
            weights = weights
            language_embeddings += weights * clip_embedding
            language_embeddings = torch.nn.functional.normalize(language_embeddings, dim=-1)


    save_path = os.path.join(scene.model_path, "point_cloud/iteration_30000", 'language_embeddings.pt')
    torch.save(language_embeddings.cpu(), save_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    # parser.add_argument('--debug_from', type=int, default=-1)
    # parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    # parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    # parser.add_argument("--start_checkpoint", type=str, default = None)
    # parser.add_argument("--comp", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    
    # print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    # safe_state(args.quiet)
    
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    # torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    project(lp.extract(args))

    # All done
    # print("\nTraining complete.")