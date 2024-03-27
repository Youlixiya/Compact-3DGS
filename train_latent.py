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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, l2_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, LatentGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.extract_latent_images import LatentImageDataset, postprocess
from scene.camera_scene import CamScene
from diffusers.models import AutoencoderKL
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, comp, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = LatentGaussianModel(dataset)
    vae = AutoencoderKL.from_pretrained(dataset.vae_model_path, subfolder='vae', torch_dtype=torch.float32
                                        ).to('cuda')
    vae.requires_grad_(False)
    latent_images = dataset.images
    images = dataset.images.replace('latents', 'images')
    dataset.images = dataset.images.replace('latents', 'images')
    # scene = Scene(dataset, gaussians)
    img_name = os.listdir(os.path.join(dataset.source_path, images))[0]
    h, w = cv2.imread(os.path.join(dataset.source_path, images, img_name)).shape[:2]
    h = h // 8
    w = w // 8
    scene = CamScene(dataset, gaussians, load_iteration=30000, h=h, w=w, eval=True)
    # scene.gaussians = gaussians
    # gaussians.load_ply('output/llff/latent_fern/point_cloud/iteration_30000/point_cloud.ply')
    cameras = scene.getTrainCameras().copy() + scene.getTestCameras().copy()
    latent_image_dataset = LatentImageDataset(dataset.source_path, cameras, vae, latent_dir=latent_images)
    gaussians.latent_training_setup(opt)
    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    latent_bg_color = [1, 1, 1, 1] if dataset.white_background else [0, 0, 0, 0]
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    latent_background = torch.tensor(latent_bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.latent_iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.latent_iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        index = randint(0, len(viewpoint_stack)-1)
        viewpoint_cam = viewpoint_stack.pop(index)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, latent_background, render_feature=True, itr=iteration, rvq_iter=False)
        # latent_image = render(viewpoint_cam, gaussians, pipe, latent_background, render_feature=True, itr=iteration, rvq_iter=False)['render_feature']
        latent_image = render_pkg["render_feature"]
        gt_latent_image = latent_image_dataset.latent_images_dict[viewpoint_cam.image_name].to('cuda')
        # Loss
        Ll1 = l2_loss(latent_image, gt_latent_image)
        # Ll1 = torch.nn.functional.mse_loss(image, gt_image)
        loss = Ll1
        # loss = Ll1 + render_pkg["vqloss"] + opt.lambda_mask*torch.mean((torch.sigmoid(gaussians._mask)))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "N_GS": gaussians._xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.latent_iterations:
                progress_bar.close()
            
            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, vae, latent_image_dataset, render, (pipe, latent_background, True))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # Optimizer step
            if iteration < opt.latent_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                # gaussians.optimizer_net.step()
                # gaussians.optimizer_net.zero_grad(set_to_none = True)
                # gaussians.scheduler_net.step()
            # if (iteration in checkpoint_iterations):
            #     print("\n[ITER {}] Saving Checkpoint".format(iteration))
            #     torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            if (iteration + 1 in saving_iterations):
                print("\n[ITER {}] Saving Latent Gaussians".format(iteration + 1))
                save_path = os.path.abspath(os.path.join(args.gs_source, os.pardir))
                # extra = ''
                gaussians.save_params(save_path, iteration + 1)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, vae, latent_image_dataset, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    latent_image = renderFunc(viewpoint, scene.gaussians, *renderArgs)["render_feature"][None]
                    image = torch.clamp((vae.decode(latent_image, return_dict=False)[0] + 1) / 2, 0.0, 1.0)
                    # image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    # h, w = image.shape[1:]
                    gt_image = latent_image_dataset.imgs_dict[viewpoint.image_name].to('cuda')
                    gt_image = torch.clamp(gt_image, 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image, global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--comp", action="store_true")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.comp, args)

    # All done
    print("\nTraining complete.")
