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

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, GaussianFeatureRasterizer
from scene.gaussian_model import GaussianModel, LatentGaussianModel
from utils.sh_utils import eval_sh
from scene.modules import triplane_sample

def render(viewpoint_camera,
           pc : GaussianModel,
           pipe,
           bg_color : torch.Tensor,
           render_feature = False,
           scaling_modifier = 1.0,
           triplane_index = -1,
           itr=-1,
           rvq_iter=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    # if render_feature:
        # bg_color = torch.tensor(
        #     [0] * override_feature.shape[-1], dtype=torch.float32, device="cuda"
        # )
    Rasterizer = GaussianFeatureRasterizer
    # else:
    #     Rasterizer = GaussianRasterizer
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = Rasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    cov3D_precomp = None
    l_vqsca=0
    l_vqrot=0
    shs = None
    if not render_feature:
        if itr == -1:
            scales = pc._scaling
            rotations = pc._rotation
            opacity = pc._opacity
            
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            # xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
            # colors_precomp = pc.mlp_head(torch.cat([pc.recolor(xyz), pc.direction_encoding(dir_pp)], dim=-1))
            # shs = pc.color_head(torch.cat([pc._feature, pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
            colors_precomp = pc.color_head(torch.cat([pc._feature, pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
            
        else:
            mask = ((torch.sigmoid(pc._mask) > 0.01).float()- torch.sigmoid(pc._mask)).detach() + torch.sigmoid(pc._mask)
            if rvq_iter:
                scales, _, l_vqsca = pc.vq_scale(pc.get_scaling.unsqueeze(0))
                rotations, _, l_vqrot = pc.vq_rot(pc.get_rotation.unsqueeze(0))
                scales = scales.squeeze()*mask
                rotations = rotations.squeeze()
                opacity = pc.get_opacity*mask

                l_vqsca = torch.sum(l_vqsca)
                l_vqrot = torch.sum(l_vqrot)
            else:
                scales = pc.get_scaling*mask
                rotations = pc.get_rotation
                opacity = pc.get_opacity*mask
                
            xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
            # xyz = xyz * 2 - 1
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
            dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            
            # triplane = pc.recolor.embeddings
            # triplane_lowres = pc.recolor.embeddings
            # triplane = pc.recolor_upsample(triplane_lowres)
            # shs = pc.mlp_head(torch.cat([triplane_sample(triplane, xyz), pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
            # colors_precomp = pc.mlp_head(torch.cat([triplane_sample(triplane, xyz), pc.direction_encoding(dir_pp)], dim=-1))
            # shs = pc.color_head(torch.cat([pc.recolor(xyz), pc.direction_encoding(dir_pp)], dim=-1)).unsqueeze(1)
            colors_precomp = pc.color_head(torch.cat([pc.recolor(xyz), pc.direction_encoding(dir_pp)], dim=-1))
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    
    
        rendered_image, radii, depth = rasterizer(
            means3D = means3D.float(),
            means2D = means2D,
            shs = None,
            colors_precomp = colors_precomp.float(),
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = None)
        rendered_feature = None
    else:
        scales = pc._scaling
        rotations = pc._rotation
        opacity = pc._opacity
        xyz = pc.contract_to_unisphere(means3D.clone().detach(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        dir_pp = (means3D - viewpoint_camera.camera_center.repeat(means3D.shape[0], 1))
        dir_pp = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        colors_precomp = pc.latent_color_head(torch.cat([pc.latent_recolor(xyz), pc.direction_encoding(dir_pp)], dim=-1))
        # xyz = xyz * 2 - 1
        # triplane_tokens = pc.triplane_tokens(1)
        # print(triplane_tokens.shape)triplane_index
        # print(encoder_hidden_states.shape)
        # triplane_lowres = pc.triplane_tokens.detokenize(pc.triplane_tokens[triplane_index])[0]
        # triplane_lowres = pc.triplane_tokens[triplane_index].embeddings
        # triplane = pc.triplane_upsample(triplane_lowres)
        # masks_precomp = pc.triplane_encoder(xyz)
        # masks_precomp = torch.sigmoid(triplane_sample(triplane, xyz))
        # masks_precomp = triplane_sample(triplane, xyz)
        # masks_precomp = pc.mask_mlp(mask_feature)
        rendered_feature, radii, depth = rasterizer(
            means3D=means3D.float(),
            means2D=means2D,
            shs=None,
            colors_precomp=colors_precomp.float(),
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
            )
        # rendered_feature = torch.sigmoid(rendered_feature)
        rendered_image = None
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "render_feature": rendered_feature,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
        "depth_3dgs": depth,
        "vqloss": l_vqsca+l_vqrot
    }