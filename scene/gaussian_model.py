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
import numpy as np
from typing import Literal, Optional, Sequence
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn, Tensor
import torch.nn.functional as F
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from jaxtyping import Float, Int, Shaped
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import tinycudann as tcnn
from scene.modules import Transformer, TriplaneTokens, triplane_sample
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from dahuffman import HuffmanCodec
import math

def camera2rasterizer(viewpoint_camera, bg_color: torch.Tensor, sh_degree: int = 0):
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    return rasterizer

class TriplaneEncoding(nn.Module):

    plane_coef: Float[Tensor, "3 num_components resolution resolution"]

    def __init__(
        self,
        in_dim: int = 3,
        resolution: int = 32,
        num_components: int = 64,
        init_scale: float = 0.1,
        reduce: Literal["sum", "product"] = "sum",
    ) -> None:
        super().__init__()
        
        self.in_dim = in_dim
        self.resolution = resolution
        self.num_components = num_components
        self.init_scale = init_scale
        self.reduce = reduce

        self.plane_coef = nn.Parameter(
            self.init_scale * torch.randn((3, self.num_components, self.resolution, self.resolution))
        )

    def get_out_dim(self) -> int:
        return self.num_components

    def forward(self, in_tensor: Float[Tensor, "*bs 3"]) -> Float[Tensor, "*bs num_components featuresize"]:
        """Sample features from this encoder. Expects in_tensor to be in range [0, resolution]"""

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]], dim=0)

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2)
        plane_features = F.grid_sample(
            self.plane_coef, plane_coord, align_corners=True
        )  # [3, num_components, flattened_bs, 1]

        if self.reduce == "product":
            plane_features = plane_features.prod(0).squeeze(-1).T  # [flattened_bs, num_components]
        else:
            plane_features = plane_features.sum(0).squeeze(-1).T

        return plane_features.reshape(*original_shape[:-1], self.num_components)

    @torch.no_grad()
    def upsample_grid(self, resolution: int) -> None:
        """Upsamples underlying feature grid

        Args:
            resolution: Target resolution.
        """
        plane_coef = F.interpolate(
            self.plane_coef.data, size=(resolution, resolution), mode="bilinear", align_corners=True
        )

        self.plane_coef = torch.nn.Parameter(plane_coef)
        self.resolution = resolution

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, model, rvq=True):
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        self._xyz = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._mask = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        
        if rvq:
            self.vq_scale = ResidualVQ(dim = 3, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, decay = 0.8, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1).cuda()
            self.vq_rot = ResidualVQ(dim = 4, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, decay = 0.8, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1).cuda()
            self.rvq_bit = math.log2(model.rvq_size)
            self.rvq_num = model.rvq_num
            
        self.recolor = tcnn.Encoding(
                 n_input_dims=3,
                 encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": model.max_hashmap,
                    "base_resolution": 16,
                    "per_level_scale": 1.447,
                },
        )

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 3 
            },
        )
        self.color_head = tcnn.Network(
                n_input_dims=(self.direction_encoding.n_output_dims+self.recolor.n_output_dims),
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        other_params = []
        for params in self.vq_rot.parameters():
            other_params.append(params)
        for params in self.vq_scale.parameters():
            other_params.append(params)
        for params in self.recolor.parameters():
            other_params.append(params)
        for params in self.color_head.parameters():
            other_params.append(params)
        # for params in self.latent_color_head.parameters():
        #     other_params.append(params)
            
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.optimizer_net = torch.optim.Adam(other_params, lr=training_args.net_lr, eps=1e-15)
        self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
        [
            torch.optim.lr_scheduler.LinearLR(
            self.optimizer_net, start_factor=0.01, total_iters=100
        ),
            torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer_net,
            milestones=training_args.net_lr_step,
            gamma=0.33,
        ),
        ]
        )
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_mask):
        d = {"xyz": new_xyz,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation,
        "mask": new_mask}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._mask = optimizable_tensors["mask"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_mask = self._mask[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_mask)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_mask = self._mask[selected_pts_mask]

        self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation, new_mask)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = torch.logical_or((torch.sigmoid(self._mask) <= 0.01).squeeze(),(self.get_opacity < min_opacity).squeeze())
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()
    
    def mask_prune(self):
        prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
        self.prune_points(prune_mask)
        torch.cuda.empty_cache()

    def post_quant(self, param, prune=False):
        max_val = torch.amax(param)
        min_val = torch.amin(param)
        param = (param - min_val)/(max_val - min_val)
        quant = torch.round(param * 255.0) / 255.0
        out = (max_val - min_val)*quant + min_val
        if prune:
            quant = quant*(torch.abs(out) > 0.1)
            out = out*(torch.abs(out) > 0.1)
        return torch.nn.Parameter(out), quant
    
    def huffman_encode(self, param):
        input_code_list = param.view(-1).tolist()
        unique, counts = np.unique(input_code_list, return_counts=True)
        num_freq = dict(zip(unique, counts))

        codec = HuffmanCodec.from_data(input_code_list)

        sym_bit_dict = {}
        for k, v in codec.get_code_table().items():
            sym_bit_dict[k] = v[0]
        total_bits = 0
        for num, freq in num_freq.items():
            total_bits += freq * sym_bit_dict[num]
        total_mb = total_bits/8/10**6
        return total_mb

    def final_prune(self, compress=False):
        prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
        self.prune_points(prune_mask)
        self._xyz = self._xyz.clone().half().float()
        self._scaling, sca_idx, _ = self.vq_scale(self.get_scaling.unsqueeze(1))
        self._rotation, rot_idx, _ = self.vq_rot(self.get_rotation.unsqueeze(1))
        self._scaling = self._scaling.squeeze()
        self._rotation = self._rotation.squeeze()
        
        position_mb = self._xyz.shape[0]*3*16/8/10**6
        scale_mb = self._xyz.shape[0]*self.rvq_bit*self.rvq_num/8/10**6 + 2**self.rvq_bit*self.rvq_num*3*32/8/10**6
        rotation_mb = self._xyz.shape[0]*self.rvq_bit*self.rvq_num/8/10**6 + 2**self.rvq_bit*self.rvq_num*4*32/8/10**6
        opacity_mb = self._xyz.shape[0]*16/8/10**6
        hash_mb = self.recolor.params.shape[0]*16/8/10**6
        # mlp_mb = (self.color_head.params.shape[0] + self.latent_color_head.params.shape[0])*16/8/10**6
        mlp_mb = self.color_head.params.shape[0]*16/8/10**6
        sum_mb = position_mb+scale_mb+rotation_mb+opacity_mb+hash_mb+mlp_mb
        
        mb_str = "Storage\nposition: "+str(position_mb)+"\nscale: "+str(scale_mb)+"\nrotation: "+str(rotation_mb)+"\nopacity: "+str(opacity_mb)+"\nhash: "+str(hash_mb)+"\nmlp: "+str(mlp_mb)+"\ntotal: "+str(sum_mb)+" MB"
        
        if compress:
            self._opacity, quant_opa = self.post_quant(self.get_opacity)
            self.recolor.params, quant_hash = self.post_quant(self.recolor.params, True)
        
            scale_mb = self.huffman_encode(sca_idx) + 2**self.rvq_bit*self.rvq_num*3*32/8/10**6
            rotation_mb = self.huffman_encode(rot_idx) + 2**self.rvq_bit*self.rvq_num*4*32/8/10**6
            opacity_mb = self.huffman_encode(quant_opa)
            hash_mb = self.huffman_encode(quant_hash)
            # mlp_mb = (self.color_head.params.shape[0] + self.latent_color_head.params.shape[0])*16/8/10**6
            mlp_mb = self.color_head.params.shape[0]*16/8/10**6
            sum_mb = position_mb+scale_mb+rotation_mb+opacity_mb+hash_mb+mlp_mb
            
            mb_str = mb_str+"\n\nAfter PP\nposition: "+str(position_mb)+"\nscale: "+str(scale_mb)+"\nrotation: "+str(rotation_mb)+"\nopacity: "+str(opacity_mb)+"\nhash: "+str(hash_mb)+"\nmlp: "+str(mlp_mb)+"\ntotal: "+str(sum_mb)+" MB"
        else:
            self._opacity = self.get_opacity.clone().half().float()
        torch.cuda.empty_cache()
        return mb_str
    
    def precompute(self):
        xyz = self.contract_to_unisphere(self.get_xyz.half(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
        self._feature = self.recolor(xyz)
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
    def contract_to_unisphere(self,
        x: torch.Tensor,
        aabb: torch.Tensor,
        ord: int = 2,
        eps: float = 1e-6,
        derivative: bool = False,
    ):
        aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
        x = (x - aabb_min) / (aabb_max - aabb_min)
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1

        if derivative:
            dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
                1 / mag**3 - (2 * mag - 1) / mag**4
            )
            dev[~mask] = 1.0
            dev = torch.clamp(dev, min=eps)
            return dev
        else:
            x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
            x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
            return x
        
    def apply_weights(self, camera, weights, weights_cnt, image_weights):
        rasterizer = camera2rasterizer(
            camera, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        )
        rasterizer.apply_weights(
            self.get_xyz,
            None,
            self.get_opacity,
            None,
            weights,
            self.get_scaling,
            self.get_rotation,
            None,
            weights_cnt,
            image_weights,
        )


# class GaussianModel:

#     def setup_functions(self):
#         def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
#             L = build_scaling_rotation(scaling_modifier * scaling, rotation)
#             actual_covariance = L @ L.transpose(1, 2)
#             symm = strip_symmetric(actual_covariance)
#             return symm
        
#         self.scaling_activation = torch.exp
#         self.scaling_inverse_activation = torch.log

#         self.covariance_activation = build_covariance_from_scaling_rotation

#         self.opacity_activation = torch.sigmoid
#         self.inverse_opacity_activation = inverse_sigmoid

#         self.rotation_activation = torch.nn.functional.normalize


#     def __init__(self, model, rvq=True):
#         self.active_sh_degree = 0
#         self.max_sh_degree = 0
#         self._xyz = torch.empty(0)
#         self._scaling = torch.empty(0)
#         self._rotation = torch.empty(0)
#         self._opacity = torch.empty(0)
#         self._mask = torch.empty(0)
#         self.max_radii2D = torch.empty(0)
#         self.xyz_gradient_accum = torch.empty(0)
#         self.denom = torch.empty(0)
#         self.optimizer = None
#         self.percent_dense = 0
#         self.spatial_lr_scale = 0
#         self.setup_functions()
        
#         if rvq:
#             self.vq_scale = ResidualVQ(dim = 3, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, decay = 0.8, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1).cuda()
#             self.vq_rot = ResidualVQ(dim = 4, codebook_size = model.rvq_size, num_quantizers = model.rvq_num, decay = 0.8, commitment_weight = 0., kmeans_init = True, kmeans_iters = 1).cuda()
#             self.rvq_bit = math.log2(model.rvq_size)
#             self.rvq_num = model.rvq_num
            
        
#         self.recolor = TriplaneTokens(resolution=32, num_components=256).cuda()
#         # self.recolor = TriplaneTokens(resolution=128, num_components=64).cuda()
        # self.recolor_upsample = nn.Sequential(
        #     # nn.ConvTranspose2d(
        #     #     512, 256, kernel_size=2, stride=2
        #     # ),
        #     nn.ConvTranspose2d(
        #         256, 128, kernel_size=2, stride=2
        #     ),
        #     nn.SiLU(inplace=True),
        #     nn.ConvTranspose2d(
        #         128, 64, kernel_size=2, stride=2
        #     ),
        #     # nn.ConvTranspose2d(
        #     #     64, 32, kernel_size=2, stride=2
        #     # ),
        # ).cuda()

#         self.direction_encoding = tcnn.Encoding(
#             n_input_dims=3,
#             encoding_config={
#                 "otype": "SphericalHarmonics",
#                 "degree": 3 
#             },
#         )
#         self.mlp_head = tcnn.Network(
#                 n_input_dims=(self.direction_encoding.n_output_dims+self.recolor.num_components // 4),
#                 n_output_dims=4,
#                 network_config={
#                     "otype": "FullyFusedMLP",
#                     "activation": "ReLU",
#                     "output_activation": "None",
#                     "n_neurons": 64,
#                     "n_hidden_layers": 2,
#                 },
#             )

#     def capture(self):
#         return (
#             self.active_sh_degree,
#             self._xyz,
#             self._scaling,
#             self._rotation,
#             self._opacity,
#             self.max_radii2D,
#             self.xyz_gradient_accum,
#             self.denom,
#             self.optimizer.state_dict(),
#             self.spatial_lr_scale,
#         )
    
#     def restore(self, model_args, training_args):
#         (self.active_sh_degree, 
#         self._xyz,
#         self._scaling, 
#         self._rotation, 
#         self._opacity,
#         self.max_radii2D, 
#         xyz_gradient_accum, 
#         denom,
#         opt_dict, 
#         self.spatial_lr_scale) = model_args
#         self.training_setup(training_args)
#         self.xyz_gradient_accum = xyz_gradient_accum
#         self.denom = denom
#         self.optimizer.load_state_dict(opt_dict)

#     @property
#     def get_scaling(self):
#         return self.scaling_activation(self._scaling)
    
#     @property
#     def get_rotation(self):
#         return self.rotation_activation(self._rotation)
    
#     @property
#     def get_xyz(self):
#         return self._xyz
    
#     @property
#     def get_opacity(self):
#         return self.opacity_activation(self._opacity)
    
#     def get_covariance(self, scaling_modifier = 1):
#         return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

#     def oneupSHdegree(self):
#         if self.active_sh_degree < self.max_sh_degree:
#             self.active_sh_degree += 1

#     def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
#         self.spatial_lr_scale = spatial_lr_scale
#         fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
#         fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
#         features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
#         features[:, :3, 0 ] = fused_color
#         features[:, 3:, 1:] = 0.0

#         print("Number of points at initialisation : ", fused_point_cloud.shape[0])

#         dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
#         scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
#         rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
#         rots[:, 0] = 1

#         opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

#         self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
#         self._scaling = nn.Parameter(scales.requires_grad_(True))
#         self._rotation = nn.Parameter(rots.requires_grad_(True))
#         self._opacity = nn.Parameter(opacities.requires_grad_(True))
#         self._mask = nn.Parameter(torch.ones((fused_point_cloud.shape[0], 1), device="cuda").requires_grad_(True))
#         self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

#     def training_setup(self, training_args):
#         self.percent_dense = training_args.percent_dense
#         self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#         self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

#         other_params = []
#         for params in self.vq_rot.parameters():
#             other_params.append(params)
#         for params in self.vq_scale.parameters():
#             other_params.append(params)
#         for params in self.recolor.parameters():
#             other_params.append(params)
#         for params in self.recolor_upsample.parameters():
#             other_params.append(params)
#         for params in self.mlp_head.parameters():
#             other_params.append(params)
            
#         l = [
#             {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
#             {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
#             {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
#             {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
#             {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"}
#         ]

#         self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
#         self.optimizer_net = torch.optim.Adam(other_params, lr=training_args.net_lr, eps=1e-15)
#         self.scheduler_net = torch.optim.lr_scheduler.ChainedScheduler(
#         [
#             torch.optim.lr_scheduler.LinearLR(
#             self.optimizer_net, start_factor=0.01, total_iters=100
#         ),
#             torch.optim.lr_scheduler.MultiStepLR(
#             self.optimizer_net,
#             milestones=training_args.net_lr_step,
#             gamma=0.33,
#         ),
#         ]
#         )
#         self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
#                                                     lr_final=training_args.position_lr_final*self.spatial_lr_scale,
#                                                     lr_delay_mult=training_args.position_lr_delay_mult,
#                                                     max_steps=training_args.position_lr_max_steps)

#     def update_learning_rate(self, iteration):
#         ''' Learning rate scheduling per step '''
#         for param_group in self.optimizer.param_groups:
#             if param_group["name"] == "xyz":
#                 lr = self.xyz_scheduler_args(iteration)
#                 param_group['lr'] = lr
#                 return lr

#     def construct_list_of_attributes(self):
#         l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
#         l.append('opacity')
#         for i in range(self._scaling.shape[1]):
#             l.append('scale_{}'.format(i))
#         for i in range(self._rotation.shape[1]):
#             l.append('rot_{}'.format(i))
#         return l

#     def save_ply(self, path):
#         mkdir_p(os.path.dirname(path))

#         xyz = self._xyz.detach().cpu().numpy()
#         normals = np.zeros_like(xyz)
#         opacities = self._opacity.detach().cpu().numpy()
#         scale = self._scaling.detach().cpu().numpy()
#         rotation = self._rotation.detach().cpu().numpy()

#         dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

#         elements = np.empty(xyz.shape[0], dtype=dtype_full)
#         attributes = np.concatenate((xyz, normals, opacities, scale, rotation), axis=1)
#         elements[:] = list(map(tuple, attributes))
#         el = PlyElement.describe(elements, 'vertex')
#         PlyData([el]).write(path)

#     def reset_opacity(self):
#         opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
#         optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
#         self._opacity = optimizable_tensors["opacity"]

#     def load_ply(self, path):
#         plydata = PlyData.read(path)

#         xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                         np.asarray(plydata.elements[0]["y"]),
#                         np.asarray(plydata.elements[0]["z"])),  axis=1)
#         opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

#         scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
#         scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
#         scales = np.zeros((xyz.shape[0], len(scale_names)))
#         for idx, attr_name in enumerate(scale_names):
#             scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

#         rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
#         rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
#         rots = np.zeros((xyz.shape[0], len(rot_names)))
#         for idx, attr_name in enumerate(rot_names):
#             rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

#         self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

#         self.active_sh_degree = self.max_sh_degree

#     def replace_tensor_to_optimizer(self, tensor, name):
#         optimizable_tensors = {}
#         for group in self.optimizer.param_groups:
#             if group["name"] == name:
#                 stored_state = self.optimizer.state.get(group['params'][0], None)
#                 stored_state["exp_avg"] = torch.zeros_like(tensor)
#                 stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

#                 del self.optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
#                 self.optimizer.state[group['params'][0]] = stored_state

#                 optimizable_tensors[group["name"]] = group["params"][0]
#         return optimizable_tensors

#     def _prune_optimizer(self, mask):
#         optimizable_tensors = {}
#         for group in self.optimizer.param_groups:
#             stored_state = self.optimizer.state.get(group['params'][0], None)
#             if stored_state is not None:
#                 stored_state["exp_avg"] = stored_state["exp_avg"][mask]
#                 stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

#                 del self.optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
#                 self.optimizer.state[group['params'][0]] = stored_state

#                 optimizable_tensors[group["name"]] = group["params"][0]
#             else:
#                 group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
#                 optimizable_tensors[group["name"]] = group["params"][0]
#         return optimizable_tensors

#     def prune_points(self, mask):
#         valid_points_mask = ~mask
#         optimizable_tensors = self._prune_optimizer(valid_points_mask)

#         self._xyz = optimizable_tensors["xyz"]
#         self._opacity = optimizable_tensors["opacity"]
#         self._scaling = optimizable_tensors["scaling"]
#         self._rotation = optimizable_tensors["rotation"]
#         self._mask = optimizable_tensors["mask"]

#         self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

#         self.denom = self.denom[valid_points_mask]
#         self.max_radii2D = self.max_radii2D[valid_points_mask]

#     def cat_tensors_to_optimizer(self, tensors_dict):
#         optimizable_tensors = {}
#         for group in self.optimizer.param_groups:
#             assert len(group["params"]) == 1
#             extension_tensor = tensors_dict[group["name"]]
#             stored_state = self.optimizer.state.get(group['params'][0], None)
#             if stored_state is not None:

#                 stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
#                 stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

#                 del self.optimizer.state[group['params'][0]]
#                 group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
#                 self.optimizer.state[group['params'][0]] = stored_state

#                 optimizable_tensors[group["name"]] = group["params"][0]
#             else:
#                 group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
#                 optimizable_tensors[group["name"]] = group["params"][0]

#         return optimizable_tensors

#     def densification_postfix(self, new_xyz, new_opacities, new_scaling, new_rotation, new_mask):
#         d = {"xyz": new_xyz,
#         "opacity": new_opacities,
#         "scaling" : new_scaling,
#         "rotation" : new_rotation,
#         "mask": new_mask}

#         optimizable_tensors = self.cat_tensors_to_optimizer(d)
#         self._xyz = optimizable_tensors["xyz"]
#         self._opacity = optimizable_tensors["opacity"]
#         self._scaling = optimizable_tensors["scaling"]
#         self._rotation = optimizable_tensors["rotation"]
#         self._mask = optimizable_tensors["mask"]

#         self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#         self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
#         self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

#     def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
#         n_init_points = self.get_xyz.shape[0]
#         # Extract points that satisfy the gradient condition
#         padded_grad = torch.zeros((n_init_points), device="cuda")
#         padded_grad[:grads.shape[0]] = grads.squeeze()
#         selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
#         selected_pts_mask = torch.logical_and(selected_pts_mask,
#                                               torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        
#         stds = self.get_scaling[selected_pts_mask].repeat(N,1)
#         means =torch.zeros((stds.size(0), 3),device="cuda")
#         samples = torch.normal(mean=means, std=stds)
#         rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
#         new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
#         new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
#         new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
#         new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
#         new_mask = self._mask[selected_pts_mask].repeat(N,1)

#         self.densification_postfix(new_xyz, new_opacity, new_scaling, new_rotation, new_mask)

#         prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
#         self.prune_points(prune_filter)

#     def densify_and_clone(self, grads, grad_threshold, scene_extent):
#         # Extract points that satisfy the gradient condition
#         selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
#         selected_pts_mask = torch.logical_and(selected_pts_mask,
#                                               torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
#         new_xyz = self._xyz[selected_pts_mask]
#         new_opacities = self._opacity[selected_pts_mask]
#         new_scaling = self._scaling[selected_pts_mask]
#         new_rotation = self._rotation[selected_pts_mask]
#         new_mask = self._mask[selected_pts_mask]

#         self.densification_postfix(new_xyz, new_opacities, new_scaling, new_rotation, new_mask)

#     def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
#         grads = self.xyz_gradient_accum / self.denom
#         grads[grads.isnan()] = 0.0
        
#         self.densify_and_clone(grads, max_grad, extent)
#         self.densify_and_split(grads, max_grad, extent)

#         prune_mask = torch.logical_or((torch.sigmoid(self._mask) <= 0.01).squeeze(),(self.get_opacity < min_opacity).squeeze())
#         if max_screen_size:
#             big_points_vs = self.max_radii2D > max_screen_size
#             big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
#             prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
#         self.prune_points(prune_mask)
#         torch.cuda.empty_cache()
    
#     def mask_prune(self):
#         prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
#         self.prune_points(prune_mask)
#         torch.cuda.empty_cache()

#     def post_quant(self, param, prune=False):
#         max_val = torch.amax(param)
#         min_val = torch.amin(param)
#         param = (param - min_val)/(max_val - min_val)
#         quant = torch.round(param * 255.0) / 255.0
#         out = (max_val - min_val)*quant + min_val
#         if prune:
#             quant = quant*(torch.abs(out) > 0.1)
#             out = out*(torch.abs(out) > 0.1)
#         return torch.nn.Parameter(out), quant
    
#     def huffman_encode(self, param):
#         input_code_list = param.view(-1).tolist()
#         unique, counts = np.unique(input_code_list, return_counts=True)
#         num_freq = dict(zip(unique, counts))

#         codec = HuffmanCodec.from_data(input_code_list)

#         sym_bit_dict = {}
#         for k, v in codec.get_code_table().items():
#             sym_bit_dict[k] = v[0]
#         total_bits = 0
#         for num, freq in num_freq.items():
#             total_bits += freq * sym_bit_dict[num]
#         total_mb = total_bits/8/10**6
#         return total_mb

#     def final_prune(self, compress=False):
#         prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
#         self.prune_points(prune_mask)
#         self._xyz = self._xyz.clone().half().float()
#         self._scaling, sca_idx, _ = self.vq_scale(self.get_scaling.unsqueeze(1))
#         self._rotation, rot_idx, _ = self.vq_rot(self.get_rotation.unsqueeze(1))
#         self._scaling = self._scaling.squeeze()
#         self._rotation = self._rotation.squeeze()
        
#         position_mb = self._xyz.shape[0]*3*16/8/10**6
#         scale_mb = self._xyz.shape[0]*self.rvq_bit*self.rvq_num/8/10**6 + 2**self.rvq_bit*self.rvq_num*3*32/8/10**6
#         rotation_mb = self._xyz.shape[0]*self.rvq_bit*self.rvq_num/8/10**6 + 2**self.rvq_bit*self.rvq_num*4*32/8/10**6
#         opacity_mb = self._xyz.shape[0]*16/8/10**6
#         triplane_mb = torch.prod(torch.tensor(self.recolor.embeddings.shape))
#         for param in self.recolor_upsample.parameters():
#             triplane_mb += torch.prod(torch.tensor(param.shape))
#         triplane_mb = triplane_mb*16/8/10**6
#         mlp_mb = self.mlp_head.params.shape[0]*16/8/10**6
#         sum_mb = position_mb+scale_mb+rotation_mb+opacity_mb+triplane_mb+mlp_mb
        
#         mb_str = "Storage\nposition: "+str(position_mb)+"\nscale: "+str(scale_mb)+"\nrotation: "+str(rotation_mb)+"\nopacity: "+str(opacity_mb)+"\ntriplane: "+str(triplane_mb)+"\nmlp: "+str(mlp_mb)+"\ntotal: "+str(sum_mb)+" MB"
        
#         if compress:
#             self._opacity, quant_opa = self.post_quant(self.get_opacity)
#             self.recolor.embeddings, quant_triplane = self.post_quant(self.recolor.embeddings, True)
        
#             scale_mb = self.huffman_encode(sca_idx) + 2**self.rvq_bit*self.rvq_num*3*32/8/10**6
#             rotation_mb = self.huffman_encode(rot_idx) + 2**self.rvq_bit*self.rvq_num*4*32/8/10**6
#             opacity_mb = self.huffman_encode(quant_opa)
#             triplane_mb = self.huffman_encode(quant_triplane)
#             mlp_mb = self.mlp_head.params.shape[0]*16/8/10**6
#             sum_mb = position_mb+scale_mb+rotation_mb+opacity_mb+triplane_mb+mlp_mb
            
#             mb_str = mb_str+"\n\nAfter PP\nposition: "+str(position_mb)+"\nscale: "+str(scale_mb)+"\nrotation: "+str(rotation_mb)+"\nopacity: "+str(opacity_mb)+"\ntriplane: "+str(triplane_mb)+"\nmlp: "+str(mlp_mb)+"\ntotal: "+str(sum_mb)+" MB"
#         else:
#             self._opacity = self.get_opacity.clone().half().float()
#         torch.cuda.empty_cache()
#         return mb_str
    
#     def precompute(self):
#         xyz = self.contract_to_unisphere(self.get_xyz.half(), torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], device='cuda'))
#         xyz = xyz * 2 - 1
#         # triplane = self.recolor.embeddings
        
#         triplane_lowres = self.recolor.embeddings
#         triplane = self.recolor_upsample(triplane_lowres)
        
#         self._feature = triplane_sample(triplane, xyz)
#         torch.cuda.empty_cache()

#     def add_densification_stats(self, viewspace_point_tensor, update_filter):
#         self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
#         self.denom[update_filter] += 1
        
#     def contract_to_unisphere(self,
#         x: torch.Tensor,
#         aabb: torch.Tensor,
#         ord: int = 2,
#         eps: float = 1e-6,
#         derivative: bool = False,
#     ):
#         aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
#         x = (x - aabb_min) / (aabb_max - aabb_min)
#         x = x * 2 - 1  # aabb is at [-1, 1]
#         mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
#         mask = mag.squeeze(-1) > 1

#         if derivative:
#             dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
#                 1 / mag**3 - (2 * mag - 1) / mag**4
#             )
#             dev[~mask] = 1.0
#             dev = torch.clamp(dev, min=eps)
#             return dev
#         else:
#             x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
#             x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
#             return x
        
# class TriplaneGaussianModel(GaussianModel):
#     use_triplane = True
#     def __init__(self,
#                  model,
#                  hidden_dim=512,
#                  rvq=True,
#                  device='cuda'):
#         super().__init__(model, rvq)
#         self.hidden_dim = hidden_dim
#         self.device = device
#         # self.triplane_encoder = TriplaneEncoding(in_dim=3,
#         #                                 resolution=64,
#         #                                 num_components=1).to(device)
#         self.triplane_tokens = TriplaneTokens().to(device)
#         self.triplane_upsample = nn.ConvTranspose2d(
#             1024, 1, kernel_size=2, stride=2
#         ).to(device)
#         self.transformer = Transformer().to(device)
#         # self.mask_mlp = tcnn.Network(
#         #         n_input_dims=(self.triplane_encoder.num_components),
#         #         n_output_dims=1,
#         #         network_config={
#         #             "otype": "FullyFusedMLP",
#         #             "activation": "ReLU",
#         #             "output_activation": "None",
#         #             "n_neurons": 16,
#         #             "n_hidden_layers": 2,
#         #         },
#         #     )
    
#     def set_instance_embeddings(self, instance_num):
#         self.instance_num = instance_num
#         self.instance_embeddings = torch.randn((instance_num, self.hidden_dim), dtype=torch.float, device=self.device).requires_grad_(True)
    
#     def set_clip_embeddings(self, clip_embeddings):
#         self.clip_embeddings = clip_embeddings
    
#     def set_instance_colors(self, instance_colors):
#         self.instance_colors = instance_colors
    
#     def load_ply(self, path):
#         plydata = PlyData.read(path)

#         xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
#                         np.asarray(plydata.elements[0]["y"]),
#                         np.asarray(plydata.elements[0]["z"])),  axis=1)
#         opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

#         scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
#         scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
#         scales = np.zeros((xyz.shape[0], len(scale_names)))
#         for idx, attr_name in enumerate(scale_names):
#             scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

#         rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
#         rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
#         rots = np.zeros((xyz.shape[0], len(rot_names)))
#         for idx, attr_name in enumerate(rot_names):
#             rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

#         self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
#         self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

#         self.active_sh_degree = self.max_sh_degree
    
#     def save_feature_params(self, path, iter, extra=''):
#         mkdir_p(os.path.dirname(path))
#         # feature_aggregator_ckpt = self.feature_aggregator.state_dict() if self.feature_aggregator else None
#         state = {
#             'instance_embeddings': self.instance_embeddings.detach().cpu().numpy(),
#             # 'triplane_encoder': self.triplane_encoder.state_dict(),
#             'triplane_tokens': self.triplane_tokens.state_dict(),
#             'triplane_upsample': self.triplane_upsample.state_dict(),
#             'transformer': self.transformer.state_dict(),
#             # 'mask_mlp': self.mask_mlp.state_dict(),
#             # 'instance_feature_decoder': self.instance_feature_decoder.state_dict(),
#             'clip_embeddings': self.clip_embeddings,
#             'instance_colors': self.instance_colors.cpu(),
#             'instance_num': self.instance_num,
#         }
#         # # rgb_decode = 'rgb_' if self.rgb_decode else ''
#         # depth_decode = 'depth_' if self.depth_decode else ''
#         save_path = os.path.join(path, f'{extra}_triplane_{iter}.pt')
#         torch.save(state, save_path)
    
#     def load_feature_params(self, params_path):
#         state = torch.load(params_path)
#         self.instance_embeddings = nn.Parameter(torch.tensor(state['instance_embeddings'], dtype=torch.float, device=self.device).requires_grad_(False))
#         self.instance_embeddings = F.normalize(self.instance_embeddings, dim=-1)
#         self.instance_colors = state['instance_colors']
#         self.instance_num = state['instance_num']
#         self.clip_embeddings = state['clip_embeddings']
#         # self.triplane_encoder.load_state_dict(state['triplane_encoder'])
#         self.triplane_tokens.load_state_dict(state['triplane_tokens'])
#         self.triplane_upsample.load_state_dict(state['triplane_upsample'])
#         self.transformer.load_state_dict(state['transformer'])
#         # self.mask_mlp.load_state_dict(state['mask_mlp'])
#         # self.instance_feature_decoder.load_state_dict(state['instance_feature_decoder'])
        
    
#     def feature_training_setup(self, training_args):
#         l = [
#             # {'params': self.triplane_encoder.parameters(), 'lr': training_args.triplane_encoder_lr, "name": "triplane_encoder"},
#             {'params': self.triplane_tokens.parameters(), 'lr': training_args.triplane_tokens_lr, "name": "triplane_tokens"},
#             {'params': self.triplane_upsample.parameters(), 'lr': training_args.triplane_upsample_lr, "name": "triplane_upsample"},
#             {'params': self.transformer.parameters(), 'lr': training_args.transformer_lr, "name": "transformer"},
#             # {'params': self.mask_mlp.parameters(), 'lr': training_args.mask_mlp_lr, "name": "mask_mlp"},
#             {'params': [self.instance_embeddings], 'lr': training_args.instance_embeddings_lr, "name": "instance_embeddings"},
#             # {'params': self.instance_feature_decoder.parameters(), 'lr': training_args.instance_feature_decoder_lr, "name": "instance_feature_decoder"},
#         ]

#         self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        

class LatentGaussianModel(GaussianModel):
    use_triplane = True
    def __init__(self,
                 model,
                 hidden_dim=512,
                 rvq=True,
                 device='cuda'):
        super().__init__(model, rvq)
        self.hidden_dim = hidden_dim
        self.device = device
        self.latent_recolor = tcnn.Encoding(
                 n_input_dims=3,
                 encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": model.max_hashmap,
                    "base_resolution": 16,
                    "per_level_scale": 1.447,
                },
        )
        self.latent_color_head = tcnn.Network(
                n_input_dims=(self.direction_encoding.n_output_dims+self.latent_recolor.n_output_dims),
                n_output_dims=4,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )
    
    def save_params(self, path, iter, extra=''):
        mkdir_p(os.path.dirname(path))
        state = {
            'latent_recolor': self.latent_recolor.state_dict(),
            'latent_color_head': self.latent_color_head.state_dict(),
        }
        save_path = os.path.join(path, f'latent_{iter}.pt')
        torch.save(state, save_path)
    
    def load_params(self, params_path):
        state = torch.load(params_path)
        self.latent_recolor.load_state_dict(state['latent_recolor'])
        self.latent_color_head.load_state_dict(state['latent_color_head'])
        
    
    def latent_training_setup(self, training_args):
        l = [
           
            {'params': self.latent_recolor.parameters(), 'lr': training_args.latent_recolor_lr, "name": "latent_recolor"},
            {'params': self.latent_color_head.parameters(), 'lr': training_args.latent_color_head_lr, "name": "latent_color_head"},
           
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)