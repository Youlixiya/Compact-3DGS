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
from torch import nn
import open_clip
from scene import Scene
import os
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Type, Tuple
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, TriplaneGaussianModel

@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    # negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    negatives: Tuple[str] = ("background")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        with torch.no_grad():
            processed_input = self.process(input).half().cuda().unsqueeze(0)
            return self.model.encode_image(processed_input)

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mask_background, args, masks_index):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    mask_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask_renders")
    mask_image_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mask_image_renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(mask_render_path, exist_ok=True)
    makedirs(mask_image_render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # mask_index = args.mask_index
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
        rendered_masks = []
        for mask_index in masks_index:
            rendered_mask = render(view, gaussians, pipeline, mask_background, render_feature=True, triplane_index=mask_index)['render_feature']
            rendered_mask = rendered_mask > 0.8
            rendered_masks.append(rendered_mask)
        rendered_masks = torch.cat(rendered_masks, 0)
        rendered_masks = torch.sum(rendered_masks, dim=0) > 0
        rendered_mask_image = rendering.clone()
        rendered_mask_image[:, ~rendered_masks] = torch.tensor([[1, 1, 1]], device='cuda', dtype=torch.float32).permute(1, 0)
        rendered_masks = rendered_masks.float()
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendered_masks, os.path.join(mask_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendered_mask_image, os.path.join(mask_image_render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, args):
    with torch.no_grad():
        gaussians = TriplaneGaussianModel(dataset, rvq=False)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        params_path = os.path.join(dataset.model_path,
                            "point_cloud",
                            "iteration_" + str(30000),
                            f"mask_triplane_{1000}.pt")
        gaussians.load_feature_params(params_path)
        
        gaussians.precompute()
        clip = OpenCLIPNetwork(OpenCLIPNetworkConfig())
        clip.set_positives(args.text_prompt.split(','))
        relevancy = clip.get_relevancy(gaussians.clip_embeddings, 0)[..., 0]
        if args.top_one:
            masks_index = torch.argmax(relevancy).unsqueeze(0)
        else:
            masks_index = torch.nonzero(relevancy > args.mask_threshold)
        print(masks_index)
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        mask_bg_color = [0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        mask_background = torch.tensor(mask_bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, mask_background, args, masks_index)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, mask_background, args, masks_index)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--mask_threshold", default=0.5, type=float)
    parser.add_argument("--top_one", default=True, type=bool)
    parser.add_argument("--text_prompt", default='', type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args)