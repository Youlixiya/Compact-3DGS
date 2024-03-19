import os
from typing import Any
import cv2
import open_clip
import torch
import math
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class MaskDataset(Dataset):
    def __init__(self,
                 source_root,
                 cameras,
                 mask_dir='masks_4',
                 clip_model_type='ViT-B-16',
                 clip_model_pretrained='laion2b_s34b_b88k',
                 semantic_similarity=0.95,
                 device='cuda'
                 ):
        self.source_root = source_root
        self.mask_dir = mask_dir
        self.semantic_similarity = semantic_similarity
        self.img_dir = mask_dir.replace('masks', 'images')
        img_suffix = os.listdir(os.path.join(source_root, self.img_dir))[0].split('.')[-1]
        self.imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in cameras]
        self.masks_path = os.path.join(source_root, f'{mask_dir}.npz')
        if os.path.exists(self.masks_path):
            colors_masks = np.load(self.masks_path, allow_pickle=True)['arr_0'].tolist()
            self.instance_masks = colors_masks['instance_masks']
            self.instance_colors = colors_masks['instance_colors']
            self.clip_embeddings = colors_masks['clip_embeddings']
        else:
            self.masks_name = [img_name.split('.')[0] + '.png' for img_name in self.imgs_name]
            self.masks_path = [os.path.join(source_root, mask_dir, 'Annotations', mask_name) for mask_name in self.masks_name]
            img_dir = mask_dir.replace('masks', 'images')
            self.imgs_path = [os.path.join(source_root, img_dir, img_name) for img_name in self.imgs_name]
            self.masks = torch.from_numpy(np.stack([cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB) for mask_path in self.masks_path], axis=0)).to(device)
            self.imgs = np.stack([cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in self.imgs_path], axis=0)
            self.device = device           
            self.clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_type,
                                                                                     pretrained=clip_model_pretrained,
                                                                                     precision="fp16",
                                                                                     device=device)
            clip_ckpt = torch.load(clip_model_pretrained)
            self.clip_model.load_state_dict(clip_ckpt)
            # self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            #     clip_model_type,  # e.g., ViT-B-16
            #     pretrained=clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            #     precision="fp16",
            # )
            self.clip_model = self.clip_model.to(device)
            # self.tokenizer = open_clip.get_tokenizer(clip_model_type)
            # self.clip_model, self.preprocess = clip.load(clip_type, device=device)
            self.set_colors()
            self.asign_masks()
            del self.masks
            self.masks = None
            save_path = os.path.join(source_root, f'{mask_dir}.npz')
            np.savez_compressed(save_path, {'instance_masks': self.instance_masks,
                                            'instance_colors':self.instance_colors,
                                            'clip_embeddings':self.clip_embeddings})
            
        # print(torch.sum(self.mask_labels))
        torch.cuda.empty_cache()
    
    def __len__(self):
        return len(self.instance_masks)
    
    def __getitem__(self, idx):
        # return self.masks[idx]
        return self.instance_masks[idx]
    
    @torch.no_grad()
    def set_colors(self):
        self.instance_colors = []
        for mask in tqdm(self.masks):
            self.instance_colors += torch.unique(mask.reshape(-1, 3), dim=0).tolist()
        self.instance_colors = torch.unique(torch.tensor(self.instance_colors, dtype=torch.uint8), dim=0)
        # self.semantic_colors = []
        
    @torch.no_grad()
    def asign_masks(self):
        n, h, w, c = self.masks.shape
        # self.semantic_masks = torch.zeros((n, h, w, 1), device=self.device)
        self.instance_masks = torch.zeros((n, h, w, 1), device=self.device)
        self.clip_embeddings = []
        for i in trange(len(self.instance_colors)):
        # for i in trange(3):
            mask = self.masks.reshape(n * h * w, -1)
            # print(self.unique_colors[i])
            target_color = self.instance_colors[i][None, :].to(self.device).expand(mask.size())
            indexs = torch.eq(mask, target_color).all(dim=1).reshape(n, h, w)
            # print(torch.sum(index.float()))
            # index = (self.masks.reshape(n * h * w, -1) == self.unique_colors[i].to(self.device)).reshape(n, h, w)
            self.instance_masks[indexs, :] = i
            tmp_clip_embedding = None
            for j, index in enumerate(indexs):
                if torch.sum(index) > 0:
                    mask_embedding = self.get_clip_embedding(index.cpu().numpy().copy(), self.imgs[j].copy())
                    if tmp_clip_embedding is None:
                        tmp_clip_embedding = mask_embedding
                    else:
                        semantic_similarity = torch.nn.functional.cosine_similarity(mask_embedding[None], tmp_clip_embedding[None])
                        if semantic_similarity >= self.semantic_similarity:
                            tmp_clip_embedding = torch.nn.functional.normalize(tmp_clip_embedding + mask_embedding, dim=-1)
                    # tmp_clip_embedding.append(mask_embedding)
            # mask_embedding = torch.nn.functional.normalize(torch.stack(tmp_clip_embedding).mean(0), dim=-1)
            # self.clip_embeddings.append(mask_embedding)
            self.clip_embeddings.append(tmp_clip_embedding)
        
        self.instance_masks = self.instance_masks.long()
        # self.semantic_masks = self.semantic_masks.long()
        # self.semantic_colors = torch.stack(self.semantic_colors)
        self.clip_embeddings = torch.stack(self.clip_embeddings)
        
        # semantic_mask_rgb_save_path = os.path.join(self.source_root, self.mask_dir, 'SemanticMasks')
        # os.makedirs(semantic_mask_rgb_save_path, exist_ok=True)
        # for i in trange(len(self.semantic_masks)):
        #     semantic_mask_rgb = Image.fromarray(self.semantic_colors[self.semantic_masks[i].cpu().reshape(h * w)].reshape(h, w, 3).numpy())
        #     semantic_mask_rgb.save(os.path.join(semantic_mask_rgb_save_path, self.imgs_name[i]))
    
    @torch.no_grad()
    def get_box_by_mask(self, mask):
        non_zero_indices = torch.nonzero(mask.float())
        min_indices = torch.min(non_zero_indices, dim=0).values
        max_indices = torch.max(non_zero_indices, dim=0).values
        top_left = min_indices
        bottom_right = max_indices + 1
        return top_left[1].item(), top_left[0].item(), bottom_right[1].item(), bottom_right[0].item()
    
    # @torch.no_grad()
    # def get_clip_embedding(self, mask, img):
    #     img_tensor = self.preprocess(Image.fromarray(img)).half().to(self.device)
    #     mask_tensor = self.mask_transform((mask * 255).astype(np.uint8))[None].half().to(self.device)
    #     mask_image_feature = self.clip_model.visual(img_tensor, mask_tensor).squeeze(0)
    #     # mask_image_feature = mask_image_feature / mask_image_feature.norm(dim=-1, keepdim=True)
    #     mask_image_feature = torch.nn.functional.normalize(mask_image_feature, dim=-1)
    #     # x1, y1, x2, y2 = self.get_box_by_mask(mask)
    #     # img[~mask.cpu().numpy(), :] = np.uint8([255, 255, 255])
    #     # img_patch = Image.fromarray(img[y1:y2, x1:x2, :])
    #     # img_patch_tensor = self.preprocess(img_patch).unsqueeze(0).cuda().half()
    #     # img_patch_feature = self.clip_model.encode_image(img_patch_tensor).squeeze(0)
    #     # img_patch_feature = torch.nn.functional.normalize(img_patch_feature, dim=-1)
    #     return mask_image_feature
    
    @torch.no_grad()
    def get_clip_embedding(self, mask, img):
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).cpu()
        mask_tensor = torch.tensor(mask)[None].cpu()
        (regions, unnorm_regions), region_masks = self.preprocess(img_tensor, mask_tensor)
        mask_image_feature = self.clip_model.encode_image(regions.cuda(), region_masks.cuda()).squeeze(0)
        # mask_image_feature = mask_image_feature / mask_image_feature.norm(dim=-1, keepdim=True)
        mask_image_feature = torch.nn.functional.normalize(mask_image_feature, dim=-1)
        # x1, y1, x2, y2 = self.get_box_by_mask(mask)
        # img[~mask.cpu().numpy(), :] = np.uint8([255, 255, 255])
        # img_patch = Image.fromarray(img[y1:y2, x1:x2, :])
        # img_patch_tensor = self.preprocess(img_patch).unsqueeze(0).cuda().half()
        # img_patch_feature = self.clip_model.encode_image(img_patch_tensor).squeeze(0)
        # img_patch_feature = torch.nn.functional.normalize(img_patch_feature, dim=-1)
        return mask_image_feature
    
    # @torch.no_grad()
    # def get_clip_embedding(self, mask, img):
    #     x1, y1, x2, y2 = self.get_box_by_mask(mask)
    #     img[~mask.cpu().numpy(), :] = np.uint8([255, 255, 255])
    #     img_patch = Image.fromarray(img[y1:y2, x1:x2, :])
    #     img_patch_tensor = self.preprocess(img_patch).unsqueeze(0).cuda().half()
    #     img_patch_feature = self.clip_model.encode_image(img_patch_tensor).squeeze(0)
    #     img_patch_feature = torch.nn.functional.normalize(img_patch_feature, dim=-1)
    #     return img_patch_feature
        
            
            
    
   
    
    
        

# if __name__ == '__main__':
    # extractor = AutoEmbeddingExtractor('Grounded-SAM/weights/sam_vit_h_4b8939.pth', 'vit_h', 
    #                               'data/360_v2/garden/images_4', )
    # extractor.run()
    # extractor = DINOV2Extractor('data/360_v2/garden/images_4', )
    # extractor.run()
    
            
            