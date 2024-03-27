import os
from typing import Any
import torch
import math
import numpy as np
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset

def preprocess(images):
    if type(images) == list:
        images_pt = []
        for image in images:
            images_pt.append(T.ToTensor()(image) * 2 - 1)
        return torch.stack(images_pt)
    else:
        return T.ToTensor()(images).unsqueeze(0) * 2 - 1

def postprocess(latent_image, pil=True):
    latent_image = (latent_image + 1) / 2
    if pil:
        latent_image = latent_image[0].permute(1, 2, 0)
        return Image.fromarray((latent_image * 255).cpu().detach().numpy().astype(np.uint8))

class LatentImageDataset(Dataset):
    def __init__(self,
                 source_root,
                 cameras,
                 vae,
                 latent_dir='latents_4',
                 device='cuda'
                 ):
        self.source_root = source_root
        self.latent_dir = latent_dir
        self.img_dir = latent_dir.replace('latents', 'images')
        img_suffix = os.listdir(os.path.join(source_root, self.img_dir))[0].split('.')[-1]
        self.imgs_name = [f'{camera.image_name}.{img_suffix}' for camera in cameras]
        self.latents_path = os.path.join(source_root, f'{latent_dir}.pt')
        self.imgs_path = [os.path.join(source_root, self.img_dir, img_name) for img_name in self.imgs_name]
        imgs = [Image.open(img_path) for img_path in self.imgs_path]
        w, h = imgs[0].size
        low_w = w // 8
        low_h = h // 8
        w = low_w * 8
        h = low_h * 8
        self.trans_low = T.Compose(
            [T.Resize((low_h, low_w)),
            T.ToTensor()]
        )
        self.trans = T.Compose(
            [T.Resize((h, w)),
            T.ToTensor()]
        )
        if os.path.exists(self.latent_dir):
            self.latent_images_dict = torch.load(self.latents_path)
        else: 
            # self.imgs = np.stack([cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in self.imgs_path], axis=0)
            self.device = device           
            self.latent_images = []
            for img in imgs:
                w, h = img.size
                w = (w // 8) * 8
                h = (h // 8) * 8
                image_pt = preprocess([img.resize((w, h))])
                encode_out = vae.encode(image_pt.to(device=vae.device, dtype=vae.dtype))
                latent_image = encode_out.latent_dist.mode()
                self.latent_images.append(latent_image)
            self.latent_images_dict = {}
            for i in range(len(self.latent_images)):
                self.latent_images_dict[cameras[i].image_name] = self.latent_images[i]
            # self.latent_images = torch.cat(self.latent_images)
            torch.save(self.latent_images_dict, self.latents_path)
            
        torch.cuda.empty_cache()
        self.imgs = torch.stack([self.trans(img) for img in imgs])
        self.imgs_low = torch.stack([self.trans_low(img) for img in imgs])
        self.imgs_dict = {}
        self.imgs_low_dict = {}
        for i in range(len(self.imgs)):
            self.imgs_dict[cameras[i].image_name] = self.imgs[i]
            self.imgs_low_dict[cameras[i].image_name] = self.imgs_low[i]
        # print(self.imgs_dict)
    
    def __len__(self):
        return len(self.latent_images)
    
    def __getitem__(self, idx):
        return self.latent_images[idx]
    
            
            