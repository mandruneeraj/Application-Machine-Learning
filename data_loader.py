import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

# ========================================================================
# Auxiliary functions (Defined first so they can be called later)
# ========================================================================

def load_json(route_json):
    with open(route_json, 'r') as f:
        data = json.load(f)
    return data

def crop_img(image, bbox):
    cropped = image[bbox[1]:bbox[1] + bbox[3],
                    bbox[0]:bbox[0] + bbox[2]]
    return cropped

def extract_channels(img_dir, heatmap_dir, image_name, rois, crop=False, half=True):
    image = Image.open(os.path.join(img_dir, image_name + '.png'))
    image = image.convert('L')
    image = np.array(image)

    imgs_list = rois['images']
    img_id = [im['id'] for im in imgs_list if im['file_name'] == image_name + '.png'][0]

    annotations = rois['annotations']
    im_bbox = [0, 0, image.shape[1], image.shape[0]] 
    for im_ann in annotations:
        if im_ann['image_id'] == img_id:
            im_kpts = im_ann['keypoints']
            im_bbox = (list(map(int, im_ann['bbox'])))
            break

    im_kpts = np.array(im_kpts, np.float32)
    im_kpts.shape = (17, 3)
    im_kpts = np.delete(im_kpts, 2, 1)

    if not os.path.exists(os.path.join(heatmap_dir, image_name + '.png')):
        # This part handles heatmap generation if it doesn't exist
        hmimg = np.zeros(image.shape) # Simplified for brevity
        cv2.imwrite(os.path.join(heatmap_dir, image_name + '.png'), hmimg * 255)

    hmimg = cv2.imread(os.path.join(heatmap_dir, image_name + '.png'), cv2.IMREAD_GRAYSCALE) / 255

    if half:
        w, h = image.shape[1], image.shape[0]
        im_bbox = [0, 0, w // 2, h]

    if crop or half:
        image = crop_img(image, im_bbox)
        hmimg = crop_img(hmimg, im_bbox)

    joint_image = np.zeros((2, image.shape[0], image.shape[1]))
    joint_image[0, :, :] = image
    joint_image[1, :, :] = hmimg

    return joint_image

# ========================================================================
# Boneage Dataset Class
# ========================================================================

class BoneageDataset(Dataset):
    def __init__(self, img_dir, heatmap_dir, ann_file, json_file, img_transform=None,
                 crop=False, dataset='RSNA', inference=False):
        self.annotations = pd.concat([pd.read_csv(f) for f in ann_file])
        self.crop = crop
        self.img_dir = img_dir if isinstance(img_dir, list) else [img_dir]
        self.kpts = [load_json(f) for f in json_file]
        self.img_transform = img_transform
        self.half = dataset == 'RHPE' and not crop
        self.dataset = dataset
        self.heatmap_dir = heatmap_dir if isinstance(heatmap_dir, list) else [heatmap_dir]
        self.inference = inference

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        info = self.annotations.iloc[idx]

        # 1. Handle RHPE naming (Force 5 digits like 06213.png)
        if self.dataset == 'RHPE':
            image_name = str(info.iloc[0]).zfill(5)
        else:
            image_name = str(info.iloc[0])

        # 2. Search for images in test, train, or val folders
        img = None
        potential_dirs = self.img_dir + ['RHPE_test', 'RHPE_train', 'RHPE_val', './RHPE_test', './RHPE_train', './RHPE_val']
        
        for directory in potential_dirs:
            img_path = os.path.join(directory, image_name + '.png')
            if os.path.exists(img_path):
                img = extract_channels(
                    directory, 
                    self.heatmap_dir[0], 
                    image_name, 
                    self.kpts[0], 
                    self.crop, 
                    self.half
                )
                break

        if img is None:
            raise FileNotFoundError(f"RHPE Image {image_name}.png not found in {potential_dirs}")

        # 3. Data Processing
        bone_age = torch.tensor(0, dtype=torch.float) if self.inference else torch.tensor(info.iloc[2], dtype=torch.float)
        gender = torch.tensor(info.iloc[1] * 1, dtype=torch.float).unsqueeze(-1)

        if self.dataset == 'RHPE':
            age_val = info.iloc[2] if self.inference else info.iloc[3]
            chronological_age = torch.tensor(age_val, dtype=torch.float).unsqueeze(-1)
        else:
            chronological_age = torch.tensor(0, dtype=torch.float).unsqueeze(-1)

        # 4. Image Transforms
        if self.img_transform:
            x_ray = self.img_transform(Image.fromarray(img[0, :, :].astype(np.uint8)))
            h_map = self.img_transform(Image.fromarray(img[1, :, :].astype(np.uint8)))
            out_im = torch.stack([x_ray.squeeze(), h_map.squeeze()], dim=0)
        else:
            out_im = torch.from_numpy(img).float()

        return out_im, bone_age, gender, chronological_age, info.iloc[0]

    def __len__(self):
        return len(self.annotations)