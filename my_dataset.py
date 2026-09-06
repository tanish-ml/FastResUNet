""" This is Data Loading script for Pascal VOC 2012 Segmentation Dataset. 
It includes a custom Dataset class with joint transformations 
for data augmentation during training. The get_dataloader function returns 
DataLoader objects for both training and validation sets. """

import os
import random
import torch
import numpy as np
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from torchvision import transforms


class JointTransform:
    def __init__(self, img_size=(448, 448)):
        self.img_size = img_size

    def __call__(self, img, mask):

        # =========================
        # 1. RANDOM SCALE (FIRST!)
        # =========================
        scale = random.uniform(0.75, 1.5)
        w, h = img.size
        new_w, new_h = int(w * scale), int(h * scale)

        img = TF.resize(img, (new_h, new_w))
        mask = TF.resize(mask, (new_h, new_w), interpolation=TF.InterpolationMode.NEAREST)

        # =========================
        # 2. RANDOM CROP (MAIN STEP)
        # =========================
        pad_w = max(0, self.img_size[1] - new_w)
        pad_h = max(0, self.img_size[0] - new_h)

        if pad_w > 0 or pad_h > 0:
            img = TF.pad(img, (0, 0, pad_w, pad_h), fill=0)
            mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=255)

        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.img_size)
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # =========================
        # 3. GEOMETRIC AUGMENTATIONS
        # =========================
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        if random.random() > 0.5:
            angle = random.randint(-15, 15)  # 🔥 reduce from 20 → 15
            img = TF.rotate(img, angle)
            mask = TF.rotate(mask, angle,
                             interpolation=TF.InterpolationMode.NEAREST,
                             fill=255)

        # =========================
        # 4. COLOR AUGMENTATIONS
        # =========================
        if random.random() > 0.5:
            img = transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.05
            )(img)

        if random.random() > 0.7:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2)))

        # =========================
        # 5. TO TENSOR
        # =========================
        img = TF.to_tensor(img)
        img = TF.normalize(img,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

        mask = torch.as_tensor(np.array(mask), dtype=torch.long)

        return img, mask

class PascalVOCSegmentation(Dataset):
    def __init__(self, root_dir, image_set='train', img_size=(224,224)):
        self.root_dir = root_dir
        self.is_train = (image_set == 'train')
        self.joint_transform = JointTransform(img_size) if self.is_train else None
        self.img_size = img_size
        
        image_dir = os.path.join(self.root_dir, 'JPEGImages')
        mask_dir = os.path.join(self.root_dir, 'SegmentationClass')
        splits_dir = os.path.join(self.root_dir, 'ImageSets', 'Segmentation')
        
        split_f = os.path.join(splits_dir, f"{image_set}.txt")
        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index])

        if self.is_train:
            img, mask = self.joint_transform(img, mask)
        else:
            # Validation transforms (No Augmentations)
            img = TF.resize(img, self.img_size)
            mask = TF.resize(mask, self.img_size, interpolation=TF.InterpolationMode.NEAREST)
            img = TF.to_tensor(img)
            img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            mask = torch.as_tensor(np.array(mask), dtype=torch.long)
            
        return img, mask

def get_dataloader(root_dir, batch_size=4, img_size=(224, 224)):
    train_ds = PascalVOCSegmentation(root_dir, 'train', img_size)
    val_ds = PascalVOCSegmentation(root_dir, 'val', img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader