import cv2
import tifffile as tiff
from torch.utils.data import Dataset, DataLoader, sampler
import albumentations as aug
from albumentations import (HorizontalFlip, VerticalFlip, ShiftScaleRotate, Normalize, Resize, Compose,Cutout, GaussNoise, RandomRotate90, Transpose, RandomBrightnessContrast, RandomCrop)
from albumentations.pytorch import ToTensor
from albumentations.augmentations.transforms import CropNonEmptyMaskIfExists, RandomResizedCrop
import numpy as np

class EndoDataset(Dataset):
    def __init__(self, phase, shape = 512, crop_type=0, train_size = 474, val_size = 99):
        self.transforms = get_transforms(phase, crop_type=crop_type, size=shape)
        self.phase = phase
        self.shape = shape
        self.train_size = train_size
        self.val_size = val_size

    def __getitem__(self, idx):
        if self.phase == 'train':
            mask = tiff.imread('./EndoCV/ead2020_semantic_segmentation_TRAIN/masks_ead2020/EAD2020_semantic_'+"{:05d}".format(idx)+'.tif')
            img = cv2.imread('./EndoCV/ead2020_semantic_segmentation_TRAIN/images_ead2020/EAD2020_semantic_'+"{:05d}".format(idx)+'.jpg')
        else:
            mask = tiff.imread('./EndoCV/EAD2020-Phase-II-Segmentation-VALIDATION/semanticMasks/EAD2020_MP1'+"{:04d}".format(idx)+'_mask.tif')
            img = cv2.imread('./EndoCV/EAD2020-Phase-II-Segmentation-VALIDATION/originalImages/EAD2020_MP1'+"{:04d}".format(idx)+'.jpg')

        if self.phase == 'train':
            if img.shape[0]<self.shape or img.shape[1]<self.shape:
                (h, w) = img.shape[:2]
                if h>w:
                    re_w = self.shape+12
                    r = float(re_w)/ w 
                    dim = (re_w, int(h * r))
                else:
                    re_h = self.shape+12
                    r = float(re_h)/ h 
                    dim = (int(w * r), re_h)
                img = cv2.resize(img, dim)
                mask_re = np.zeros((5, dim[1], dim[0]))
                for i in range(5):
                    mask_re[i] = cv2.resize(mask[i], dim, interpolation = cv2.INTER_NEAREST)
                mask = (mask_re.transpose(1,2,0) > 0).astype('int')
            else:
                mask = (mask.transpose(1,2,0) > 0).astype('int')
        else:
            (H, W) = img.shape[:2]
            mask = mask[:5, ...]
            pad_h = 0 if (H%128)==0 else 128-(H%128)
            pad_w = 0 if (W%128)==0 else 128-(W%128)
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)))
            mask = np.pad(mask, ((0, 0), (0, pad_h), (0, pad_w)))
            mask = (mask.transpose(1, 2, 0) > 0).astype('int')
        
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        mask = mask[0].permute(2, 0, 1)
        return img, mask

    def __len__(self):
        if self.phase == 'train':
            return self.train_size
        else:
            return self.val_size

def get_transforms(phase, crop_type=0, size=512):
    list_transforms = []
    if phase == "train":
        list_transforms.extend([
             aug.Flip(),
             aug.Cutout(num_holes=4, p=0.5),
             aug.OneOf([
                 aug.RandomContrast(),
                 aug.RandomGamma(),
                 aug.RandomBrightness(),
                 ], p=1),

             aug.ShiftScaleRotate(rotate_limit=90),
             aug.OneOf([
                    aug.GaussNoise(p=.35),
                    ], p=.5),
            ])
        if crop_type==0:
            list_transforms.extend([
                CropNonEmptyMaskIfExists(size, size)
            ])

        elif crop_type==1:
            list_transforms.extend([
                RandomResizedCrop(size, size, scale=1, ratio=1)
            ])

    list_transforms.extend(
        [
            Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225], p=1),
            ToTensor(),
        ]
    )
    list_trfms = Compose(list_transforms)
    return list_trfms

def provider(phase, shape, crop_type, batch_size=8, num_workers=4):
    '''Returns dataloader for the model training'''
    if phase == 'train':
        image_dataset = EndoDataset(phase, shape=shape, crop_type=crop_type)
    else:
        image_dataset = EndoDataset(phase, shape=shape, crop_type=crop_type)
        
    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,   
    )

    return dataloader