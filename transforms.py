import time

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

from util import get_data, np_loader


if __name__ == "__main__":

    root = "data/memes256"

    path_list, label_list, num_to_cat = get_data(root)
    print("Total images :", len(path_list))

    input_size = 96

    train_transforms = A.Compose([
        # Resizing
        A.RandomResizedCrop(input_size, input_size, scale=(0.5, 1.0), ratio=(3./4., 4./3.), interpolation=cv2.INTER_LINEAR),
        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        # A.RandomRotate90(p=0.25),
        # A.OneOf([
        #     # A.GridDistortion(num_steps=5, distort_limit=0.03, always_apply=True),
        #     A.IAAPerspective(scale=(0.05, 0.1), keep_size=True, always_apply=True),
        #     A.IAAAffine(shear=(-15, 15), always_apply=True),
        # ], p=1.0),
        # Color transforms
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, always_apply=True),
            # A.Posterize(num_bits=[4,6], always_apply=True),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, always_apply=True),
            A.RandomGamma(gamma_limit=(50, 150), always_apply=True),
            A.ToGray(p=0.4),
       ], p=0.5),
        # Blurring and sharpening
        A.OneOf([
            A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True),
            A.GaussianBlur(blur_limit=(1, 5), always_apply=True),
            # A.GlassBlur(sigma=0.7, max_delta=4, iterations=1, always_apply=True, mode='fast'),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=True),
            # A.MedianBlur(blur_limit=5, always_apply=True),
            # A.MotionBlur(blur_limit=3, always_apply=True),
            # A.RandomShadow(shadow_roi=(0, 0.0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=3, always_apply=True),    
        ], p=0.5),
        A.GaussNoise(var_limit=5.0, mean=0, p=1.0),
        # ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.SmallestMaxSize(input_size),
        A.CenterCrop(input_size, input_size),
        # ToTensorV2(),
    ])


    cols = 6
    rows = 3
    scale = 3

    img_path = np.random.choice(path_list)
    print("Path :", img_path)
    img_color = np_loader(img_path)
    print("img_color :", img_color.shape)

    # ten = ToTensorV2()(image=img_color)["image"]
    # print(ten.unsqueeze_(0).shape)
    total = 0

    fig = plt.figure(figsize=(scale*cols, scale*rows))
    for i in range(1, cols*rows+1):
        if i==1:
            img = val_transforms(image=img_color)["image"]
        else:
            t0 = time.time()
            img_color = np_loader(img_path)
            img = train_transforms(image=img_color)["image"]
            dt = time.time() - t0
            total += dt
            rate = 1/dt if dt!=0 else 'a lot'
            # print("dt={}. {} per second".format(dt, rate))
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)

    dt = total/(cols*rows-1)
    rate = 1/dt
    print("avg dt={}. {} per second".format(dt, rate))
    
    fig.tight_layout()
    plt.show()
