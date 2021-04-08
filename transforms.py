import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np

from util import get_data, np_loader


if __name__ == "__main__":

    root = "data"
    root = r"C:\Users\LUKE_SARGEN\projects\classifier\data\subset"

    path_list, label_list, num_to_cat = get_data(root)
    print("Total images :", len(path_list))

    input_size = 256

    train_transforms = A.Compose([
        # Resizing
        A.RandomResizedCrop(input_size, input_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), interpolation=cv2.INTER_LINEAR),
        # Spatial transforms
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.Rotate(limit=45, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        # A.RandomRotate90(p=1.0),
        A.OneOf([
            A.OpticalDistortion(distort_limit=0.05, shift_limit=0.0, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, always_apply=True, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, always_apply=True, p=0.5),
            A.IAAPerspective(scale=(0.05, 0.1), keep_size=True, always_apply=True, p=0.5),
       ], p=0.5),
        # Color transforms
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, always_apply=True, p=0.5),
            A.Posterize(num_bits=[4,6], always_apply=True, p=0.5),
            A.RGBShift(r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, always_apply=True, p=0.5),
            A.RandomGamma(gamma_limit=(50, 150), always_apply=True, p=0.5),
       ], p=0.5),
        A.RandomShadow(shadow_roi=(0, 0.0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=3, always_apply=False, p=0.5),
        # Blurring and sharpening
        A.OneOf([
            A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True, p=0.5),
            A.GaussianBlur(blur_limit=(1, 7), always_apply=True, p=0.5),
            A.GaussNoise(var_limit=100.0, mean=0, always_apply=True, p=0.5),
            A.GlassBlur(sigma=0.7, max_delta=4, iterations=1, always_apply=True, mode='fast', p=0.5),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=True, p=0.5),
            A.MedianBlur(blur_limit=5, always_apply=True, p=0.5),
            A.MotionBlur(blur_limit=5, p=1.0),
        ], p=0.5),
        # A.ToGray(p=0.1),
        # ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.SmallestMaxSize (input_size),
        A.CenterCrop(input_size, input_size),
        # ToTensorV2(),
    ])


    cols = 4
    rows = 3
    scale = 3

    img_path = np.random.choice(path_list)
    print("Path :", img_path)
    img_color = np_loader(img_path)

    # ten = ToTensorV2()(image=img_color)["image"]
    # print(ten.unsqueeze_(0).shape)

    fig = plt.figure(figsize=(scale*cols, scale*rows))
    for i in range(1, cols*rows+1):
        if i==1:
            img = val_transforms(image=img_color)["image"]
        else:
            img = train_transforms(image=img_color)["image"]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
    fig.tight_layout()
    plt.show()
