import os
import glob

import numpy as np
from PIL import Image
import torchvision.transforms as T


def get_data(root):

    # TODO : check for train/val folders

    num_to_cat = {}
    path_list = []
    label_list = []
    for label, folder in enumerate(glob.glob(os.path.join(root, '*'))):
        folder_name = os.path.split(folder)[1]
        num_to_cat.update({label: folder_name})
        for img_path in glob.glob(os.path.join(folder, '*')):
            path_list.append(img_path)
            label_list.append(label)
    return path_list, label_list, num_to_cat


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        return Image.open(f).convert('RGB')


def custom_resize(img, size):
    width, height = img.size[:2]
    # print("width, height :", width, height)
    scale = size/min(width, height)
    # print("scale :", scale)
    img = img.resize((int(width*scale), int(height*scale)))
    return img


def prepare_image(img, size=None):
    if size:
        img = custom_resize(img, size)
    return T.ToTensor()(img).unsqueeze_(0)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))


def normalize(data):
    vmin = np.min(data)
    vmax = np.max(data)
    return (data - vmin) / (vmax - vmin + 1e-7)


def time_to_string(t):
    if t > 3600: return "{:.2f} hours".format(t/3600)
    if t > 60: return "{:.2f} minutes".format(t/60)
    else: return "{:.2f} seconds".format(t)

