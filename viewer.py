import os
import glob
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import load_model
from util import pil_loader, prepare_image, get_data


if __name__ == "__main__":

    root = "data"

    model_path = "runs/demo.pth"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device : {}".format(device))

    print("Loading model...")
    model = load_model(model_path, device)

    path_list, label_list, num_to_cat = get_data(root)
    print("Total images :", len(path_list))

    rows = 3
    scale = 3

    fig, ax = plt.subplots(rows, 1+model.out_channels, figsize=(scale*(1+model.out_channels), scale*rows))  # w, h
    idxs = np.random.choice(range(len(path_list)), rows, replace=False)
    for i in range(rows):
        img_path = path_list[idxs[i]]
        print(i, img_path)
        img_color = pil_loader(img_path)
        img = prepare_image(img_color, model.input_size)
        ymask, yclass = model.predict(img.to(device))
        yprob, yhat = torch.max(yclass, dim=1)

        ax[i][0].imshow(img.squeeze().numpy().transpose((1,2,0)))
        ax[i][0].set_title("{} ({:.02f}%)".format(model.num_to_cat[int(yhat)], float(100*yprob)))

        for j in range(model.out_channels):
            ax[i][1+j].imshow(ymask.detach().cpu().numpy().squeeze().transpose((1,2,0))[:,:,j])
            ax[i][1+j].set_title("{} ({:.02f}%)".format(model.num_to_cat[int(j)], float(100*yclass[0,j])))

    fig.tight_layout()
    plt.show()
