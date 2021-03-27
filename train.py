import os
import time
import glob

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from model import FunnyNet, save_model
from util import time_to_string, pil_loader, prepare_image

class FolderDataset(Dataset):
    def __init__(self, data, labels, transforms=None, load_in_ram=False):
        self.transforms = T.Compose([T.ToTensor()])
        if transforms:
            self.transforms = transforms
        self.data = data
        self.labels = labels
        self.load_in_ram = load_in_ram
        self.length = len(self.data)
        if load_in_ram:
            self.pil = [pil_loader(path) for path in data]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.load_in_ram:
            img = self.pil[idx]
        else:
            img = pil_loader(self.data[idx])
        img = self.transforms(img)
        label = self.labels[idx]
        mask = torch.empty(img.shape[1], img.shape[2], dtype=torch.long).fill_(label)
        target = torch.tensor((label), dtype=torch.long)
        return img, mask, target


if __name__ == "__main__":
    MANUAL_SEED = 42
    torch.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed(MANUAL_SEED)
    np.random.seed(MANUAL_SEED)

    save_final = True
    view_results = True
    graph_metrics = True

    # Model Config
    in_channels = 3
    filters = 8  # 16
    activation = "relu"  # relu, leaky_relu, silu, mish

    # Training Hyperparameters
    input_size = 64
    num_epochs = 600

    # Dataloader parameters
    batch_size = 8
    shuffle = True
    num_workers = 3
    drop_last = False

    # Optimization
    optim_type = 'adamw'  # sgd 1e-5, adam 4e-4, adamw 4e-4
    base_lr = 4e-3
    momentum = 0.9
    nesterov = True
    weight_decay = 5e-4  # 0, 1e-5, 3e-5, *1e-4, 3e-4, *5e-4, 3e-4, 1e-3, 1e-2
    scheduler_type = 'step'  # step, plateau
    lr_milestones = [150, 200, 450]  # for step
    lr_gamma = 0.2  # for step
    plateau_patience = 10
    
    # Dataset parameters
    data_root = "data"
    validation_split = 0.1  # percent used for validation as a decimal
    load_in_ram = True
    set_mean = [0, 0, 0]
    set_std = [1, 1, 1]
    train_transforms = T.Compose([
        T.RandomResizedCrop(input_size, scale=(0.9, 1.0), ratio=(3./4., 4./3.)),
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.RandomChoice([
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.01),
            T.ColorJitter(brightness=0.16, contrast=0.15, saturation=0.5, hue=0.04),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ColorJitter(brightness=0.32),
            T.ColorJitter(contrast=0.3),
            T.ColorJitter(saturation=0.5),
            T.ColorJitter(hue=0.04),
        ]),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
    ])
    val_transforms = T.Compose([
        T.Resize(input_size),
        T.CenterCrop(input_size),
        T.ToTensor(),
    ])

    """ Prepare the data """
    # Load the images from the data_root folder
    print(" * Loading data...")
    num_to_cat = {}
    cat_to_num = {}
    path_list = []
    label_list = []
    t0 = time.time()
    for label, folder in enumerate(glob.glob(os.path.join(data_root, '*'))):
        folder_name = os.path.split(folder)[1]
        num_to_cat.update({label: folder_name})
        cat_to_num.update({folder_name: label})
        for img_path in glob.glob(os.path.join(folder, '*')):
            path_list.append(img_path)
            label_list.append(label)
    num_cat = label+1
    print("Number of categories :", num_cat)
    print("Categories :", num_to_cat)
    print(f"{len(path_list)} Images")

    print("Spliting with {:.0f}% for validation.".format(100*validation_split))
    x_train, x_val, y_train, y_val = train_test_split(path_list, label_list, test_size=validation_split, random_state=MANUAL_SEED , stratify=label_list)

    train_dataset = FolderDataset(data=x_train, labels=y_train, transforms=train_transforms, load_in_ram=load_in_ram)
    print("Train length :", len(train_dataset))
    val_dataset = FolderDataset(data=x_val, labels=y_val, transforms=val_transforms, load_in_ram=load_in_ram)
    print("Validation length :", len(val_dataset))

    duration = time.time() - t0
    rate = 'a lot of' if duration==0 else "{:.1f}".format(len(path_list)/duration)
    print("Setup {} images in {}. {} images per second.".format(len(path_list), time_to_string(duration), rate))

    print(" * Creating dataloaders...")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size ,shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last, persistent_workers=(True if num_workers > 0 else False))
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size ,shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last, persistent_workers=(True if num_workers > 0 else False))


    """ Setup the model, optimizer, and scheduler """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device :', device)

    print(" * Creating model...")
    model = FunnyNet(in_channels, num_cat, filters, activation, num_to_cat, set_mean, set_std).to(device)


    # Setup optimizer
    print(" * Creating optimizer...")
    if optim_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    
    # Setup scheduler
    print(" * Creating scheduler...")
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_gamma, patience=plateau_patience, verbose=True)
    else:
        scheduler = None

    # Setup run data
    base_save = f"runs/save"
    save_path = lambda r, t : base_save + '/run{:05d}_{}.pth'.format(r, t)
    if not os.path.exists(base_save):
        os.makedirs(base_save)

    current_run = 0
    with open('runs/LASTRUN.txt') as f:
        current_run = int(f.read()) + 1
    with open('runs/LASTRUN.txt', 'w') as f:
        f.write("%s\n" % current_run)
    print("current run :", current_run)

    """ Training Stage 1 - Segementation """
    print(" * Start training...")

    criterion = nn.CrossEntropyLoss()

    run_stats = []

    t0 = time.time()
    iterations = 0
    model.train()
    for epoch in range(num_epochs):
        t1 = time.time()
        train_loss_total = 0.0
        train_seg_correct_total = 0.0
        train_seg_seen_total = 0.0
        train_cla_correct_total = 0.0
        train_cla_seen_total = 0.0
        for batch_idx, (data, true_masks, true_labels) in enumerate(train_loader):
            data = data.to(device)
            true_masks = true_masks.to(device)
            true_labels = true_labels.to(device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, true_masks)
            loss.backward()
            optimizer.step()
            iterations += 1
            # Update running metrics
            train_loss_total += loss
            _, tags = torch.max(logits, dim=1)
            train_seg_correct_total += (tags == true_masks).sum()
            train_seg_seen_total += true_masks.numel()
            class_logits = torch.mean(torch.mean(logits, dim=2), dim=2)
            _, tags = torch.max(class_logits, dim=1)
            train_cla_correct_total += (tags == true_labels).sum()
            train_cla_seen_total += true_labels.numel()

        model.eval()
        val_loss_total = 0.0
        val_seg_correct_total = 0.0
        val_seg_seen_total = 0.0
        val_cla_correct_total = 0.0
        val_cla_seen_total = 0.0
        for batch_idx, (data, true_masks, true_labels) in enumerate(val_loader):
            data = data.to(device)
            true_masks = true_masks.to(device)
            true_labels = true_labels.to(device)
            logits = model(data)
            loss = criterion(logits, true_masks)
            # Update running metrics
            val_loss_total += loss
            _, tags = torch.max(logits, dim=1)
            val_seg_correct_total += (tags == true_masks).sum()
            val_seg_seen_total += true_masks.numel()
            class_logits = torch.mean(torch.mean(logits, dim=2), dim=2)
            _, tags = torch.max(class_logits, dim=1)
            val_cla_correct_total += (tags == true_labels).sum()
            val_cla_seen_total += true_labels.numel()

        # END EPOCH LOOP
        train_loss = train_loss_total/len(train_loader)
        train_seg_acc = 100*train_seg_correct_total/train_seg_seen_total
        train_cla_acc = 100*train_cla_correct_total/train_cla_seen_total
        val_loss = val_loss_total/len(val_loader)
        val_seg_acc = 100*val_seg_correct_total/val_seg_seen_total
        val_cla_acc = 100*val_cla_correct_total/val_cla_seen_total
        epoch_metrics = {
            "epoch": epoch+1,
            "iterations": iterations,
            "elapsed_time": time.time()-t0,
            "lr": optimizer.param_groups[0]['lr'],
            "train_loss": train_loss.item(),
            "train_seg_acc": train_seg_acc.item(),
            "train_cla_acc": train_cla_acc.item(),
            "val_loss": val_loss.item(),
            "val_seg_acc": val_seg_acc.item(),
            "val_cla_acc": val_cla_acc.item(),
        }
        run_stats.append(epoch_metrics)

        if scheduler:
            if type(scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(train_loss)
            elif type(scheduler) == optim.lr_scheduler.MultiStepLR:
                scheduler.step()

        duration = time.time()-t1
        remaining = duration*(num_epochs-epoch)
        print("epoch {}. {} iters: {}. loss={:.06f}. s={:.2f}%. c={:.2f}%. lr={:.06f}. elapsed={}. remaining={}.".format(epoch+1, iterations, time_to_string(duration), train_loss, train_seg_acc, train_cla_acc, optimizer.param_groups[0]['lr'], time_to_string(time.time()-t0), time_to_string(remaining)))

    print('Finished Stage 1 - Segementation. Duration={}. {} iterations'.format(time_to_string(time.time()-t0), iterations))

    """ Training Stage 1 - Classification """

    if save_final:
        save_model(model, save_path(current_run, "final"))

    if graph_metrics:
        line_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        fig, ax = plt.subplots(2, 2, figsize=(10,8))  # w, h
        font_size = 14

        # loss, top 0
        ax[0][0].set_xlabel('epochs', fontsize=font_size)
        ax[0][0].set_ylabel('loss', fontsize=font_size)
        ax[0][0].set_yscale('log')
        ax[0][0].tick_params(axis='y')
        ax[0][0].plot(range(1, num_epochs+1), [x["train_loss"] for x in run_stats], color=line_colors[0], label="train_loss")
        ax[0][0].plot(range(1, num_epochs+1), [x["val_loss"] for x in run_stats], color=line_colors[0], label="val_loss", linestyle='dashed')
        ax[0][0].legend()

        # acc, top 1
        ax[0][1].set_xlabel('epochs', fontsize=font_size)
        ax[0][1].set_ylabel('accuracy', fontsize=font_size)
        # ax[0][1].set_yscale('log')
        ax[0][1].tick_params(axis='y')
        train_acc = ["train_seg_acc", "train_cla_acc"]
        val_acc = ["val_seg_acc", "val_cla_acc"]
        for i in range(len(train_acc)):
            ax[0][1].plot(range(1, num_epochs+1), [x[train_acc[i]] for x in run_stats], color=line_colors[i], label=train_acc[i])
        for i in range(len(val_acc)):
            ax[0][1].plot(range(1, num_epochs+1), [x[val_acc[i]] for x in run_stats], color=line_colors[i], label=val_acc[i], linestyle='dashed')
        ax[0][1].legend()

        # lr, bot 0
        ax[1][0].set_xlabel('epochs', fontsize=font_size)
        ax[1][0].set_ylabel('lr', fontsize=font_size)
        ax[1][0].set_yscale('log')
        ax[1][0].tick_params(axis='y')
        ax[1][0].plot(range(1, num_epochs+1), [x["lr"] for x in run_stats], color=line_colors[0], label="lr")

        # coef, bot 1
        # ax[1][1].set_xlabel('epochs', fontsize=font_size)
        # ax[1][1].set_ylabel('coefficients', fontsize=font_size)
        # ax[1][1].tick_params(axis='y')
        # coef = ["jaccard", "dice", "tversky"]
        # for i in range(len(coef)):
        #     ax[1][1].plot(range(1, num_epochs+1), [x[coef[i]] for x in run_stats], color=line_colors[i], label=coef[i])
        # ax[1][1].legend()

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.savefig("{}/run{:05d}_metrics.png".format(base_save, current_run), bbox_inches='tight')
        plt.show()

    if view_results:
        rows = 3
        fig, ax = plt.subplots(rows, 1+model.out_channels, figsize=(4*rows,2.5*(1+model.out_channels)))  # w, h
        idxs = np.random.choice(range(len(x_val)), rows, replace=False)
        for i in range(rows):
            img = prepare_image(x_val[idxs[i]])
            y = y_val[idxs[i]]
            ymask = model.predict(img.to(device))
            yclass = torch.mean(torch.mean(ymask, dim=2), dim=2)
            yprob, yhat = torch.max(yclass, dim=1)

            ax[i][0].imshow(img.squeeze().numpy().transpose((1,2,0)))
            ax[i][0].set_title("{} ({:.02f} % {})".format(num_to_cat[int(y)], float(100*yprob), num_to_cat[int(yhat)]))

            for j in range(model.out_channels):
                ax[i][1+j].imshow(ymask.detach().cpu().numpy().squeeze().transpose((1,2,0))[:,:,j])
                ax[i][1+j].set_title("{} Mask".format(num_to_cat[int(j)]))

        fig.tight_layout()
        fig.savefig("{}/run{:05d}_samples.png".format(base_save, current_run), bbox_inches='tight')
        plt.show()
