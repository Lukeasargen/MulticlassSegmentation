import os
import time

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp
import torchvision.transforms as T

from model import SegmentationModel, save_model
from util import time_to_string, prepare_image, get_data, pil_loader, np_loader, add_weight_decay


class FolderDataset(Dataset):
    def __init__(self, data, labels, transforms=None, load_in_ram=False):
        self.transforms = T.Compose([T.ToTensor()])
        if transforms:
            self.transforms = transforms
        self.data = data
        self.labels = labels
        self.load_in_ram = load_in_ram
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.load_in_ram:
            img = self.data[idx]
        else:
            img = np_loader(self.data[idx])
        img = self.transforms(image=img)["image"] / 255.0
        label = self.labels[idx]
        mask = torch.empty(img.shape[1], img.shape[2], dtype=torch.long).fill_(label)
        target = torch.tensor((label), dtype=torch.long)
        return img, mask, target


if __name__ == "__main__":
    MANUAL_SEED = 42
    np.random.seed(MANUAL_SEED)
    torch.manual_seed(MANUAL_SEED)
    torch.cuda.manual_seed(MANUAL_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    save_epochs = True  # Write a single copy that gets updated every epoch, like a checkpoint that gets overwritten each epoch
    graph_metrics = True
    view_results = True
    checkpoint_epoch = 20  # epoch save interval
    use_amp = True

    # Model Config
    in_channels = 3
    filters = 16
    activation = "silu"  # relu, leaky_relu, silu, mish

    # Training Hyperparameters
    input_size = 128
    num_epochs = 100

    # Dataloader parameters
    batch_size = 256
    shuffle = True
    num_workers = 6
    drop_last = False
    pin_memory = True

    # Optimization
    optim_type = 'adamw'  # sgd 1e-5, adam 4e-4, adamw 4e-4
    base_lr = 4e-4
    momentum = 0.98
    nesterov = True
    weight_decay = 1e-5  # 0, 1e-5, 3e-5, *1e-4, 3e-4, *5e-4, 3e-4, 1e-3, 1e-2
    scheduler_type = 'plateau'  # step, plateau, exp
    lr_milestones = [150, 180]  # for step
    lr_gamma = 0.8
    plateau_patience = 20
    use_classifer_grad = True  # Uses the classifer gradients to update the encoder
    class_multi = 100.0
    cutmix_beta = 0.5

    # Dataset parameters
    data_root = "data/memes256"
    validation_split = 0.06  # percent used for validation as a decimal
    load_in_ram = False  # can speed up small datasets <2000 images, num_workers=0
    set_mean = [0.527, 0.487, 0.459]
    set_std = [0.247, 0.236, 0.245]
    train_transforms = A.Compose([
        # Resizing
        A.RandomResizedCrop(input_size, input_size, scale=(0.2, 1.0), ratio=(3./4., 4./3.), interpolation=cv2.INTER_LINEAR),
        # Spatial transforms
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Rotate(limit=20, p=0.5),
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
       ], p=0.5),
        # Blurring and sharpening
        A.OneOf([
            A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True),
            A.GaussianBlur(blur_limit=(1, 5), always_apply=True),
            # A.GlassBlur(sigma=0.7, max_delta=4, iterations=1, always_apply=True, mode='fast'),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), always_apply=True),
            # A.MedianBlur(blur_limit=5, always_apply=True),
            # A.MotionBlur(blur_limit=3, always_apply=True),
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=3, always_apply=True),    
            A.GaussNoise(var_limit=5.0, mean=0, p=1.0),
        ], p=1.0),
        A.ToGray(p=0.25),
        ToTensorV2(),
    ])
    val_transforms = A.Compose([
        A.SmallestMaxSize(input_size),
        A.CenterCrop(input_size, input_size),
        ToTensorV2(),
    ])

    """ Prepare the data """
    # Load the images from the data_root folder
    print(" * Loading data from {}...".format(data_root))
    t0 = time.time()
    path_list, label_list, num_to_cat = get_data(data_root)
    num_cat = len(num_to_cat)
    print("Number of categories :", num_cat)
    print("Categories :", num_to_cat)
    print(f"{len(path_list)} Images")

    print(" * Creating datasets...")
    print("Spliting with {:.0f}% for validation.".format(100*validation_split))
    x_train, x_val, y_train, y_val = train_test_split(path_list, label_list, test_size=validation_split, random_state=MANUAL_SEED , stratify=label_list)
    if load_in_ram:
        x_train = [pil_loader(path) for path in x_train]
        x_val = [pil_loader(path) for path in x_val]
    train_dataset = FolderDataset(data=x_train, labels=y_train, transforms=train_transforms, load_in_ram=load_in_ram)
    print("Train length :", len(train_dataset))
    val_dataset = FolderDataset(data=x_val, labels=y_val, transforms=val_transforms, load_in_ram=load_in_ram)
    print("Validation length :", len(val_dataset))
    duration = time.time() - t0
    rate = 'a lot of' if duration==0 else "{:.1f}".format(len(path_list)/duration)
    print("Found {} images in {}. {} images per second.".format(len(path_list), time_to_string(duration), rate))

    print(" * Creating dataloaders...")
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size ,shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last, persistent_workers=(True if num_workers > 0 else False),
            pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size ,shuffle=False,
            num_workers=num_workers, drop_last=drop_last, persistent_workers=(True if num_workers > 0 else False),
            pin_memory=pin_memory)

    """ Setup the model, optimizer, and scheduler """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device :', device)

    print(" * Creating model...")
    model = SegmentationModel(in_channels, num_cat, filters, activation, set_mean, set_std,
                num_to_cat=num_to_cat, input_size=input_size).to(device)

    if weight_decay != 0:
        params = add_weight_decay(model, weight_decay)
    else:
        params = model.parameters()

    # Setup optimizer
    print(" * Creating optimizer...")
    if optim_type == 'sgd':
        optimizer = optim.SGD(params, lr=base_lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    elif optim_type == 'adam':
        optimizer = optim.Adam(params, lr=base_lr)
    elif optim_type == 'adamw':
        optimizer = optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
    
    # Setup scheduler
    print(" * Creating scheduler...")
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=lr_gamma)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_gamma, patience=plateau_patience, verbose=True)
    elif scheduler_type == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
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

    scalar = amp.GradScaler()

    """ Training """
    print(" * Start training...")

    criterion = nn.CrossEntropyLoss()

    run_stats = []

    t1 = time.time()
    iterations = 0
    model.train()
    try:
        for epoch in range(num_epochs):
            t2 = time.time()
            train_seg_loss_total = 0.0
            train_cla_loss_total = 0.0
            train_seg_correct_total = 0.0
            train_seg_seen_total = 0.0
            train_cla_correct_total = 0.0
            train_cla_seen_total = 0.0
            for batch_idx, (data, true_masks, true_labels) in enumerate(train_loader):
                data = data.to(device)
                true_masks = true_masks.to(device)
                true_labels = true_labels.to(device)
                optimizer.zero_grad()

                if cutmix_beta > 0:
                    cutmix_beta = 1.0
                    lam = np.random.beta(cutmix_beta, cutmix_beta)
                    rand_index = torch.randperm(data.size()[0]).to(data.device)
                    target_a = true_labels
                    target_b = true_labels[rand_index]
                    # Now the bboxes for the input and mask
                    _, _, w, h = data.size()
                    cut_rat = np.sqrt(1. - lam)
                    # Box size
                    cut_w, cut_h = int(w*cut_rat), int(h*cut_rat)
                    # Box center
                    cx, cy = np.random.randint(w), np.random.randint(h)
                    bbx1 = np.clip(cx - cut_w // 2, 0, w)
                    bbx2 = np.clip(cx + cut_w // 2, 0, w)
                    bby1 = np.clip(cy - cut_h // 2, 0, h)
                    bby2 = np.clip(cy + cut_h // 2, 0, h)
                    data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]
                    true_masks[:, bbx1:bbx2, bby1:bby2] = true_masks[rand_index, bbx1:bbx2, bby1:bby2]
                    # Adjust the classification loss based on pixel area ratio
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w*h))
                    # The data and labels are now cutmixed

                # if use_amp:
                with amp.autocast(enabled=use_amp):
                    logits, encoding = model(data)
                    seg_loss = criterion(logits, true_masks)
                    encoding = encoding if use_classifer_grad else encoding.detach()
                    class_logits = model.classifier(encoding)
                    if cutmix_beta > 0:
                        class_loss = criterion(class_logits, target_a)*lam + criterion(class_logits, target_b)*(1.0-lam)
                    else:
                        class_loss = criterion(class_logits, true_labels)
                    loss = seg_loss+class_multi*class_loss
                # else:
                #     logits, encoding = model(data)
                #     seg_loss = criterion(logits, true_masks)
                #     encoding = encoding if use_classifer_grad else encoding.detach()
                #     class_logits = model.classifier(encoding)
                #     if cutmix_beta > 0:
                #         class_loss = criterion(class_logits, target_a)*lam + criterion(class_logits, target_b)*(1.0-lam)
                #     else:
                #         class_loss = criterion(class_logits, true_labels)
                #     loss = seg_loss+class_multi*class_loss

                if use_amp:
                    with amp.autocast():
                        scalar.scale(loss).backward()
                        scalar.step(optimizer)
                        scalar.update()
                else:
                    loss.backward()
                    optimizer.step()

                iterations += 1
                # Update running metrics
                train_seg_loss_total += seg_loss
                train_cla_loss_total += class_loss
                _, tags = torch.max(logits, dim=1)
                train_seg_correct_total += (tags == true_masks).sum()
                train_seg_seen_total += true_masks.numel()
                _, tags = torch.max(class_logits, dim=1)
                train_cla_correct_total += (tags == true_labels).sum()
                train_cla_seen_total += true_labels.numel()

            model.eval()
            val_seg_loss_total = 0.0
            val_cla_loss_total = 0.0
            val_seg_correct_total = 0.0
            val_seg_seen_total = 0.0
            val_cla_correct_total = 0.0
            val_cla_seen_total = 0.0
            with torch.set_grad_enabled(False):
                for batch_idx, (data, true_masks, true_labels) in enumerate(val_loader):
                    data = data.to(device)
                    true_masks = true_masks.to(device)
                    true_labels = true_labels.to(device)
                    logits, encoding = model(data)
                    class_logits = model.classifier(encoding)
                    loss = criterion(logits, true_masks)
                    class_loss = criterion(class_logits, true_labels)
                    # Update running metrics
                    val_seg_loss_total += loss
                    val_cla_loss_total += class_loss
                    _, tags = torch.max(logits, dim=1)
                    val_seg_correct_total += (tags == true_masks).sum()
                    val_seg_seen_total += true_masks.numel()
                    _, tags = torch.max(class_logits, dim=1)
                    val_cla_correct_total += (tags == true_labels).sum()
                    val_cla_seen_total += true_labels.numel()

            # EPOCH TRAIN AND VALIDATE
            train_loss = (train_seg_loss_total+train_cla_loss_total)/len(train_loader)
            train_seg_loss = train_seg_loss_total/len(train_loader)
            train_cla_loss = train_cla_loss_total/len(train_loader)
            train_seg_acc = 100*train_seg_correct_total/train_seg_seen_total
            train_cla_acc = 100*train_cla_correct_total/train_cla_seen_total
            val_loss = (val_seg_loss_total+val_cla_loss_total)/len(val_loader)
            val_seg_loss = val_seg_loss_total/len(val_loader)
            val_cla_loss = val_cla_loss_total/len(val_loader)
            val_seg_acc = 100*val_seg_correct_total/val_seg_seen_total
            val_cla_acc = 100*val_cla_correct_total/val_cla_seen_total
            epoch_metrics = {
                "epoch": epoch+1,
                "iterations": iterations,
                "elapsed_time": time.time()-t1,
                "lr": optimizer.param_groups[0]['lr'],
                "train_loss": train_loss.item(),
                "train_seg_loss": train_seg_loss.item(),
                "train_cla_loss": train_cla_loss.item(),
                "train_seg_acc": train_seg_acc.item(),
                "train_cla_acc": train_cla_acc.item(),
                "val_loss": val_loss.item(),
                "val_seg_loss": val_seg_loss.item(),
                "val_cla_loss": val_cla_loss.item(),
                "val_seg_acc": val_seg_acc.item(),
                "val_cla_acc": val_cla_acc.item(),
            }
            run_stats.append(epoch_metrics)

            duration = time.time()-t2
            remaining = duration*(num_epochs-epoch-1)
            print("epoch {}. {}. loss={:.3f}/{:.3f}. s={:.1f}/{:.1f}. c={:.1f}/{:.1f}. lr={:2e}. elapsed={}. remaining={}.".format(epoch+1, time_to_string(duration), train_loss, val_loss, train_seg_acc, val_seg_acc, train_cla_acc, val_cla_acc, optimizer.param_groups[0]['lr'], time_to_string(time.time()-t1), time_to_string(remaining)))

            if scheduler:
                if type(scheduler) == optim.lr_scheduler.ReduceLROnPlateau:
                    scheduler.step(train_loss)
                elif type(scheduler) in [optim.lr_scheduler.MultiStepLR, optim.lr_scheduler.ExponentialLR]:
                    scheduler.step()

            if save_epochs:
                save_model(model, save_path(current_run, "latest"))
            
            if (epoch+1) % checkpoint_epoch == 0:
                save_model(model, save_path(current_run, f"check{epoch+1}"))
    except KeyboardInterrupt:
        print("KeyboardInterrupt Exit")
        pass

    print('Finished Training. Duration={}. {} iterations'.format(time_to_string(time.time()-t1), iterations))

    """ Training Stage 1 - Classification """

    save_model(model, save_path(current_run, "final"))
    print("saving to: {}".format(save_path(current_run, "final")))


    # Graph Metrics
    line_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    fig, ax = plt.subplots(2, 2, figsize=(10,8))  # w, h
    font_size = 14
    train_steps = len(run_stats)

    # loss, top 0
    ax[0][0].set_xlabel('epochs', fontsize=font_size)
    ax[0][0].set_ylabel('loss', fontsize=font_size)
    ax[0][0].set_yscale('log')
    ax[0][0].tick_params(axis='y')
    train_loss_list = ["train_seg_loss", "train_cla_loss"]
    val_loss_list = ["val_seg_loss", "val_cla_loss"]
    for i in range(len(train_loss_list)):
        ax[0][0].plot(range(1, train_steps+1), [x[train_loss_list[i]] for x in run_stats], color=line_colors[i], label=train_loss_list[i])
        ax[0][0].plot(range(1, train_steps+1), [x[val_loss_list[i]] for x in run_stats], color=line_colors[i], label=val_loss_list[i], linestyle='dashed')
    ax[0][0].legend()

    # acc, top 1
    ax[0][1].set_xlabel('epochs', fontsize=font_size)
    ax[0][1].set_ylabel('accuracy', fontsize=font_size)
    # ax[0][1].set_yscale('log')
    ax[0][1].tick_params(axis='y')
    train_acc_list = ["train_seg_acc", "train_cla_acc"]
    val_acc_list = ["val_seg_acc", "val_cla_acc"]
    for i in range(len(train_acc_list)):
        ax[0][1].plot(range(1, train_steps+1), [x[train_acc_list[i]] for x in run_stats], color=line_colors[i], label=train_acc_list[i])
        ax[0][1].plot(range(1, train_steps+1), [x[val_acc_list[i]] for x in run_stats], color=line_colors[i], label=val_acc_list[i], linestyle='dashed')
    ax[0][1].legend()

    # lr, bot 0
    ax[1][0].set_xlabel('epochs', fontsize=font_size)
    ax[1][0].set_ylabel('lr', fontsize=font_size)
    ax[1][0].set_yscale('log')
    ax[1][0].tick_params(axis='y')
    ax[1][0].plot(range(1, train_steps+1), [x["lr"] for x in run_stats], color=line_colors[0], label="lr")

    # coef, bot 1
    ax[1][1].set_xlabel('epochs', fontsize=font_size)
    ax[1][1].set_ylabel('loss', fontsize=font_size)
    ax[1][1].tick_params(axis='y')
    ax[1][1].plot(range(1, train_steps+1), [x["train_loss"] for x in run_stats], color=line_colors[0], label="train_loss")
    ax[1][1].plot(range(1, train_steps+1), [x["val_loss"] for x in run_stats], color=line_colors[0], label="val_loss", linestyle='dashed')
    ax[1][1].legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig("{}/run{:05d}_metrics.png".format(base_save, current_run), bbox_inches='tight')
    if graph_metrics:
        plt.show()


    # View validation predictions
    rows = 4
    scale = 2.2
    fig, ax = plt.subplots(rows, 1+model.out_channels, figsize=(scale*(1+model.out_channels), scale*rows))
    idxs = np.random.choice(range(len(x_val)), rows, replace=False)
    for i in range(rows):
        img = pil_loader(x_val[idxs[i]])
        width, height = img.size[:2]
        scale = input_size/width 
        img = img.resize((int(width*scale), int(height*scale)))
        img = prepare_image(img, model.input_size, square=True)
        y = y_val[idxs[i]]
        ymask, yclass = model.predict(img.to(device))
        yprob, yhat = torch.max(yclass, dim=1)

        ax[i][0].imshow(img.squeeze().numpy().transpose((1,2,0)))
        color = "g" if int(y)==int(yhat) else "r"
        title_str = "{} ({:.02f} % {})".format(num_to_cat[int(y)], float(100*yprob), num_to_cat[int(yhat)])
        ax[i][0].set_title(title_str, color=color)

        for j in range(model.out_channels):
            ax[i][1+j].imshow(ymask.detach().cpu().numpy().squeeze().transpose((1,2,0))[:,:,j])
            ax[i][1+j].set_title("{}".format(num_to_cat[int(j)]))

    fig.tight_layout()
    fig.savefig("{}/run{:05d}_samples.png".format(base_save, current_run), bbox_inches='tight')
    if view_results:
        plt.show()
