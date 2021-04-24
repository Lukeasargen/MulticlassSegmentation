import os
import time

from torchvision import transforms as T
from torchvision.datasets import ImageFolder

in_root = "data/memes"
out_root = "data/memes256"
out_size = int(256)

resize = T.Resize(out_size)

train_ds = ImageFolder(root=in_root)

# Make output folders
if not os.path.exists(out_root):
    os.mkdir(out_root)
out_folders = []  # Path to destination folder
for c in train_ds.classes:
    c_path = os.path.join(out_root, c)
    out_folders.append(c_path)
    if not os.path.exists(c_path):
        os.mkdir(c_path)

start_time = time.time()
for i in range(len(train_ds)):
    x, y = train_ds[i]
    name = os.path.split(train_ds.samples[i][0])[1]  # get the file path, split, get filename w extension
    x = resize(x)
    out = os.path.join(out_root, train_ds.classes[y], name)
    x.save(out)

    if (i+1) % 50 == 0:
        count = i+1
        t2 = time.time() - start_time
        rate = count/t2
        est = t2/count * (len(train_ds)-count)
        print("{}/{} images. {:.2f} seconds. {:.2f} images per seconds. {:.2f} seconds remaining.".format(count, len(train_ds), t2, rate, est))

duration = time.time() - start_time
print(" * Resize Complete")
print(" * Duration {:.2f} Seconds".format(duration))
print(" * {:.2f} Images per Second".format(len(train_ds)/duration))

