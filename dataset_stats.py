import time

import numpy as np

from util import get_data, np_loader


if __name__ == "__main__":

    root = "data"

    num = 1000000

    path_list, label_list, num_to_cat = get_data(root)
    print("Total images :", len(path_list))

    mean = 0.0
    var = 0.0
    n = min(len(path_list), num)  # Go through the whole dataset if possible
    t0 = time.time()
    t1 = t0
    for i in range(n):
        # img in shape [W, H, C]
        img = np_loader(path_list[i]) / 255.0
        mean += np.mean(img, axis=(0, 1))
        var += np.var(img, axis=(0, 1))  # you can add var, not std
        if (i+1) % 100 == 0:
            t2 = time.time()
            print("{}/{} measured. Total time={:.2f}s. Images per second {:.2f}.".format(i+1, n, t2-t0, 100/(t2-t1)))
            t1 = t2
    print("set_mean = [{:4.3f}, {:4.3f}, {:4.3f}]".format(*(mean/n)))
    print("set_std = [{:4.3f}, {:4.3f}, {:4.3f}]".format(*np.sqrt(var/n)))
    print("var :", var/n)
