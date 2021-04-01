

# Setup conda environment with pytorch

Conda environment
```
conda create --name pytorch
conda activate pytorch
conda install cython
```
Conda install command generated here: [https://pytorch.org/](https://pytorch.org/)
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

```
# requirements
pip install -r requirements.txt
```

# What is this model

The model is the original UNet with an additional classifier on the last output from the encoder. The UNet does a pixel wise classification. The classifier does the same thing but the label is for the whole image. The classifier first concatenates both AdaptiveAvgPooling and AdaptiveMaxPool operations on the encoder feature maps to give a 1x1 output. Then the data goes through a `conv-bn-act-conv` to convert the data to label space.

![model_arch](/images/readme/model_arch.png)

# How to train

**You have to change the training parameters in train.py** I'm to lazy to setup cli args so you have to change the parameters in the file. This works well for me since I just train on a local machine and usually have the IDE open anyway. Most parameter names are obvious, but if you can't tell what something is doing you should just explore the code; it's pretty short and simple. The first 150 or so lines setup the data, dataloaders, model, optmizer, scheduler, etc., the next 100 lines is the training loop, and the rest uses matplotlib to graph the metrics and visualize a few validation images.

Setup your `data_root` folder like this:
```
  data_root/
  ├── class 1
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── class 2
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  └── class ...
      ├── img1.jpg
      └── ...
  ```

Right now validation is done randomly with sklearn's `train_test_split`. Maybe one day I'll fix this.

Here's what the training graph looks like:

![train_metrics](/images/readme/train_metrics.png)

I found that training goes faster if the RandomResizedCrop (line 91) has a larger scale, like scale=(0.8, 1.0). But then it usually has worse validation performance. Still good to know for running quicker experiments to see if the model fit the data.

# Use sort_folder.py to help label data

I had 150,000 images to label into 8 classes. I'm pretty lazy and I only labeled about 200 for each class. I used this labeled set to train a classifier to help label the rest of the images. It takes way less time to fix labels than make them from scratch.

Simply go into `sort_folder.py` and change the varables `root`, `num`, and `model_paths` to match your project. The script first ensembles the models found in `model_paths` by checking if they have the same classes. It makes folders in `root` for each class. Then it uses glob to load images from `root` and the goes through `min(num, len(images))` and sorts them.

# Run the Flask app

```
python app.py
```

Flask app runs on <pc_ip>:5000/upload. If you can't access the app, try changing your firewall settings and allow python on private networks.

This is the terminal when you start `app.py`:

![app_working](/images/readme/app_working.png)

Without any changes, `app.py` uses the model in `runs/demo.pth`. This was trained on a collection of 3,000 memes split into 1,500 Funny and 1,500 Not Funny.

This is what it renders in the browser:

![app_screenshot](/images/readme/app_screenshot.png)

The "Choose file" and "Upload" buttons do exactly what they say. The "Click for random sample" button checks in `static/samples` for images and runs prediction on them. "Deep Analysis" button returns the segmentation map and an overlay on the original along with the class prediction. Here is what an overlay looks like:

![app_overlay](/images/readme/app_overlay.png)

This app script was for fun and is not a good way to "deploy" these models. I use the app so I can send pictures from my phone to see how "funny" a picture is.
