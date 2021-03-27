import os
import torch
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torchvision.transforms as T

from flask import Flask
from flask import render_template
from flask import url_for
from flask import request
from flask import redirect
from werkzeug.utils import secure_filename

from model import load_model
from util import pil_loader

app = Flask(__name__)

SAVE_FOLDER = 'static/uploads/'
SAMPLE_DATA_FOLDER = 'static/samples/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

MODEL_SAVE_PATH = "runs/save/run00159_final.pth"

CURRENT_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_FOLDER_PATH = os.path.join(CURRENT_DIR_PATH, SAVE_FOLDER)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def load_model_global():
    global model
    print(" * Loading model...")
    print(" * Device : {}".format(device))
    model = load_model(MODEL_SAVE_PATH, device)
    print(model.num_to_cat)
    print(" * Model loaded")

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_sample_path_and_upload_name():
    selection_folder_path = os.path.join(CURRENT_DIR_PATH, SAMPLE_DATA_FOLDER)
    rand_sample = random.choice(os.listdir(selection_folder_path))
    current_img_path = os.path.join(selection_folder_path, rand_sample)
    upload_name = SAMPLE_DATA_FOLDER + '/' + rand_sample
    return current_img_path, upload_name

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == "POST":
        current_img_path = None
        if 'btn_sample' in request.form:
            current_img_path, upload_name = get_sample_path_and_upload_name()
        elif 'btn_upload' in request.form or 'btn_analysis' in request.form:
            if 'image' not in request.files:
                error_str = "Failed to upload image."
                return render_template('upload.html', error_str=error_str)
            file = request.files['image']
            if file.filename == "":
                error_str = "Invalid character in filename."
                return render_template('upload.html', error_str=error_str)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_name = filename
                upload_name = SAVE_FOLDER + filename
                current_img_path = os.path.join(SAVE_FOLDER_PATH, filename)
                file.save(current_img_path)
                print("Saving {}".format(filename))
            else:
                error_str = "Invalid file. Allowed types -> png, jpg, jpeg"
                return render_template('upload.html', error_str=error_str)

        # Use the model to make a prediction and create the relevant strings
        print("upload_name :", upload_name)
        if current_img_path and upload_name:
            try:
                # Pass image through model
                img_color = pil_loader(current_img_path)  # Returns image [1, 3, h, w]
                print("img_color :", img_color)
                img_ten = T.ToTensor()(img_color).unsqueeze_(0).to(device)
                print("img_ten.shape :", img_ten.shape)
                mask = model.predict(img_ten)
                print("mask.shape :", mask.shape)

                # Convert to probabilities and binary masks
                mask_top_prob, mask_top_bin = torch.max(mask, dim=1)  # Top class probability and label per pixel
                print("mask_top_prob.shape :", mask_top_prob.shape)
                print("mask_top_bin.shape :", mask_top_bin.shape)

                class_pred = torch.mean(torch.mean(mask, dim=2), dim=2) # Average maps
                print("class_pred :", class_pred, class_pred.shape)
                class_prob, class_num = torch.max(class_pred, dim=1)  # Top class average probability and label
                print("class_prob :", class_prob, class_prob.shape)
                print("class_num :", class_num, class_num.shape)
                cat_name = model.num_to_cat[int(class_num)]
                print("cat_name :", cat_name)
                # Simple rating label
                rating_str = "{:.06f} % {}".format(float(100*class_prob), cat_name)
                print("rating_str :", rating_str)

            except Exception as e:
                print(e)
                error_str = "Failed to process image."
                return render_template('upload.html', error_str=error_str)
        else:
            error_str = "Failed to load image."
            return render_template('upload.html', error_str=error_str)

        # Now the image is through the model and the simple predictions are complete

        # If the button was analysis, do the heatmap, else just return normal page
        if 'btn_analysis' in request.form:
            name, extension = os.path.splitext(filename)  # Split the filename for use later

            print("mask :", torch.min(mask), torch.max(mask))
            print("mask.shape :", mask.shape)
            mask = mask.squeeze().detach().cpu().numpy()
            print("mask.shape :", mask.shape)
            mask = mask[class_num]
            print("mask.shape :", mask.shape)
            mask = np.expand_dims(mask, axis=2)
            print("mask.shape :", mask.shape)
            mask = np.repeat(mask[:, :], 3, axis=2) * 255
            print("mask.shape :", mask.shape)

            # # Save the heatmap as an image file
            heatmap_filename = name + "_heatmap" + extension
            heatmap_upload_name = SAVE_FOLDER + heatmap_filename
            print("heatmap_upload_name :", heatmap_upload_name)
            heatmap_path = os.path.join(SAVE_FOLDER_PATH, heatmap_filename)
            print("heatmap_path :", heatmap_path)

            cv2.imwrite(heatmap_path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            # # Save the overlay as an image file
            # overlay_filename = name + "_overlay" + extension
            # overlay_upload_name = SAVE_FOLDER + overlay_filename
            # print("overlay_upload_name :", overlay_upload_name)
            # overlay_path = os.path.join(SAVE_FOLDER_PATH, overlay_filename)
            # cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            # # Send the path to the created images
            return render_template('upload.html',
                                    filename=upload_name,
                                    rating=rating_str,
                                    heatmap=heatmap_upload_name,
                                    cat_name=cat_name,
                                    # overlay=overlay_upload_name
                                    )
        else:  # Normal upload
            return render_template('upload.html',
                                    filename=upload_name,
                                    rating=rating_str
                                    )

    return render_template('upload.html')  # Failure case

if __name__ == "__main__":
    load_model_global()
    app.run(host='0.0.0.0', port=5000, debug=True)