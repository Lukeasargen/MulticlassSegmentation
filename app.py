import os
import time
import random

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from flask import Flask
from flask import render_template
from flask import url_for
from flask import request
from flask import redirect
from werkzeug.utils import secure_filename

from model import load_model
from util import pil_loader, custom_resize, prepare_image, normalize, crop_center

app = Flask(__name__)

SAVE_FOLDER = 'static/uploads/'
SAMPLE_DATA_FOLDER = 'static/samples/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

MODEL_SAVE_PATH = "runs/demo.pth"

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


def report_time(t0, msg):
    duration = time.time() - t0
    print("{} : {:.03f} seconds".format(msg, duration))


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    t0 = time.time()
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
        # print("upload_name :", upload_name)
        if current_img_path and upload_name:
            try:
                # Pass image through model
                img_color = pil_loader(current_img_path)
                # print("img_color.size :", img_color.size)
                img_color_resize = custom_resize(img_color, model.input_size)
                # print("img_color_resize.size :", img_color_resize.size)
                img_ten = prepare_image(img_color_resize).to(device)
                # print("img_ten.shape :", img_ten.shape)
                ymask, yclass = model.predict(img_ten)
                class_prob, class_num = torch.max(yclass, dim=1)
                # print("ymask.shape :", ymask.shape)
                # print("yclass.shape :", yclass.shape)
                # print("class_prob.shape :", class_prob.shape)
                # print("class_num.shape :", class_num.shape)

                # Convert to probabilities and binary masks
                ymask_top_prob, ymask_top_bin = torch.max(ymask, dim=1)  # Top class probability and label per pixel
                # print("ymask_top_prob.shape :", ymask_top_prob.shape)
                # print("ymask_top_bin.shape :", ymask_top_bin.shape)

                cat_name = model.num_to_cat[int(class_num)]
                # print("cat_name :", cat_name)
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

            # print("ymask :", torch.min(ymask), torch.max(ymask))
            # print("ymask.shape :", ymask.shape)
            ymask = ymask.squeeze().detach().cpu().numpy()
            # print("ymask.shape :", ymask.shape)
            ymask = ymask[class_num]
            # print("ymask.shape :", ymask.shape)
            ymask = normalize(ymask)
            # print("ymask.shape :", ymask.shape)
            # print("ymask :", np.min(ymask), np.max(ymask))

            # intensity = 1
            # ymask = torch.pow(torch.from_numpy(ymask), intensity).numpy()

            # # Save the heatmap as an image file
            # bool_mask = (ymask > 0.5).astype(int)  # makes a binary mask, sharper edges
            bool_mask = ymask

            # print("bool_mask :", np.min(bool_mask), np.max(bool_mask))
            bool_img = Image.fromarray(np.uint8(bool_mask*255)).convert('RGB')
            heatmap_filename = name + "_heatmap" + extension
            # print("heatmap_filename :", heatmap_filename)
            heatmap_upload_name = SAVE_FOLDER + heatmap_filename
            # print("heatmap_upload_name :", heatmap_upload_name)
            heatmap_path = os.path.join(SAVE_FOLDER_PATH, heatmap_filename)
            # print("heatmap_path :", heatmap_path)
            bool_img.save(heatmap_path)

            # # Save the overlay as an image file
            mask_img = Image.fromarray(ymask*255).convert('RGB')
            # print("mask_img.size :", mask_img.size)
            mask_width, mask_height = mask_img.size
            # print("img_color.size :", img_color.size)
            in_width, in_height = img_color.size
            scale_h = (in_height/mask_height)
            scale_w = (in_width/mask_width)
            # print("scale_h :", scale_h)
            # print("scale_w :", scale_w)
            scale = min(scale_w, scale_h)
            # print("scale :", scale)
            mask_img_up = mask_img.resize((int(scale*mask_width), int(scale*mask_height)))
            # print("mask_img_up.size :", mask_img_up.size)
            up_width, up_height = mask_img_up.size
            img_color_crop = crop_center(img_color, up_width, up_height)
            # print("img_color_crop.size :", img_color_crop.size)
            mask_arr = np.array(mask_img_up)/255
            # print("mask_arr.shape :", mask_arr.shape)
            # print("mask_arr :", np.min(mask_arr), np.max(mask_arr))
            overlay_img = img_color_crop*mask_arr
            # print("overlay_img :", np.min(overlay_img), np.max(overlay_img))
            overlay_img = Image.fromarray(np.uint8(overlay_img)).convert('RGB')
            # print("overlay_img.shape :", overlay_img.size)
            overlay_filename = name + "_overlay" + extension
            # print("overlay_filename :", overlay_filename)
            overlay_upload_name = SAVE_FOLDER + overlay_filename
            # print("overlay_upload_name :", overlay_upload_name)
            overlay_path = os.path.join(SAVE_FOLDER_PATH, overlay_filename)
            # print("heatmap_path :", heatmap_path)
            overlay_img.save(overlay_path)

            report_time(t0, "Deep Analysis")
            # Send the path to the created images
            return render_template('upload.html',
                                    filename=upload_name,
                                    rating=rating_str,
                                    heatmap=heatmap_upload_name,
                                    cat_name=cat_name,
                                    overlay=overlay_upload_name
                                    )
        else:  # Normal upload
            report_time(t0, "Normal Upload")
            return render_template('upload.html',
                                    filename=upload_name,
                                    rating=rating_str
                                    )

    return render_template('upload.html')  # Failure case

if __name__ == "__main__":
    load_model_global()
    app.run(host='0.0.0.0', port=5000, debug=True)