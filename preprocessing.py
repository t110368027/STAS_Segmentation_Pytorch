import json
import cv2
import matplotlib.pyplot as plt
import os
import glob
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass
    return path


def contour_fill(a_points, a_height, a_width):
    mask_list = []
    for xy in a_points:
        a_x, a_y = int(xy[0]), int(xy[1])
        mask_list.append((a_x, a_y))
    img = Image.new('L', (a_width, a_height), 0)
    ImageDraw.Draw(img).polygon(mask_list, outline=255, fill=255)
    img = np.array(img)
    img[img == 255] = 1
    return img


def main(path_train_images, path_train_mask, path_train_annotations, path_dataset):
    # get filename
    list_train_images = glob.glob(os.path.join(path_train_images, '*'))
    list_train_annotations = glob.glob(os.path.join(path_train_annotations, '*'))

    print("Transformed data annotations to mask images")
    # get mask from annotations    
    for idx, annotations in tqdm(enumerate(list_train_annotations)):
        with open(annotations) as f:
            p = json.load(f)
            a_height, a_width = p["imageHeight"], p["imageWidth"]
            a_name = p['imagePath'].split('.')[0]+'.png'
            mask = np.zeros((a_height, a_width))
            for i in range(len(p["shapes"])):
                a_points = p["shapes"][i]['points']
                mask = contour_fill(a_points, a_height, a_width)+mask
            if np.amax(mask) == 2:
                mask[mask == 2] = 1
            if np.amax(mask) != 1:
                print(idx)
            cv2.imwrite(os.path.join(path_train_mask, a_name), mask)

    # get filename        
    list_train_mask = glob.glob(os.path.join(path_train_mask, '*'))

    train_images = []
    train_images_name = []

    print("Reading training raw images")
    for path in tqdm(list_train_images):
        raw_image = cv2.imread(path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        train_images.append(raw_image)
        train_images_name.append(path)
    train_images = np.array(train_images)
    train_images_name = np.array(train_images_name)

    train_mask = []
    train_mask_name = []

    print("Reading training mask images")
    for path in tqdm(list_train_mask):
        raw_image = cv2.imread(path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
        train_mask.append(raw_image)
        train_mask_name.append(path)
    train_mask = np.array(train_mask)
    train_mask_name = np.array(train_mask_name)

    # add one diamention to mask
    y = np.expand_dims(train_mask, axis=-1)

    # save preprocessing dataset
    np.savez(path_dataset,
             image=train_images, mask=y,
             image_name=train_images_name,
             mask_name=train_mask_name)
    print("It's already save dataset at {}".format(_path_dataset))


if __name__ == "__main__":
    # define path
    _path_train_images = make_path('./SEG_Train_Datasets/Train_Images/')
    _path_train_mask = make_path('./SEG_Train_Datasets/Train_Masks/')
    _path_train_annotations = make_path('./SEG_Train_Datasets/Train_Annotations/')
    _path_dataset = './SEG_Train_Datasets/set.npz'

    main(path_train_images=_path_train_images,
         path_train_mask=_path_train_mask,
         path_train_annotations=_path_train_annotations,
         path_dataset=_path_dataset)