import os
import cv2
import yaml
import glob
import torch
import numpy as np
import albumentations as A
from tqdm import tqdm
from pprint import pprint
from IPython.display import clear_output
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from train import Model, ImageDataset, get_val_augmentation
from preprocessing import make_path


def main(path, path_test_images):
    with open(os.path.join(path.split('checkpoints')[0], "hparams.yaml"), 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            print('Model Definition :')
            pprint(parsed_yaml)
            print()
        except yaml.YAMLError as exc:
            print(exc)

    model_arch = parsed_yaml['arch']
    model_backbone = parsed_yaml['encoder_name']
    in_ch = parsed_yaml['in_channels']
    out_ch = parsed_yaml['out_classes']

    model = Model.load_from_checkpoint(path, arch=model_arch, encoder_name=model_backbone,
                                       in_channels=in_ch, out_classes=out_ch)
    print('\nmodel load ok')

    b_th = 0.5
    list_test_images = glob.glob(os.path.join(path_test_images, '*'))

    test_images = []
    test_images_name = []
    test_folder_name = path_test_images.split('/')[-1]+'_Prediction'
    path_test_predictions = make_path(path_test_images.replace(path_test_images.split('/')[-1], test_folder_name))

    print('reading raw images......')
    for path in tqdm(list_test_images):
        raw_image = cv2.imread(path)
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        test_images.append(raw_image)
        test_images_name.append(path)

    test_dataset = ImageDataset(np.array(test_images), transforms=get_val_augmentation())

    print('predict labels......')
    for i in tqdm(range(len(test_dataset))):
        image = test_dataset[i]
        save_path = test_images_name[i].replace(path_test_images.split('/')[-1], test_folder_name)
        save_path = save_path.replace("jpg", "png")
        with torch.no_grad():
            model.eval()
            logits = model(image)
        pr_mask = logits.sigmoid()
        pr_mask = pr_mask.numpy().squeeze()
        pr_mask = A.resize(pr_mask, height=942, width=1716, interpolation=cv2.INTER_CUBIC)

        th = b_th
        pr_mask[pr_mask >= th] = 1
        pr_mask[pr_mask < th] = 0
        pr_mask = cv2.cvtColor(pr_mask, cv2.COLOR_GRAY2RGB)*255
        pr_mask = pr_mask.astype(np.uint8)
        cv2.imwrite(save_path, pr_mask)

    return path_test_predictions


if __name__ == "__main__":
    _path_model = './logs/best/version_0/checkpoints/best_model.ckpt'
    _path_test_images = './SEG_Train_Datasets/Private_Image'
    _path_test_predictions = main(path=_path_model, path_test_images=_path_test_images)

    print("Model Prediction done!, save to {}".format(_path_test_predictions))
