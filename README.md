# STAS Competition by using Image Segmentation 

###### tags: `Github`

**Competition Url : https://tbrain.trendmicro.com.tw/Competitions/Details/22**

##  Before You Start
* Clone repo and install [**requirements.txt**](https://github.com/t110368027/STAS_Segmentation_Pytorch/blob/main/requirements.txt "requirements.txt") in a [**Python>=3.7.0**](https://www.python.org/) environment, including [**PyTorch>=1.10**](https://pytorch.org/get-started/locally/), [**PyTorch Lightining>=1.5**](https://www.pytorchlightning.ai/).
* Or you can using conda environment [**conda_environ_python_3.7.yml**](https://github.com/t110368027/STAS_Segmentation_Pytorch/blob/main/conda_environ_python_3.7.yml "conda_environ_python_3.7.yml").

```
git clone https://github.com/t110368027/STAS_Segmentation_Pytorch  # clone
cd STAS_Segmentation_Pytorch
```
## 1. Create your Environment

Install Package.
```
pip install -r requirements.txt  # install
```
or Conda Environment.
```
conda env create -f conda_environ_python_3.7.yml
```
## 2. Prepare Dataset

### 2.1 Download Dataset
Go to [**competition website**](https://tbrain.trendmicro.com.tw/Competitions/Details/22) download **SEG_Train_Datasets.zip**, **Public_Image.zip** and **Public_Image.zip**.

---
<details>   
<summary>Click to Open Image</summary>   
<pre>
    <img src="https://i.imgur.com/VsgqDTn.png">
</pre>   
</details>

---

### 2.2 Unzip Dataset
Unzip these three zip files to `./SEG_Train_Datasets/`.

### 2.3 Organize Directories
Organize your **images** and **labels** according to the following examples, including **training, validation, and testing datasets**.

---
<details>   
<summary>Click to Open Directories Structure</summary>   
<pre><code>
STAS_Segmentation_Pytorch        
├── SEG_Train_Datasets
│   ├── Private_Image              ## Unzip by Private_Image.zip 
│   │   ├── Private_00000000.jpg
│   │   ├── Private_00000001.jpg
│   │   ├── ...
│   │   └── Private_00000183.jpg
│   │   
│   ├── Public_Image               ## Unzip by Public_Image.zip 
│   │   ├── Public_00000000.jpg
│   │   ├── Public_00000001.jpg
│   │   ├── ...
│   │   └── Public_00000130.jpg
│   │
│   ├── Train_Annotations          ## Unzip by SEG_Train_Datasets.zip
│   │   ├── 00000000.json
│   │   ├── 00000001.json
│   │   ├── ...
│   │   └── 00001052.json
│   │   
│   ├── Train_Images               ## Unzip by SEG_Train_Datasets.zip
│   │   ├── 00000000.jpg
│   │   ├── 00000001.jpg
│   │   ├── ...
│   │   └── 00001052.jpg
│   │   
│   └── set.npz                    ## Create by preprocessing.py
│
├── conda_environ_python_3.7.yml   ## Conda environment
├── preprocessing.py
├── requirements.txt
├── train.py
├── test.py
│
├── SEG_Train_Datasets.zip         ## Download these three zip files on competition url
├── Private_Image.zip
└── Public_Image.zip
</code></pre>   
</details>

---

## 3. Preprocessing
Before running `preprocessing.py`, please check your `./SEG_Train_Datasets/` include all datasets `.jpg` and `.json` file follow [**2.3 Organize Directories**](#23-Organize-Directories).
```
$ python preprocessing.py
```

## 4. Train
* All results on Public/Private are training on **remote training device** by  the GPU and hyperparameter is below [**4.0 Hyperparameter**](#40-Hyperparameter).
* Validation and tuning model are on **local training device**.

```
$ python train.py
```

### 4.0 Hyperparameter
---
<details>   
<summary>Hyperparameter on Remote Traing Device</summary>   
<pre>
    1. Model Architecture
        -   Architecture : Unet
        -   Backbone : efficientnet-b7
    2. Hyperparameter
        -   Batch Size : 16
        -   Optimizer : Adam
        -   Learning Rate : 0.0001
        -   Epochs : 100
        -   Loss = Dice loss
        -   Data Augmentation = True
</pre>   
</details>

---

### 4.1 Training Device
---
<details>   
<summary>Our Local Training Device</summary>   
<pre>
    OS : Windows 10
    IDE : Pycharm 2021.2.2
    python : 3.7
    GPU : Nvidia TITAN Xp (driver : 456.71)
    Cuda versuin : 11.1
</pre>   
</details>

---
---
<details>   
<summary>Our Remote Training Device</summary>   
<pre>
    OS : Ubuntu 18.04
    python : 3.7
    GPU : Nvidia V100*4 (driver : 460.91.03)
    Cuda versuin : 11.2
</pre>   
</details>

---

### 4.2 Training on Your Device
:fire: **Please tune hyperparameters and set batch size depending on your GPU VRAM size**.:fire: 

```python=3.7
if __name__ == "__main__":
    cfg_load_npz_path = './SEG_Train_Datasets/set.npz'
    cfg_model_arch = "Unet"
    cfg_model_backbone = "efficientnet-b7"
    cfg_batch_size = 16
    cfg_epoch = 100
    cfg_lr = 0.0001
    main(cfg_model_arch, cfg_model_backbone, cfg_batch_size, cfg_lr, cfg_epoch, cfg_load_npz_path)
```
:ballot_box_with_check: **Please confirm the device you have [CPU/GPU]** [**Pytorch Lightning Document**](https://pytorch-lightning.readthedocs.io/en/latest/accelerators/gpu_basic.html).
```python=3.7
trainer = pl.Trainer(accelerator="gpu", devices=[0])
```

## 5. Inference
```
$ python test.py
```
:fire: Choose your model `.ckpt`, model is save `./logs/.../version_/checkpoints/.ckpt`.:fire: 

Replace path of `_path_model` and `_path_test_images`, which model you choose or Image Folder you want to predict.
```python=3.7
if __name__ == "__main__":
    _path_model = './logs/.../version_/checkpoints/.ckpt'
    _path_test_images = './SEG_Train_Datasets/Public_Image'
    _path_test_predictions = main(path=_path_model, path_test_images=_path_test_images)

    print("Model Prediction done!, save to {}".format(_path_test_predictions))
```

## Reference
* [Pytorch](https://github.com/pytorch/pytorch)
* [Pytorch-Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [Segmentation_Models.Pytorch](https://github.com/qubvel/segmentation_models.pytorch)\
* [U-Net: Convolutional Networks for Biomedical Image Segmentation
](https://arxiv.org/pdf/1505.04597.pdf)
* [V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
](https://arxiv.org/pdf/1606.04797.pdf)
* [DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation
](https://arxiv.org/pdf/2006.04868.pdf)
## 📝 Citing

```
@misc{STAS_Segmentation_Pytorch,
  author = {Jia-Ming Hou},
  title = {{STAS_Segmentation_Pytorch}},
  year = {2022},
  month = {06},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/t110368027/STAS_Segmentation_Pytorch}},
  version = {1.0}, 
}
```

## 🛡️ License
 
Project is distributed under [AGPL v3](https://)
