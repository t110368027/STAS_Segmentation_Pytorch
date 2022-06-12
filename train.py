"""
References:
    https://pytorch-lightning.readthedocs.io/en/latest/?_ga=2.23257402.1423241754.1654949510-1134220784.1646556985
    https://pytorch.org/docs/stable/index.html
    https://albumentations.ai/docs/api_reference/augmentations/transforms/
    https://github.com/qubvel/segmentation_models.pytorch
"""

import cv2
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import smp_metrics
from datetime import datetime


def get_val_augmentation():
    """Add paddings to make image shape divisible by 32"""
    transform = [
        A.Resize(height=512, width=512, interpolation=cv2.INTER_AREA),
        ToTensorV2(transpose_mask=True),
    ]
    return A.Compose(transform)


def get_train_augmentation():
    """Add paddings to make image shape divisible by 32"""
    transform = [
        A.Resize(height=512, width=512, interpolation=cv2.INTER_AREA),
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            # A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1),
            ], p=0.5),
        A.RandomBrightnessContrast(p=0.8),
        A.RandomGamma(p=0.8),
        A.ShiftScaleRotate(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        ToTensorV2(transpose_mask=True)
    ]
    return A.Compose(transform)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_tensor, target_tensor=None, transforms=None):

        if target_tensor is not None:
            assert data_tensor.shape[:1] == target_tensor.shape[:1]

        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transforms = transforms

    def __getitem__(self, index):
        data_tensor = self.data_tensor[index]

        if self.target_tensor is None:
            if self.transforms:
                sample = self.transforms(image=data_tensor)
                data_tensor = sample['image']
            else:
                data_tensor = torch.tensor(data_tensor).permute(2, 0, 1)
            return data_tensor

        else:
            target_tensor = self.target_tensor[index]
            target_tensor = target_tensor.astype(float)

            if self.transforms:
                sample = self.transforms(image=data_tensor, mask=target_tensor)
                data_tensor, target_tensor = sample['image'], sample['mask']
            else:
                data_tensor = torch.tensor(data_tensor).permute(2, 0, 1)
                target_tensor = torch.tensor(target_tensor).permute(2, 0, 1)
            return data_tensor, target_tensor

    def __len__(self):
        return len(self.data_tensor)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 8, numworkers: int = 0):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image = None
        self.mask = None
        self.numworkers = numworkers

    def prepare_data(self):
        return

    def setup(self, stage=None):

        with np.load(self.data_dir, allow_pickle=True) as f:
            self.image, self.mask = f['image'], f['mask']

        x_train,x_val,y_train,y_val = train_test_split(self.image, self.mask, test_size=0.05, random_state=1)

        self.train_dataset = ImageDataset(x_train, y_train, get_train_augmentation())
        self.valid_dataset = ImageDataset(x_val, y_val, get_val_augmentation())

    def train_dataloader(self):
        #  Generating train_dataloader
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.numworkers)

    def val_dataloader(self):
        #  Generating val_dataloader
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.numworkers)


class Model(pl.LightningModule):

    def __init__(self, arch, encoder_name, in_channels, out_classes, lr=0.0001, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameters for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.lr = lr
        self.save_hyperparameters()

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image, mask = batch

        assert image.ndim == 4

        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4

        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        tp, fp, fn, tn = smp_metrics.get_stats(pred_mask.long(), mask.long(), mode="binary", threshold=0.5)

        f1_score = smp_metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        recall = smp_metrics.recall(tp, fp, fn, tn, reduction="micro")
        precision = smp_metrics.precision(tp, fp, fn, tn, reduction="micro")

        if stage == 'train':
            logs = {'train_loss': loss, 'train_precision': precision, 'train_recall': recall,
                    'train_f1_score': f1_score, 'lr': self.optimizer.param_groups[0]['lr']}
            self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()

        f1_score = smp_metrics.f1_score(tp, fp, fn, tn, reduction="micro")
        recall = smp_metrics.recall(tp, fp, fn, tn, reduction="micro")
        precision = smp_metrics.precision(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_precision": precision,
            f"{stage}_recall": recall,
            f"{stage}_f1_score": f1_score,
            f"{stage}_avg_loss": avg_loss,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3),
            'monitor': 'valid_avg_loss'}
        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}


def main(model_arch, model_backbone, batch_size, lr, epoch, load_npz_path):

    model = Model(model_arch, model_backbone, in_channels=3, out_classes=1, lr=lr)
    dataset_stas = DataModule(data_dir=load_npz_path, batch_size=batch_size)
    checkpoint_callback = ModelCheckpoint(monitor='valid_avg_loss',
                                          save_top_k=2,
                                          save_weights_only=True,
                                          filename='{epoch:02d}-{valid_avg_loss:.4f}-{valid_f1_score:.4f}',
                                          verbose=False,
                                          mode='min')

    trainer = pl.Trainer(
        accelerator="gpu", devices=[0],
        max_epochs=epoch,
        precision=16,
        logger=CSVLogger(save_dir="logs/", name=model_arch + "_" + model_backbone+"_"+str(batch_size)),
        enable_progress_bar=True,
        callbacks=[checkpoint_callback],
    )

    time_start = datetime.now()

    trainer.fit(model, dataset_stas)

    time_end = datetime.now()
    print("Training execution time is: ", (time_end - time_start))


if __name__ == "__main__":
    cfg_load_npz_path = './SEG_Train_Datasets/set.npz'
    cfg_model_arch = "Unet"
    cfg_model_backbone = "efficientnet-b7"
    cfg_batch_size = 16
    cfg_epoch = 100
    cfg_lr = 0.0001
    main(cfg_model_arch, cfg_model_backbone, cfg_batch_size, cfg_lr, cfg_epoch, cfg_load_npz_path)
    