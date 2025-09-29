from lightning.pytorch.loggers import TensorBoardLogger
import torch
import json
from chestxray.models import lit_module
from chestxray.data.datamodule import ChestXray14DataModule
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

ROOT = Path(__file__).resolve().parents[1]
RESIZED = ROOT / "data" / "processed" / "resized" / "512"

LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
          "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
          "Emphysema","Fibrosis","Pleural_Thickening","Hernia"]

def main():
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_csv = RESIZED / "train.csv"
    val_csv = RESIZED / "val.csv"
    test_csv = RESIZED / "test.csv"
    cached_dir = RESIZED
    img_size = 512

    pos = json.loads(Path( ROOT / "configs" / "pos_weight.json").read_text())["pos_weight"]
    print(pos)

    dm = ChestXray14DataModule(
        train_csv=train_csv, 
        val_csv=val_csv, 
        test_csv=test_csv,
        img_size=img_size, 
        batch_size=32, 
        num_workers=6,
        cached_dir=cached_dir,
        pin_memory=True, 
        persistent_workers=True,
        mean=0.5, 
        std=0.25,
    )

    logger = TensorBoardLogger(
        save_dir="logs",
        name="resnet_cbam_AdamW"
    )

    model = lit_module.LitChestXray(
        in_chans=1, 
        num_classes=len(LABELS), 
        lr=1e-4, 
        weight_decay=1e-4, 
        pos_weight=pos,
        total_epochs=100,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=ROOT / "logs" / "checkpoint",
        monitor="val_mAP", 
        mode="max", 
        save_top_k=1, 
        filename="resnet_cbam_AdamW_{epoch:03d}-{val_mAP:.4f}",
        save_weights_only=False
    )
    
    es_cb = EarlyStopping(monitor="val_AUROC", mode="max", patience=10)
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=100,
        precision="32",
        accelerator="gpu", 
        devices=1,
        log_every_n_steps=100,
        callbacks=[ckpt_cb, es_cb, lr_cb],
        gradient_clip_val=1.0,
        logger=logger
    )

    trainer.fit(model, datamodule=dm)

if __name__ == "__main__":
    main()