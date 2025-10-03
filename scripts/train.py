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
        batch_size=12, 
        num_workers=6,
        cached_dir=cached_dir,
        pin_memory=True, 
        persistent_workers=True,
        mean=0.5, 
        std=0.25,
    )

    logger = TensorBoardLogger(
        save_dir="logs",
        name="resnet50_cbam_AdamW_100"
    )

    model = lit_module.LitChestXray(
        in_chans=1, 
        num_classes=len(LABELS), 
        lr=1e-4, 
        weight_decay=1e-4, 
        total_epochs=300,
        # thresholds_path="../configs/thresholds.json"
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=ROOT / "logs" / "checkpoint_resnet50_100",
        monitor="val_mAP", 
        mode="max", 
        save_top_k=1, 
        filename="resnet50_cbam_AdamW_{epoch:03d}_100-{val_mAP:.4f}",
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
        callbacks=[ckpt_cb, lr_cb],
        gradient_clip_val=1.0,
        logger=logger
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()