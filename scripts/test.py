from pathlib import Path
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from chestxray.models import lit_module
from chestxray.data.datamodule import ChestXray14DataModule
from pathlib import Path
import pytorch_lightning as pl
import argparse
from typing import List

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nih_kaggle"
ALL_IMG_DIR = ROOT / "data" / "processed" / "images_all"
OUT = ROOT / "data" / "processed"/ "resized" / "512"
LABELS_CSV = ROOT / "data" / "processed"/ "labels_full.csv"

LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
          "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
          "Emphysema","Fibrosis","Pleural_Thickening","Hernia"]

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def main(args):
    train_csv = OUT / "train.csv"
    val_csv = OUT / "val.csv"
    test_csv = OUT / "test.csv"
    cached_dir = OUT
    img_size = 512

    ckpt_path = args.ckpt_dir + "/" + args.ckpt_name 
    thresholds_path = args.thresholds_path

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

    trainer = pl.Trainer(
        max_epochs=100,
        precision="32",
        accelerator="gpu", 
        devices=1,
        gradient_clip_val=1.0,
    )

    model = lit_module.LitChestXray.load_from_checkpoint(
        str(ckpt_path), 
        map_location=None, 
        strict=False,
        thresholds_path= thresholds_path,
        widths=args.model_width
    )

    trainer.test(model=model, datamodule=dm)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-dir", type=str, default=str(ROOT / "logs" / "checkpoints"))
    p.add_argument("--ckpt-name", type=str, default="residual_epoch=039_100-val_mAP=0.2342.ckpt")
    p.add_argument("--thresholds-path", type=str, default=None)
    p.add_argument(
        "--model-width",
        type=int, 
        nargs=4,
        metavar=("C1","C2","C3","C4"),
        default=[ 64, 128, 256, 512]
    )
    args = p.parse_args()
    main(args)