from lightning.pytorch.loggers import TensorBoardLogger
import torch
import json
from chestxray.models import lit_module, cnn
from chestxray.data.datamodule import ChestXray14DataModule
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import argparse

ROOT = Path(__file__).resolve().parents[1]
RESIZED = ROOT / "data" / "processed" / "resized" / "512"

LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
          "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
          "Emphysema","Fibrosis","Pleural_Thickening","Hernia"]

def print_args(args, pos):
    print("\n=== Training Configuration ===")
    print(f"Data paths:")
    print(f"  Train CSV: {args.train_csv}")
    print(f"  Val CSV:   {args.val_csv}") 
    print(f"  Test CSV:  {args.test_csv}")
    print(f"  Cache dir: {args.cached_dir}")
    print(f"\nTraining settings:")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Num workers:    {args.num_worker}")
    print(f"  Total epochs:   {args.total_epochs}")
    print(f"  Loss function:  {args.loss_func}")
    print(f"\nLogging:")
    print(f"  Log dir:        {args.log_dir}")
    print(f"  Log name:       {args.log_name}")
    print(f"  Checkpoint dir: {args.ckpt_dir}")
    print(f"  Thresholds:     {args.thresholds_path}")
    print(f"  pos_weight:     {pos}")
    print("===========================\n")

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_csv = args.train_csv
    val_csv   = args.val_csv
    test_csv  = args.test_csv
    cached_dir = args.cached_dir
    img_size = 512

    pos_weight_path = Path(args.pos_weight_path) if args.pos_weight_path \
        else (ROOT / "configs" / "pos_weight.json")
    pos = json.loads(pos_weight_path.read_text())["pos_weight"]
    

    dm = ChestXray14DataModule(
        train_csv=train_csv, 
        val_csv=val_csv,
        test_csv=test_csv,
        img_size=img_size, 
        batch_size=args.batch_size, 
        num_workers=args.num_worker,
        cached_dir=cached_dir,
        pin_memory=True, 
        persistent_workers=True,
        mean=0.5, 
        std=0.25,
    )

    logger = TensorBoardLogger(
        save_dir=args.log_dir,
        name=args.log_name
    )

    model = lit_module.LitChestXray(
        in_chans=1, 
        num_classes=len(LABELS), 
        lr=1e-4, 
        weight_decay=1e-4, 
        total_epochs=args.total_epochs,
        thresholds_path=args.thresholds_path,
        loss_func=args.loss_func,
        block = cnn.ResidualBlock,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath= args.ckpt_dir,
        monitor="val_mAP", 
        mode="max", 
        save_top_k=1, 
        filename="resnet50_cbam_AdamW_{epoch:03d}_100-{val_mAP:.4f}",
        save_weights_only=False
    )
    
    es_cb = EarlyStopping(monitor="val_mAP", mode="max", patience=10, verbose=True)
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    print_args(args, pos)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        precision="32",
        accelerator="gpu", 
        devices=1,
        log_every_n_steps=100,
        callbacks=[ckpt_cb, lr_cb, es_cb],
        gradient_clip_val=1.0,
        logger=logger
    )

    trainer.fit(model, datamodule=dm)

    if args.run_test:
        trainer.test(model, datamodule=dm, ckpt_path="best")

if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--data-dir", default=str(RESIZED))
    p.add_argument("--train-csv", type=str, default=str(RESIZED / "train.csv"))
    p.add_argument("--val-csv",   type=str, default=str(RESIZED / "val.csv"))
    p.add_argument("--test-csv",  type=str, default=str(RESIZED / "test.csv"))
    p.add_argument("--cached-dir", default=str(RESIZED))
    p.add_argument("--ckpt-dir",  default=str(ROOT / "logs" / "checkpoint_resnet50_100"))
    p.add_argument("--log-dir",  default="logs")
    p.add_argument("--log-name",  default="resnet50_cbam_AdamW_100")

    p.add_argument("--pos-weight-path", type=str, default=None)
    p.add_argument("--thresholds-path", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-worker", type=int, default=12)

    p.add_argument("--total-epochs", type=int, default=100)
    p.add_argument("--loss-func", type=str, default="binary_cross_entropy_with_logits")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--run_test", action="store_true", default=True)
    args = p.parse_args()
    main(args)