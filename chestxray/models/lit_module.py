import torch
import math
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision.ops import sigmoid_focal_loss
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
import numpy as np
from .cnn import CNN, ResidualBlock, BottleneckBlock
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from ..utils.confusion_matrix import plot_multilabel_confusion_matrix
from ..utils.tune_thresholds import tune_thresholds
from  typing import Optional, Union
import json
from pathlib import Path
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]

LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
          "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
          "Emphysema","Fibrosis","Pleural_Thickening","Hernia"]

class LitChestXray(pl.LightningModule):
    def __init__(
            self,
            in_chans=1, 
            num_classes=14,
            block: ResidualBlock | BottleneckBlock = BottleneckBlock,
            lr=3e-4, 
            weight_decay=1e-4, 
            # pos_weight=None,
            total_epochs=100,
            thresholds_path: str | None = None
        ):
        super().__init__()
        self.total_epochs=total_epochs
        
        if thresholds_path is None:
            path = ROOT / "configs" /"thresholds.json"
            base = path.stem
            suffix = path.suffix
            counter = 1
            while path.exists():
                path = path.with_name(f"{base}_{counter}{suffix}")
                counter += 1
            self.thresholds_path = str(path)
        else:
            self.thresholds_path = thresholds_path

        self.save_hyperparameters()
        
        self._val_probs, self._val_targs = [], []
        self.model = CNN(
            num_classes=num_classes, 
            in_chans=in_chans,
            block=block
        )
        self.register_buffer(
            "thresholds", 
            torch.full((num_classes,), 0.5, dtype=torch.float32), 
            persistent=True
        )

        # self.register_buffer("pos_weight", None)
        # if pos_weight is not None:
        #     self.pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
            # self.loss_fn = MultilabelFocalLoss(self.pos_weight)
        self.auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.ap    = MultilabelAveragePrecision(num_labels=num_classes, average="macro")

        self.test_auroc = MultilabelAUROC(num_labels=num_classes, average="macro")
        self.test_ap    = MultilabelAveragePrecision(num_labels=num_classes, average="macro")
        self.test_probs_buf = []
        self.test_targs_buf  = []

    def setup(self, stage: str | None = None):
        if stage in {"validate", "test"} and self.thresholds_path:
            p = Path(self.thresholds_path)
            if p.exists():
                data = json.loads(p.read_text())
                ths = torch.tensor(data["thresholds"], dtype=torch.float32, device=self.device)
                print(ths)
                if ths.numel() == self.thresholds.numel():
                    self.thresholds.copy_(ths)
                else:
                    self.print(f"[warn] thresholds size mismatch: "
                            f"file={ths.numel()} vs model={self.thresholds.numel()}. Using in-memory thresholds.")

    
    def forward(self, x):
        return self.model(x)


    def training_step(self, batch, _):
        x, y = batch["image"], batch["target"].float()
        logits = self.model(x)
        # loss = F.binary_cross_entropy_with_logits(
        #     input=logits,
        #     target= y, 
        #     pos_weight=self.pos_weight
        # )
        loss = sigmoid_focal_loss(
            inputs=logits,
            targets= y, 
            # pos_weight=self.pos_weight
            reduction="mean"
        )
        # loss = self.loss_fn(logits, y)
        bs = x.size(0)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=bs, logger=True)
        return loss
    

    def validation_step(self, batch, _):
        x, y = batch["image"], batch["target"].float()
        logits = self.model(x)
        # loss = F.binary_cross_entropy_with_logits(
        #     input=logits,
        #     target= y, 
        #     pos_weight=self.pos_weight
        # )
        loss = sigmoid_focal_loss(
            inputs=logits,
            targets= y, 
            # pos_weight=self.pos_weight
            reduction="mean"
        )
        # loss = self.loss_fn(logits, y)
        probs = logits.sigmoid()
        bs = x.size(0)
        self.auroc.update(probs, y.int()); self.ap.update(probs, y.int())
        self._val_probs.append(probs.detach().cpu())
        self._val_targs.append(y.detach().cpu())
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=bs, logger=True)


    def on_validation_epoch_end(self):
        auroc = self.auroc.compute()
        ap = self.ap.compute()
        self.log("val_AUROC", auroc, prog_bar=True)
        self.log("val_mAP", ap, prog_bar=True)
        self.auroc.reset(); self.ap.reset()

        probs = torch.cat(self._val_probs, 0).numpy()
        targs = torch.cat(self._val_targs, 0).numpy().astype(int)
        self._val_probs.clear(); self._val_targs.clear()

        ths = tune_thresholds(probs, targs)            # np.array [C]
        # store in a buffer for later (moves with device; saved in checkpoint)
        self.thresholds = torch.tensor(ths, dtype=torch.float32, device=self.device)

        if self.thresholds_path:
            Path(self.thresholds_path).parent.mkdir(parents=True, exist_ok=True)
            Path(self.thresholds_path).write_text(json.dumps({"thresholds": ths.tolist()}, indent=2))

        yhat = (probs >= ths).astype(int)
        macro_f1 = f1_score(targs, yhat, average="macro", zero_division=0)
        self.log("val_F1_macro@tuned", macro_f1, prog_bar=True)


    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, 
            T_max=self.total_epochs
        )
        return {"optimizer": opt, "lr_scheduler": sch}
    

    def on_test_start(self):
        if self.thresholds_path:
            p = Path(self.thresholds_path)
            if p.exists():
                data = json.loads(p.read_text())
                ths = torch.tensor(data["thresholds"], dtype=torch.float32, device=self.device)
                if ths.numel() == self.thresholds.numel():
                    self.thresholds.copy_(ths)
    

    def test_step(self, batch, _):
        x, y = batch["image"], batch["target"].float()
        logits = self.model(x)
        loss = sigmoid_focal_loss(
            inputs=logits,
            targets= y, 
            # pos_weight=self.pos_weight
            reduction="mean"
        )
        # loss = F.binary_cross_entropy_with_logits(
        #     input=logits,
        #     target= y, 
        #     pos_weight=self.pos_weight
        # )
        # loss = self.loss_fn(logits, y)
        probs = logits.sigmoid()
        self.test_auroc.update(probs, y.int())
        self.test_ap.update(probs, y.int())
        bs = x.size(0)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, batch_size=bs)
        self.test_probs_buf.append(probs.detach().cpu())
        self.test_targs_buf.append(y.detach().cpu())
        return None


    def on_test_epoch_end(self):
        self.log("test_AUROC", self.test_auroc.compute(), prog_bar=True)
        self.log("test_mAP", self.test_ap.compute(), prog_bar=True)

        probs = torch.cat(self.test_probs_buf).numpy()
        ytrue = torch.cat(self.test_targs_buf).numpy().astype(int)
        self.test_probs_buf.clear(); self.test_targs_buf.clear()

        ths = self.thresholds.detach().cpu().numpy()
        if ths.size == 0 and self.thresholds_path and Path(self.thresholds_path).exists():
            ths = np.array(json.loads(Path(self.thresholds_path).read_text())["thresholds"], dtype=np.float32)
        if ths.size == 0:     # fallback
            ths = np.full(probs.shape[1], 0.5, dtype=np.float32)

        print(ths)

        ypred = (probs >= ths).astype(int)
        cms = multilabel_confusion_matrix(ytrue, ypred)


        report = classification_report(ytrue, ypred, target_names=LABELS, zero_division=0)
        print(report)

        prev = ytrue.mean(axis=0)
        posr = ypred.mean(axis=0)
        print("prevalence:", prev.round(4))
        print("pred+ rate:", posr.round(4))
        plot_multilabel_confusion_matrix(cms, LABELS)

        # np.save("test_confusion_matrices.npy", cms)
        # open("test_classification_report.txt","w").write(report)

        self.test_auroc.reset()
        self.test_ap.reset()
        self.test_probs_buf.clear()
        self.test_targs_buf.clear()


if __name__ == "__main__":
    print(ROOT)