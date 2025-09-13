from pathlib import Path
import pandas as pd
import os
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nih_kaggle"
ENRTY_CSV = RAW / "Data_Entry_2017.csv"
ALL_IMG_DIR = ROOT / "data" / "processed" / "images_all"

LABELS = ["Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
          "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
          "Emphysema","Fibrosis","Pleural_Thickening","Hernia"]


def combine_all_img():
    ALL_IMG_DIR.mkdir(parents=True, exist_ok=True)
    for p in RAW.glob("images_*/images/*"):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        d = ALL_IMG_DIR / p.name
        if d.exists(): continue
        try:
            d.symlink_to(p.resolve())
        except OSError:
            os.link(p.resolve(), d)


def main():
    combine_all_img()
    df = pd.read_csv(ENRTY_CSV)
    df = df.iloc[:,:-1]
    df["labels"] = df["Finding Labels"].fillna("").str.split("|")

    rows = []
    for _, row in df.iterrows():
        y = {k:0 for k in LABELS}
        for t in row["labels"]:
            if t == "No Finding" or t == "": continue
            if t in y: y[t] = 1
        print(f"{str((ALL_IMG_DIR / row["Image Index"]))}")
        rows.append({
            "image_path": str((ALL_IMG_DIR / row["Image Index"])),
            "patient_id": row["Patient ID"],
            "ori_img_w": row["OriginalImage[Width"],
            "ori_img_h": row["Height]"],
            "ori_img_spacing_x": row["OriginalImagePixelSpacing[x"],
            "ori_img_spacing_y": row["y]"],
            **y
        })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "processed" / "labels_full.csv", index=False)
    print("Wrote labels_full.csv:", len(out))

if __name__ == "__main__":
    main()