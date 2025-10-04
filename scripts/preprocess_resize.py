from pathlib import Path
import pandas as pd
import cv2
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nih_kaggle"
ALL_IMG_DIR = ROOT / "data" / "processed" / "images_all"
OUT = ROOT / "data" / "processed"/ "resized"
LABELS_CSV = ROOT / "data" / "processed"/ "labels_full.csv"


def resize(img_path, img_name, size=512):
    out_dir = OUT / str(size)
    os.makedirs(out_dir, exist_ok=True)
    out_path = out_dir / str(img_name)
    if os.path.exists(out_path):
        print(f"Skip {out_path} existed")
        return out_path
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"file {img_path} not found.")
    h, w = img.shape
    scale = size / max(h, w)
    resized_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    print(f"Saving to: {out_path}")
    success = cv2.imwrite(str(out_path), resized_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    if not success:
        raise RuntimeError(f"Failed to save image to {out_path}")
    return out_path
    

def main():
    print("preprocess_resize")
    try:
        with open(RAW / "train_val_list.txt", "r") as f:
            entries = f.readlines()
    except Exception as e:
        print(f"An error occurred: {e}")   

    train_val = [line.strip() for line in entries]

    df = pd.read_csv(LABELS_CSV)

    train_val_data = []
    test_data = []
    size = 512

    for _, img in df.iterrows():
        row_data = img.copy()
        out_path = resize(img_name=img["image_idx"], img_path=img["image_path"], size=size)
        row_data["image_path"] = out_path
        if img["image_idx"] in train_val :
            train_val_data.append(row_data)
        else:
            test_data.append(row_data)

    train_val_df = pd.DataFrame(train_val_data)
    test_df = pd.DataFrame(test_data)

    csv_dir = OUT / str(size)
    csv_dir.mkdir(parents=True, exist_ok=True)
    train_val_df.to_csv(csv_dir / "train_val.csv", index=False)
    test_df.to_csv(csv_dir / "test.csv", index=False)

    if input(f"Are you sure you want to delete raw data at {RAW}? (y/N) ").lower() == 'y':
        try:
            shutil.rmtree(RAW)
            print(f"Directory '{RAW}' and its contents removed successfully.")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}")
    else:
        print("Deletion cancelled.")

if __name__ == "__main__":
    main()
