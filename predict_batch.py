# predict_batch.py (放在 Project 資料夾裡面)
import sys
import os
import csv
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from unet import Unet  # 確保同資料夾下有 unet.py

SEAGRASS_CLASS = 1
IMG_EXTENSIONS = ('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')

def compute_coverage(pr_mask):
    return np.sum(pr_mask == SEAGRASS_CLASS) / pr_mask.size

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("LOG|[錯誤] 參數不足")
        sys.exit(1)

    csv_in = sys.argv[1]
    csv_out = sys.argv[2]
    img_dir = sys.argv[3]
    model_path = sys.argv[4]
    col_name = sys.argv[5]+"_coverage"

    print("LOG|[系統] 正在將模型載入 GPU... (這可能需要幾秒鐘，請稍候)", flush=True)
    try:
        # 關鍵：模型在這裡只會載入一次！
        unet = Unet(model_path=model_path)
    except Exception as e:
        print(f"LOG|[錯誤] 模型載入失敗: {e}", flush=True)
        sys.exit(1)

    print("LOG|[系統] 模型載入完成！開始高速批次辨識...", flush=True)

    # 讀取 CSV
    with open(csv_in, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if col_name not in fieldnames:
            fieldnames.append(col_name)
        rows = list(reader)

    # 遞迴搜尋圖片路徑
    available_images = {}
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.lower().endswith(IMG_EXTENSIONS):
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, img_dir).replace('\\', '/')
                available_images[rel_path] = abs_path

    total_rows = len(rows)
    processed_count = 0

    # 在 Docker 內部跑迴圈辨識
    for i, row in enumerate(rows):
        img_name = row.get("seagrass_image_name", "").strip()
        
        if not img_name:
            # 空行略過，但還是要更新進度條
            progress = int(((i + 1) / total_rows) * 100)
            print(f"PROGRESS|{progress}", flush=True)
            continue
            
        if img_name in available_images:
            img_path = available_images[img_name]
            print(f"LOG|正在辨識: {img_name}...", flush=True)
            try:
                image = Image.open(img_path)
                _, pred_mask = unet.detect_image(image, return_mask=True)
                coverage = compute_coverage(pred_mask)
                row[col_name] = f"{coverage*100:.6f}"
                processed_count += 1
            except Exception as e:
                print(f"LOG|[錯誤] 辨識失敗 ({img_name}): {e}", flush=True)
                row[col_name] = "Error"
        else:
            print(f"LOG|[跳過] 找不到圖片: {img_name}", flush=True)
            if col_name not in row or row[col_name] == "":
                row[col_name] = "Not Found"

        # 回報進度給外部的 UI 程式
        progress = int(((i + 1) / total_rows) * 100)
        print(f"PROGRESS|{progress}", flush=True)

    # 將結果寫入新 CSV
    print("LOG|[系統] 正在儲存結果...", flush=True)
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"DONE|{processed_count}", flush=True)