import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from unet import Unet

def calculate_metrics_from_cm(cm):
    """直接從累加好的混淆矩陣計算各項評估指標"""
    print("Confusion Matrix:\n", cm)

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    # 各類別指標 (加上 1e-6 避免除以零)
    recall = TP / (TP + FN + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6)
    fallout = FP / (FP + TN + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)

    # 平均指標
    metrics = {
        "mRecall": np.mean(recall),
        "mPrecision": np.mean(precision),
        "mAcc": np.mean(accuracy),
        "mFallout": np.mean(fallout),
        "mIoU": np.mean(iou)
    }

    print("\n--- Evaluation Metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics


def predict_directory_from_txt(model, dir_origin_path, txt_path, dir_save_jpeg_path, dir_save_mask_path, dir_mask_path, name_classes):
    """讀取 txt 檔內的圖片名稱進行批量預測並計算指標 (優化記憶體版)"""
    num_classes = len(name_classes)
    total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    has_valid_gt = False 

    # 建立兩個輸出資料夾
    os.makedirs(dir_save_jpeg_path, exist_ok=True)
    os.makedirs(dir_save_mask_path, exist_ok=True)

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"找不到指定的 txt 檔案: {txt_path}")
        
    with open(txt_path, 'r') as f:
        img_names = [line.strip() for line in f.readlines() if line.strip()]

    print(f"Total images to predict: {len(img_names)}")

    for img_name in tqdm(img_names):
        img_filename = img_name if img_name.endswith('.jpg') else f"{img_name}.jpg"
        mask_filename = img_name.replace('.jpg', '.png') if img_name.endswith('.jpg') else f"{img_name}.png"

        image_path = os.path.join(dir_origin_path, img_filename)
        
        try:
            # 讀取原始圖片
            image = Image.open(image_path)
        except FileNotFoundError:
            print(f"\nWarning: Image not found {image_path}. Skipping...")
            continue

        # 模型推論 (r_image 為混合圖, pred_mask 為預測遮罩)
        r_image, pred_mask = model.detect_image(image, return_mask=True)

        # 讀取 Ground Truth 進行指標計算
        gt_mask_path = os.path.join(dir_mask_path, mask_filename)
        try:
            gt_mask = Image.open(gt_mask_path)
            gt_mask = gt_mask.resize(pred_mask.shape[::-1] if isinstance(pred_mask, np.ndarray) else pred_mask.size, Image.NEAREST)
            
            cm = confusion_matrix(
                np.array(gt_mask).flatten(), 
                np.array(pred_mask).flatten(), 
                labels=list(range(num_classes))
            )
            total_cm += cm
            has_valid_gt = True
            
        except FileNotFoundError:
            pass 

        # ===== 儲存推論結果圖片 =====
        # 1. 儲存原圖至 JPEGImages 資料夾
        image.save(os.path.join(dir_save_jpeg_path, img_filename))
        
        # 2. 儲存預測的遮罩 (mask) 至 SegmentationClass 資料夾
        mask_save_path = os.path.join(dir_save_mask_path, mask_filename)
        if isinstance(pred_mask, Image.Image):
            pred_mask.save(mask_save_path)
        else:
            Image.fromarray(np.uint8(pred_mask)).save(mask_save_path)

    # 計算全域指標
    if has_valid_gt:
        print("\nCalculating global metrics...")
        calculate_metrics_from_cm(total_cm)
    else:
        print("\nNo valid ground truths found to calculate metrics.")


def main():
    # ================= 參數設定區 =================
    # 類別設定
    name_classes    = ["background", "seagrass"]
    
    # 路徑設定
    dir_origin_path = "VOCdevkit/VOC2007/JPEGImages/"
    txt_path        = "VOCdevkit/VOC2007/ImageSets/Segmentation/test.txt"
    
    # 修改輸出路徑為兩個獨立資料夾
    dir_save_jpeg_path  = "imgout/JPEGImages"
    dir_save_mask_path  = "imgout/SegmentationClass"
    
    # Ground Truth Mask 的路徑 (用於計算 mIoU 等指標)
    dir_mask_path   = "VOCdevkit/VOC2007/SegmentationClass/" 
    # ============================================

    print("Initializing Unet model...")
    model = Unet()

    print(f"Starting directory prediction based on {txt_path}...")
    predict_directory_from_txt(
        model=model, 
        dir_origin_path=dir_origin_path, 
        txt_path=txt_path, 
        dir_save_jpeg_path=dir_save_jpeg_path,   # 傳入圖片輸出路徑
        dir_save_mask_path=dir_save_mask_path,   # 傳入遮罩輸出路徑
        dir_mask_path=dir_mask_path,
        name_classes=name_classes
    )
    print("Prediction completely finished!")

if __name__ == "__main__":
    main()