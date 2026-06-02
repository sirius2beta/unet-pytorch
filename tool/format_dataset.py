import os
import random
import shutil
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# -------------------------------------------------------#
#   設定參數與比例
# -------------------------------------------------------#
trainval_percent = 1
train_percent = 0.8

# 基底路徑與 A 資料夾路徑（拿掉 FOLDER_B）
base_path = r"C:\project\unet-pytorch"
FOLDER_A = r"C:\project\unet-pytorch\202507_DS"

# 最終 VOC 資料夾結構的根目錄
VOCdevkit_path = os.path.join(base_path, 'VOCdevkit')

# 目的地 VOC 子資料夾路徑
dst_img_dir = os.path.join(VOCdevkit_path, 'VOC2007', 'JPEGImages')
dst_mask_dir = os.path.join(VOCdevkit_path, 'VOC2007', 'SegmentationClass')
saveBasePath = os.path.join(VOCdevkit_path, 'VOC2007', 'ImageSets', 'Segmentation')


def create_voc_folder_structure(voc_path):
    """建立 VOC 資料夾結構"""
    folders = [
        'VOC2007/JPEGImages',
        'VOC2007/SegmentationClass',
        'VOC2007/ImageSets/Segmentation',
    ]
    for folder in folders:
        path = os.path.join(voc_path, folder)
        os.makedirs(path, exist_ok=True)
        print(f"已建立: {path}")
    print("VOC 資料夾結構建立完成")


def process_and_merge_to_voc(folder_a):
    """
    直接讀取 CVAT 的 folder_a，將其處理、歸一化後，直接寫入到 VOCdevkit 之中。
    """
    splits = ['train', 'test']
    pic_basenames = []  # 用來記錄所有處理過的圖片 base_name，以便後續檢查/補齊 mask

    for split in splits:
        src_jpeg_split = os.path.join(folder_a, 'JPEGImages', split)
        src_seg_split = os.path.join(folder_a, 'SegmentationClass', split)

        if not os.path.exists(src_jpeg_split):
            print(f"[{split}] 找不到來源圖片資料夾 {src_jpeg_split}，跳過此部分。")
            continue

        print(f"\n========== 正在處理 {split} 資料，直接寫入 VOC 結構 ==========")

        # 1. 處理 JPEGImages (直接扁平化命名並複製到 VOCdevkit/VOC2007/JPEGImages)
        print(f"[{split}] 正在複製並重新命名圖片...")
        for root, dirs, files in os.walk(src_jpeg_split):
            for file in files:
                if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue

                src_pic_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_pic_path, src_jpeg_split)
                rel_parts = rel_path.split(os.sep)
                
                # 新檔名加上 split 前綴以避免 train 和 test 內有同名資料夾導致衝突
                new_filename = f"{split}_" + "_".join(rel_parts)
                dst_pic_path = os.path.join(dst_img_dir, new_filename)

                shutil.copy2(src_pic_path, dst_pic_path)

                base_name = os.path.splitext(new_filename)[0]
                pic_basenames.append((base_name, dst_pic_path))

        # 2. 處理 SegmentationClass (直接歸一化並寫入 VOCdevkit/VOC2007/SegmentationClass)
        if os.path.exists(src_seg_split):
            print(f"[{split}] 正在讀取原始標籤、進行歸一化(0/1)並寫入...")
            for root, dirs, files in os.walk(src_seg_split):
                for file in files:
                    if not file.lower().endswith('.png'):
                        continue

                    src_mask_path = os.path.join(root, file)
                    rel_path = os.path.relpath(src_mask_path, src_seg_split)
                    rel_parts = rel_path.split(os.sep)
                    
                    new_filename = f"{split}_" + "_".join(rel_parts)
                    dst_mask_path = os.path.join(dst_mask_dir, new_filename)

                    # 讀取並直接進行二值化歸一化 (0 或 1)
                    img = cv2.imread(src_mask_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = np.clip(img, 0, 1)
                        cv2.imwrite(dst_mask_path, img)
                    else:
                        print(f"讀取遮罩失敗: {src_mask_path}")

    # 3. 以已生成的圖片為基準，檢查並自動補齊缺失的全黑 mask
    print("\n========== 開始檢查缺失的遮罩並自動補件 ==========")
    generated_count = 0
    for base_name, dst_pic_path in tqdm(pic_basenames, desc="檢查缺失的 mask"):
        expected_mask_name = base_name + '.png'
        expected_mask_path = os.path.join(dst_mask_dir, expected_mask_name)

        if not os.path.exists(expected_mask_path):
            try:
                with Image.open(dst_pic_path) as img:
                    w, h = img.size

                # 建立 0-1 格式的純黑單通道圖 (UNet 訓練用)
                mask = Image.new('L', (w, h), 0)
                mask.save(expected_mask_path)
                generated_count += 1
            except Exception as e:
                print(f"無法讀取圖片 {dst_pic_path} 以生成 mask: {e}")

    print(f"補件完成！共自動生成了 {generated_count} 張純黑 mask。")


if __name__ == "__main__":
    random.seed(0)

    print(f"來源資料夾 (A): {FOLDER_A}")
    print(f"目標 VOCdevkit 路徑: {VOCdevkit_path}")

    # 執行步驟 1: 建立 VOC 資料夾結構
    create_voc_folder_structure(VOCdevkit_path)

    # 執行步驟 2: 整理、歸一化並直接寫入 VOC 結構 (不再經由 FOLDER_B)
    process_and_merge_to_voc(FOLDER_A)

    # 執行步驟 3: 產生 ImageSets 中的 txt 劃分檔案
    print("\n========== 開始產生 ImageSets 中的 txt 檔案 ==========")
    temp_seg = os.listdir(dst_mask_dir)
    total_seg = [seg for seg in temp_seg if seg.endswith(".png")]

    num = len(total_seg)
    list_index = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list_index, tv)
    train = random.sample(trainval, tr)

    print(f"總標籤數量: {num}")
    print(f"訓練與驗證集大小 (trainval size): {tv}")
    print(f"訓練集大小 (train size): {tr}")

    with open(os.path.join(saveBasePath, 'trainval.txt'), 'w') as ftrainval, \
            open(os.path.join(saveBasePath, 'test.txt'), 'w') as ftest, \
            open(os.path.join(saveBasePath, 'train.txt'), 'w') as ftrain, \
            open(os.path.join(saveBasePath, 'val.txt'), 'w') as fval:

        for i in list_index:
            name = total_seg[i][:-4] + '\n'
            if i in trainval:
                ftrainval.write(name)
                if i in train:
                    ftrain.write(name)
                else:
                    fval.write(name)
            else:
                ftest.write(name)

    print("txt 檔案產生完成。")

    # 執行步驟 4: 檢查資料集像素格式
    print("\n正在檢查資料集格式是否符合要求...")
    classes_nums = np.zeros([256], np.int64)
    for i in tqdm(list_index, desc="檢查像素"):
        name = total_seg[i]
        png_file_name = os.path.join(dst_mask_dir, name)

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print(f"\n警告：標籤圖片 {name} 的 shape 為 {str(np.shape(png))}，不屬於灰階圖。")

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("\n印出像素點的值與數量：")
    print('-' * 37)
    print(f"| {'Key':>15} | {'Value':>15} |")
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print(f"| {str(i):>15} | {str(classes_nums[i]):>15} |")
            print('-' * 37)

    if classes_nums[255] > 0:
        print("警告：檢測到標籤中像素點的值仍包含 255，請確認二值化步驟是否正常運作。")
    elif classes_nums[0] > 0 and classes_nums[1] > 0:
        print("成功：標籤格式正確！像素值僅包含 0 (背景) 與 1 (目標)，適合二分類 UNet 訓練。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("提示：檢測到標籤中僅包含背景像素點 (0)。")

    print("\n全部流程整合完畢！您已跳過中間過渡資料夾，直接生成了標準 VOCdevkit 資料夾。")