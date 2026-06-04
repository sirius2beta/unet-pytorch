import os
import zipfile
import shutil
import json
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

CONFIG_FILE = "config.json"

class DatasetMergerVOCApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CVAT 合併與 VOC 資料集生成工具 (Dark Theme)")
        self.root.geometry("650x550")
        
        self._apply_dark_theme()
        
        self.project_dir = tk.StringVar()
        self.dataset_vars = {} 

        self._build_ui()
        self._load_config()

    def _apply_dark_theme(self):
        """配置整體深色風格"""
        bg_color = "#2b2b2b"
        fg_color = "#e8e8e8"
        panel_bg = "#3c3f41"
        accent_color = "#4b6eaf"
        export_bg = "#388E3C"
        export_hover = "#4CAF50"

        self.root.configure(bg=bg_color)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure(".", background=bg_color, foreground=fg_color, font=("Segoe UI", 10))
        style.configure("TFrame", background=bg_color)
        style.configure("TLabelframe", background=bg_color, bordercolor="#555555")
        style.configure("TLabelframe.Label", background=bg_color, foreground="#a9b7c6", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=bg_color, foreground=fg_color)
        style.configure("TEntry", fieldbackground=panel_bg, foreground=fg_color, insertcolor=fg_color, bordercolor="#555555")
        style.map("TEntry", fieldbackground=[("readonly", panel_bg)], foreground=[("readonly", fg_color)])
        style.configure("TCheckbutton", background=bg_color, foreground=fg_color)
        style.map("TCheckbutton", background=[("active", bg_color)], indicatorcolor=[("selected", accent_color)])
        style.configure("TButton", background=panel_bg, foreground=fg_color, borderwidth=1, bordercolor="#555555", focuscolor=panel_bg)
        style.map("TButton", background=[("active", accent_color)])
        style.configure("Export.TButton", font=("Segoe UI", 11, "bold"), background=export_bg, foreground="white")
        style.map("Export.TButton", background=[("active", export_hover)])
        style.configure("Vertical.TScrollbar", background=panel_bg, troughcolor=bg_color, bordercolor=bg_color, arrowcolor=fg_color)
        style.map("Vertical.TScrollbar", background=[("active", accent_color)])

    def _build_ui(self):
        # --- 頂部：選擇專案資料夾 ---
        top_frame = ttk.Frame(self.root, padding=15)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="專案根目錄 (Project):").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Entry(top_frame, textvariable=self.project_dir, width=40, state='readonly').pack(side=tk.LEFT, padx=5, ipady=3)
        ttk.Button(top_frame, text="選擇資料夾...", command=self.browse_folder).pack(side=tk.LEFT, padx=5)

        # --- 中間：資料集列表 ---
        middle_frame = ttk.LabelFrame(self.root, text=" 資料集列表 (將自動排除 export) ", padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        self.canvas = tk.Canvas(middle_frame, bg="#2b2b2b", highlightthickness=0)
        scrollbar = ttk.Scrollbar(middle_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- 底部：匯出按鈕 ---
        bottom_frame = ttk.Frame(self.root, padding=15)
        bottom_frame.pack(fill=tk.X)
        ttk.Button(bottom_frame, text="🚀 執行合併與生成 VOC", style="Export.TButton", command=self.process_pipeline).pack(ipady=5, ipadx=10)

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    saved_dir = config.get("project_dir", "")
                    if saved_dir and os.path.isdir(saved_dir):
                        self.set_project(saved_dir)
            except Exception:
                pass

    def _save_config(self, path):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump({"project_dir": path}, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"儲存設定檔失敗: {e}")

    def browse_folder(self):
        folder_selected = filedialog.askdirectory(title="請選擇專案目錄 (如 unet-pytorch)")
        if folder_selected:
            self.set_project(folder_selected)

    def set_project(self, path):
        anno_dir = os.path.join(path, "annotation")
        if not os.path.exists(anno_dir):
            messagebox.showerror("錯誤", f"選擇的專案目錄下找不到 'annotation' 資料夾！\n\n您選擇的路徑: {path}\n請確保裡面包含 annotation 資料夾。")
            return
            
        self.project_dir.set(path)
        self._save_config(path)
        self.scan_datasets(anno_dir)

    def scan_datasets(self, anno_path):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.dataset_vars.clear()

        # 標題行
        ttk.Label(self.scrollable_frame, text="資料集名稱", width=35, anchor="w", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Label(self.scrollable_frame, text="Train", width=12, anchor="center", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, pady=(0, 5))
        ttk.Label(self.scrollable_frame, text="Test", width=12, anchor="center", font=("Segoe UI", 10, "bold")).grid(row=0, column=2, pady=(0, 5))
        ttk.Separator(self.scrollable_frame, orient="horizontal").grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 5))

        row = 2
        for item in sorted(os.listdir(anno_path)):
            item_path = os.path.join(anno_path, item)
            if os.path.isdir(item_path) and item.lower() != 'export':
                zip_path = os.path.join(item_path, 'cvat.zip')
                if os.path.exists(zip_path):
                    var_train = tk.BooleanVar(value=False)
                    var_test = tk.BooleanVar(value=False)
                    self.dataset_vars[item] = {'train': var_train, 'test': var_test}

                    ttk.Label(self.scrollable_frame, text=f"📁 {item}", width=35, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
                    ttk.Checkbutton(self.scrollable_frame, variable=var_train).grid(row=row, column=1)
                    ttk.Checkbutton(self.scrollable_frame, variable=var_test).grid(row=row, column=2)
                    row += 1

    def process_pipeline(self):
        project_path = self.project_dir.get()
        if not project_path:
            messagebox.showwarning("警告", "請先選擇 Project 專案資料夾！")
            return

        anno_dir = os.path.join(project_path, 'annotation')
        export_dir = os.path.join(anno_dir, 'export')
        voc_dir = os.path.join(project_path, 'VOCdevkit')

        # === 1. 處理 export 目錄清理 ===
        if os.path.exists(export_dir):
            if not messagebox.askyesno("確認", f"準備清空 {export_dir} 並重新合併資料集，是否繼續？"):
                return
            shutil.rmtree(export_dir)

        # === 2. 處理 VOCdevkit 目錄清理 ===
        if os.path.exists(voc_dir):
            if not messagebox.askyesno("確認", f"發現已存在的 VOCdevkit 目錄：\n{voc_dir}\n\n為了確保資料純淨，將會清空重建，是否繼續？"):
                return
            shutil.rmtree(voc_dir)

        print("\n" + "="*50)
        print("階段一：開始解壓並合併 CVAT 資料至 export")
        print("="*50)
        
        try:
            self._extract_and_merge(anno_dir, export_dir)
        except Exception as e:
            messagebox.showerror("錯誤", f"合併資料集至 export 時發生錯誤:\n{str(e)}")
            return

        print("\n" + "="*50)
        print("階段二：將 export 資料轉換並封裝為 VOCdevkit")
        print("="*50)
        
        try:
            stats = self._build_voc_structure(export_dir, voc_dir)
            
            # 整合成功訊息
            success_msg = (
                f"✅ 全部流程執行完畢！\n\n"
                f"📊 [ VOC 資料集統計 ]\n"
                f"- 自動補齊黑圖: {stats['generated']} 張\n"
                f"- 總樣本數量: {stats['total']}\n"
                f"  └ Train: {stats['train']}\n"
                f"  └ Val:   {stats['val']}\n"
                f"  └ Test:  {stats['test']}\n\n"
                f"📁 VOCdevkit 已生成於:\n{voc_dir}"
            )
            messagebox.showinfo("完成", success_msg)

        except Exception as e:
            messagebox.showerror("錯誤", f"生成 VOCdevkit 時發生錯誤:\n{str(e)}")
            import traceback
            traceback.print_exc()

    def _extract_and_merge(self, base_dir, export_dir):
        """將勾選的 ZIP 檔解壓並合併到 export_dir 中"""
        for ds_name, vars_dict in self.dataset_vars.items():
            if not vars_dict['train'].get() and not vars_dict['test'].get():
                continue

            selected_splits = []
            if vars_dict['train'].get(): selected_splits.append('train')
            if vars_dict['test'].get(): selected_splits.append('test')

            zip_path = os.path.join(base_dir, ds_name, 'cvat.zip')
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for file_info in zf.infolist():
                    if file_info.is_dir(): continue
                    parts = file_info.filename.split('/')
                    if len(parts) >= 3:
                        root_folder, split = parts[0], parts[1]
                        filename = parts[-1]

                        if root_folder in ['JPEGImages', 'SegmentationClass'] and split in selected_splits:
                            subfolders = parts[2:-1]
                            target_dir = os.path.join(export_dir, root_folder, split, *subfolders)
                            os.makedirs(target_dir, exist_ok=True)

                            # 檔名前綴防碰撞
                            prefix = "_".join(subfolders) if subfolders else ""
                            new_filename = f"{ds_name}_{prefix}_{filename}" if prefix else f"{ds_name}_{filename}"
                            target_path = os.path.join(target_dir, new_filename)

                            with zf.open(file_info) as source, open(target_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
        print(f"-> 成功輸出至: {export_dir}")

    def _build_voc_structure(self, folder_a, voc_path):
        """完全依照使用者的腳本邏輯，處理歸一化、建立 VOC 與切割資料集"""
        trainval_percent = 1
        train_percent = 0.8
        random.seed(0)

        # 建立結構
        dst_img_dir = os.path.join(voc_path, 'VOC2007', 'JPEGImages')
        dst_mask_dir = os.path.join(voc_path, 'VOC2007', 'SegmentationClass')
        saveBasePath = os.path.join(voc_path, 'VOC2007', 'ImageSets', 'Segmentation')
        
        for path in [dst_img_dir, dst_mask_dir, saveBasePath]:
            os.makedirs(path, exist_ok=True)

        splits = ['train', 'test']
        pic_basenames = []

        for split in splits:
            src_jpeg_split = os.path.join(folder_a, 'JPEGImages', split)
            src_seg_split = os.path.join(folder_a, 'SegmentationClass', split)

            if not os.path.exists(src_jpeg_split):
                continue

            print(f"[{split}] 正在處理並重新命名圖片...")
            for root, dirs, files in os.walk(src_jpeg_split):
                for file in files:
                    if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        continue
                    src_pic_path = os.path.join(root, file)
                    rel_parts = os.path.relpath(src_pic_path, src_jpeg_split).split(os.sep)
                    new_filename = f"{split}_" + "_".join(rel_parts)
                    dst_pic_path = os.path.join(dst_img_dir, new_filename)
                    shutil.copy2(src_pic_path, dst_pic_path)
                    pic_basenames.append((os.path.splitext(new_filename)[0], dst_pic_path))

            if os.path.exists(src_seg_split):
                print(f"[{split}] 正在讀取標籤、進行歸一化(0/1)並寫入...")
                for root, dirs, files in os.walk(src_seg_split):
                    for file in files:
                        if not file.lower().endswith('.png'): continue
                        src_mask_path = os.path.join(root, file)
                        rel_parts = os.path.relpath(src_mask_path, src_seg_split).split(os.sep)
                        new_filename = f"{split}_" + "_".join(rel_parts)
                        dst_mask_path = os.path.join(dst_mask_dir, new_filename)

                        # 使用 imdecode 解決 Windows 中文路徑報錯問題
                        img = cv2.imdecode(np.fromfile(src_mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = np.clip(img, 0, 1)
                            # 使用 imencode 解決 Windows 中文路徑報錯問題
                            cv2.imencode('.png', img)[1].tofile(dst_mask_path)
                        else:
                            print(f"讀取遮罩失敗: {src_mask_path}")

        print("-> 開始檢查缺失的遮罩並自動補件 (純黑)")
        generated_count = 0
        for base_name, dst_pic_path in pic_basenames:
            expected_mask_path = os.path.join(dst_mask_dir, f"{base_name}.png")
            if not os.path.exists(expected_mask_path):
                try:
                    with Image.open(dst_pic_path) as img:
                        w, h = img.size
                    mask = Image.new('L', (w, h), 0)
                    mask.save(expected_mask_path)
                    generated_count += 1
                except Exception as e:
                    print(f"無法讀取圖片 {dst_pic_path} 以生成 mask: {e}")

        # 切割資料集 (生成 txt)
        temp_seg = os.listdir(dst_mask_dir)
        total_seg = [seg for seg in temp_seg if seg.endswith(".png")]
        num = len(total_seg)
        list_index = list(range(num))
        tv = int(num * trainval_percent)
        tr = int(tv * train_percent)
        trainval = random.sample(list_index, tv)
        train = random.sample(trainval, tr)

        with open(os.path.join(saveBasePath, 'trainval.txt'), 'w') as ftrainval, \
             open(os.path.join(saveBasePath, 'test.txt'), 'w') as ftest, \
             open(os.path.join(saveBasePath, 'train.txt'), 'w') as ftrain, \
             open(os.path.join(saveBasePath, 'val.txt'), 'w') as fval:
            
            for i in list_index:
                name = total_seg[i][:-4] + '\n'
                if i in trainval:
                    ftrainval.write(name)
                    if i in train: ftrain.write(name)
                    else: fval.write(name)
                else:
                    ftest.write(name)

        # 像素檢查 (列印至終端機)
        print("\n-> 正在檢查資料集像素格式...")
        classes_nums = np.zeros([256], np.int64)
        for name in total_seg:
            png_file_name = os.path.join(dst_mask_dir, name)
            png = np.array(Image.open(png_file_name), np.uint8)
            classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

        print("\n[ 像素檢查結果 ]")
        for i in range(256):
            if classes_nums[i] > 0:
                print(f"像素值 {i}: 數量 {classes_nums[i]}")
                
        if classes_nums[255] > 0:
            print("⚠️ 警告：標籤中仍包含值為 255 的像素，請確認二值化。")

        return {
            "generated": generated_count,
            "total": num,
            "train": len(train),
            "val": tv - len(train),
            "test": num - tv
        }

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetMergerVOCApp(root)
    root.mainloop()