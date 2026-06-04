import os
import sys
import zipfile
import shutil
import json
import random
import cv2
import numpy as np
from PIL import Image
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

CONFIG_FILE = "config.json"

class PrintLogger:
    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root

    def write(self, text):
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END) 
        self.text_widget.configure(state="disabled")
        self.root.update_idletasks() 

    def flush(self):
        pass

class DatasetMergerVOCApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CVAT 合併與 VOC 資料集生成工具 (資料庫對接版) - 分離匯出")
        self.root.geometry("750x850") 
        
        self._apply_dark_theme()
        
        self.project_dir = tk.StringVar()
        self.database_dir = tk.StringVar() 
        self.dataset_vars = {} 
        self.total_info_var = tk.StringVar(value="📊 目前選取包含: 0 張 Train 照片, 0 張 Test 照片")

        self._build_ui()
        self._load_config()

    def _apply_dark_theme(self):
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
        style.map("Export.TButton", background=[("active", export_hover), ("disabled", "#555555")])
        style.configure("Vertical.TScrollbar", background=panel_bg, troughcolor=bg_color, bordercolor=bg_color, arrowcolor=fg_color)
        style.map("Vertical.TScrollbar", background=[("active", accent_color)])

        neon_green = "#00FF7F" 
        style.configure("Striking.Horizontal.TProgressbar",
                        troughcolor=panel_bg,
                        background=neon_green,
                        bordercolor=bg_color,
                        lightcolor=neon_green,
                        darkcolor=neon_green)
                        
    def _build_ui(self):
        # --- 頂部：路徑選擇區塊 ---
        top_frame = ttk.Frame(self.root, padding=15)
        top_frame.pack(fill=tk.X)

        proj_frame = ttk.Frame(top_frame)
        proj_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(proj_frame, text="專案目錄 (輸出 VOCdevkit 處):", width=26).pack(side=tk.LEFT)
        ttk.Entry(proj_frame, textvariable=self.project_dir, width=44, state='readonly').pack(side=tk.LEFT, padx=5, ipady=3)
        ttk.Button(proj_frame, text="選擇專案...", command=self.browse_project).pack(side=tk.LEFT, padx=5)

        db_frame = ttk.Frame(top_frame)
        db_frame.pack(fill=tk.X)
        ttk.Label(db_frame, text="Database 目錄 (來源資料庫):", width=26).pack(side=tk.LEFT)
        ttk.Entry(db_frame, textvariable=self.database_dir, width=44, state='readonly').pack(side=tk.LEFT, padx=5, ipady=3)
        ttk.Button(db_frame, text="選擇 Database...", command=self.browse_database).pack(side=tk.LEFT, padx=5)

        # --- 中間：資料集列表 ---
        middle_frame = ttk.LabelFrame(self.root, text=" 資料集列表 (已自動進入 annotation 抓取) ", padding=10)
        middle_frame.pack(fill=tk.BOTH, expand=False, padx=15, pady=5)

        self.canvas = tk.Canvas(middle_frame, bg="#2b2b2b", highlightthickness=0, height=180) 
        scrollbar = ttk.Scrollbar(middle_frame, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- 按鈕與統計資訊 ---
        btn_frame = ttk.Frame(self.root, padding=10)
        btn_frame.pack(fill=tk.X)
        
        ttk.Label(btn_frame, textvariable=self.total_info_var, font=("Segoe UI", 11, "bold"), foreground="#4CAF50").pack(pady=(0, 5))
        
        # 🌟 分離為兩個按鈕
        action_btn_frame = ttk.Frame(btn_frame)
        action_btn_frame.pack(fill=tk.X, pady=5)
        
        self.train_btn = ttk.Button(action_btn_frame, text="🚀 匯出 Train 資料集", style="Export.TButton", command=lambda: self.process_pipeline('train'))
        self.train_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5), ipady=5)
        
        self.test_btn = ttk.Button(action_btn_frame, text="🚀 匯出 Test 資料集", style="Export.TButton", command=lambda: self.process_pipeline('test'))
        self.test_btn.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0), ipady=5)

        # --- 底部：進度條區塊 ---
        progress_frame = ttk.Frame(self.root, padding=(15, 0, 15, 15))
        progress_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.progress_text_var = tk.StringVar(value="等待執行...")
        ttk.Label(progress_frame, textvariable=self.progress_text_var, font=("Segoe UI", 10, "bold"), foreground="#00FFFF").pack(side=tk.TOP, anchor="w", pady=(0, 4))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, style="Striking.Horizontal.TProgressbar")
        self.progress_bar.pack(side=tk.TOP, fill=tk.X)

        # --- 底部：終端機輸出日誌框 ---
        log_frame = ttk.LabelFrame(self.root, text=" 執行進度日誌 ", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=(0, 10))

        self.log_text = tk.Text(log_frame, bg="#1e1e1e", fg="#cccccc", font=("Consolas", 10), state="disabled", wrap="word")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        sys.stdout = PrintLogger(self.log_text, self.root)
        sys.stderr = PrintLogger(self.log_text, self.root)
    
    def update_progress(self, percent, text=None):
        self.progress_var.set(percent)
        if text:
            self.progress_text_var.set(text)
        self.root.update() 

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    proj_dir = config.get("project_dir", "")
                    db_dir = config.get("database_dir", "")
                    
                    if proj_dir and os.path.isdir(proj_dir):
                        self.project_dir.set(proj_dir)
                    
                    if db_dir and os.path.isdir(db_dir):
                        self.database_dir.set(db_dir)
                        anno_dir = os.path.join(db_dir, "annotation")
                        if os.path.isdir(anno_dir):
                            self.scan_datasets(anno_dir)
            except Exception:
                pass

    def _save_config(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "project_dir": self.project_dir.get(),
                    "database_dir": self.database_dir.get()
                }, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"儲存設定檔失敗: {e}")

    def browse_project(self):
        folder_selected = filedialog.askdirectory(title="請選擇專案目錄 (用來存放 VOCdevkit)")
        if folder_selected:
            self.project_dir.set(folder_selected)
            self._save_config()

    def browse_database(self):
        folder_selected = filedialog.askdirectory(title="請選擇 Database 資料庫目錄")
        if folder_selected:
            self.database_dir.set(folder_selected)
            self._save_config()
            
            anno_path = os.path.join(folder_selected, "annotation")
            if os.path.isdir(anno_path):
                self.scan_datasets(anno_path)
            else:
                self._clear_dataset_list()
                messagebox.showwarning("找不到資料夾", f"找不到 'annotation' 資料夾！\n\n路徑: {anno_path}")

    def _clear_dataset_list(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.dataset_vars.clear()
        self.update_totals()

    def scan_datasets(self, anno_path):
        self._clear_dataset_list()
        if not os.path.isdir(anno_path): return

        ttk.Label(self.scrollable_frame, text="資料集名稱", width=32, anchor="w", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 5))
        ttk.Label(self.scrollable_frame, text="Train", width=8, anchor="center", font=("Segoe UI", 10, "bold")).grid(row=0, column=1, pady=(0, 5))
        ttk.Label(self.scrollable_frame, text="張數", width=10, anchor="w", foreground="#a9b7c6", font=("Segoe UI", 9)).grid(row=0, column=2, pady=(0, 5))
        ttk.Label(self.scrollable_frame, text="Test", width=8, anchor="center", font=("Segoe UI", 10, "bold")).grid(row=0, column=3, pady=(0, 5))
        ttk.Label(self.scrollable_frame, text="張數", width=10, anchor="w", foreground="#a9b7c6", font=("Segoe UI", 9)).grid(row=0, column=4, pady=(0, 5))
        ttk.Separator(self.scrollable_frame, orient="horizontal").grid(row=1, column=0, columnspan=5, sticky="ew", pady=(0, 5))

        row = 2
        for item in sorted(os.listdir(anno_path)):
            item_path = os.path.join(anno_path, item)
            if os.path.isdir(item_path) and item.lower() != 'export':
                zip_path = os.path.join(item_path, 'cvat.zip')
                if os.path.exists(zip_path):
                    train_cnt = 0
                    test_cnt = 0
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zf:
                            for file_info in zf.infolist():
                                if file_info.is_dir(): continue
                                parts = file_info.filename.split('/')
                                if len(parts) >= 3 and parts[0] == 'JPEGImages':
                                    split = parts[1]
                                    ext = parts[-1].lower()
                                    if ext.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                        if split == 'train': train_cnt += 1
                                        elif split == 'test': test_cnt += 1
                    except Exception as e:
                        print(f"無法讀取 {zip_path} 的圖片數量: {e}")

                    var_train = tk.BooleanVar(value=False)
                    var_test = tk.BooleanVar(value=False)
                    
                    self.dataset_vars[item] = {
                        'train': var_train, 
                        'test': var_test,
                        'train_count': train_cnt,
                        'test_count': test_cnt
                    }

                    ttk.Label(self.scrollable_frame, text=f"📁 {item}", width=32, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
                    ttk.Checkbutton(self.scrollable_frame, variable=var_train, command=self.update_totals).grid(row=row, column=1)
                    ttk.Label(self.scrollable_frame, text=f"{train_cnt} 張", width=10, anchor="w", foreground="#a9b7c6").grid(row=row, column=2)
                    ttk.Checkbutton(self.scrollable_frame, variable=var_test, command=self.update_totals).grid(row=row, column=3)
                    ttk.Label(self.scrollable_frame, text=f"{test_cnt} 張", width=10, anchor="w", foreground="#a9b7c6").grid(row=row, column=4)
                    
                    row += 1

    def update_totals(self):
        total_train = 0
        total_test = 0
        for ds_name, info in self.dataset_vars.items():
            if info['train'].get(): total_train += info['train_count']
            if info['test'].get(): total_test += info['test_count']
        
        self.total_info_var.set(f"📊 目前選取包含: {total_train} 張 Train 照片, {total_test} 張 Test 照片")

    def process_pipeline(self, target_split):
        project_path = self.project_dir.get()
        db_path = self.database_dir.get()

        if not project_path:
            messagebox.showwarning("警告", "請先選擇專案目錄！")
            return
        if not db_path:
            messagebox.showwarning("警告", "請先選擇 Database 目錄！")
            return
            
        anno_dir = os.path.join(db_path, "annotation")
        if not os.path.isdir(anno_dir):
            messagebox.showerror("錯誤", f"在 Database 下找不到 annotation 資料夾！\n路徑: {anno_dir}")
            return

        has_selection = any(v[target_split].get() for v in self.dataset_vars.values())
        if not has_selection:
            messagebox.showwarning("警告", f"您尚未勾選任何 {target_split.capitalize()} 資料集！")
            return

        export_dir = os.path.join(anno_dir, 'export')
        voc_dir = os.path.join(project_path, 'VOCdevkit')

        if not messagebox.askyesno("確認", f"準備開始匯出並覆寫 【{target_split.upper()}】 資料。\n\n另一半的資料將會被保留，是否繼續？"):
            return

        # 清空日誌與鎖定按鈕
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        self.train_btn.configure(state='disabled')
        self.test_btn.configure(state='disabled')
        self.update_progress(0, f"初始化 {target_split.upper()} 匯出中...")

        try:
            print("\n" + "="*50)
            print(f"階段一：開始解壓並合併 CVAT [{target_split.upper()}] 資料至 export")
            print("="*50)
            self._extract_and_merge(anno_dir, export_dir, target_split)

            print("\n" + "="*50)
            print(f"階段二：將 export [{target_split.upper()}] 轉換並更新至 VOCdevkit")
            print("="*50)
            stats = self._build_voc_structure(export_dir, voc_dir, target_split)
            
            success_msg = (
                f"✅ {target_split.upper()} 流程執行完畢！\n\n"
                f"📊 [ VOCdevkit 當前總計 ]\n"
                f"- 自動補齊黑圖: {stats['generated']} 張 (本次)\n"
                f"- VOC 資料庫總樣本數量: {stats['total']}\n"
                f"  └ Train: {stats['train']} 張 (由 Train 資料隨機 80%)\n"
                f"  └ Val:   {stats['val']} 張 (由 Train 資料隨機 20%)\n"
                f"  └ Test:  {stats['test']} 張 (保留 Test 資料)\n\n"
                f"📁 目錄:\n{voc_dir}"
            )
            print(f"\n✅ {target_split.upper()} 處理成功完成！")
            self.update_progress(100, "✅ 執行完畢！")
            messagebox.showinfo("完成", success_msg)

        except Exception as e:
            print(f"錯誤: {str(e)}")
            self.update_progress(0, "❌ 發生錯誤")
            messagebox.showerror("錯誤", f"執行過程中發生錯誤:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.train_btn.configure(state='normal')
            self.test_btn.configure(state='normal')

    def _extract_and_merge(self, base_dir, export_dir, target_split):
        # 先清除 export 裡面舊的目標 split 資料，避免幽靈檔案
        for folder in ['JPEGImages', 'SegmentationClass']:
            target_path = os.path.join(export_dir, folder, target_split)
            if os.path.exists(target_path):
                shutil.rmtree(target_path)

        for ds_name, vars_dict in self.dataset_vars.items():
            if not vars_dict[target_split].get():
                continue

            zip_path = os.path.join(base_dir, ds_name, 'cvat.zip')
            print(f"-> 正在準備解壓: {ds_name} (包含: {target_split})")
            
            with zipfile.ZipFile(zip_path, 'r') as zf:
                info_list = zf.infolist()
                total_files = len(info_list)
                
                for i, file_info in enumerate(info_list):
                    if i % 10 == 0 or i == total_files - 1:
                        self.update_progress((i+1)/total_files * 100, f"正在解壓縮 {ds_name} ({i+1}/{total_files})...")

                    if file_info.is_dir(): continue
                    parts = file_info.filename.split('/')
                    if len(parts) >= 3:
                        root_folder, split = parts[0], parts[1]
                        filename = parts[-1]

                        if root_folder in ['JPEGImages', 'SegmentationClass'] and split == target_split:
                            subfolders = parts[2:-1]
                            target_dir = os.path.join(export_dir, root_folder, split, *subfolders)
                            os.makedirs(target_dir, exist_ok=True)

                            prefix = "_".join(subfolders) if subfolders else ""
                            new_filename = f"{ds_name}_{prefix}_{filename}" if prefix else f"{ds_name}_{filename}"
                            target_path = os.path.join(target_dir, new_filename)

                            with zf.open(file_info) as source, open(target_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
        print(f"-> {export_dir} ({target_split}) 解壓合併完成")

    def _build_voc_structure(self, folder_a, voc_path, target_split):
        train_percent = 0.8
        random.seed(0)

        dst_img_dir = os.path.join(voc_path, 'VOC2007', 'JPEGImages')
        dst_mask_dir = os.path.join(voc_path, 'VOC2007', 'SegmentationClass')
        saveBasePath = os.path.join(voc_path, 'VOC2007', 'ImageSets', 'Segmentation')
        
        for path in [dst_img_dir, dst_mask_dir, saveBasePath]:
            os.makedirs(path, exist_ok=True)

        # 清除 VOCdevkit 中屬於此次 target_split 的舊檔案
        print(f"-> 正在清理 VOCdevkit 舊有的 {target_split} 檔案...")
        for target_dir in [dst_img_dir, dst_mask_dir]:
            if os.path.exists(target_dir):
                for f in os.listdir(target_dir):
                    if f.startswith(f"{target_split}_"):
                        os.remove(os.path.join(target_dir, f))

        src_jpeg_split = os.path.join(folder_a, 'JPEGImages', target_split)
        src_seg_split = os.path.join(folder_a, 'SegmentationClass', target_split)
        
        pic_basenames = []
        generated_count = 0

        if os.path.exists(src_jpeg_split):
            img_tasks = []
            for root, dirs, files in os.walk(src_jpeg_split):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_tasks.append((root, file))
            
            total_imgs = len(img_tasks)
            print(f"\n[{target_split}] 正在處理並轉換圖片為 .jpg (共 {total_imgs} 張)...")
            
            for i, (root_dir, file) in enumerate(img_tasks):
                if i % 5 == 0 or i == total_imgs - 1:
                    self.update_progress((i+1)/total_imgs * 100, f"正在轉換圖片格式 ({i+1}/{total_imgs})...")

                src_pic_path = os.path.join(root_dir, file)
                rel_parts = os.path.relpath(src_pic_path, src_jpeg_split).split(os.sep)
                
                raw_filename = f"{target_split}_" + "_".join(rel_parts)
                base_name, _ = os.path.splitext(raw_filename)
                new_filename = base_name + ".jpg"
                dst_pic_path = os.path.join(dst_img_dir, new_filename)
                
                try:
                    with Image.open(src_pic_path) as img:
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img.save(dst_pic_path, 'JPEG', quality=100)
                    pic_basenames.append((base_name, dst_pic_path))
                except Exception as e:
                    print(f"⚠️ 無法讀取或轉換圖片 {src_pic_path}: {e}")
            print(f"[{target_split}] 圖片處理完畢。")

            # 2. 處理標籤
            if os.path.exists(src_seg_split):
                mask_tasks = []
                for root, dirs, files in os.walk(src_seg_split):
                    for file in files:
                        if file.lower().endswith('.png'):
                            mask_tasks.append((root, file))
                            
                total_masks = len(mask_tasks)
                print(f"[{target_split}] 正在處理並歸一化標籤 (共 {total_masks} 張)...")
                
                for i, (root_dir, file) in enumerate(mask_tasks):
                    if i % 10 == 0 or i == total_masks - 1:
                        self.update_progress((i+1)/total_masks * 100, f"處理遮罩標籤 ({i+1}/{total_masks})...")

                    src_mask_path = os.path.join(root_dir, file)
                    rel_parts = os.path.relpath(src_mask_path, src_seg_split).split(os.sep)
                    
                    raw_maskname = f"{target_split}_" + "_".join(rel_parts)
                    base_maskname, _ = os.path.splitext(raw_maskname)
                    new_maskname = base_maskname + ".png"
                    dst_mask_path = os.path.join(dst_mask_dir, new_maskname)

                    img = cv2.imdecode(np.fromfile(src_mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = np.clip(img, 0, 1)
                        cv2.imencode('.png', img)[1].tofile(dst_mask_path)
                    else:
                        print(f"讀取遮罩失敗: {src_mask_path}")
                print(f"[{target_split}] 標籤處理完畢。")

            # 3. 檢查並補齊黑圖
            print(f"\n-> 開始檢查缺失的遮罩並自動補件 (純黑)...")
            total_check = len(pic_basenames)
            
            for i, (base_name, dst_pic_path) in enumerate(pic_basenames):
                if i % 10 == 0 or i == total_check - 1:
                    self.update_progress((i+1)/total_check * 100, f"檢查缺失標籤 ({i+1}/{total_check})...")

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
            
            if generated_count > 0:
                print(f"自動補齊了 {generated_count} 張黑圖。")

        # 4. 生成 txt (動態掃描 VOCdevkit 裡的所有圖片)
        self.update_progress(95, "正在掃描總目錄並重建 txt 清單檔案...")
        print("\n-> 正在掃描 VOCdevkit 更新總索引...")
        
        test_list = []
        trainval_list = []
        
        all_voc_images = [f for f in os.listdir(dst_img_dir) if f.endswith('.jpg')]
        for img_file in all_voc_images:
            base_name = os.path.splitext(img_file)[0]
            if base_name.startswith("test_"):
                test_list.append(base_name)
            elif base_name.startswith("train_"):
                trainval_list.append(base_name)

        tr_count = int(len(trainval_list) * train_percent)
        train_list = random.sample(trainval_list, tr_count)
        val_list = [x for x in trainval_list if x not in train_list]

        def write_txt(filename, data_list):
            with open(os.path.join(saveBasePath, filename), 'w') as f:
                if data_list:
                    f.write('\n'.join(data_list) + '\n')

        write_txt('trainval.txt', trainval_list)
        write_txt('test.txt', test_list)
        write_txt('train.txt', train_list)
        write_txt('val.txt', val_list)


        return {
            "generated": generated_count,
            "total": len(all_voc_images),
            "train": len(train_list),
            "val": len(val_list),
            "test": len(test_list)
        }

if __name__ == "__main__":
    root = tk.Tk()
    app = DatasetMergerVOCApp(root)
    root.mainloop()