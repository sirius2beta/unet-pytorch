import os
import sys
import zipfile
import shutil
import json
import random
import cv2
import numpy as np
from PIL import Image

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea, QCheckBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox, QGroupBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

CONFIG_FILE = "config.json"

# --- 攔截 print() 輸出的 Stream 類別 ---
class EmittingStream(QObject):
    textWritten = pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))

    def flush(self):
        pass


class DatasetMergerVOCApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CVAT 合併與 VOC 資料集生成工具 (資料庫對接版) - 分離匯出")
        self.resize(750, 780)
        
        self.dataset_vars = {} 
        
        self._apply_dark_theme()
        self._build_ui()
        self._load_config()

    def _apply_dark_theme(self):
        # 使用 QSS (Qt Style Sheets) 設定深色主題，高度還原您的配色設計
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #e8e8e8;
                font-family: "Segoe UI", "Microsoft JhengHei";
                font-size: 10pt;
            }
            QGroupBox {
                border: 1px solid #555555;
                margin-top: 1ex;
                padding-top: 10px;
                font-weight: bold;
                color: #a9b7c6;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
            }
            QLineEdit {
                background-color: #3c3f41;
                border: 1px solid #555555;
                padding: 4px;
                color: #e8e8e8;
            }
            QLineEdit:read-only {
                background-color: #323232;
                color: #aaaaaa;
            }
            QPushButton {
                background-color: #3c3f41;
                border: 1px solid #555555;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #4b6eaf;
            }
            QPushButton#exportBtn {
                background-color: #388E3C;
                color: white;
                font-weight: bold;
                font-size: 11pt;
                padding: 8px;
            }
            QPushButton#exportBtn:hover {
                background-color: #4CAF50;
            }
            QPushButton#exportBtn:disabled {
                background-color: #555555;
                color: #999999;
            }
            QProgressBar {
                border: 1px solid #555555;
                background-color: #3c3f41;
                text-align: center;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #00FF7F;
            }
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                font-family: "Consolas", monospace;
                border: 1px solid #555555;
            }
            QScrollArea {
                border: none;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
        """)

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- 頂部：路徑選擇區塊 ---
        top_frame = QWidget()
        top_layout = QVBoxLayout(top_frame)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # 專案目錄
        proj_layout = QHBoxLayout()
        proj_label = QLabel("unet-pytorch資料夾:")
        proj_label.setFixedWidth(200)
        self.project_dir_input = QLineEdit()
        self.project_dir_input.setReadOnly(True)
        proj_btn = QPushButton("瀏覽...")
        proj_btn.clicked.connect(self.browse_project)
        proj_layout.addWidget(proj_label)
        proj_layout.addWidget(self.project_dir_input)
        proj_layout.addWidget(proj_btn)

        # Database 目錄
        db_layout = QHBoxLayout()
        db_label = QLabel("Database 目錄 (來源資料庫):")
        db_label.setFixedWidth(200)
        self.database_dir_input = QLineEdit()
        self.database_dir_input.setReadOnly(True)
        db_btn = QPushButton("瀏覽...")
        db_btn.clicked.connect(self.browse_database)
        db_layout.addWidget(db_label)
        db_layout.addWidget(self.database_dir_input)
        db_layout.addWidget(db_btn)

        top_layout.addLayout(proj_layout)
        top_layout.addLayout(db_layout)
        main_layout.addWidget(top_frame)

        # --- 中間：資料集列表 ---
        middle_group = QGroupBox(" 資料集列表 (已自動進入 annotation 抓取) ")
        middle_layout = QVBoxLayout(middle_group)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.grid_layout = QGridLayout(self.scroll_widget)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.scroll_area.setWidget(self.scroll_widget)
        middle_layout.addWidget(self.scroll_area)
        main_layout.addWidget(middle_group, stretch=2)

        # --- 按鈕與統計資訊 ---
        btn_frame = QWidget()
        btn_layout = QVBoxLayout(btn_frame)
        btn_layout.setContentsMargins(0, 0, 0, 0)
        
        self.total_info_label = QLabel("📊 目前選取包含: 0 張 Train 照片, 0 張 Test 照片")
        self.total_info_label.setStyleSheet("color: #4CAF50; font-weight: bold; font-size: 11pt;")
        self.total_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        btn_layout.addWidget(self.total_info_label)
        
        action_layout = QHBoxLayout()
        self.train_btn = QPushButton("🚀 匯出 Train 資料集")
        self.train_btn.setObjectName("exportBtn")
        self.train_btn.clicked.connect(lambda: self.process_pipeline('train'))
        
        self.test_btn = QPushButton("🚀 匯出 Test 資料集")
        self.test_btn.setObjectName("exportBtn")
        self.test_btn.clicked.connect(lambda: self.process_pipeline('test'))
        
        action_layout.addWidget(self.train_btn)
        action_layout.addWidget(self.test_btn)
        btn_layout.addLayout(action_layout)
        main_layout.addWidget(btn_frame)

        # --- 底部：進度條區塊 ---
        progress_frame = QWidget()
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_text_label = QLabel("等待執行...")
        self.progress_text_label.setStyleSheet("color: #00FFFF; font-weight: bold;")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        
        progress_layout.addWidget(self.progress_text_label)
        progress_layout.addWidget(self.progress_bar)
        main_layout.addWidget(progress_frame)

        # --- 底部：終端機輸出日誌框 ---
        log_group = QGroupBox(" 執行進度日誌 ")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        main_layout.addWidget(log_group, stretch=1)

        # 將標準輸出導向自訂的 Stream
        self.stream = EmittingStream()
        self.stream.textWritten.connect(self.normal_output_written)
        sys.stdout = self.stream
        sys.stderr = self.stream

    def normal_output_written(self, text):
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_text.setTextCursor(cursor)
        self.log_text.ensureCursorVisible()
        # 強制刷新 UI
        QApplication.processEvents()

    def update_progress(self, percent, text=None):
        self.progress_bar.setValue(int(percent))
        if text:
            self.progress_text_label.setText(text)
        # 讓介面即時響應，類似 tkinter 的 self.root.update()
        QApplication.processEvents()

    def _load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    proj_dir = config.get("project_dir", "")
                    db_dir = config.get("database_dir", "")
                    
                    if proj_dir and os.path.isdir(proj_dir):
                        self.project_dir_input.setText(proj_dir)
                    
                    if db_dir and os.path.isdir(db_dir):
                        self.database_dir_input.setText(db_dir)
                        anno_dir = os.path.join(db_dir, "annotation")
                        if os.path.isdir(anno_dir):
                            self.scan_datasets(anno_dir)
            except Exception:
                pass

    def _save_config(self):
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "project_dir": self.project_dir_input.text(),
                    "database_dir": self.database_dir_input.text()
                }, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"儲存設定檔失敗: {e}")

    def browse_project(self):
        folder_selected = QFileDialog.getExistingDirectory(self, "請選擇專案目錄 (用來存放 VOCdevkit)")
        if folder_selected:
            self.project_dir_input.setText(folder_selected)
            self._save_config()

    def browse_database(self):
        folder_selected = QFileDialog.getExistingDirectory(self, "請選擇 Database 資料庫目錄")
        if folder_selected:
            self.database_dir_input.setText(folder_selected)
            self._save_config()
            
            anno_path = os.path.join(folder_selected, "annotation")
            if os.path.isdir(anno_path):
                self.scan_datasets(anno_path)
            else:
                self._clear_dataset_list()
                QMessageBox.warning(self, "找不到資料夾", f"找不到 'annotation' 資料夾！\n\n路徑: {anno_path}")

    def _clear_dataset_list(self):
        # 清除 QGridLayout 中所有的 widget
        while self.grid_layout.count():
            child = self.grid_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.dataset_vars.clear()
        self.update_totals()

    def scan_datasets(self, anno_path):
        self._clear_dataset_list()
        if not os.path.isdir(anno_path): return

        # 建立表頭
        headers = [
            ("資料集名稱", 0, Qt.AlignmentFlag.AlignLeft),
            ("Train", 1, Qt.AlignmentFlag.AlignCenter),
            ("張數", 2, Qt.AlignmentFlag.AlignLeft),
            ("Test", 3, Qt.AlignmentFlag.AlignCenter),
            ("張數", 4, Qt.AlignmentFlag.AlignLeft)
        ]
        
        for text, col, align in headers:
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight: bold;" if "張數" not in text else "color: #a9b7c6;")
            self.grid_layout.addWidget(lbl, 0, col, alignment=align)

        # 橫線 (Separator)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #555555;")
        self.grid_layout.addWidget(line, 1, 0, 1, 5)

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

                    chk_train = QCheckBox()
                    chk_test = QCheckBox()
                    
                    chk_train.stateChanged.connect(self.update_totals)
                    chk_test.stateChanged.connect(self.update_totals)
                    
                    self.dataset_vars[item] = {
                        'train': chk_train, 
                        'test': chk_test,
                        'train_count': train_cnt,
                        'test_count': test_cnt
                    }

                    # 加入畫面
                    self.grid_layout.addWidget(QLabel(f"📁 {item}"), row, 0, alignment=Qt.AlignmentFlag.AlignLeft)
                    self.grid_layout.addWidget(chk_train, row, 1, alignment=Qt.AlignmentFlag.AlignCenter)
                    
                    lbl_tr_cnt = QLabel(f"{train_cnt} 張")
                    lbl_tr_cnt.setStyleSheet("color: #a9b7c6;")
                    self.grid_layout.addWidget(lbl_tr_cnt, row, 2, alignment=Qt.AlignmentFlag.AlignLeft)
                    
                    self.grid_layout.addWidget(chk_test, row, 3, alignment=Qt.AlignmentFlag.AlignCenter)
                    
                    lbl_ts_cnt = QLabel(f"{test_cnt} 張")
                    lbl_ts_cnt.setStyleSheet("color: #a9b7c6;")
                    self.grid_layout.addWidget(lbl_ts_cnt, row, 4, alignment=Qt.AlignmentFlag.AlignLeft)
                    
                    row += 1

    def update_totals(self):
        total_train = 0
        total_test = 0
        for ds_name, info in self.dataset_vars.items():
            if info['train'].isChecked(): total_train += info['train_count']
            if info['test'].isChecked(): total_test += info['test_count']
        
        self.total_info_label.setText(f"📊 目前選取包含: {total_train} 張 Train 照片, {total_test} 張 Test 照片")

    def process_pipeline(self, target_split):
        project_path = self.project_dir_input.text()
        db_path = self.database_dir_input.text()

        if not project_path:
            QMessageBox.warning(self, "警告", "請先選擇專案目錄！")
            return
        if not db_path:
            QMessageBox.warning(self, "警告", "請先選擇 Database 目錄！")
            return
            
        anno_dir = os.path.join(db_path, "annotation")
        if not os.path.isdir(anno_dir):
            QMessageBox.critical(self, "錯誤", f"在 Database 下找不到 annotation 資料夾！\n路徑: {anno_dir}")
            return

        has_selection = any(v[target_split].isChecked() for v in self.dataset_vars.values())
        if not has_selection:
            QMessageBox.warning(self, "警告", f"您尚未勾選任何 {target_split.capitalize()} 資料集！")
            return

        export_dir = os.path.join(anno_dir, 'export')
        voc_dir = os.path.join(project_path, 'VOCdevkit')

        reply = QMessageBox.question(self, "確認", 
                                     f"準備開始匯出並覆寫 【{target_split.upper()}】 資料。\n\n另一半的資料將會被保留，是否繼續？",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply != QMessageBox.StandardButton.Yes:
            return

        # 清空日誌與鎖定按鈕
        self.log_text.clear()
        self.train_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
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
            QMessageBox.information(self, "完成", success_msg)

        except Exception as e:
            print(f"錯誤: {str(e)}")
            self.update_progress(0, "❌ 發生錯誤")
            QMessageBox.critical(self, "錯誤", f"執行過程中發生錯誤:\n{str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            self.train_btn.setEnabled(True)
            self.test_btn.setEnabled(True)

    def _extract_and_merge(self, base_dir, export_dir, target_split):
        for folder in ['JPEGImages', 'SegmentationClass']:
            target_path = os.path.join(export_dir, folder, target_split)
            if os.path.exists(target_path):
                shutil.rmtree(target_path)

        for ds_name, vars_dict in self.dataset_vars.items():
            if not vars_dict[target_split].isChecked():
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
    app = QApplication(sys.argv)
    window = DatasetMergerVOCApp()
    window.show()
    sys.exit(app.exec())