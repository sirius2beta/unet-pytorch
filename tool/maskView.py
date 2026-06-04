import os
import shutil
import sys
from PIL import Image, ImageQt
from PyQt6.QtCore import Qt, QRectF, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QColor, QBrush, QWheelEvent, QImage
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QLabel, QScrollArea, QGridLayout,
    QCheckBox, QFrame, QMessageBox, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem
)

# ---------------------------------------------------------
# 1. ⚡️ 背景載入縮圖的專屬執行緒 (Worker)
# ---------------------------------------------------------
class ThumbnailWorker(QThread):
    thumbnail_ready = pyqtSignal(int, str, QImage)
    
    def __init__(self, img_dir, mask_dir, image_list):
        super().__init__()
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.image_list = image_list
        self.is_running = True

    def run(self):
        target_width = 125

        for idx, img_rel_path in enumerate(self.image_list):
            if not self.is_running:
                break
            
            img_path = os.path.join(self.img_dir, img_rel_path)
            if not os.path.exists(img_path):
                continue

            try:
                img = Image.open(img_path).convert("RGBA")
                ratio = target_width / img.width
                new_h = int(img.height * ratio)
                
                if new_h <= 0:
                    continue
                    
                img = img.resize((target_width, new_h), Image.Resampling.BILINEAR)

                if os.path.exists(self.mask_dir):
                    base_name_no_ext = os.path.splitext(img_rel_path)[0]
                    mask_path = os.path.join(self.mask_dir, base_name_no_ext + ".png")
                    
                    if os.path.exists(mask_path):
                        mask_img = Image.open(mask_path).convert("L")
                        mask_img = mask_img.resize((target_width, new_h), Image.Resampling.NEAREST)
                        binary_mask = mask_img.point(lambda p: 255 if p > 0 else 0)
                        
                        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                        red_layer = Image.new("RGBA", img.size, (255, 0, 0, 100))
                        overlay.paste(red_layer, (0, 0), mask=binary_mask)
                        img = Image.alpha_composite(img, overlay)

                qimg = QImage(ImageQt.ImageQt(img)).copy()
                self.thumbnail_ready.emit(idx, img_rel_path, qimg)
                
            except Exception as e:
                print(f"背景處理縮圖失敗 {img_rel_path}: {e}")
                continue


# ---------------------------------------------------------
# 自訂的圖片檢視器
# ---------------------------------------------------------
class PhotoViewer(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(self)
        self.photo = QGraphicsPixmapItem()
        self.scene.addItem(self.photo)
        self.setScene(self.scene)
        
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def set_image(self, pixmap):
        self.photo.setPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))
        self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: QWheelEvent):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.photo.pixmap().isNull():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# ---------------------------------------------------------
# 主程式
# ---------------------------------------------------------
class ImageApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("海草資料集篩選器 (實體按鍵與鍵盤快捷版)")
        self.setGeometry(100, 100, 1200, 800)

        self.base_dir = ""
        self.img_dir = ""
        self.mask_dir = ""
        self.image_extensions = ('.jpg', '.jpeg', '.JPG', '.JPEG')
        
        self.image_list = []
        self.selected_images = set()
        self.current_index = -1
        self.show_mask = True

        self.thumb_checkboxes = {}
        self.thumb_labels = {}
        self.cache_thumb_masked = {}
        
        self.worker_thread = None

        self.init_ui()
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def init_ui(self):
        main_layout = QVBoxLayout()

        # --- Top ---
        top_layout = QHBoxLayout()
        self.btn_select_dir = QPushButton("選擇 Segmentation 資料夾")
        self.btn_select_dir.clicked.connect(self.select_directory)
        self.lbl_dir_path = QLabel("尚未選擇資料夾")
        top_layout.addWidget(self.btn_select_dir)
        top_layout.addWidget(self.lbl_dir_path, 1)
        main_layout.addLayout(top_layout)

        # --- Middle ---
        middle_layout = QHBoxLayout()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedWidth(340)
        self.scroll_widget = QWidget()
        self.scroll_widget.setStyleSheet("background-color: #1E1E1E;") 
        self.grid_layout = QGridLayout(self.scroll_widget)
        self.grid_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        self.grid_layout.setSpacing(4)
        self.scroll_area.setWidget(self.scroll_widget)
        middle_layout.addWidget(self.scroll_area)

        right_layout = QVBoxLayout()
        display_layout = QHBoxLayout()
        
        self.btn_prev = QPushButton("◀\n前\n一\n張")
        self.btn_prev.setFixedWidth(40)
        self.btn_prev.setFixedHeight(200)
        self.btn_prev.clicked.connect(self.prev_image)
        display_layout.addWidget(self.btn_prev, 0, Qt.AlignmentFlag.AlignVCenter)
        
        self.main_viewer = PhotoViewer()
        display_layout.addWidget(self.main_viewer, 1)

        self.btn_next = QPushButton("▶\n下\n一\n張")
        self.btn_next.setFixedWidth(40)
        self.btn_next.setFixedHeight(200)
        self.btn_next.clicked.connect(self.next_image)
        display_layout.addWidget(self.btn_next, 0, Qt.AlignmentFlag.AlignVCenter)

        right_layout.addLayout(display_layout, 1)

        # ✅ 需求: 新增下方的操作按鈕列 (包含選取與遮罩切換)
        action_layout = QHBoxLayout()

        # 1. 實體選取按鈕
        self.btn_toggle_select = QPushButton("選取照片 [空白鍵]")
        self.btn_toggle_select.setCheckable(True) # 讓它擁有「被按下」的狀態
        self.btn_toggle_select.setFixedHeight(40)
        # 設定按鈕被選取時的亮藍色外觀
        self.btn_toggle_select.setStyleSheet("""
            QPushButton:checked {
                background-color: #0D6EFD;
                color: white;
                font-weight: bold;
                border: none;
                border-radius: 4px;
            }
        """)
        self.btn_toggle_select.clicked.connect(self.toggle_current_selection)
        action_layout.addWidget(self.btn_toggle_select)
        self.update_main_image_border(False)
        # 2. 遮罩開關按鈕
        self.btn_toggle_mask = QPushButton("關閉遮罩 [H]")
        self.btn_toggle_mask.setCheckable(True)
        self.btn_toggle_mask.setFixedHeight(40)
        self.btn_toggle_mask.clicked.connect(self.toggle_mask)
        action_layout.addWidget(self.btn_toggle_mask)

        right_layout.addLayout(action_layout)

        middle_layout.addLayout(right_layout, 1)
        main_layout.addLayout(middle_layout, 1)

        # --- Bottom ---
        bottom_layout = QHBoxLayout()
        self.lbl_count = QLabel("已選擇: 0 張照片")
        self.btn_export = QPushButton("Export 已選照片 (保留結構)")
        self.btn_export.clicked.connect(self.export_data)
        bottom_layout.addWidget(self.lbl_count)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.btn_export)
        main_layout.addLayout(bottom_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # ✅ 確保所有按鈕不會搶走鍵盤焦點，維持熱鍵順暢運作
        for widget in [self.btn_prev, self.btn_next, self.btn_toggle_mask, self.btn_toggle_select, self.btn_select_dir, self.btn_export, self.scroll_area, self.main_viewer]:
            widget.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    # ---------------------------------------------------------
    # 🌟 鍵盤快捷鍵攔截與事件處理
    # ---------------------------------------------------------
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Left:
            self.prev_image()
        elif event.key() == Qt.Key.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key.Key_Space:
            # ✅ 空白鍵直接觸發選取按鈕的點擊事件
            self.btn_toggle_select.click()
        elif event.key() == Qt.Key.Key_H:
            self.btn_toggle_mask.click()
        else:
            super().keyPressEvent(event)

    def toggle_current_selection(self):
        """觸發當前照片的勾選狀態改變"""
        if self.image_list and 0 <= self.current_index < len(self.image_list):
            img_rel_path = self.image_list[self.current_index]
            chk = self.thumb_checkboxes.get(img_rel_path)
            if chk:
                chk.toggle()

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "選取 Segmentation 資料夾")
        if not dir_path:
            return

        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.is_running = False
            self.worker_thread.quit()
            self.worker_thread.wait()

        self.base_dir = dir_path
        self.img_dir = os.path.join(dir_path, "JPEGImages")
        self.mask_dir = os.path.join(dir_path, "SegmentationClass")

        if not os.path.exists(self.img_dir):
            QMessageBox.critical(self, "錯誤", "找不到 JPEGImages 資料夾！")
            return

        self.lbl_dir_path.setText(dir_path)
        self.cache_thumb_masked.clear()

        self.image_list = []
        for root, dirs, files in os.walk(self.img_dir):
            for f in files:
                if f.endswith(self.image_extensions):
                    full_path = os.path.join(root, f)
                    rel_path = os.path.relpath(full_path, self.img_dir)
                    self.image_list.append(rel_path)
        
        self.image_list.sort()
        self.selected_images.clear()
        self.update_count_label()
        
        if self.image_list:
            self.current_index = 0
            self.build_empty_thumbnail_grid()
            self.update_main_image()
            
            self.worker_thread = ThumbnailWorker(self.img_dir, self.mask_dir, self.image_list)
            self.worker_thread.thumbnail_ready.connect(self.update_single_thumbnail)
            self.worker_thread.start()
        else:
            QMessageBox.information(self, "提示", "JPEGImages 資料夾內無支援的照片。")

    def build_empty_thumbnail_grid(self):
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        
        self.thumb_checkboxes.clear()
        self.thumb_labels.clear()

        checkbox_style = """
            QCheckBox {
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border: 2px solid #666666;
                border-radius: 4px;
                background-color: #2B2B2B;
            }
            QCheckBox::indicator:hover {
                border: 2px solid #AAAAAA;
            }
            QCheckBox::indicator:checked {
                background-color: #0D6EFD; 
                border: 2px solid #0D6EFD;
                image: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>');
            }
        """

        for idx, img_rel_path in enumerate(self.image_list):
            row = idx // 2
            col = idx % 2

            frame = QFrame()
            # ✅ 需求1: 強制設定每張縮圖卡片的固定寬度 (150px 剛好能完美塞入左側捲動區的兩排)
            frame.setFixedWidth(150)
            frame.setStyleSheet("""
                QFrame {
                    background-color: #2D2D2D;
                    border: 2px solid #000000;
                    margin: 0px;
                }
            """)
            
            frame_layout = QVBoxLayout(frame)
            frame_layout.setContentsMargins(0, 0, 0, 0)
            frame_layout.setSpacing(0)

            top_bar_widget = QWidget()
            # ✅ 排版優化: 給予固定高度，確保不論檔名斷成幾行，卡片高度都一致
            top_bar_widget.setFixedHeight(45) 
            top_bar_widget.setStyleSheet("background-color: #1A1A1A; border: none;")
            top_bar = QHBoxLayout(top_bar_widget)
            top_bar.setContentsMargins(4, 4, 4, 4) 

            chk = QCheckBox()
            chk.setStyleSheet(checkbox_style) 
            chk.setChecked(img_rel_path in self.selected_images)
            chk.stateChanged.connect(lambda state, name=img_rel_path: self.on_thumbnail_checked(state, name))
            self.thumb_checkboxes[img_rel_path] = chk

            pure_filename = os.path.basename(img_rel_path)
            lbl_name = QLabel(pure_filename)
            # ✅ 需求2: 開啟自動換行，遇到長檔名會自動斷行顯示
            lbl_name.setWordWrap(True)
            lbl_name.setStyleSheet("color: #FFFFFF; font-size: 11px; border: none; padding-left: 2px;")
            lbl_name.setToolTip(img_rel_path)

            top_bar.addWidget(chk)
            top_bar.addWidget(lbl_name, 1)
            frame_layout.addWidget(top_bar_widget)

            lbl_thumb = QLabel("載入中...")
            lbl_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl_thumb.setStyleSheet("color: #888888; border: none; background-color: #2D2D2D; min-height: 80px;")
            lbl_thumb.mousePressEvent = lambda event, idx=idx: self.jump_to_image(idx)
            
            self.thumb_labels[img_rel_path] = lbl_thumb
            frame_layout.addWidget(lbl_thumb)

            self.grid_layout.addWidget(frame, row, col)

    def update_single_thumbnail(self, index, img_rel_path, qimg):
        pixmap = QPixmap.fromImage(qimg)
        self.cache_thumb_masked[img_rel_path] = pixmap
        
        if img_rel_path in self.thumb_labels:
            self.thumb_labels[img_rel_path].setText("")
            self.thumb_labels[img_rel_path].setPixmap(pixmap)

    def generate_pixmap(self, img_rel_path, use_mask):
        img_path = os.path.join(self.img_dir, img_rel_path)
        if not os.path.exists(img_path):
            return QPixmap()

        img = Image.open(img_path).convert("RGBA")
        
        if use_mask and os.path.exists(self.mask_dir):
            base_name_no_ext = os.path.splitext(img_rel_path)[0]
            mask_path = os.path.join(self.mask_dir, base_name_no_ext + ".png")
            
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path).convert("L")
                binary_mask = mask_img.point(lambda p: 255 if p > 0 else 0)
                
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                red_layer = Image.new("RGBA", img.size, (255, 0, 0, 100))
                overlay.paste(red_layer, (0, 0), mask=binary_mask)
                img = Image.alpha_composite(img, overlay)

        img_qt = ImageQt.ImageQt(img)
        return QPixmap.fromImage(img_qt)

    def update_main_image(self):
        if 0 <= self.current_index < len(self.image_list):
            img_rel_path = self.image_list[self.current_index]
            pixmap = self.generate_pixmap(img_rel_path, use_mask=self.show_mask)
            self.main_viewer.set_image(pixmap)
            
            is_selected = img_rel_path in self.selected_images
            self.update_main_image_border(is_selected)

            self.setWindowTitle(f"海草資料集篩選器 - [{self.current_index + 1}/{len(self.image_list)}] {img_rel_path}")

    def update_main_image_border(self, is_selected):
        """🌟 同步更新右側大圖邊框，以及實體按鈕的狀態與文字"""
        if is_selected:
            self.main_viewer.setStyleSheet("QGraphicsView { border: 6px solid #0D6EFD; border-radius: 4px; }")
        else:
            self.main_viewer.setStyleSheet("QGraphicsView { border: 6px solid #1E1E1E; border-radius: 4px; }")
            
        # ✅ 同步按鈕狀態 (blockSignals 避免按鈕自己又觸發了一次選取事件造成無限迴圈)
        self.btn_toggle_select.blockSignals(True)
        self.btn_toggle_select.setChecked(is_selected)
        self.btn_toggle_select.setText("✓ 已選取 [空白鍵]" if is_selected else "選取照片 [空白鍵]")
        self.btn_toggle_select.blockSignals(False)

    def toggle_mask(self):
        self.show_mask = not self.show_mask
        self.btn_toggle_mask.setText("開啟遮罩 [H]" if not self.show_mask else "關閉遮罩 [H]")
        self.update_main_image()

    def jump_to_image(self, index):
        self.current_index = index
        self.update_main_image()

    def next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.update_main_image()

    def prev_image(self):
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.update_main_image()

    def on_thumbnail_checked(self, state, img_rel_path):
        if state == 2:
            self.selected_images.add(img_rel_path)
        else:
            self.selected_images.discard(img_rel_path)
            
        if self.image_list and 0 <= self.current_index < len(self.image_list):
            if img_rel_path == self.image_list[self.current_index]:
                self.update_main_image_border(state == 2)
                
        self.update_count_label()

    def update_count_label(self):
        self.lbl_count.setText(f"已選擇: {len(self.selected_images)} 張照片")

    def export_data(self):
        if not self.selected_images:
            QMessageBox.warning(self, "提示", "請至少勾選一張照片再進行 Export！")
            return

        export_base = os.path.join(self.base_dir, "export")

        if os.path.exists(export_base) and os.listdir(export_base):
            reply = QMessageBox.question(
                self, 
                "Export 資料夾不為空", 
                "export 資料夾內已經有其他檔案或資料夾。\n\n你要先「清空」它再進行匯出嗎？\n\n"
                "• [Yes]：先清空所有舊資料，再匯出\n"
                "• [No]：不清空，直接匯出 (會覆蓋同名檔案)\n"
                "• [Cancel]：取消本次匯出",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel
            )

            if reply == QMessageBox.StandardButton.Yes:
                try:
                    shutil.rmtree(export_base)
                except Exception as e:
                    QMessageBox.critical(self, "錯誤", f"清空資料夾失敗：\n{e}")
                    return
            elif reply == QMessageBox.StandardButton.Cancel:
                return

        success_count = 0

        for img_rel_path in self.selected_images:
            src_img = os.path.join(self.img_dir, img_rel_path)
            dst_img = os.path.join(export_base, "JPEGImages", img_rel_path)
            
            os.makedirs(os.path.dirname(dst_img), exist_ok=True)
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)

            base_name_no_ext = os.path.splitext(img_rel_path)[0]
            mask_rel_path = base_name_no_ext + ".png"
            
            src_mask = os.path.join(self.mask_dir, mask_rel_path)
            dst_mask = os.path.join(export_base, "SegmentationClass", mask_rel_path)
            
            if os.path.exists(src_mask):
                os.makedirs(os.path.dirname(dst_mask), exist_ok=True)
                shutil.copy2(src_mask, dst_mask)
            
            success_count += 1

        QMessageBox.information(self, "成功", f"已成功匯出 {success_count} 組資料至 export 資料夾內！")

    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.is_running = False
            self.worker_thread.quit()
            self.worker_thread.wait()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageApp()
    window.show()
    sys.argv.append('--style=Fusion')
    sys.exit(app.exec())