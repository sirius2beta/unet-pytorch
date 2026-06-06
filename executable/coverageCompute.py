import sys
import os
import subprocess
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QProgressBar, 
                             QTextEdit, QMessageBox, QLineEdit)
from PyQt6.QtCore import QThread, pyqtSignal, QSettings

class InferenceWorker(QThread):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, proj_dir, img_dir, model_path, csv_path):
        super().__init__()
        self.proj_dir = proj_dir
        self.img_dir = img_dir
        self.model_path = model_path
        self.csv_path = csv_path
        self.docker_image = "unet-train:headless"  
        self.is_running = True
        self.process = None

    def run(self):
        try:
            # 準備輸出檔案與欄位名稱
            model_filename = os.path.basename(self.model_path)
            col_name = os.path.splitext(model_filename)[0]

            csv_dir = os.path.dirname(os.path.abspath(self.csv_path))
            csv_basename = os.path.splitext(os.path.basename(self.csv_path))[0]
            output_csv_filename = f"{csv_basename}_{col_name}.csv"
            output_csv_full_path = os.path.join(csv_dir, output_csv_filename)
            input_csv_filename = os.path.basename(self.csv_path)

            # 強制轉換所有路徑，避免 Windows 爛斜線搞事
            proj_mount = os.path.abspath(self.proj_dir).replace('\\', '/')
            img_mount = os.path.abspath(self.img_dir).replace('\\', '/')
            model_mount = os.path.abspath(self.model_path).replace('\\', '/')
            csv_mount_dir = csv_dir.replace('\\', '/')

            # 呼叫 Docker 的指令 (注意加上了 -u 確保 Python 即時輸出)
            args = [
                "docker", "run", "--gpus", "all",
                "--shm-size=8g", "--rm",
                "-v", f"{proj_mount}:/app/project",
                "-v", f"{img_mount}:/app/img",
                "-v", f"{model_mount}:/app/model.pth",
                "-v", f"{csv_mount_dir}:/app/csv",   # 把 CSV 所在的資料夾也掛載進去
                self.docker_image,
                "python", "-u", "/app/project/predict_batch.py",  # 呼叫新的批次腳本
                f"/app/csv/{input_csv_filename}",                 # 輸入的 CSV
                f"/app/csv/{output_csv_filename}",                # 輸出的 CSV
                "/app/img",                                       # 圖片路徑
                "/app/model.pth",                                 # 模型路徑
                col_name                                          # 欄位名稱
            ]

            self.log_signal.emit("[系統] 正在啟動 Docker 容器，請稍候...")

            # 使用 Popen 來即時讀取 Docker 的輸出
            self.process = subprocess.Popen(
                args, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                encoding="utf-8"
            )

            processed_count = 0

            # 即時逐行讀取 Docker 內部的回報
            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    self.process.terminate()
                    self.log_signal.emit("[系統] 收到停止指令，強制終止中...")
                    break
                
                line = line.strip()
                if not line:
                    continue

                # 解析我們在 predict_batch.py 設計好的標籤
                if line.startswith("PROGRESS|"):
                    pct = int(line.split("|")[1])
                    self.progress_signal.emit(pct)
                elif line.startswith("LOG|"):
                    msg = line.split("|", 1)[1]
                    self.log_signal.emit(msg)
                elif line.startswith("DONE|"):
                    processed_count = line.split("|")[1]
                else:
                    # 如果有非預期的 Python 報錯，直接印出來
                    self.log_signal.emit(line)

            self.process.wait()

            if self.is_running:
                if self.process.returncode == 0:
                    self.finished_signal.emit(True, f"共辨識了 {processed_count} 張。\n結果已儲存至: {output_csv_full_path}")
                else:
                    self.finished_signal.emit(False, "Docker 執行過程中發生嚴重錯誤，請查看上方 Log。")

        except Exception as e:
            self.finished_signal.emit(False, str(e))

    def stop(self):
        self.is_running = False
        if self.process:
            self.process.terminate()

class SeagrassApp(QWidget):
    def __init__(self):
        super().__init__()
        # 初始化 QSettings，會在同資料夾產生 config.ini
        self.settings = QSettings('config.ini', QSettings.Format.IniFormat)
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Seagrass Coverage Detector')
        self.resize(700, 450)

        layout = QVBoxLayout()

        # 1. 選擇 Project 資料夾
        proj_layout = QHBoxLayout()
        self.lbl_proj = QLabel("unet-pytorch資料夾:")
        self.txt_proj = QLineEdit()
        self.btn_proj = QPushButton("瀏覽...")
        self.btn_proj.clicked.connect(self.select_proj_dir)
        proj_layout.addWidget(self.lbl_proj)
        proj_layout.addWidget(self.txt_proj)
        proj_layout.addWidget(self.btn_proj)
        layout.addLayout(proj_layout)

        # 2. 選擇 模型檔 (.pth)
        model_layout = QHBoxLayout()
        self.lbl_model = QLabel("模型權重 (.pth):")
        self.txt_model = QLineEdit()
        self.btn_model = QPushButton("瀏覽...")
        self.btn_model.clicked.connect(self.select_model_path)
        model_layout.addWidget(self.lbl_model)
        model_layout.addWidget(self.txt_model)
        model_layout.addWidget(self.btn_model)
        layout.addLayout(model_layout)

        # 3. 選擇圖片資料夾 (img)
        img_layout = QHBoxLayout()
        self.lbl_img = QLabel("圖片資料夾 (img):")
        self.txt_img = QLineEdit()
        self.btn_img = QPushButton("瀏覽...")
        self.btn_img.clicked.connect(self.select_img_dir)
        img_layout.addWidget(self.lbl_img)
        img_layout.addWidget(self.txt_img)
        img_layout.addWidget(self.btn_img)
        layout.addLayout(img_layout)

        # 4. 選擇輸入 CSV
        csv_layout = QHBoxLayout()
        self.lbl_csv = QLabel("輸入 CSV 檔案:")
        self.txt_csv = QLineEdit()
        self.btn_csv = QPushButton("瀏覽...")
        self.btn_csv.clicked.connect(self.select_csv)
        csv_layout.addWidget(self.lbl_csv)
        csv_layout.addWidget(self.txt_csv)
        csv_layout.addWidget(self.btn_csv)
        layout.addLayout(csv_layout)

        # 進度條與 Log
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        # 執行與停止按鈕
        btn_layout = QHBoxLayout()
        self.btn_start = QPushButton("開始辨識")
        self.btn_start.setStyleSheet("font-weight: bold; color: white;")
        self.btn_start.clicked.connect(self.start_inference)
        self.btn_stop = QPushButton("強制停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_inference)
        
        btn_layout.addWidget(self.btn_start)
        btn_layout.addWidget(self.btn_stop)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        
        # UI 建立完成後，載入上次儲存的設定
        self.load_settings()

    def load_settings(self):
        # 讀取 config.ini，如果沒有設定過預設回傳空字串
        self.txt_proj.setText(self.settings.value('proj_dir', ''))
        self.txt_model.setText(self.settings.value('model_path', ''))
        self.txt_img.setText(self.settings.value('img_dir', ''))
        self.txt_csv.setText(self.settings.value('csv_path', ''))

    def save_settings(self):
        # 將目前的文字框內容寫入 config.ini
        self.settings.setValue('proj_dir', self.txt_proj.text().strip())
        self.settings.setValue('model_path', self.txt_model.text().strip())
        self.settings.setValue('img_dir', self.txt_img.text().strip())
        self.settings.setValue('csv_path', self.txt_csv.text().strip())

    # 覆寫關閉視窗的事件，在打 X 關閉程式時自動存檔
    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def select_proj_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇 Project 資料夾")
        if directory:
            self.txt_proj.setText(directory)

    def select_model_path(self):
        file, _ = QFileDialog.getOpenFileName(self, "選擇模型權重", "", "PyTorch Models (*.pth *.pt)")
        if file:
            self.txt_model.setText(file)

    def select_img_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "選擇圖片資料夾")
        if directory:
            self.txt_img.setText(directory)

    def select_csv(self):
        file, _ = QFileDialog.getOpenFileName(self, "選擇輸入 CSV", "", "CSV Files (*.csv)")
        if file:
            self.txt_csv.setText(file)

    def log(self, message):
        self.log_text.append(message)

    def start_inference(self):
        proj_dir = self.txt_proj.text().strip()
        model_path = self.txt_model.text().strip()
        img_dir = self.txt_img.text().strip()
        csv_path = self.txt_csv.text().strip()

        if not all([proj_dir, model_path, img_dir, csv_path]):
            QMessageBox.warning(self, "警告", "請確保所有路徑皆已填寫！")
            return

        # 執行前也強制存檔一次，確保萬無一失
        self.save_settings()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        # 啟動 Worker
        self.worker = InferenceWorker(proj_dir, img_dir, model_path, csv_path)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.log_signal.connect(self.log)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def stop_inference(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.stop()
            self.btn_stop.setEnabled(False)

    def on_finished(self, success, message):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if success:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.critical(self, "錯誤", message)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SeagrassApp()
    ex.show()
    sys.exit(app.exec())