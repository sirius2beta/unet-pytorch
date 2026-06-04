import sys
import os
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, 
                             QPushButton, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QTextEdit)
from PyQt6.QtCore import QProcess
from PyQt6.QtGui import QFont

class DockerRunnerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.process = None # 用來儲存當前執行的 QProcess
        self.initUI()
        
    def initUI(self):
        # 1. 專案目錄選擇欄位
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Project Directory:")
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select your project path...")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_directory)
        
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(self.btn_browse)
        
        # 2. 控制按鈕 (Train / Test / Stop)
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Train")
        self.btn_train.clicked.connect(self.run_train)
        
        self.btn_test = QPushButton("Test")
        self.btn_test.clicked.connect(self.run_test)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_process)
        self.btn_stop.setEnabled(False) # 預設無法點擊
        
        # 美化按鈕樣式
        self.btn_train.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.btn_test.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        
        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_test)
        btn_layout.addWidget(self.btn_stop)
        
        # 3. 模擬 Terminal 的輸出視窗
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Courier New", 10)) # 使用等寬字型
        self.terminal_output.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        
        # 主佈局
        main_layout = QVBoxLayout()
        main_layout.addLayout(dir_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(QLabel("Console Output:"))
        main_layout.addWidget(self.terminal_output)
        
        self.setLayout(main_layout)
        self.setWindowTitle("Docker Task Runner")
        self.resize(700, 450)
        
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            self.dir_input.setText(directory)
            
    
    def run_train(self):
        project_dir = self.dir_input.text().strip()
        if not project_dir:
            self.terminal_output.append("<font color='red'>[Error] Please select a project directory first!</font>")
            return
            
        # 取得最後一個資料夾的名稱 (例如從 C:\project\unet-pytorch 取得 unet-pytorch)
        folder_name = os.path.basename(os.path.normpath(project_dir))
        container_workspace = f"/workspace/{folder_name}"
        
        # 完美還原並自動化你的 Docker 指令
        cmd = "docker"
        args = [
            "run", "--gpus", "all",
            "--shm-size=8g",
            "--rm",
            "-v", f"{project_dir}:{container_workspace}",
            "-w", "/workspace/unet-pytorch",
            "unet-train:headless",
            "python3", "train.py"
        ]
        
        self.start_command(cmd, args)

    def run_test(self):
        project_dir = self.dir_input.text().strip()
        if not project_dir:
            self.terminal_output.append("<font color='red'>[Error] Please select a project directory first!</font>")
            return
            
        folder_name = os.path.basename(os.path.normpath(project_dir))
        container_workspace = f"/workspace/{folder_name}"
        
        cmd = "docker"
        args = [
            "run", "--gpus", "all",
            "--shm-size=8g",
            "--rm",
            "-v", f"{project_dir}:{container_workspace}",
            "-w", "/workspace/unet-pytorch",
            "unet-train:headless",
            "python3", "test.py"
        ]
        
        self.start_command(cmd, args)

    def start_command(self, cmd, args):
        if self.process is not None and self.process.state() == QProcess.ProcessState.Running:
            return # 防呆：避免重複執行
            
        self.terminal_output.clear()
        self.terminal_output.append(f"<font color='#f1c40f'>[Running]: {cmd} {' '.join(args)}</font>\n")
        
        # 初始化 QProcess
        self.process = QProcess(self)
        # 合併標準輸出與標準錯誤輸出
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        
        # 綁定訊號 (當有新輸出時觸發)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        
        # 開始執行
        self.process.start(cmd, args)
        
        # 切換按鈕狀態
        self.set_ui_running_state(True)

    def handle_stdout(self):
        # 讀取輸出的資料
        data = self.process.readAllStandardOutput()
        # 嘗試解碼 (Windows 通常是 cp950 或 utf-8，Linux/Mac 為 utf-8)
        try:
            text = data.data().decode('utf-8')
        except UnicodeDecodeError:
            text = data.data().decode('cp950', errors='ignore')
            
        # 將內容移至最末端並加入新文字，確保滾動條會自動往下
        self.terminal_output.moveCursor(self.terminal_output.textCursor().MoveOperation.End)
        self.terminal_output.insertPlainText(text)
        self.terminal_output.moveCursor(self.terminal_output.textCursor().MoveOperation.End)

    def stop_process(self):
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.terminal_output.append("\n<font color='red'>[Stopping process...]</font>")
            self.process.terminate() # 嘗試溫和關閉
            if not self.process.waitForFinished(3000): # 如果 3 秒內沒反應
                self.process.kill() # 強制關閉

    def process_finished(self, exit_code, exit_status):
        self.terminal_output.append(f"\n<font color='#f1c40f'>[Finished] Process exited with code {exit_code}</font>")
        self.set_ui_running_state(False)
        self.process = None

    def set_ui_running_state(self, is_running):
        """根據執行狀態鎖定或解鎖按鈕"""
        self.btn_train.setEnabled(not is_running)
        self.btn_test.setEnabled(not is_running)
        self.btn_browse.setEnabled(not is_running)
        self.dir_input.setEnabled(not is_running)
        self.btn_stop.setEnabled(is_running)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DockerRunnerApp()
    ex.show()
    sys.exit(app.exec())