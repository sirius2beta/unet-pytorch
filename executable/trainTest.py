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
        self.process = None 
        self.initUI()
        
    def initUI(self):
        # 1. 專案目錄選擇欄位
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Project Directory:")
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("選擇unet-pytorch專案目錄...")
        
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_directory)
        
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(self.btn_browse)
        
        # 2. 模型權值選擇欄位 (.pth)
        model_layout = QHBoxLayout()
        self.model_label = QLabel("Model Weight:")
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("選擇模型權重(.pth)...")
        
        self.btn_browse_model = QPushButton("Browse")
        self.btn_browse_model.clicked.connect(self.browse_model_file)
        
        model_layout.addWidget(self.model_label)
        model_layout.addWidget(self.model_input)
        model_layout.addWidget(self.btn_browse_model)
        
        # 3. 控制按鈕
        btn_layout = QHBoxLayout()
        self.btn_train = QPushButton("Train")
        self.btn_train.clicked.connect(self.run_train)
        
        self.btn_test = QPushButton("Test")
        self.btn_test.clicked.connect(self.run_test)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_process)
        self.btn_stop.setEnabled(False) 
        
        self.btn_train.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold;")
        self.btn_test.setStyleSheet("background-color: #3498db; color: white; font-weight: bold;")
        self.btn_stop.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold;")
        
        btn_layout.addWidget(self.btn_train)
        btn_layout.addWidget(self.btn_test)
        btn_layout.addWidget(self.btn_stop)
        
        # 4. 模擬 Terminal 的輸出視窗
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Courier New", 10)) 
        self.terminal_output.setStyleSheet("background-color: #1e1e1e; color: #ffffff;")
        
        main_layout = QVBoxLayout()
        main_layout.addLayout(dir_layout)
        main_layout.addLayout(model_layout)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(QLabel("Console Output:"))
        main_layout.addWidget(self.terminal_output)
        
        self.setLayout(main_layout)
        self.setWindowTitle("Docker Task Runner")
        self.resize(750, 500)
        
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if directory:
            directory = directory.replace('\\', '/')
            self.dir_input.setText(directory)
            
    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Weight File", "", "PyTorch Model (*.pth);;All Files (*)"
        )
        if file_path:
            file_path = file_path.replace('\\', '/')
            self.model_input.setText(file_path)

    def build_docker_args(self, script_name):
        project_dir = self.dir_input.text().strip().replace('\\', '/')
        model_path = self.model_input.text().strip().replace('\\', '/')
        
        if not project_dir:
            self.terminal_output.append("<font color='red'>[Error] Please select a project directory first!</font>")
            return None
            
        folder_name = os.path.basename(os.path.normpath(project_dir))
        container_workspace = f"/workspace/{folder_name}"
        
        args = [
            "run", "--gpus", "all",
            "--shm-size=8g",
            "--rm",
            "-v", f"{project_dir}:{container_workspace}",
            "-w", "/workspace/unet-pytorch"
        ]

        # 如果有選擇模型檔案，將其直接掛載到 /workspace/weight/ 下
        if model_path and os.path.exists(model_path):
            filename = os.path.basename(model_path)
            docker_model_path = f"/workspace/weight/{filename}"
            args.extend(["-v", f"{model_path}:{docker_model_path}"])
            
            # (選擇性) 將路徑設為環境變數，方便你在 Python 腳本中透過 os.environ 讀取
            args.extend(["-e", f"MODEL_PATH={docker_model_path}"])

        args.extend([
            "unet-train:headless",
            "python3", script_name
        ])
        
        return args

    def run_train(self):
        args = self.build_docker_args("train.py")
        if args:
            self.start_command("docker", args)

    def run_test(self):
        args = self.build_docker_args("test.py")
        if args:
            self.start_command("docker", args)

    def start_command(self, cmd, args):
        if self.process is not None and self.process.state() == QProcess.ProcessState.Running:
            return 
            
        self.terminal_output.clear()
        self.terminal_output.append(f"<font color='#f1c40f'>[Running]: {cmd} {' '.join(args)}</font>\n")
        
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        
        self.process.start(cmd, args)
        self.set_ui_running_state(True)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput()
        try:
            text = data.data().decode('utf-8')
        except UnicodeDecodeError:
            text = data.data().decode('cp950', errors='ignore')
            
        self.terminal_output.moveCursor(self.terminal_output.textCursor().MoveOperation.End)
        self.terminal_output.insertPlainText(text)
        self.terminal_output.moveCursor(self.terminal_output.textCursor().MoveOperation.End)

    def stop_process(self):
        if self.process and self.process.state() == QProcess.ProcessState.Running:
            self.terminal_output.append("\n<font color='red'>[Stopping process...]</font>")
            self.process.terminate() 
            if not self.process.waitForFinished(3000): 
                self.process.kill() 

    def process_finished(self, exit_code, exit_status):
        self.terminal_output.append(f"\n<font color='#f1c40f'>[Finished] Process exited with code {exit_code}</font>")
        self.set_ui_running_state(False)
        self.process = None

    def set_ui_running_state(self, is_running):
        self.btn_train.setEnabled(not is_running)
        self.btn_test.setEnabled(not is_running)
        self.btn_browse.setEnabled(not is_running)
        self.dir_input.setEnabled(not is_running)
        self.model_input.setEnabled(not is_running)
        self.btn_browse_model.setEnabled(not is_running)
        self.btn_stop.setEnabled(is_running)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = DockerRunnerApp()
    ex.show()
    sys.exit(app.exec())