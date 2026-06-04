import sys
import os
import json
import shutil # 引入 shutil 用於複製檔案
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, 
                             QPushButton, QVBoxLayout, QHBoxLayout, 
                             QFileDialog, QTextEdit)
from PyQt6.QtCore import QProcess
from PyQt6.QtGui import QFont

class DockerRunnerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.process = None 
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_file_path = os.path.join(current_dir, "testconfig.txt")
        
        self.initUI()
        self.load_config_to_ui() 
        
    def initUI(self):
        # 1. 專案目錄選擇欄位
        dir_layout = QHBoxLayout()
        self.dir_label = QLabel("Project Directory:")
        self.dir_input = QLineEdit()
        self.dir_input.setPlaceholderText("Select your project path...")
        self.dir_input.returnPressed.connect(self.save_config_from_ui)
        
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_directory)
        
        dir_layout.addWidget(self.dir_label)
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(self.btn_browse)
        
        # 2. 模型權值選擇欄位 (.pth)
        model_layout = QHBoxLayout()
        self.model_label = QLabel("Model Weight (.pth):")
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Select or text your .pth file path...")
        self.model_input.returnPressed.connect(self.save_config_from_ui)
        
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
            self.save_config_from_ui() 
            
    def browse_model_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Weight File", "", "PyTorch Model (*.pth);;All Files (*)"
        )
        if file_path:
            file_path = file_path.replace('\\', '/')
            self.model_input.setText(file_path)
            self.save_config_from_ui() 

    def load_config_to_ui(self):
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, "r", encoding="utf-8") as f:
                    config_data = json.load(f)
                    if "project_directory" in config_data:
                        self.dir_input.setText(config_data["project_directory"])
                    # 介面顯示原本選取的模型絕對路徑/外部路徑
                    if "model_original_path" in config_data:
                        self.model_input.setText(config_data["model_original_path"])
                    elif "model_path" in config_data:
                        self.model_input.setText(config_data["model_path"])
            except Exception as e:
                print(f"Error reading config: {e}")
        else:
            self.save_config_from_ui(quiet=True)

    # ====== 核心優化：處理外部模型複製與 Docker 相對路徑轉換 ======
    def save_config_from_ui(self, quiet=False):
        project_dir = self.dir_input.text().strip().replace('\\', '/')
        model_path = self.model_input.text().strip().replace('\\', '/')
        
        # 預設 Docker 內部的模型路徑跟原本填的一樣
        docker_model_path = model_path
        model_original_path = model_path
        
        # 只有在「專案路徑」和「模型路徑」都有填寫，且模型檔案確實存在時才進行判斷
        if project_dir and model_path and os.path.exists(model_path):
            # 檢查模型是否在專案資料夾內部
            # os.path.commonpath 會判斷共同的最上層目錄是否為 project_dir
            try:
                is_inside = os.path.commonpath([project_dir, model_path]) == os.path.normpath(project_dir)
            except ValueError:
                is_inside = False # 跨磁碟（例如 C 槽跟 D 槽）會丟出 ValueError，代表一定在外面
                
            if not is_inside:
                # 1. 規劃專案內部的臨時複製目標資料夾 (tmp_model)
                tmp_model_dir = os.path.join(project_dir, "tmp_model").replace('\\', '/')
                os.makedirs(tmp_model_dir, exist_ok=True)
                
                filename = os.path.basename(model_path)
                dest_file_path = os.path.join(tmp_model_dir, filename).replace('\\', '/')
                
                # 2. 複製檔案（防呆：如果檔案已經存在且大小相同就不重複複製）
                if not (os.path.exists(dest_file_path) and os.path.getsize(model_path) == os.path.getsize(dest_file_path)):
                    if not quiet:
                        self.terminal_output.append(f"<font color='#f1c40f'>[File] Copying model to project temporary folder: tmp_model/{filename} ...</font>")
                        QApplication.processEvents() # 讓 UI 刷新顯示提示，避免凍結
                    try:
                        shutil.copy2(model_path, dest_file_path)
                        if not quiet:
                            self.terminal_output.append(f"<font color='#2ecc71'>[File] Copy completed!</font>")
                    except Exception as e:
                        self.terminal_output.append(f"<font color='red'>[Error] Failed to copy model file: {e}</font>")
                
                # 3. 關鍵：寫入 testconfig.txt 的路徑必須是 Docker 看得見的相對路徑！
                docker_model_path = f"tmp_model/{filename}"
            else:
                # 如果原本就在專案裡面，轉成相對於專案目錄的相對路徑
                docker_model_path = os.path.relpath(model_path, project_dir).replace('\\', '/')

        config_data = {
            "project_directory": project_dir,
            "model_path": docker_model_path,          # 這是給 Docker 內的程式碼（Unet類別）讀的相對路徑
            "model_original_path": model_original_path # 這是給 UI 介面記錄原本選取位置用的
        }
        
        try:
            with open(self.config_file_path, "w", encoding="utf-8") as f:
                json.dump(config_data, f, indent=4, ensure_ascii=False)
            
            if not quiet:
                self.terminal_output.append(
                    f"<font color='#2ecc71'>[Config] Settings saved. Docker path -> {docker_model_path}</font>"
                )
        except Exception as e:
            self.terminal_output.append(f"<font color='red'>[Error] Failed to write config file: {e}</font>")

    def run_train(self):
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