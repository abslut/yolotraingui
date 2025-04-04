# _  __              _          _____ _   _   ___   _   _
#| |/ /  ___ __   __(_) _ __   |__  /| | | | / _ \ | | | |
#| ' /  / _ \\ \ / /| || '_ \    / / | |_| || | | || | | |
#| . \ |  __/ \ V / | || | | |  / /_ |  _  || |_| || |_| |
#|_|\_\ \___|  \_/  |_||_| |_| /____||_| |_| \___/  \___/
# APRIL 2025
# Version 1.0
# A GUI TOOL
# For YOLO Training
import os
import sys
import re
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
import torch

def resource_path(relative_path):
    """获取资源的绝对路径，适用于开发和 PyInstaller 打包后的情况"""
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

class TrainWorker(QObject):
    update_log = pyqtSignal(str)
    training_finished = pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.process = QProcess()
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stderr)
        self.process.finished.connect(self.handle_finished)

    def start_training(self, args):
        if sys.platform == "win32":
            yolo_command = "yolo"
        elif sys.platform.startswith("darwin"):
            yolo_command = "yolo"
        else:
            raise OSError("Unsupported operating system")
        self.process.start(yolo_command, args)

    def handle_stdout(self):
        data = self.process.readAllStandardOutput().data()
        decoded_data = self.decode_data(data)
        self.filter_and_emit(decoded_data)

    def handle_stderr(self):
        data = self.process.readAllStandardError().data()
        decoded_data = self.decode_data(data)
        self.filter_and_emit(decoded_data)

    def decode_data(self, data):
        encodings = ['utf-8', 'gbk', 'latin-1']
        for encoding in encodings:
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                pass
        return data.decode('utf-8', errors='ignore')

    def filter_and_emit(self, text):
        cleaned_text = re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', text)
        self.update_log.emit(cleaned_text.strip())

    def handle_finished(self, exit_code):
        self.training_finished.emit(exit_code)

class ConsoleDisplay(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setStyleSheet("""
            background-color: #1E1E1E;
            color: #D4D4D4;
            font-family: Consolas;
            font-size: 10pt;
            border: none;
        """)

    def append_message(self, message):
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertPlainText(message + "\n")
        self.ensureCursorVisible()

class TrainingWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO训练控制台")
        self.setGeometry(100, 50, 1200, 800)
        self.init_ui()
        self.setup_worker()

    def init_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()
        
        # 设置窗口图标并调试
        icon_path = resource_path('ico.icns' if sys.platform.startswith("darwin") else 'ico.ico')
        icon = QIcon(icon_path)
        self.setWindowIcon(icon)

        # 参数配置区
        param_group = QGroupBox("训练参数配置")
        grid_layout = QGridLayout()

        # 模型选择
        self.model_path = self.create_file_input("pt")
        grid_layout.addWidget(QLabel("模型文件 (.pt):"), 0, 0)
        grid_layout.addWidget(self.model_path, 0, 1)

        # 数据集配置
        self.data_yaml = self.create_file_input("yaml")
        grid_layout.addWidget(QLabel("数据集 (.yaml):"), 0, 2)
        grid_layout.addWidget(self.data_yaml, 0, 3)

        # 训练轮次
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 1000)
        self.epochs.setValue(100)
        grid_layout.addWidget(QLabel("训练轮次:"), 0, 4)
        grid_layout.addWidget(self.epochs, 0, 5)

        # 输入尺寸
        self.imgsz = QSpinBox()
        self.imgsz.setRange(32, 2048)
        self.imgsz.setValue(640)
        grid_layout.addWidget(QLabel("输入尺寸:"), 0, 6)
        grid_layout.addWidget(self.imgsz, 0, 7)

        # 批处理大小
        self.batch = QSpinBox()
        self.batch.setRange(-1, 256)
        self.batch.setValue(16)
        grid_layout.addWidget(QLabel("批处理大小:"), 1, 0)
        grid_layout.addWidget(self.batch, 1, 1)

        # 设备选择
        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "mps", "0", "0,1"])
        grid_layout.addWidget(QLabel("训练设备:"), 1, 2)
        grid_layout.addWidget(self.device, 1, 3)

        # 输出路径
        self.output_path = self.create_file_input("folder")
        grid_layout.addWidget(QLabel("输出文件夹:"), 1, 4)
        grid_layout.addWidget(self.output_path, 1, 5)

        # 优化器
        self.optimizer = QComboBox()
        self.optimizer.addItems(["SGD", "Adam", "AdamW"])
        grid_layout.addWidget(QLabel("优化器:"), 1, 6)
        grid_layout.addWidget(self.optimizer, 1, 7)

        # 学习率
        self.lr = QDoubleSpinBox()
        self.lr.setRange(0.00001, 1.0)
        self.lr.setValue(0.01)
        self.lr.setSingleStep(0.00001)
        self.lr.setDecimals(5)
        grid_layout.addWidget(QLabel("学习率:"), 2, 0)
        grid_layout.addWidget(self.lr, 2, 1)

        # 正则化强度
        self.weight_decay = QDoubleSpinBox()
        self.weight_decay.setRange(0.0, 1.0)
        self.weight_decay.setValue(0.0005)
        self.weight_decay.setSingleStep(0.00001)
        self.weight_decay.setDecimals(5)
        grid_layout.addWidget(QLabel("正则化强度:"), 2, 2)
        grid_layout.addWidget(self.weight_decay, 2, 3)

        # 马赛克增强
        self.mosaic = QCheckBox("启用")
        self.mosaic.setChecked(True)
        grid_layout.addWidget(QLabel("马赛克增强:"), 2, 4)
        grid_layout.addWidget(self.mosaic, 2, 5)

        # MixUp
        self.mixup = QDoubleSpinBox()
        self.mixup.setRange(0.0, 1.0)
        self.mixup.setValue(0.0)
        self.mixup.setSingleStep(0.1)
        grid_layout.addWidget(QLabel("MixUp:"), 2, 6)
        grid_layout.addWidget(self.mixup, 2, 7)

        # 翻转方向
        self.flip = QComboBox()
        self.flip.addItems(["none", "horizontal", "vertical", "both"])
        grid_layout.addWidget(QLabel("翻转方向:"), 3, 0)
        grid_layout.addWidget(self.flip, 3, 1)

        # 旋转范围
        self.degrees = QDoubleSpinBox()
        self.degrees.setRange(0.0, 180.0)
        self.degrees.setValue(0.0)
        self.degrees.setSingleStep(1.0)
        grid_layout.addWidget(QLabel("旋转范围:"), 3, 2)
        grid_layout.addWidget(self.degrees, 3, 3)

        # 平移幅度
        self.translate = QDoubleSpinBox()
        self.translate.setRange(0.0, 1.0)
        self.translate.setValue(0.1)
        self.translate.setSingleStep(0.01)
        grid_layout.addWidget(QLabel("平移幅度:"), 3, 4)
        grid_layout.addWidget(self.translate, 3, 5)

        # 错切幅度
        self.shear = QDoubleSpinBox()
        self.shear.setRange(0.0, 180.0)
        self.shear.setValue(0.0)
        self.shear.setSingleStep(1.0)
        grid_layout.addWidget(QLabel("错切幅度:"), 3, 6)
        grid_layout.addWidget(self.shear, 3, 7)

        # 缩放
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.0, 1.0)
        self.scale.setValue(0.5)
        self.scale.setSingleStep(0.1)
        grid_layout.addWidget(QLabel("缩放:"), 4, 0)
        grid_layout.addWidget(self.scale, 4, 1)

        # 早停轮次
        self.patience = QSpinBox()
        self.patience.setRange(0, 100)
        self.patience.setValue(50)
        grid_layout.addWidget(QLabel("早停轮次:"), 4, 2)
        grid_layout.addWidget(self.patience, 4, 3)

        # 混合精度
        self.amp = QCheckBox("启用")
        self.amp.setChecked(True)
        grid_layout.addWidget(QLabel("混合精度:"), 4, 4)
        grid_layout.addWidget(self.amp, 4, 5)

        # 继续训练
        self.resume = QCheckBox("启用")
        grid_layout.addWidget(QLabel("继续训练:"), 4, 6)
        grid_layout.addWidget(self.resume, 4, 7)

        # 启用个性化模型
        self.enable_personal_model = QCheckBox("启用个性化模型")
        self.enable_personal_model.stateChanged.connect(self.toggle_personal_model_yaml)
        grid_layout.addWidget(self.enable_personal_model, 5, 0, 1, 2)

        # 个性化模型 YAML 文件
        self.personal_model_yaml = self.create_file_input("yaml")
        self.personal_model_yaml_line_edit = self.personal_model_yaml.findChild(QLineEdit)
        self.personal_model_yaml_button = self.personal_model_yaml.findChild(QPushButton)
        self.personal_model_yaml_line_edit.setEnabled(False)
        self.personal_model_yaml_button.setEnabled(False)
        grid_layout.addWidget(QLabel("个性化YAML:"), 5, 2)
        grid_layout.addWidget(self.personal_model_yaml, 5, 3)

        param_group.setLayout(grid_layout)
        layout.addWidget(param_group)

        # 控制台输出
        self.console = ConsoleDisplay()
        layout.addWidget(self.console)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始训练", clicked=self.start_training)
        self.stop_btn = QPushButton("终止训练", clicked=self.stop_training)
        self.stop_btn.setEnabled(False)
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        layout.addLayout(btn_layout)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

    def create_file_input(self, file_type):
        h_layout = QHBoxLayout()
        line_edit = QLineEdit()
        btn = QPushButton("选择")
        btn.clicked.connect(lambda: self.select_file(line_edit, file_type))
        h_layout.addWidget(line_edit)
        h_layout.addWidget(btn)
        container = QWidget()
        container.setLayout(h_layout)
        return container

    def select_file(self, target_field, file_type):
        filters = {
            "pt": ("PyTorch模型文件 (*.pt)", ".pt"),
            "yaml": ("YAML配置文件 (*.yaml)", ".yaml"),
            "folder": ("文件夹", "")
        }
        filter_desc, extension = filters[file_type]
        if file_type == "folder":
            path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", filter_desc)
        if path and (file_type == "folder" or path.endswith(extension)):
            target_field.setText(path)

    def toggle_personal_model_yaml(self, state):
        enabled = (state == Qt.CheckState.Checked.value)
        self.personal_model_yaml_line_edit.setEnabled(enabled)
        self.personal_model_yaml_button.setEnabled(enabled)

    def setup_worker(self):
        self.worker = TrainWorker()
        self.worker.update_log.connect(self.console.append_message)
        self.worker.training_finished.connect(self.handle_training_end)

    def validate_inputs(self):
        errors = []
        if not self.model_path.findChild(QLineEdit).text().endswith('.pt'):
            errors.append("请选择有效的模型文件 (.pt)")
        if not self.data_yaml.findChild(QLineEdit).text().endswith('.yaml'):
            errors.append("请选择有效的数据集配置文件 (.yaml)")
        if not self.output_path.findChild(QLineEdit).text():
            errors.append("请选择有效的输出路径")
        if self.enable_personal_model.isChecked() and not self.personal_model_yaml_line_edit.text():
            errors.append("启用个性化模型时必须选择 YAML 配置文件")
        if errors:
            QMessageBox.critical(self, "输入错误", "\n".join(errors))
            return False
        return True

    def start_training(self):
        if not self.validate_inputs():
            return

        device = self.device.currentText()
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif sys.platform == "darwin" and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        args = [
            "train",
            f"model={self.model_path.findChild(QLineEdit).text()}",
            f"data={self.data_yaml.findChild(QLineEdit).text()}",
            f"epochs={self.epochs.value()}",
            f"imgsz={self.imgsz.value()}",
            f"batch={self.batch.value()}",
            f"device={device}",
            "save_period=1",
            "exist_ok=True",
            "verbose=False",
            f"project={self.output_path.findChild(QLineEdit).text()}",
            f"optimizer={self.optimizer.currentText()}",
            f"lr0={self.lr.value()}",
            f"weight_decay={self.weight_decay.value()}",
            f"mosaic={1.0 if self.mosaic.isChecked() else 0.0}",
            f"mixup={self.mixup.value()}",
            f"flipud={1.0 if self.flip.currentText() in ['vertical', 'both'] else 0.0}",
            f"fliplr={1.0 if self.flip.currentText() in ['horizontal', 'both'] else 0.0}",
            f"degrees={self.degrees.value()}",
            f"translate={self.translate.value()}",
            f"shear={self.shear.value()}",
            f"scale={self.scale.value()}",
            f"patience={self.patience.value()}",
            f"amp={self.amp.isChecked()}",
            f"resume={self.resume.isChecked()}"
        ]

        if self.enable_personal_model.isChecked():
            args.append(f"cfg={self.personal_model_yaml_line_edit.text()}")

        self.console.clear()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.worker.start_training(args)

    def stop_training(self):
        if self.worker.process.state() == QProcess.ProcessState.Running:
            self.worker.process.kill()
            self.console.append_message("\n[系统] 训练已手动终止")

    def handle_training_end(self, exit_code):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        status = "成功完成" if exit_code == 0 else f"异常退出 (代码 {exit_code})"
        self.console.append_message(f"\n[系统] 训练{status}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 设置全局应用程序图标
    icon_path = resource_path('ico.icns' if sys.platform.startswith("darwin") else 'ico.ico')
    app_icon = QIcon(icon_path)
    app.setWindowIcon(app_icon)
    
    window = TrainingWindow()
    window.show()
    sys.exit(app.exec())