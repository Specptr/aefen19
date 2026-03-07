# app.py
# 26.3.7
import sys
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QTextEdit,
    QLineEdit, QPushButton, QVBoxLayout, QGridLayout,
    QHBoxLayout, QFrame
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QTextCursor, QTextBlockFormat, QTextCharFormat, QColor
from inference.infer import generate_stream
from inference.infer import runtime_stats, runtime
from PySide6.QtCore import QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QStackedLayout
from config import *

from PySide6.QtCore import QThread, Signal

class ModelThread(QThread):
    token_signal = Signal(str)

    def __init__(self, prompt, history):
        super().__init__()
        self.prompt = prompt
        self.history = history

    def run(self):
        for token in generate_stream(self.prompt, self.history):
            self.token_signal.emit(token)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        self.resize(800, 520)
        self.history = []

        self.setStyleSheet("""
            QWidget {
                background-color: #050505;
                font-family: "Segoe UI";
                color: #e0e0e0;
                font-size: 16px;
            }

            QFrame {
                background-color: #000000;
                border: 1.5px solid #c1c1c1;
                border-radius: 16px;
            }

            QTextEdit {
                background-color: #050505;
                border: none;
                padding: 20px;
            }

            QLineEdit {
                background-color: #000000;
                border: none;
                padding: 14px;
            }

            QPushButton {
                background-color: #000000;
                border-radius: 10px;
                border: 2px solid #c0c0c0;
                padding: 8px 14px;
            }

            QPushButton:hover {
                background-color: #1a1a1a;
            }

            QLineEdit {
                padding-left: 16px;
            }

            QPushButton {
                min-width: 20px;
                max-width: 20px;
                height: 15px;
            }

            #no_frame {
                border: none;
            }
        """)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(30, 20, 30, 20)

        # ===== 标题 =====
        title = QLabel("Æfen 19")
        title.setStyleSheet("""
            font-family: "Script MT";
            font-size: 20px;
            letter-spacing: 2px;
            """)
        title.setAlignment(Qt.AlignLeft)
        title.setContentsMargins(10, 0, 0, 0)
        title.setObjectName("no_frame")
        main_layout.addWidget(title)

        # ===== 中间区域 =====
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(40)

        # 左侧照片区
        self.photo_frame = QFrame()
        photo_layout = QVBoxLayout()
        photo_layout.setContentsMargins(0, 0, 0, 0)

        self.photo_label = QLabel()
        self.photo_label.setAlignment(Qt.AlignCenter)

        pixmap = QPixmap(PHOTO_PATH)
        if not pixmap.isNull():
            self.photo_label.setPixmap(
                pixmap.scaled(
                    250, 400,
                    Qt.KeepAspectRatio,
                    Qt.FastTransformation
                )
            )
        else:
            self.photo_label.setText("photo")
        self.photo_label.setObjectName("no_frame")

        photo_layout.addWidget(self.photo_label)
        self.photo_frame.setLayout(photo_layout)
        self.photo_frame.setFixedWidth(240)

        # 右侧对话区
        self.chat_frame = QFrame()
        chat_layout = QVBoxLayout()
        chat_layout.setContentsMargins(0, 0, 0, 0)

        self.chat_area = QTextEdit()
        self.chat_area.setReadOnly(True)

        chat_layout.addWidget(self.chat_area)
        self.chat_frame.setLayout(chat_layout)

        middle_layout.addWidget(self.photo_frame)
        middle_layout.addWidget(self.chat_frame)

        main_layout.addLayout(middle_layout)

        # ===== 状态栏 =====
        self.status_frame = QFrame()
        status_layout = QHBoxLayout()
        status_layout.setContentsMargins(10, 2, 10, 2)
        status_layout.setSpacing(10)
        self.status_frame.setObjectName("no_frame")
        self.status_frame.setFrameShape(QFrame.NoFrame)

        # 文本状态
        self.status_label = QLabel(f"")
        self.status_label.setStyleSheet("""
            font-size: 12px;
            color: #9a9a9a;
            border: none;
            background: transparent;
        """)

        # bar
        self.bar_label = QLabel("")
        self.bar_label.setStyleSheet("""
            font-size: 12px;
            color: #9a9a9a;
            border: none;
            background: transparent;
        """)

        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.bar_label, alignment=Qt.AlignRight)

        self.status_frame.setLayout(status_layout)
        main_layout.addWidget(self.status_frame)

        # ===== 底部区域 =====
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(40)

        # ===== 左侧按钮区 =====
        self.control_frame = QFrame()
        control_layout = QVBoxLayout()
        control_layout.setSpacing(3)
        control_layout.setContentsMargins(0,0,0,0)

        # ---------- 第一排（占位按钮 保留边框） ----------
        row1 = QHBoxLayout()

        btn1 = QPushButton("")
        btn2 = QPushButton("")
        btn3 = QPushButton("")

        row1.addWidget(btn1)
        row1.addWidget(btn2)
        row1.addWidget(btn3)

        control_layout.addLayout(row1)


        # ---------- 第二排（T P M 无框按钮） ----------
        row2 = QHBoxLayout()

        self.btn_t = QPushButton("T")
        self.btn_p = QPushButton("P")
        self.btn_m = QPushButton("M")

        for b in [self.btn_t, self.btn_p, self.btn_m]:
            b.setStyleSheet("""
                border:none;
                background:transparent;
                color:#e0e0e0;
                font-size:14px;
            """)

        row2.addWidget(self.btn_t)
        row2.addWidget(self.btn_p)
        row2.addWidget(self.btn_m)

        control_layout.addLayout(row2)


        # ---------- 第三排（参数区 panel） ----------
        self.param_panel = QWidget()
        self.param_panel.setMaximumHeight(0)

        panel_layout = QStackedLayout()
        panel_layout.setContentsMargins(0,0,0,0)

        # ===== T 参数 =====
        t_widget = QWidget()
        t_layout = QHBoxLayout()
        t_layout.setContentsMargins(0,0,0,0)

        t_minus = QPushButton("-")
        t_val = QLabel("0.75")
        t_plus = QPushButton("+")

        for w in [t_minus, t_plus]:
            w.setStyleSheet("border:none;background:transparent;")

        t_val.setStyleSheet("border:none;background:transparent;color:#e0e0e0")

        t_layout.addWidget(t_val)
        t_layout.addWidget(t_minus)
        t_layout.addWidget(t_plus)

        t_widget.setLayout(t_layout)


        # ===== P 参数 =====
        p_widget = QWidget()
        p_layout = QHBoxLayout()
        p_layout.setContentsMargins(0,0,0,0)

        p_minus = QPushButton("-")
        p_val = QLabel("0.90")
        p_plus = QPushButton("+")

        for w in [p_minus, p_plus]:
            w.setStyleSheet("border:none;background:transparent;")

        p_val.setStyleSheet("border:none;background:transparent;color:#e0e0e0")

        p_layout.addWidget(p_val)
        p_layout.addWidget(p_minus)
        p_layout.addWidget(p_plus)

        p_widget.setLayout(p_layout)


        # ===== M 参数 =====
        m_widget = QWidget()
        m_layout = QHBoxLayout()
        m_layout.setContentsMargins(0,0,0,0)

        m_input = QLineEdit("256")
        m_input.setStyleSheet("border:none;background:transparent;color:#e0e0e0;")

        m_layout.addWidget(m_input)
        m_widget.setLayout(m_layout)


        panel_layout.addWidget(t_widget)
        panel_layout.addWidget(p_widget)
        panel_layout.addWidget(m_widget)

        self.param_panel.setLayout(panel_layout)
        self.param_stack = panel_layout

        control_layout.addWidget(self.param_panel)

        self.control_frame.setLayout(control_layout)
        self.control_frame.setFixedWidth(240)
        self.control_frame.setObjectName("no_frame")

        # 右侧输入区
        self.input_frame = QFrame()
        input_layout = QHBoxLayout()
        input_layout.setContentsMargins(10, 10, 10, 10)
        input_layout.setSpacing(10)

        self.input_box = QLineEdit()
        self.input_box.setPlaceholderText("say something...")

        self.send_btn = QPushButton("⁜")
        self.send_btn.setFixedHeight(40)

        self.send_btn.clicked.connect(self.send_message)
        self.input_box.returnPressed.connect(self.send_message)

        input_layout.addWidget(self.input_box)
        input_layout.addWidget(self.send_btn)

        self.input_frame.setLayout(input_layout)
        self.input_frame.setFixedHeight(70)

        bottom_layout.addWidget(self.control_frame)
        bottom_layout.addWidget(self.input_frame)

        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

        self.update_token_status()
        self.update_bar_status()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_bar_status)
        self.timer.start(100)

        self.btn_t.clicked.connect(lambda: self.toggle_param(0))
        self.btn_p.clicked.connect(lambda: self.toggle_param(1))
        self.btn_m.clicked.connect(lambda: self.toggle_param(2))

    def send_message(self):
        text = self.input_box.text().strip()
        if not text:
            return

        self.input_box.clear()

        cursor = self.chat_area.textCursor()
        cursor.movePosition(QTextCursor.End)

        block_format = QTextBlockFormat()
        block_format.setAlignment(Qt.AlignRight)
        cursor.insertBlock(block_format)

        char_format_user = QTextCharFormat()
        char_format_user.setForeground(QColor("#9a9a9a"))

        cursor.insertText(text + " ◅", char_format_user)

        block_format2 = QTextBlockFormat()
        block_format2.setAlignment(Qt.AlignLeft)
        cursor.insertBlock(block_format2)

        char_format_assistant = QTextCharFormat()
        char_format_assistant.setForeground(QColor("#ffffff"))

        cursor.insertText("\n▻ ", char_format_assistant)

        self.chat_area.setTextCursor(cursor)
        self.chat_area.ensureCursorVisible()

        self.thread = ModelThread(text, self.history)
        self.thread.token_signal.connect(self.append_token)
        self.thread.finished.connect(self.close_assistant_block)
        self.thread.start()

    def append_token(self, token):
        cursor = self.chat_area.textCursor()
        cursor.movePosition(QTextCursor.End)

        char_format = QTextCharFormat()
        char_format.setForeground(QColor("#ffffff"))

        cursor.insertText(token, char_format)

        self.chat_area.setTextCursor(cursor)
        self.chat_area.ensureCursorVisible()

        self.update_token_status()

    def close_assistant_block(self):
        cursor = self.chat_area.textCursor()
        cursor.movePosition(QTextCursor.End)

        char_format = QTextCharFormat()
        char_format.setForeground(QColor("#ffffff"))

        cursor.insertText("\n", char_format)

        self.chat_area.setTextCursor(cursor)
        self.chat_area.ensureCursorVisible()

    def update_token_status(self):
        stats = runtime

        tok_s = stats["tok_s"]
        tokens = stats["tokens"]
        gen = stats["gen_tokens"]
        total = stats["total_tokens"]
        device = stats["device"]

        self.status_label.setText(
            f"{tok_s:.1f} tok/s | ctx {tokens} | total {total} | gen {gen} | {device} |"
        )

    def update_bar_status(self):
        stats = runtime_stats()

        gpu = stats["gpu"]

        def make_gpu_bar(x):
            n = int(x * 20)
            return "█" * n + "░" * (20 - n)

        gpu_bar = make_gpu_bar(gpu)

        gpu_pct = int(gpu * 100)

        self.bar_label.setText(
            f"GPU {gpu_bar} {gpu_pct:>3}% "
        )

    def toggle_param(self, index):
        start = self.param_panel.maximumHeight()

        if start == 0:
            end = 30
        else:
            end = 0

        self.param_stack.setCurrentIndex(index)

        anim = QPropertyAnimation(self.param_panel, b"maximumHeight")
        anim.setDuration(180)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.OutCubic)

        anim.start()
        self.anim = anim

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
