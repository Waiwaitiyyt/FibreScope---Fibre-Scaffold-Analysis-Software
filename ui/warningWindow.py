from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QDialogButtonBox, QCheckBox, QTextEdit
)
from PySide6.QtCore import Qt


class NoModelWarning(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Warning")
        self.setFixedSize(200, 50)
        layout = QVBoxLayout()
        label = QLabel("Invalid Model!")
        layout.addWidget(label)
        label.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)

class NoImgWarning(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Warning")
        self.setFixedSize(200, 50)
        layout = QVBoxLayout()
        label = QLabel("Invalid Image!")
        layout.addWidget(label)
        label.setAlignment(Qt.AlignCenter)
        self.setLayout(layout)