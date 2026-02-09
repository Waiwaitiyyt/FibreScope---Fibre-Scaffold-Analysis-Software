from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QDialogButtonBox, QCheckBox, QTextEdit
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QMovie


class FactorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Change Scale Factor")
        # self.setFixedSize(300, 200)
        self.factor_edit = QLineEdit(self)
        self.factor_edit.setPlaceholderText("e.g., 1.25")
        self.setDefaultCheckbox = QCheckBox("Set as default?", self)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        form_layout = QFormLayout()
        form_layout.addRow(QLabel("Scale Factor:"), self.factor_edit)
        form_layout.addRow(self.setDefaultCheckbox) 
        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)
        self.setLayout(main_layout)

    def getResult(self):
        result = {"scaleFactor": self.factor_edit.text(), "setDefault": self.setDefaultCheckbox.isChecked()}
        return result

class ProcessingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowFlags(
            Qt.Dialog |
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint
        )
        self.setModal(True)
        self.setFixedSize(185, 60)

        spinner = QLabel(alignment=Qt.AlignCenter)
        movie = QMovie("media/spinner.gif")
        movie.setScaledSize(QSize(50, 28))
        spinner.setMovie(movie)
        movie.start()

        text = QLabel("Processing…")
        text.setAlignment(Qt.AlignVCenter)

        row = QHBoxLayout()
        row.addWidget(spinner)
        row.addWidget(text)

        layout = QVBoxLayout(self)
        layout.addLayout(row)

        self._center_on_parent()

    def _center_on_parent(self):
        if self.parent():
            self.move(
                self.parent().geometry().center()
                - self.rect().center()
            )



class AcceptValue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Notice")
        self.setFixedSize(200, 50)
        layout = QVBoxLayout()
        label = QLabel("Change Made!")
        layout.addWidget(label)
        self.setLayout(layout)

class RejectValue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Warning")
        self.setFixedSize(200, 50)
        layout = QVBoxLayout()
        label = QLabel("Invalid Input!")
        layout.addWidget(label)
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

class JERDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Change Junction Exclusion Radius (JER)")
        # self.setFixedSize(300, 200)

        self.factor_edit = QLineEdit(self)
        self.factor_edit.setPlaceholderText("e.g., 40")

        self.setDefaultCheckbox = QCheckBox("Set as default?", self)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            parent=self
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        form_layout = QFormLayout()
        form_layout.addRow(QLabel("JER:"), self.factor_edit)
        form_layout.addRow(self.setDefaultCheckbox) 

        main_layout = QVBoxLayout()
        main_layout.addLayout(form_layout)
        main_layout.addWidget(self.button_box)
        
        self.setLayout(main_layout)

    def getResult(self):
        result = {"JER": self.factor_edit.text(), "setDefault": self.setDefaultCheckbox.isChecked()}
        return result