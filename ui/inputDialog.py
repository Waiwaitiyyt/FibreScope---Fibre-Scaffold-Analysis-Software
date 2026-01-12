from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
    QDialogButtonBox, QCheckBox, QTextEdit
)

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

class AcceptValue(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Notice")
        self.setFixedSize(200, 50)
        layout = QVBoxLayout()
        label = QLabel("Scale Factor Changed!")
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

