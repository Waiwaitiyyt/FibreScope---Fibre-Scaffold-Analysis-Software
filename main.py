import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QDialog, QWidget, QVBoxLayout, QTableWidgetItem, QSplashScreen
from PySide6.QtCore import Qt, QSize, QThread, Signal
from PySide6.QtGui import QIcon, QPixmap, QMovie
from ui.mainWindow import Ui_MainWindow   
from ui.subUi import FactorDialog, AcceptValue, RejectValue, NoModelWarning, NoImgWarning, ProcessingDialog, JERDialog
from ui.about import AboutDialog
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import core.io as io
from core.fibreCore import fibreMeasure
from core.poreCore import poresMeasure
from core.render import fibreResultVisualise, poreResultVisualise
import json
import cv2
from datetime import datetime


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowIcon(QIcon("icon.ico"))
        self._connectSignals()
        self._canvaSet()

        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)
        self.mode = data["mode"]
        
        if self.mode == "f":
            self.setWindowTitle("FibreScope - fibre mode")
            try:
                self.modelPath = data["fibreModel"]
                self.model = fibreMeasure.FibreModel(self.modelPath)
            except:
                modelWarning = NoModelWarning(parent = self)
                modelWarning.exec()
        if self.mode == "p":
            self.setWindowTitle("FibreScope - pore mode")
            try:
                self.modelPath = data["poreModel"]
                self.model = poresMeasure.PoreModel(self.modelPath)
            except:
                modelWarning = NoModelWarning(parent = self)
                modelWarning.exec()

        data["imgPath"] = ""
        with open("data.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent = 2)

    def _connectSignals(self):
        self.ui.actionOpen_Image.triggered.connect(io.selectImg)
        self.ui.actionExit.triggered.connect(self._closeWindow)
        self.ui.actionChange_Fibre_Model.triggered.connect(io.changeFibreModel)
        self.ui.actionChange_Pore_Model.triggered.connect(io.changePoreModel)
        self.ui.actionChange_Scale_Factor.triggered.connect(self._factorDialog)
        self.ui.actionChange_JER.triggered.connect(self._jerDialog)
        self.ui.actionFibre_Measure.triggered.connect(self._toggleFibreMode)
        self.ui.actionPore_Measure.triggered.connect(self._togglePoreMode)
        self.ui.actionRun_Analysis.triggered.connect(self._startAnalysis)
        self.ui.actionSave_Result.triggered.connect(self._saveResult)
        self.ui.actionAbout.triggered.connect(self._about)

    def _factorDialog(self):
        dialog = FactorDialog(parent = self)
        result = dialog.exec()
        if result == QDialog.Accepted:
            inputData = dialog.getResult()
            newFactor = inputData["scaleFactor"]
            try:
                newFactor = float(newFactor)
                accept = AcceptValue(parent = self)
                accept.exec()
                with open("data.json", "r") as jsonFile:
                    data = json.load(jsonFile)
                data["scaleFactor"] = newFactor
                with open("data.json", "w") as jsonFile:
                    json.dump(data, jsonFile, indent = 2)

            except ValueError:
                reject = RejectValue(parent = self)
                reject.exec()

    def _jerDialog(self):
        dialog = JERDialog(parent = self)
        result = dialog.exec()
        if result == QDialog.Accepted:
            inputData = dialog.getResult()
            newFactor = inputData["JER"]
            try:
                newFactor = float(newFactor)
                accept = AcceptValue(parent = self)
                accept.exec()
                with open("data.json", "r") as jsonFile:
                    data = json.load(jsonFile)
                data["scaleFactor"] = newFactor
                with open("data.json", "w") as jsonFile:
                    json.dump(data, jsonFile, indent = 2)

            except ValueError:
                reject = RejectValue(parent = self)
                reject.exec()

    def _toggleFibreMode(self):
        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)
        data["mode"] = "f"
        with open("data.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent = 2)
        self.modelPath = data["fibreModel"]
        self.model = fibreMeasure.FibreModel(self.modelPath)
        self.setWindowTitle("FibreScope - fibre mode")

    def _togglePoreMode(self):
        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)
        data["mode"] = "p"
        with open("data.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent = 2)
        self.modelPath = data["poreModel"]
        self.model = poresMeasure.PoreModel(self.modelPath)
        self.setWindowTitle("FibreScope - pore mode")

    def _canvaSet(self):
        self.fig = Figure() 
        self.canvas = FigureCanvas(self.fig)
        layout = self.ui.resultFrame.layout()
        if layout is None:
            layout = QVBoxLayout(self.ui.resultFrame)
            self.ui.resultFrame.setLayout(layout)
        layout.addWidget(self.canvas)

    def _startAnalysis(self):
        self.processing = ProcessingDialog(self)
        self.processing.show()
        QApplication.processEvents()
        self.worker = AnalysisWorker(self)
        self.worker.resultReady.connect(self._renderResults)
        self.worker.finished.connect(self._finishAnalysis)
        self.worker.start()

    def _finishAnalysis(self):
        self.processing.close()
        self.worker.deleteLater()

    def _computeAnalysis(self):
        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)
        imgPath = data["imgPath"]
        mode = data["mode"]
        self.fig.clear()
        if imgPath != "":
            if mode == "f": 
                return fibreMeasure.measure(imgPath, self.model)
            if mode == "p":
                return poresMeasure.measure(imgPath, self.model)
        else:
            imgWarning = NoImgWarning(parent = self)
            imgWarning.exec()

    def _renderResults(self, result):
        if result is None:
            NoImgWarning(parent=self).exec()
            return
        self.fig.clear()
        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)
        if data["mode"] == "f":
            diameterList, measuredImg, cleanImg = result
            fibreResultVisualise(
                diameterList,
                measuredImg,
                cleanImg,
                originalImg=cv2.imread(data["imgPath"]),
                fig=self.fig
            )
        else:
            areaList, mask_color, refinedContours = result
            poreResultVisualise(
                areaList,
                mask_color,
                refinedContours,
                originalImg=cv2.imread(data["imgPath"]),
                fig=self.fig
            )
        self.canvas.draw()        
        self._showResult()        

    def _showResult(self):
        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)
        if data["mode"] == "f":
            modeSetting = "Fibre Param"
        elif data["mode"] == "p":
            modeSetting = "Pores Param"

        average = QTableWidgetItem(str(round(data[modeSetting]["Average"], 4)))
        stdev = QTableWidgetItem(str(round(data[modeSetting]["Standard Deviation"], 4)))
        var = QTableWidgetItem(str(round(data[modeSetting]["KDE Peak"], 4)))
        sem = QTableWidgetItem(str(round(data[modeSetting]["SEM"], 4)))
        median = QTableWidgetItem(str(round(data[modeSetting]["median"], 4)))
        q1, q3 = round(data[modeSetting]["Q1, Q3"][0], 4), round(data[modeSetting]["Q1, Q3"][1], 4)
        q1q3 = QTableWidgetItem(f"{q1}, {q3}")
        iqr = QTableWidgetItem(str(round(data[modeSetting]["IQR"], 4)))
        ciLow, ciHigh = round(data[modeSetting]["95% CI"][0], 4), round(data[modeSetting]["95% CI"][1], 4)
        ci95 = QTableWidgetItem(f"{ciLow}, {ciHigh}")
        jerAndFactor = round(data["JER"]), data["scaleFactor"]
        kernel = QTableWidgetItem(f"{jerAndFactor}")

        average.setTextAlignment(Qt.AlignCenter)
        stdev.setTextAlignment(Qt.AlignCenter)
        var.setTextAlignment(Qt.AlignCenter)
        sem.setTextAlignment(Qt.AlignCenter)
        median.setTextAlignment(Qt.AlignCenter)
        q1q3.setTextAlignment(Qt.AlignCenter)
        iqr.setTextAlignment(Qt.AlignCenter)
        ci95.setTextAlignment(Qt.AlignCenter)
        kernel.setTextAlignment(Qt.AlignCenter)

        self.ui.tableWidget.setItem(0, 0, average)
        self.ui.tableWidget.setItem(0, 1, stdev)
        self.ui.tableWidget.setItem(0, 2, var)
        self.ui.tableWidget.setItem(0, 3, sem)
        self.ui.tableWidget.setItem(0, 4, median)
        self.ui.tableWidget.setItem(0, 5, q1q3)
        self.ui.tableWidget.setItem(0, 6, iqr)
        self.ui.tableWidget.setItem(0, 7, ci95)
        self.ui.tableWidget.setItem(0, 8, kernel)

    def _saveResult(self):
        now = datetime.now()
        year = now.year
        month = now.month
        day = now.day
        hour = now.hour
        minute = now.minute
        second = now.second
        self.fig.savefig(f"{year}_{month}_{day}_{hour}_{minute}_{second}.png")
        print("Result saved!")

    def _about(self):
        about = AboutDialog(parent = self)
        about.exec()

    def _closeWindow(self):
        QApplication.quit()
        with open("data.json", "r") as jsonFile:
            data = json.load(jsonFile)
        data["imgPath"] = ""
        with open("data.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent = 2)

class AnalysisWorker(QThread):
    resultReady  = Signal(object)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

    def run(self):
        results = self.main_window._computeAnalysis()
        self.resultReady.emit(results)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    splash = QSplashScreen(
        QPixmap("media/Splash2.png"),
        Qt.WindowStaysOnTopHint
    )
    splash.show()
    app.processEvents()
    window = MainWindow()
    window.show()
    splash.finish(window)
    sys.exit(app.exec())
