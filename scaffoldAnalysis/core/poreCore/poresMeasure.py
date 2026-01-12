import torch
import numpy as np
import cv2
from core.poreCore.model import PointNetLike
from core.poreCore.predict import predict_single_contour
from core.poreCore.extract_to_data import getContour
from core.poreCore.area import contourArea
import matplotlib.pyplot as plt
from typing import Tuple
import json
from scipy.stats import sem

class PoreModel:
    def __init__(self, modelPath, num_points = 64):
        print("Loading Pore Model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_points = num_points
        self.model = PointNetLike(
            num_classes=2,
            num_points=num_points
        )
        self.model.load_state_dict(
            torch.load(modelPath, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"Model Loading Sucessful, device: {self.device}")

    @torch.no_grad()
    def process(self, contours_list: list) -> list:
        poreContours = []
        for contour in contours_list:
            contour = contour.squeeze(1)
            cls, conf = predict_single_contour(self.model, contour, self.device, self.num_points)
            if cls == 1:
                poreContours.append(contour)
        return poreContours


def measure(imgPath: str, model: PoreModel) -> Tuple[list, np.ndarray, list]:
    refinedContours = getContour(imgPath)
    poreContours = model.process(refinedContours)
    refinedContours_int32 = [cnt.astype(np.int32) for cnt in refinedContours]
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    mask_color = cv2.cvtColor(np.zeros_like(img), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_color, refinedContours_int32, -1, (128, 128, 128), 1) 
    cv2.drawContours(mask_color, poreContours, -1, (0, 0, 255), 2) 
    areaList = []
    with open("data.json", "r") as jsonFile:
        data = json.load(jsonFile)
    scaleFactor = data["scaleFactor"]

    for contour in poreContours:
        area, _, _, _ = contourArea(contour, threshold = 0.1)
        areaList.append(int(area) * scaleFactor ** 2)
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            centreGravity = (cx, cy)
        cv2.putText(mask_color, f"{int(area)}", centreGravity, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    resultAnalyse(areaList)
    return areaList, mask_color, refinedContours

def resultAnalyse(areaList: list):
    areaArr = np.asarray(areaList, dtype = np.float64)
    average = np.mean(areaArr)
    stdev = np.std(areaArr, ddof = 1)
    var = np.var(areaArr)
    semValue = sem(areaArr, ddof = 1)
    median = np.median(areaArr)
    quantileLow, quantileHigh = np.quantile(areaArr, 0).astype(float), np.quantile(areaArr, 1).astype(float)
    boot = np.random.choice(areaArr, (10000, len(areaArr)), replace=True)
    ciLow, ciHigh = np.percentile(np.median(boot, axis=1), [2.5, 97.5])

    with open("data.json", "r") as jsonFile:
        data = json.load(jsonFile)

    fibreDict = {"Average": average,
                 "Standard Deviation": stdev,
                 "Variance": var,
                 "SEM": semValue,
                 "median": median,
                 "Q1, Q3": (quantileLow, quantileHigh),
                 "IQR": quantileHigh - quantileLow,
                 "95% CI": (ciLow, ciHigh),
                 "Raw": areaList
                 }

    data["Pores Param"] = fibreDict
    with open("data.json", "w") as jsonFile:
        json.dump(data, jsonFile, indent = 2)

if __name__ == "__main__":
    pass
