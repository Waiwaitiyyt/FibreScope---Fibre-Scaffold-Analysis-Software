import cv2
import numpy as np
from skimage.morphology import skeletonize
from typing import Tuple
from scipy.ndimage import distance_transform_edt
from scipy.stats import sem, gaussian_kde
from core.fibreCore.predict import predict_single_contour
import torch
from core.fibreCore.model import PointNetLike
from core.fibreCore.contourPrefix import morphFix
from typing import Tuple
import json

class FibreModel:
    def __init__(self, modelPath, num_points = 64):
        print("Loading Fibre Model...")
        self.device = torch.device("cpu")
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
    def process(self, contours: list, cleanImg: np.ndarray) -> np.ndarray:
        for i, contour in enumerate(contours):
            contour = contour.squeeze(1)
            cls, conf = predict_single_contour(self.model, contour, self.device, self.num_points)
            if cls == 1:
                cv2.drawContours(cleanImg, [contour], -1, 255, -1)
        return cleanImg


def _distanceMeasure(binary_mask: np.ndarray,
                     scaleFactor: float,
                     jer: float,
                     min_area=50,
                     width_jump_ratio=1.5,
                     local_window=10,
                     max_radius_factor=1.4,
                     ) -> Tuple[list, np.ndarray, list]:
    """
    width_jump_ratio : reject if local width deviates too much from neighbors
    local_window     : neighborhood size along skeleton
    max_radius_factor: reject points that are too thick relative to median
    """

    fg = (binary_mask > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    large_mask = np.zeros_like(fg)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            large_mask[labels == i] = 1
    if large_mask.sum() == 0:
        return [], None, []

    dt = distance_transform_edt(large_mask)
    skel_bool = skeletonize(large_mask > 0)
    skelImg = (skel_bool.astype(np.uint8) * 255)


    # --- Neighbor count ---
    padded = np.pad(skel_bool.astype(np.uint8), 1)
    neighbor_counts = np.zeros_like(skel_bool, dtype=np.int32)
    offsets = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for dy, dx in offsets:
        neighbor_counts += padded[1+dy:1+dy+skel_bool.shape[0],
                                  1+dx:1+dx+skel_bool.shape[1]]

    junction_mask = skel_bool & (neighbor_counts >= 3)

    # --- Junction exclusion zone ---
    if junction_mask.any():
        junc_map = np.zeros_like(skel_bool, dtype=np.uint8)
        junc_map[junction_mask] = 1
        dist_to_junc = distance_transform_edt(1 - junc_map)
        junction_exclusion_radius = jer  # tune
        near_junction = dist_to_junc <= junction_exclusion_radius
    else:
        near_junction = np.zeros_like(skel_bool, dtype=bool)

    keep_mask = skel_bool & (neighbor_counts == 2) & (~near_junction)

    coords = np.argwhere(keep_mask)
    if coords.size == 0:
        return [], skelImg

    widths = 2.0 * dt[keep_mask]

    # --- Global thickness sanity check ---
    median_width = np.median(widths)
    valid = widths < (max_radius_factor * median_width)
    coords = coords[valid]
    widths = widths[valid]
    if len(widths) < local_window:
        return [], skelImg

    order = np.lexsort((coords[:,1], coords[:,0]))
    widths = widths[order]
    good = np.ones(len(widths), dtype=bool)
    half = local_window // 2

    for i in range(half, len(widths) - half):
        local = widths[i-half:i+half+1]
        local_med = np.median(local)
        if widths[i] > width_jump_ratio * local_med:
            good[i] = False
        if widths[i] < local_med / (2 * width_jump_ratio):
            good[i] = False

    widths = widths[good]
    if len(widths) > 2000:
        idx = np.random.choice(len(widths), 2000, replace=False)
        widths = widths[idx]
        coords = coords[idx]

    widths *= scaleFactor
    return widths.tolist(), skelImg, coords.tolist()

def contourIsolate(contour: np.ndarray) -> np.ndarray:
    pts = contour.reshape(-1, 2)           
    x0, y0 = pts.min(axis=0).astype(int)
    x1, y1 = pts.max(axis=0).astype(int)
    h, w = (y1 - y0 + 50), (x1 - x0 + 50)
    mask = np.zeros((h, w), dtype=np.uint8)
    normalised = (25 + pts - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [normalised], -1, 255, 1)
    return mask


def _imgProcess(imgPath: str) -> Tuple[np.ndarray, list]:
    grayImg = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
    binary = cv2.threshold(cv2.bitwise_not(grayImg), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernelSize = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelSize, kernelSize))
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    # Remove small objects
    mask = np.zeros_like(closing)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closing, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 200:
            mask[labels == i] = 255
    # Remove small holse
    invImg = cv2.bitwise_not(mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(invImg, connectivity=8)
    filterImg = np.zeros_like(invImg)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > 200:
            filterImg[labels == i] = 255

    cleanImg = cv2.bitwise_not(filterImg)

    # imgCopy = filterImg.copy()
    # border_color = 255 
    # border_thickness = 1 
    # h, w = imgCopy.shape[:2]
    # cv2.rectangle(imgCopy, (0, 0), (w - border_thickness, h - border_thickness), border_color, border_thickness)
    # contours, _ = cv2.findContours(imgCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # for contour in contours[:-1]:
    #     contourMask = contourIsolate(contour)
    #     widths, skelImg, _ = _distanceMeasure(contourMask, 1, 5, min_area=5, local_window=2)
    #     if (0 < np.std(widths) < 10):
    #         cv2.drawContours(cleanImg, [contour], -1, (255, 255, 255), -1)

    # Re-extract from new img
    imgCopy = cleanImg.copy()
    border_color = 255   
    border_thickness = 1 
    h, w = imgCopy.shape[:2]
    cv2.rectangle(imgCopy, (0, 0), (w - border_thickness, h - border_thickness), border_color, border_thickness)
    contours, _ = cv2.findContours(imgCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return cleanImg, contours


def measure(imgPath: str, model: FibreModel) -> Tuple[list, np.ndarray, np.ndarray]:
    cleanImg, contours = _imgProcess(imgPath)
    cleanImg = model.process(contours, cleanImg)
    with open("data.json", "r") as jsonFile:
        data = json.load(jsonFile)
    scaleFactor = data["scaleFactor"]
    jer = data["JER"]
    diameterList, skeleton, sampleCoordsList = _distanceMeasure(cleanImg, scaleFactor, jer)
    resultAnalyse(diameterList)
    measuredImg = cv2.cvtColor(cleanImg, cv2.COLOR_GRAY2BGR)
    skeletonIndice = np.where(skeleton == 255)
    measuredImg[skeletonIndice] = (0, 0, 255)
    for coord in sampleCoordsList:
        cv2.circle(measuredImg, (coord[1], coord[0]), 3, (255, 0, 0), -1)
    return diameterList, measuredImg, cleanImg

def resultAnalyse(diameterList: list):
    diameterArr = np.asarray(diameterList, dtype = np.float64)
    average = np.mean(diameterArr)
    stdev = np.std(diameterArr, ddof = 1)

    kde = gaussian_kde(diameterArr)
    x = np.linspace(diameterArr.min(), diameterArr.max(), 1000)
    y = kde(x)
    fibrePeak = x[np.argmax(y)]

    semValue = sem(diameterArr, ddof = 1)
    median = np.median(diameterArr)
    quantileLow, quantileHigh = np.quantile(diameterArr, 0).astype(float), np.quantile(diameterArr, 1).astype(float)
    boot = np.random.choice(diameterArr, (10000, len(diameterArr)), replace=True)
    ciLow, ciHigh = np.percentile(np.median(boot, axis=1), [2.5, 97.5])

    with open("data.json", "r") as jsonFile:
        data = json.load(jsonFile)

    fibreDict = {"Average": average,
                 "Standard Deviation": stdev,
                 "KDE Peak": fibrePeak,
                 "SEM": semValue,
                 "median": median,
                 "Q1, Q3": (quantileLow, quantileHigh),
                 "IQR": quantileHigh - quantileLow,
                 "95% CI": (ciLow, ciHigh),
                 "Raw": diameterList
                 }

    data["Fibre Param"] = fibreDict

    with open("data.json", "w") as jsonFile:
        json.dump(data, jsonFile, indent = 2)
     

if __name__ == "__main__":
    imgPath = r"demo/2.JPG"
    fibreModel = FibreModel(r"core/fibreCore/fibreModel.pth")
    diameterList, measuredImg, cleanImg = measure(imgPath, fibreModel)
    print(np.mean(diameterList))
    cv2.imshow("a", measuredImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    
