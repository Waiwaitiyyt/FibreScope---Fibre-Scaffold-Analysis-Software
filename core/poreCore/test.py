import os
from unittest import IsolatedAsyncioTestCase
import cv2
import numpy as np
import pickle
from skimage.io import imread
from skimage.morphology import remove_small_holes
from skimage.measure import label
import torch
from model import PointNetLike
from scipy.interpolate import interp1d
from predict import resample_contour, normalize_contour, compute_scalars, predict_single_contour, getContourList
from extract_to_data import getContour
import time
import matplotlib.pyplot as plt
from label_data import contourIsolate
import poresMeasure



def getContour(image_path:str) -> list:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 画边框使边缘孔洞闭合
    border_color = 255   
    border_thickness = 1         
    h, w = img.shape[:2]
    cv2.line(img, (0, 0), (w, 0), border_color, border_thickness)
    cv2.line(img, (0, h-1), (w, h-1), border_color, border_thickness)
    cv2.line(img, (0, 0), (0, h), border_color, border_thickness)
    cv2.line(img, (w-1, 0), (w-1, h), border_color, border_thickness)
    # 从图像中提取所有轮廓，初步提取
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 25, 50)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contourImg = np.zeros_like(img)
    cv2.drawContours(contourImg, contours, -1, 255, 3)
    labeled_mask = label(contourImg > 0, connectivity=1)
    # 二次提取轮廓，使用skimage中的remove_small_holes()函数去除小洞进行降噪处理，二次提取的轮廓将用于训练
    cleaned = remove_small_holes(labeled_mask, 50)
    binary_clean = (cleaned > 0).astype(np.uint8) * 255 
    refinedEdges = cv2.Canny(binary_clean, 50, 100)
    refinedContours, _ = cv2.findContours(refinedEdges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return refinedContours

# imgPath = "images\\6.JPG"
# img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
# refinedContours = getContour(imgPath)
# mask = np.zeros_like(img)
# cv2.drawContours(mask, refinedContours, -1, 255, 1)
# cv2.imshow("contours", mask)
# # cv2.imwrite("extractedContours.jpg", mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

imgPath = "images\\6.JPG"
poresMeasure.setup()
areas, resultImg = poresMeasure.measure(imgPath)
poresMeasure.resultAnalyse(areas)
cv2.imshow("result", resultImg)
cv2.imwrite("pore Result.jpg", resultImg)
cv2.waitKey(0)
