import os
import cv2
import numpy as np
import pickle
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from typing import Tuple

IMAGE_DIR = "images"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def imgProcess(imgPath: str) -> np.ndarray:
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
    # Here is a re-process for the binary image
    # Contours are extarcted to distinguish the unwanted hole and remove them
    cleanBinaryImg = cleanImg.copy()
    border_color = 255   
    border_thickness = 1         
    h, w = cleanImg.shape[:2]
    cv2.line(cleanImg, (0, 0), (w, 0), border_color, border_thickness)
    cv2.line(cleanImg, (0, h-1), (w, h-1), border_color, border_thickness)
    cv2.line(cleanImg, (0, 0), (0, h), border_color, border_thickness)
    cv2.line(cleanImg, (w-1, 0), (w-1, h), border_color, border_thickness)
    # contours, _ = cv2.findContours(cleanImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    return cleanBinaryImg 

def _distanceMeasure(binary_mask: np.ndarray,
                     min_area=5,
                     width_jump_ratio=1.5,
                     local_window=2,
                     max_radius_factor=1.4) -> Tuple[list, np.ndarray, list]:
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
        junction_exclusion_radius = 5  # tune
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
        if widths[i] < local_med / width_jump_ratio:
            good[i] = False

    widths = widths[good]
    if len(widths) > 100:
        idx = np.random.choice(len(widths), 100, replace=False)
        widths = widths[idx]
        coords = coords[idx]

    return widths.tolist(), skelImg, coords.tolist()

def contourIsolate(contour: np.ndarray) -> np.ndarray:
    pts = contour.reshape(-1, 2)           
    x0, y0 = pts.min(axis=0).astype(int)
    x1, y1 = pts.max(axis=0).astype(int)
    h, w = (y1 - y0 + 50), (x1 - x0 + 50)
    mask = np.zeros((h, w), dtype=np.uint8)
    normalised = (25 + pts - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [normalised], -1, 128, -1)
    return mask


if __name__ == "__main__":
    all_items = []
    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = imgProcess(img_path)
        if img is None:
            continue

        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for cnt in contours:
            contourMask = contourIsolate(cnt)
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            equiv_diam = np.average(_distanceMeasure(contourMask)[0])
            item = {
                'contour': cnt.reshape(-1, 2).astype(np.float32),
                'area': float(area),
                'perimeter': float(perimeter),
                'equiv_diam': float(equiv_diam),
                'label': -1  # Label to be -1 as default, will be re-labelled in label_data.py c
            }
            all_items.append(item)
    # save as .pkl
    output_file = os.path.join(OUTPUT_DIR, "raw_contours.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(all_items, f)
    print(f"Extracted {len(all_items)} contours, saved to {output_file}")