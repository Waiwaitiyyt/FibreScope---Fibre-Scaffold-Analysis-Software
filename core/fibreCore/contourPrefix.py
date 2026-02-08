import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt
from typing import Tuple

def contourIsolate(contour: np.ndarray) -> np.ndarray:
    pts = contour.reshape(-1, 2)           
    x0, y0 = pts.min(axis=0).astype(int)
    x1, y1 = pts.max(axis=0).astype(int)
    h, w = (y1 - y0 + 50), (x1 - x0 + 50)
    mask = np.zeros((h, w), dtype=np.uint8)
    normalised = (25 + pts - [x0, y0]).astype(np.int32).reshape(-1, 1, 2)
    cv2.drawContours(mask, [normalised], -1, 128, -1)
    return mask

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
    
def morphFix(cleanImg):
    imgCopy = cleanImg.copy()
    border_color = (255, 255, 255)   
    border_thickness = 1 
    h, w = imgCopy.shape[:2]
    cv2.rectangle(imgCopy, (0, 0), (w - border_thickness, h - border_thickness), border_color, border_thickness)
    contours, _ = cv2.findContours(imgCopy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for contour in contours[:-1]:
        contourMask = contourIsolate(contour)
        widths = _distanceMeasure(contourMask, scaleFactor=1, jer=5,min_area=5, local_window=2)[0]
        if (0 < np.std(widths) < 10):
            cv2.drawContours(cleanImg, [contour], -1, (255, 255, 255), -1)
    return cleanImg


