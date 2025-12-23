import numpy as np
import cv2

# ---------------------------------------------------------
# Convert a single contour to a binary mask, formatting
# ---------------------------------------------------------
def contour_to_mask(contour):
    """
    contour: numpy array of shape (N,1,2) from cv2.findContours()
    shape: (H, W)
    """

    x0, y0 = contour.min(axis = 0).astype(int)
    x1, y1 = contour.max(axis = 0).astype(int)
    h, w = y1 - y0 + 50, x1 - x0 + 50
    mask = np.zeros((h, w)).astype(np.int8)
    normalisedContour = (contour - [x0, y0]).astype(np.int32)

    cv2.drawContours(mask, [normalisedContour], -1, 1, thickness=1)
    return mask


# The idea here is to fill the contour point-by-point from both column and row direction
# And check if the area obtained is similar, if so, take the average to be the "true" area, else discard this.


def filled_area_by_columns(mask):
    h, w = mask.shape
    filled = np.zeros_like(mask)
    for x in range(w):
        ys = np.where(mask[:, x] == 1)[0]
        if len(ys) >= 2:
            filled[ys.min():ys.max() + 1, x] = 1
    return filled.sum(), filled

def filled_area_by_rows(mask):
    h, w = mask.shape
    filled = np.zeros_like(mask)
    for y in range(h):
        xs = np.where(mask[y, :] == 1)[0]
        if len(xs) >= 2:
            filled[y, xs.min():xs.max() + 1] = 1
    return filled.sum(), filled


def contourArea(contour, threshold=0.1):
    mask = contour_to_mask(contour)
    area_col, filled_col = filled_area_by_columns(mask)
    area_row, filled_row = filled_area_by_rows(mask)
    mean_area = 0.5 * (area_col + area_row)
    if mean_area == 0:
        return 0, False, filled_col, filled_row
    relative_error = abs(area_col - area_row) / mean_area
    is_closed = relative_error <= threshold
    return mean_area, is_closed, filled_col, filled_row


if __name__ == "__main__":

    img = cv2.imread("a.png", cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 30, 100)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)

    for i, cnt in enumerate(contours):
        area, closed, filled_col, filled_row = contourArea(
            cnt,
            threshold=0.1
        )

        print(f"Contour {i}: area = {area}, closed = {closed}")

        # Optional: visualize
        cv2.imshow("Filled by Columns", filled_col * 255)
        cv2.imshow("Filled by Rows", filled_row * 255)
        cv2.waitKey(0)

    cv2.destroyAllWindows()
