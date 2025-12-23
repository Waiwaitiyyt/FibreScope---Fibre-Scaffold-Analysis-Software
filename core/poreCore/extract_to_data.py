import os
import cv2
import numpy as np
import pickle
from skimage.morphology import remove_small_holes, label

IMAGE_DIR = "images"
OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def getContour(image_path:str) -> list:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Draw border for the img to enclose contours near the edges
    border_color = 255   
    border_thickness = 1         
    h, w = img.shape[:2]
    cv2.line(img, (0, 0), (w, 0), border_color, border_thickness)
    cv2.line(img, (0, h-1), (w, h-1), border_color, border_thickness)
    cv2.line(img, (0, 0), (0, h), border_color, border_thickness)
    cv2.line(img, (w-1, 0), (w-1, h), border_color, border_thickness)
    # Obtain all contours from the img, the preliminary extraction
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contourImg = np.zeros_like(img)
    cv2.drawContours(contourImg, contours, -1, 255, 3)
    labeled_mask = label(contourImg > 0, connectivity=1)
    # Secondary extraction, try to de-noise via the remove_small_holes() function in skimage
    cleaned = remove_small_holes(labeled_mask, 50)
    binary_clean = (cleaned > 0).astype(np.uint8) * 255 
    refinedEdges = cv2.Canny(binary_clean, 50, 100)
    refinedContours, _ = cv2.findContours(refinedEdges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    return refinedContours

if __name__ == "__main__":
    all_items = []
    for img_name in os.listdir(IMAGE_DIR):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img_path = os.path.join(IMAGE_DIR, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        contours = getContour(img_path)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            equiv_diam = np.sqrt(4 * area / np.pi)
            item = {
                'contour': cnt.reshape(-1, 2).astype(np.float32),
                'area': float(area),
                'perimeter': float(perimeter),
                'equiv_diam': float(equiv_diam),
                'label': -1  # Label to be -1 as default, will be re-labelled in label_data.py 
            }
            all_items.append(item)
    # save as .pkl
    output_file = os.path.join(OUTPUT_DIR, "raw_contours.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(all_items, f)
    print(f"Extracted {len(all_items)} contours, saved to {output_file}")