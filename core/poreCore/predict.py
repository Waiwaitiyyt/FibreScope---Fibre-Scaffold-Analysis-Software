# predict.py
import torch
import numpy as np
import cv2
from core.poreCore.model import PointNetLike
from scipy.interpolate import interp1d
from skimage.morphology import remove_small_holes
from skimage.measure import label

# --- Helper Functions ---
def resample_contour(contour, num_points=64):
    n_points = len(contour)
    if n_points == 0:
        raise ValueError("Contour has zero points.")
    if n_points < num_points:
        padding_needed = num_points - n_points
        repeated_last_point = np.tile(contour[-1], (padding_needed, 1))
        contour = np.vstack([contour, repeated_last_point])
    elif n_points > num_points:
        old_indices = np.arange(n_points)
        new_indices = np.linspace(0, n_points - 1, num_points)
        fx = interp1d(old_indices, contour[:, 0], kind='linear')
        fy = interp1d(old_indices, contour[:, 1], kind='linear')
        contour = np.column_stack((fx(new_indices), fy(new_indices)))
    return contour

def normalize_contour(contour):
    center = contour.mean(axis=0)
    contour_centered = contour - center
    max_dist = np.max(np.linalg.norm(contour_centered, axis=1))
    if max_dist == 0: max_dist = 1e-8
    return contour_centered / max_dist

def compute_scalars(contour):
    cnt_int = contour.astype(np.int32).reshape(-1, 1, 2)
    area = cv2.contourArea(cnt_int)
    perimeter = cv2.arcLength(cnt_int, closed=True)
    min_coords = contour.min(axis=0)
    max_coords = contour.max(axis=0)
    diameter = np.linalg.norm(max_coords - min_coords)
    return np.array([area, perimeter, diameter], dtype=np.float32)


def predict_single_contour(model, contour, device, num_points=64):
    # Re-sample
    contour_rs = resample_contour(contour, num_points=num_points)
    
    # Normalisation
    contour_norm = normalize_contour(contour_rs)
    
    # Obtain scalars
    scalars_np = compute_scalars(contour_rs)
    
    # Tensor transformation
    points_tensor = torch.tensor(contour_norm, dtype=torch.float32).unsqueeze(0).to(device) # (1, N, 2)
    scalars_tensor = torch.tensor(scalars_np, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model(points_tensor, scalars_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item()
    
    return pred_class, confidence

# --- Main Prediction Logic ---
def extract_contours_from_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)
    contours, _ = cv2.findContours(edges, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    contour_points_list = [cnt.squeeze(1) for cnt in contours if len(cnt) > 5] 
    return contour_points_list

def getContourList(image_path:str) -> list:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    border_color = 255   
    border_thickness = 1         
    h, w = img.shape[:2]
    cv2.line(img, (0, 0), (w, 0), border_color, border_thickness)
    cv2.line(img, (0, h-1), (w, h-1), border_color, border_thickness)
    cv2.line(img, (0, 0), (0, h), border_color, border_thickness)
    cv2.line(img, (w-1, 0), (w-1, h), border_color, border_thickness)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img, 30, 100)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contourImg = np.zeros_like(img)
    cv2.drawContours(contourImg, contours, -1, 255, 2)
    labeled_mask = label(contourImg > 0, connectivity=1)
    cleaned = remove_small_holes(labeled_mask, 50)
    binary_clean = (cleaned > 0).astype(np.uint8) * 255 
    refinedEdges = cv2.Canny(binary_clean, 50, 100)
    refinedContours, _ = cv2.findContours(refinedEdges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contourList = [cnt.squeeze(1) for cnt in refinedContours]
    return contourList

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "pointnet_pore_classifier.pth" # Use the name from train.py
    NUM_POINTS = 64
    print("Loading model...")
    model = PointNetLike(num_classes=2, num_points=NUM_POINTS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")
    IMAGE_PATH = "images\\6.JPG" 
    try:
        contours_list = getContourList(IMAGE_PATH)
        print(f"Found {len(contours_list)} contours in the image.")
        # for i, cnt in enumerate(contours_list[:5]): # Predict first 5 for demo
        #     try:
        #         cls, conf = predict_single_contour(model, cnt, DEVICE, NUM_POINTS)
        #         print(f"Contour {i}: Predicted class={cls} ({'Pore' if cls==1 else 'Noise/Fiber'}), Confidence={conf:.4f}")
        #     except Exception as e:
        #          print(f"Failed to predict contour {i}: {e}")
        cls, conf = predict_single_contour(model, contours_list[149], DEVICE, NUM_POINTS)
        print(f"Contour 149: Predicted class={cls} ({'Pore' if cls==1 else 'Noise/Fiber'}), Confidence={conf:.4f}")

    except FileNotFoundError as e:
        print(e)

        # Or from pkl 
        # from your_data_module import get_your_contour_list
        # contours_list = get_your_contour_list()
        # example_contour = contours_list[0] 
        # cls, conf = predict_single_contour(model, example_contour, DEVICE, NUM_POINTS)
        # print(f"Prediction result: Class={cls} ({'Pore' if cls==1 else 'Noise/Fiber'}), Confidence={conf:.4f}")