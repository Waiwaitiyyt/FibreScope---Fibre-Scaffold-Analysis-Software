import numpy as np
import cv2
import seaborn as sns
from matplotlib.figure import Figure

def fibreResultVisualise(diameterList: list, measuredImg: np.ndarray, cleanImg: np.ndarray, originalImg: np.ndarray, fig: Figure | None = None) -> Figure:
    areaArr = np.asarray(diameterList, dtype=np.float64)
    if fig is None:
        fig = Figure(figsize=(12, 6))

    axes = fig.subplots(2, 3)

    ax_hist, ax_kde, ax_box = axes[0]
    ax_originalImg, ax_mask, ax_refinedContours = axes[1]

    # Titles
    ax_hist.set_title("Diameter Histogram")
    ax_kde.set_title("Kernel Density Estimation")
    ax_box.set_title("Boxplot")
    ax_originalImg.set_title("Original Image")
    ax_mask.set_title("Skeleton and Measurement Sites")
    ax_refinedContours.set_title("Binary Mask")

    # Labels
    ax_hist.set_xlabel("Fibre Diameter (µm)")
    ax_kde.set_xlabel("Fibre Diameter (µm)")
    ax_box.set_xlabel("Fibre Diameter (µm)")

    # Image axes
    for ax in (ax_originalImg, ax_mask, ax_refinedContours):
        ax.axis("off")

    # Plots
    sns.histplot(areaArr, binwidth=3, stat="count",
                 color="lightgray", ax=ax_hist)
    sns.kdeplot(areaArr, color="#7092BE", lw=2, ax=ax_kde)
    sns.boxplot(x=areaArr, ax=ax_box)

    ax_originalImg.imshow(cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB))
    ax_mask.imshow(cv2.cvtColor(measuredImg, cv2.COLOR_BGR2RGB))
    ax_refinedContours.imshow(cv2.cvtColor(cleanImg, cv2.COLOR_BGR2RGB))

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.95,
        bottom=0.08,
        hspace=0.35,
        wspace=0.25
    )

    return fig

def poreResultVisualise(areaList: list, mask_color: np.ndarray, refinedContours: list, originalImg: np.ndarray, fig: Figure | None = None) -> Figure:
    areaArr = np.asarray(areaList, dtype=np.float64)
    if fig is None:
        fig = Figure(figsize=(12, 6))

    axes = fig.subplots(2, 3)

    ax_hist, ax_kde, ax_box = axes[0]
    ax_originalImg, ax_mask, ax_refinedContours = axes[1]

    # Titles
    ax_hist.set_title("Pore Size Histogram")
    ax_kde.set_title("Kernel Density Estimation")
    ax_box.set_title("Boxplot")
    ax_originalImg.set_title("Original Image")
    ax_mask.set_title("Labelled Pores")
    ax_refinedContours.set_title("Scaffold Contours")

    # Labels
    ax_hist.set_xlabel("Pore Size (µm²)")
    ax_kde.set_xlabel("Pore Size (µm²)")
    ax_kde.ticklabel_format(style='scientific', axis='y', scilimits=(-2,2))
    ax_box.set_xlabel("Pore Size (µm²)")

    # Image axes
    for ax in (ax_originalImg, ax_mask, ax_refinedContours):
        ax.axis("off")

    # Plots
    sns.histplot(areaArr, bins = "auto", stat="count",
                 color="lightgray", ax=ax_hist)
    sns.kdeplot(areaArr, color="#7092BE", lw=2, ax=ax_kde)
    sns.boxplot(x=areaArr, ax=ax_box)

    ax_originalImg.imshow(cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB))
    ax_mask.imshow(cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB))

    contourMask = np.zeros_like(originalImg)
    refinedContours_int32 = [cnt.astype(np.int32) for cnt in refinedContours]
    cv2.drawContours(contourMask, refinedContours_int32, -1, (255, 255, 255), 1) 
    ax_refinedContours.imshow(contourMask)

    fig.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.95,
        bottom=0.08,
        hspace=0.35,
        wspace=0.25
    )

    return fig