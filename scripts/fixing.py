import cv2 as cv 
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import try_all_threshold, threshold_minimum, threshold_triangle, threshold_yen, threshold_li
from scipy.ndimage import binary_fill_holes

# photos\Day 40 Glass 1 700x Partial Coaxial upper left.jpg
# photos\Day 3 asyn on glass slide 2500x partial coaxial upper left.jpg
# photos\Day 6 asyn 100x full coaxial.jpg
# photos\Day 40 Asyn GCS 2 Partial coaxial lower left 2500x.jpg

# Load grayscale image, scale to 0-255 uint8
img = io.imread(r"learning\D16a1 cover slip 2500x Full coaxial.jpg", as_gray=True)
range_img = (img * 255).astype(np.uint8)
plt.imshow(range_img, cmap='gray')
plt.title("Original Image")
plt.show()

# Step 1: NLM denoising
nlm = cv.fastNlMeansDenoising(range_img, h=10, templateWindowSize=7,
                               searchWindowSize=21)

# Step 2: CLAHE on NLM output
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
img_clahe = clahe.apply(nlm).astype(np.uint8)

# Step 3: Compute entropy on CLAHE image
entropy_img = entropy(img_as_ubyte(nlm), disk(2))

# Step 4: Normalize entropy to 0-255 uint8
norm_entropy = cv.normalize(entropy_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
norm_entropy = cv.bitwise_not(norm_entropy)
plt.imshow(norm_entropy, cmap='gray')
plt.title("Normalized Entropy Map")
plt.show()

# Step 5: Fuse CLAHE image + normalized entropy map
fused = cv.addWeighted(img_clahe, 0.6, norm_entropy, 0.4, 0)
fused = cv.GaussianBlur(fused, (3, 3), 0)

# Step 6: Threshold fused image using 2 thresholding methods
# Compute thresholds using scikit-image methods
threshold_value_yen = threshold_yen(fused)
threshold_value_minimum = threshold_minimum(fused)

# Weighted average of thresholds
weighted_threshold = (.9 * threshold_value_yen) + (.1 * threshold_value_minimum)
print(f"Triangle Threshold: {threshold_value_yen}")
print(f"Minimum Threshold: {threshold_value_minimum}")
print(f"Weighted Threshold: {weighted_threshold}")

# Apply the weighted threshold to create a binary mask
binary = (fused > weighted_threshold).astype(np.uint8)
plt.imshow(binary, cmap='gray')
plt.title('Binary Mask (Weighted Threshold)')
plt.show()

# Step 7
# Invert binary so fibrils are white (foreground), background is black
inverted_binary = np.logical_not(binary.astype(bool))

# Fill holes inside fibrils
filled_mask = binary_fill_holes(inverted_binary)

# Invert back so fibrils are black, background is white
filled_mask = np.logical_not(filled_mask)

# Convert back to uint8 (0, 255) for OpenCV
filled_mask = (filled_mask * 255).astype(np.uint8)
plt.imshow(filled_mask, cmap='gray')
plt.title('Filled Mask')
plt.show()
# Visualization
fig, ax = plt.subplots(2, 3, figsize=(10, 8))
ax[0,0].imshow(range_img, cmap='gray')
ax[0,0].set_title("Original")
ax[0,0].axis('off')
ax[0,1].imshow(nlm, cmap='gray')
ax[0,1].set_title("NLM Denoised")
ax[0,1].axis('off')
ax[0,2].imshow(img_clahe, cmap='gray')
ax[0,2].set_title("CLAHE")
ax[0,2].axis('off')
ax[1,0].imshow(norm_entropy, cmap='gray')
ax[1,0].set_title("Entropy Map")
ax[1,0].axis('off')
ax[1,1].imshow(fused, cmap='gray')
ax[1,1].set_title("Fused (CLAHE + Entropy)")
ax[1,1].axis('off')
ax[1,2].imshow(binary, cmap='gray')
ax[1,2].set_title("Final Binary")
ax[1,2].axis('off')
plt.tight_layout()
plt.show()

# Invert the filled mask so fibrils are white (foreground) and background is black
inverted_filled_mask = cv.bitwise_not(filled_mask)

# Run connected components
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(inverted_filled_mask, connectivity=8)
output = cv.cvtColor(filled_mask, cv.COLOR_GRAY2BGR)

# Prepare CSV output
with open("connected_components_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "X", "Y", "Width", "Height", "Area", "Centroid_X", "Centroid_Y"])

    # Loop through components (skip background: i = 0)
    for i in range(0, num_labels):
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        # Filter out small noise
        if area < 25:
            continue

        # Draw bounding box (red) and centroid (green)
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.circle(output, (int(cX), int(cY)), 3, (0, 255, 0), -1)

        # Add ID number as text near the centroid
        cv.putText(output, f"ID {i}", (int(x), int(y) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 1, cv.LINE_AA)

        # Write to CSV
        writer.writerow([i, x, y, w, h, area, cX, cY])

# Show result
plt.imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
plt.title("Connected Components with Bounding Boxes and IDs")
plt.axis('off')
plt.show()