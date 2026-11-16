import cv2 as cv 
import csv
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import try_all_threshold, threshold_minimum, threshold_triangle, threshold_yen
from scipy.ndimage import binary_fill_holes

# photos\Day 40 Glass 1 700x Partial Coaxial upper left.jpg
# photos\Day 3 asyn on glass slide 2500x partial coaxial upper left.jpg
# photos\Day 6 asyn 100x full coaxial.jpg

# Load grayscale image, scale to 0-255 uint8
img = io.imread(r"photos/Day 6 asyn 100x full coaxial.jpg", as_gray=True)
range_img = (img * 255).astype(np.uint8)
plt.imshow(range_img, cmap='gray')
plt.title("Original Image")
plt.show()

# Step 1: NLM denoising
nlm = cv.fastNlMeansDenoising(range_img, h=10, templateWindowSize=7, searchWindowSize=21)

# Step 2: CLAHE on NLM output
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
img_clahe = clahe.apply(nlm).astype(np.uint8)

# Step 3: Compute entropy on CLAHE image
entropy_img = entropy(img_as_ubyte(nlm), disk(3))

# Step 4: Normalize entropy to 0-255 uint8
norm_entropy = cv.normalize(entropy_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
norm_entropy = cv.bitwise_not(norm_entropy)
plt.imshow(norm_entropy, cmap='gray')
plt.title("Normalized Entropy Map")
plt.show()

# Step 5: Fuse CLAHE image + normalized entropy map
fused = cv.addWeighted(img_clahe, 0.6, norm_entropy, 0.4, 0)
fused = cv.GaussianBlur(fused, (3, 3), 0)  # Optional smoothing

# Step 6: Threshold fused image using Minimum thresholding
threshold_value = threshold_triangle(fused)  # FIXED: get the actual threshold
binary = (fused > threshold_value).astype(np.uint8)  # Make binary mask

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

# Prepare binary image for connected components
binary = cv.bitwise_not(binary)  # Invert binary mask for connected components
binary = binary.astype(np.float32) / 255.0  # Convert to float for processing
# Binary image: make sure it's 0/1 not 0/255 for fill_holes
binary_bool = (binary > 0)

# Fill holes
filled_mask = binary_fill_holes(binary_bool)

# Convert back to uint8 for OpenCV usage
filled_mask = (filled_mask * 255).astype(np.uint8)

plt.imshow(filled_mask, cmap='gray')
plt.title("Hole-Filled Binary Mask")
plt.show()


# Run connected components on filled mask
num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(filled_mask, connectivity=8)

# Convert to BGR for visualization
output = cv.cvtColor(filled_mask, cv.COLOR_GRAY2BGR)

# Prepare CSV output
with open("connected_components_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "X", "Y", "Width", "Height", "Area", "Centroid_X", "Centroid_Y"])

    # Loop through components (skip background: i = 0)
    for i in range(1, num_labels):
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        # Filter out small noise
        if area < 20:
            continue

        # Draw bounding box (red) and centroid (green)
        cv.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.circle(output, (int(cX), int(cY)), 3, (0, 255, 0), -1)

        # Write to CSV
        writer.writerow([i, x, y, w, h, area, cX, cY])

# Show result
plt.imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
plt.title("Connected Components with Bounding Boxes")
plt.axis('off')
plt.show()