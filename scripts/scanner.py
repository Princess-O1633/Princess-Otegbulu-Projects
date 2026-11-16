import pytesseract
import cv2 as cv
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.restoration import denoise_nl_means
from skimage.filters import threshold_minimum, threshold_yen
from scipy.ndimage import binary_fill_holes
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

scale_ratios = {
    100: 1.030065,    # µm / pixel
    500: 2.049180,    
    700: 0.146305,    
    1000: 0.102749,   
    2500: 0.041072    
}

# Load and preprocess image for OCR
img_path = os.path.join("photos", "M1 1 Alpha-syn 1 700x Full Coaxial.jpg")
img = cv.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Image not found: {img_path}")

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Run OCR
data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DATAFRAME)
data = data.dropna(subset=['text']) # Drop rows with no text

# Group by line number (block_num + par_num + line_num ensures unique lines)
for _, line_df in data.groupby(['block_num', 'par_num', 'line_num']):
    left = line_df['left'].min()
    top = line_df['top'].min()
    right = (line_df['left'] + line_df['width']).max()
    bottom = (line_df['top'] + line_df['height']).max()

    # Paint the line white
    cv.rectangle(img, (left, top), (right, bottom), (255, 255, 255), -1)


# Step 1: NLM denoising
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
nlm = cv.fastNlMeansDenoising(gray_img, h=10, templateWindowSize=7,
                               searchWindowSize=21)

# Step 2: CLAHE on NLM output
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
img_clahe = clahe.apply(nlm).astype(np.uint8)

# Step 3: Compute entropy on CLAHE image
entropy_img = entropy(img_as_ubyte(nlm), disk(2))

# Step 4: Normalize entropy to 0-255 uint8
norm_entropy = cv.normalize(entropy_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
norm_entropy = cv.bitwise_not(norm_entropy)

# Step 5: Fuse CLAHE image + normalized entropy map
fused = cv.addWeighted(img_clahe, 0.6, norm_entropy, 0.4, 0)
fused = cv.GaussianBlur(fused, (3, 3), 0)

# Step 6: Threshold fused image using 2 thresholding methods
threshold_value_yen = threshold_yen(fused)
threshold_value_minimum = threshold_minimum(fused)

# Weighted average of thresholds
weighted_threshold = (.9 * threshold_value_yen) + (.1 * threshold_value_minimum)
binary = (fused > weighted_threshold).astype(np.uint8)

# Step 7: Invert binary so fibrils are white (foreground), background is black
inverted_binary = np.logical_not(binary.astype(bool))
filled_mask = binary_fill_holes(inverted_binary)
filled_mask = np.logical_not(filled_mask) # Invert back

# Convert back to uint8 (0, 255) for OpenCV
filled_mask = (filled_mask * 255).astype(np.uint8)

# Visualization
fig, ax = plt.subplots(2, 3, figsize=(8, 8))
ax[0,0].imshow(img, cmap='gray')
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
plt.show()
inverted_filled_mask = cv.bitwise_not(filled_mask)

# Run connected components
scale = 5
scale_length = 100

# micron-per-pixel ratio (example: 1 micron = 2 pixels → ratio = 0.5)
micron_per_pixel = scale / scale_length  

num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(inverted_filled_mask, connectivity=8)
output = cv.cvtColor(filled_mask, cv.COLOR_GRAY2BGR)

with open("connected_components_output.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "X", "Y", "Width", "Height", "Area", "Centroid_X", "Centroid_Y", "LongestLine_px", "LongestLine_um"])

    for i in range(1, num_labels):  # skip background
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        
        if area < 35 or w < 8 or h < 8 :
            continue

        # Get pixel coordinates of the current component
        fibril_pixels = np.column_stack(np.where(labels == i))

        # Find the longest pair of points
        max_dist = 0
        longest_pair = None
        for p1 in fibril_pixels:
            distances = np.sum((fibril_pixels - p1) ** 2, axis=1)  # squared distances
            idx = np.argmax(distances)
            dist = np.sqrt(distances[idx])  # Euclidean distance
            if dist > max_dist:
                max_dist = dist
                longest_pair = (tuple(p1), tuple(fibril_pixels[idx]))

        longest_line_px = max_dist
        longest_line_um = longest_line_px * micron_per_pixel

        # Draw bounding box
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.circle(output, (int(cX), int(cY)), 3, (0, 255, 0), -1)
        cv.putText(output, f"ID {i}", (int(x), int(y) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 1, cv.LINE_AA)

        # Draw longest line in blue
        if longest_pair is not None:
            cv.line(output, (longest_pair[0][1], longest_pair[0][0]), (longest_pair[1][1], longest_pair[1][0]), (255, 0, 0), 1)

        # Write to CSV
        writer.writerow([i, x, y, w, h, area, cX, cY, longest_line_px, longest_line_um])

# Show result
plt.imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
plt.title("Connected Components with Bounding Boxes and IDs")
plt.show()