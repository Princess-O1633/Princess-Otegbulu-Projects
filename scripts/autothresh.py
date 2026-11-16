import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology

# Load grayscale image and convert to uint8
img = io.imread("Day 6 asyn on glass slide 2500x partial coaxial upper left.jpg" 
, as_gray=True)
img_uint8 = (img * 255).astype(np.uint8)

# Step 1: CLAHE for contrast enhancement
clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img_uint8)

# Step 3: Gaussian blur for further smoothing
blurred = cv.GaussianBlur(img_clahe, (3, 3), 0)
blurred = (blurred * 255).astype(np.uint8)
# Step 4: Fixed thresholding at 120
_, binary = cv.thresholdWithMask(blurred, 80, 255, cv.THRESH_BINARY)

#Step 5: Morphological cleanup (remove small objects and holes)
binary_bool = binary.astype(bool)
cleaned = morphology.remove_small_objects(binary_bool, min_size=64)
cleaned = morphology.remove_small_holes(cleaned, area_threshold=128)

# Convert final mask to uint8 for display
binary_final = (cleaned * 255).astype(np.uint8)

# Display
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
ax = axes.ravel()

ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original")
ax[0].axis('off')

ax[1].imshow(img_clahe, cmap='gray')
ax[1].set_title("CLAHE")
ax[1].axis('off')

ax[2].imshow(blurred, cmap='gray')
ax[2].set_title("Gaussian Blur")
ax[2].axis('off')

ax[3].imshow(binary, cmap='gray')
ax[3].set_title("Thresholded (Fixed @ 110)")
ax[3].axis('off')

ax[4].imshow(binary_final, cmap='gray')
ax[4].set_title("Cleaned Mask")
ax[4].axis('off')

plt.tight_layout()
plt.show()
