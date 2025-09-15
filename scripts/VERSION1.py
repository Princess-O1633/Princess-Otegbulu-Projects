import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage import io
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import try_all_threshold 
# photos\Day 6 asyn 100x full coaxial.jpg
# 

# Load grayscale image, scale to 0-255 uint8
img = io.imread(r"photos\Day 6 asyn 100x full coaxial.jpg", as_gray=True)
range_img = (img * 255).astype(np.uint8)
plt.imshow(range_img, cmap='gray')
plt.title("Original Image")
plt.show()

# Step 1: NLM denoising
nlm = cv.fastNlMeansDenoising(range_img, h=10, templateWindowSize=7, searchWindowSize=21)
plt.imshow(nlm, cmap='gray')
plt.title("NLM Denoised")
plt.show()

# Step 2: CLAHE on NLM output
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
img_clahe = clahe.apply(nlm)

plt.imshow(img_clahe, cmap='gray')
plt.title("CLAHE")
plt.show()

# Step 3: Compute entropy on CLAHE image
entropy_img = entropy(img_as_ubyte(nlm), disk(2))
plt.imshow(entropy_img, cmap='gray')
plt.title('Entropy Image')
plt.show()

# Step 4: Normalize entropy to 0-255 uint8
norm_entropy = cv.normalize(entropy_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
norm_entropy = cv.bitwise_not(norm_entropy)
plt.imshow(norm_entropy, cmap='gray')
plt.title("Normalized Entropy Map")
plt.show()

# Step 5: Fuse CLAHE image + normalized entropy map
fused = cv.addWeighted(img_clahe, 0.7, norm_entropy, 0.3, 0)
fused = cv.GaussianBlur(fused, (3, 3), 0)  # Optional smoothing

# Step 6: Threshold fused image
_, binary = cv.threshold(fused, 0, 255, cv.THRESH_BINARY + cv.THRESH_TRIANGLE)

# Step 7: Morphological opening to clean small noise
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3,3))
cleaned = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel)

# Visualization
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
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

ax[1,2].imshow(cleaned, cmap='gray')
ax[1,2].set_title("Final Binary (cleaned)")
ax[1,2].axis('off')

plt.tight_layout()
plt.show()