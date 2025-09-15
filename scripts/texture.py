import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage. filters.rank import entropy
from skimage.morphology import disk
from skimage import io, morphology
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import try_all_threshold

# Grayscale
img = io.imread("photos\Day 3 asyn on glass slide 2500x partial coaxial upper left.jpg" , as_gray=True)
range_img = (img * 255).astype(np.uint8)
plt.imshow(range_img, cmap='gray')
plt.show()

# CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
img_clahe = clahe.apply(range_img)
plt.imshow(img_clahe, cmap='gray')
plt.show()
img_clahe = img_clahe.astype(np.float32)

# NLM denoising
img_clahe = np.clip(img_clahe, 0, 255).astype(np.uint8)
nlm = cv.fastNlMeansDenoising(img_clahe, h=10, templateWindowSize=7, searchWindowSize=21)
plt.imshow(nlm, cmap='gray')
plt.title("NLM Denoised")
plt.show()
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original")
ax[0].axis('off')
ax[1].imshow(img_clahe, cmap='gray')
ax[1].set_title("CLAHE")
ax[1].axis('off')
ax[2].imshow(nlm, cmap='gray')
ax[2].set_title("NLM Denoised")
ax[2].axis('off')
plt.tight_layout()
plt.show()

entropy_img = entropy(range_img, disk(3))
plt.imshow(entropy_img, cmap='gray')
plt.title('Entropy Image')
plt.show()

blurred = cv.GaussianBlur(entropy_img, (5, 5), 0)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Entropy Image')  
plt.show()

try_all_threshold(blurred, figsize=(10, 8), verbose=False)
plt.show()

