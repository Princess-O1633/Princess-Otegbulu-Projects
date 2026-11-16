import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage. filters.rank import entropy
from skimage import io, morphology
from skimage.morphology import disk
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import try_all_threshold 

# Grayscale
img = io.imread("photos\Day 6 asyn 100x full coaxial.jpg" , as_gray=True)
range_img = (img * 255).astype(np.uint8)
plt.imshow(range_img, cmap='gray')
plt.show()

# NLM denoising
range_img = np.clip(range_img, 0, 255).astype(np.uint8)
nlm = cv.fastNlMeansDenoising(range_img, h=10, templateWindowSize=7, searchWindowSize=21)
plt.imshow(nlm, cmap='gray')
plt.title("NLM Denoised")
plt.show()

# Not yet incorporated
# entropy_img = entropy(range_img, disk(3))
# plt.imshow(entropy_img, cmap='gray')
# plt.title('Entropy Image')
# plt.show()

# CLAHE
clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12))
img_clahe = clahe.apply(range_img)
plt.imshow(img_clahe, cmap='gray')
plt.show()
img_clahe = cv.fastNlMeansDenoising(img_clahe, h=5, templateWindowSize=7, searchWindowSize=21)
plt.imshow(img_clahe, cmap='gray')
plt.show()
img_clahe = img_clahe.astype(np.float32)


fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(img, cmap='gray')
ax[0].set_title("Original")
ax[0].axis('off')
ax[2].imshow(img_clahe, cmap='gray')
ax[2].set_title("CLAHE")
ax[2].axis('off')
ax[1].imshow(nlm, cmap='gray')
ax[1].set_title("NLM Denoised")
ax[1].axis('off')
plt.tight_layout()
plt.show()


