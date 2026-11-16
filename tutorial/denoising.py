import numpy as np
from matplotlib import pyplot as plt
from skimage import io, img_as_float
from scipy import ndimage as nd

img = img_as_float(io.imread('noisy_img.jpg'))
gaussian_filter = nd.gaussian_filter(img, sigma=3)

plt.imsave('gaussian.jpg', gaussian_filter)

median_img = nd.median_filter(img, size=5)
plt.imsave("median.jpg", median_img)

# NL Means denoising
from skimage.restoration import denoise_nl_means, estimate_sigma
# Estimate sigma
sigma_est = np.mean(estimate_sigma(img, channel_axis=-1))  # use -1 for RGB

# Define patch parameters
patch_kw = dict(patch_size=5, patch_distance=3, channel_axis=-1)

# Apply denoising
nlm = denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                       patch_size=5, patch_distance=3, channel_axis=-1)

# Save output
plt.imsave("nlm.jpg", nlm)