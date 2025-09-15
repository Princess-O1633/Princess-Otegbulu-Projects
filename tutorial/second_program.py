from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import cv2 as cv
from skimage import restoration

img = io.imread('test_image_cropped.jpg', as_gray=True)


def gkern(kernlen=21, nsig=2):    #Returns a 2D Gaussian kernel.

    lim = kernlen//2 + (kernlen % 2)/2
    x = np.linspace(-lim, lim, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

psf = gkern(5,3)   #Kernel length and sigma
print(psf)

deconvolved, _ = restoration.unsupervised_wiener(img, psf)
plt.imsave('deconvolved_image.jpg', deconvolved, cmap='gray')
