import cv2 as cv
import numpy as np
import skimage 
import matplotlib.pyplot as plt

# Load image and crop out scale bar (assumed bottom-right ~100x100 px)
img = cv.imread('D16a cover slip 2500x Full coaxial.jpg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
resized = cv.resize(gray, (900, 600), interpolation = cv.INTER_CUBIC)
cv.imshow('Resize', resized)

# Apply Gaussian blur
blurred = cv.GaussianBlur(resized, (5, 5), 0)
cv.imshow('Blurred', blurred)

# Thresholding
_, thresh = cv.threshold(blurred, 120, 255, cv.THRESH_BINARY_INV)
cv.imshow('Thresholded', thresh)

# Apply Canny edge detection
edges = cv.Canny(blurred, 50, 150)
cv.imshow('Edges', edges)

roberts_edges = skimage.filters.roberts(resized)
plt.imshow(roberts_edges, cmap='gray')  
plt.show()

cv.waitKey(0)