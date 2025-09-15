import cv2 as cv

img = cv.imread('cats.jpg')
cv.imshow('cats', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Laplacion
lap = cv.Laplacian(gray, cv.CV_64F)

cv.waitKey(0)