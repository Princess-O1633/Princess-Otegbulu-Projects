import cv2 as cv
import matplotlib.pyplot as plt

# Reading video.
img = cv.imread('cat.jpg')

cv.imshow("Cat",img)

cv.waitKey(0)


# Reading Video
"""

capture = cv.VideoCapture("dog.mp4")

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)# display each frame

    if cv.waitKey(20) and 0xFF==ord('d'):#If the letter d pressed
        break

capture.release()
cv.destroyAllWindows()"""