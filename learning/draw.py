import cv2 as cv
import numpy as np

blank= np.zeros((500,1000,3), dtype='uint8')
# cv.imshow ('Blank', blank)

# # 1. Paint the image a certain colour
# #blank[200:300, 300:400] = 0,255,0
# #cv.imshow('Green', blank)

# #2. Draw a rectangle
# cv.rectangle(blank, (0,0),(blank.shape[1]//2, blank.shape[0]//2),(0,255,0), thickness = cv.FILLED) 
# cv.imshow('Rectangle', blank)

# # 3. Draw A Circle
# cv.circle(blank, (250,250),40,(0,0,255),thickness = -1)
# cv.imshow('Circle', blank)

# # 4. Draw a Line
# cv.line(blank,(0,0),(blank.shape[1]//2, blank.shape[0]//2), (255,255,255), thickness =3)
# cv.imshow('Line', blank)

# 5. Write Text
cv.putText(blank, 'Madieliene She Likes Manderines', (220,225), cv.FONT_HERSHEY_DUPLEX, 1.0,(0,165,255), 2)
cv.imshow('Text', blank)
cv.waitKey(0)
