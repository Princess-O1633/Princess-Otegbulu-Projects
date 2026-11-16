# import
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-c", "--connectivity", type=int, default=4,
	help="connectivity for connected component analysis")
args = vars(ap.parse_args())

# load the input image, convert it to grayscale, and apply a binary
# threshold to it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale", gray)
thresh = cv2.threshold(gray, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

cv2.imshow("Thresholded", thresh)


# apply connnected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(
    thresh, args["connectivity"], cv2.CV_32S)
(numLabels, labels, stats, centroids) = output
# The total number of detected components
# Labeled map of all objects in your binary image (1,2,3..)
# Area, bounding box
# Centroid

# Loop over the # of unique connected component labels
for i in range(0, numLabels):
    # if this is the first component, ignore it
    if i == 0:
        text = "examining component: {}/{}".format(
            i + 1, numLabels)
    # otherwise, we are examining an actual connected component
    else:
        text = "examing component: {}/{}".format( i + 1, numLabels)

# print a update for current component
print("[INFO] {}".format(text))

# Extract the connected component statistics and centroid

x = stats[i, cv2.CC_STAT_LEFT]
y = stats[i, cv2.CC_STAT_TOP]
w = stats[i, cv2.CC_STAT_WIDTH]
h = stats[i, cv2.CC_STAT_HEIGHT]
area = stats[i, cv2.CC_STAT_AREA]
(cX,cY) = centroids[i]

#Clone the original image so we can draw on it
# A bounding Box surrounding the connected component w/ a circle at the centroid

output = image.copy()
cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

# construct a mask for the current connected component 
# by find all pixlels in the labels array that have the label ID

component_mask = (labels == i).astype("uint8") * 255

# show our output image and connected component mask
cv2.imshow('Output', output)
cv2.imshow("Component Mask", component_mask)

cv2.waitKey(0)