import cv2 as cv
import numpy as np

# Load image
image = cv.imread('GeeksOPENCV.png')
if image is None:
    raise ValueError("Image not found or path is incorrect.")

# Convert to grayscale
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Apply Gaussian blur
blur = cv.GaussianBlur(gray, (7, 7), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Thresholding with Otsu's method
threshold = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
cv.imshow('Thresholded', threshold)

# Connected components analysis
analysis = cv.connectedComponentsWithStats(threshold, 4, cv.CV_32S)
(totalLabels, label_ids, values, centroids) = analysis

# Output image to show filtered components
output = np.zeros(gray.shape, dtype="uint8")

# Loop over each component
for i in range(1, totalLabels):
    area = values[i, cv.CC_STAT_AREA]

    if area >= 50:
        # Bounding box
        x = values[i, cv.CC_STAT_LEFT]
        y = values[i, cv.CC_STAT_TOP]
        w = values[i, cv.CC_STAT_WIDTH]
        h = values[i, cv.CC_STAT_HEIGHT]

        pt1 = (x, y)
        pt2 = (x + w, y + h)
        (X, Y) = centroids[i]

        new = image.copy()
        cv.rectangle(new, pt1, pt2, (0, 255, 0), 2)
        cv.circle(new, (int(X), int(Y)), 4, (0, 0, 255), -1)

        # Mask for current component
        componentMask = (label_ids == i).astype("uint8") * 255
        output = cv.bitwise_or(output, componentMask)

        # Show current component
        cv.imshow("Component", new)
        cv.imshow("Component Mask", componentMask)

        key = cv.waitKey(0)
        if key == ord("q"):
            break

# Show final output
cv.imshow("Filtered Components", output)
cv.waitKey(0)
cv.destroyAllWindows()
