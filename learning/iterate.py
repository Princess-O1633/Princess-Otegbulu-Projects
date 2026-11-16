import argparse
import cv2

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-c", "--connectivity", type=int, default=4, help="connectivity for connected component analysis")
args = vars(ap.parse_args())

# Load image and preprocess
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale", gray)

thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresholded", thresh)

# Connected components
numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
    thresh, args["connectivity"], cv2.CV_32S)

# Loop over each component
for i in range(1, numLabels):  # skip background (label 0)
    print(f"[INFO] Examining component {i}/{numLabels - 1}")

    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    cX, cY = centroids[i]

    print(f" - Bounding Box: x={x}, y={y}, w={w}, h={h}")
    print(f" - Area: {area}")
    print(f" - Centroid: ({cX:.2f}, {cY:.2f})")

    # Draw and show
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)

    mask = (labels == i).astype("uint8") * 255

    cv2.imshow("Component", output)
    cv2.imshow("Component Mask", mask)

    # Wait for key to move to next
    key = cv2.waitKey(0)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
