import argparse
import cv2 as cv

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Apply Otsu thresholding to an image')
parser.add_argument('-i', '--image', type=str, required=True, help='Path to the input image')
args = parser.parse_args()

# Load and check image
image = cv.imread(args.image)
if image is None:
    raise FileNotFoundError(f"Image not found at {args.image}")

# Resize for consistency
resized = cv.resize(image, (500, 500), interpolation=cv.INTER_CUBIC)

# Convert to grayscale
gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

# Optional: apply CLAHE (helps when contrast is low)
# clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# gray = clahe.apply(gray)

# Otsu thresholding
otsu_val, thresh = cv.threshold(gray, 125, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
print(f"[INFO] Otsu's threshold value: {otsu_val}")

# Display
cv.imshow('Resized Image', resized)
cv.imshow('Grayscale', gray)
cv.imshow('Otsu Threshold', thresh)

# Optional: save result
cv.imwrite("otsu_result.png", thresh)

cv.waitKey(0)
cv.destroyAllWindows()
