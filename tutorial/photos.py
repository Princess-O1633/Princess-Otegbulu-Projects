import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io

folder_path = 'A-Syn Images\Batch 1\Batch 1\A Syn 0913224'  # Raw string to handle backslashes

failed_paths = []

image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    
    image = cv.imread(image_path)
    if image is None:
        failed_paths.append(image_path)
        print(f"Failed to load with OpenCV: {image_path}")
        continue

    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow('OpenCV Image', image)
    cv.waitKey(500)  # shows each image for 0.5 sec (change as needed)

cv.destroyAllWindows()

# Now try fallback with skimage for failed ones
for path in failed_paths:
    try:
        fallback_image = io.imread(path, as_gray=True)
        print(type(fallback_image))
        plt.imshow(fallback_image, cmap='gray')
        plt.title(f"Fallback load: {os.path.basename(path)}")
        plt.axis('off')
        plt.show()
    except Exception as e:
        print(f"Still failed: {path}")

