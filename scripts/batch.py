import glob
import pytesseract
import re
import os
import csv
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.restoration import denoise_nl_means
from skimage.filters import threshold_minimum, threshold_yen
from scipy.ndimage import binary_fill_holes
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

reprocess = []

# Path for batch images
path = os.path.join("photos", "Batches", "*.*")
# Scale calibration data
scale_data = {
    100: {"scale_length_um": 1000, "ratio_um_per_pixel": 1000 / 968},
    500: {"scale_length_um": 500, "ratio_um_per_pixel": 500 / 244},
    700: {"scale_length_um": 50, "ratio_um_per_pixel": 50 / 341},
    1000: {"scale_length_um": 10, "ratio_um_per_pixel": 10 / 98},
    2500: {"scale_length_um": 10, "ratio_um_per_pixel": 10 / 242}
}

# Open CSV once for all files
with open("output.csv", "w", newline="") as f:
    writer = csv.writer(f)

    # Loop through all images
    for file in glob.glob(path):
        print(f"Processing file: {file}")
        img = cv.imread(file)
        if img is None:
            print(f"Warning: Could not read image {file}. Skipping.")
            reprocess.append(file)
            continue

        # Detect magnification from filename
        match = re.search(r"\b(\d{2,4})x\b", os.path.basename(file), re.IGNORECASE)
        if match:
            magnification = int(match.group(1))
            print(f"Detected magnification: {magnification}x")
        else:
            print(f"Warning: Could not detect magnification from filename {file}. Saving for later.")
            reprocess.append(file)
            continue
        if magnification not in scale_data:
            print(f"Warning: No scale data available for {magnification}x magnification. Saving for later.")
            reprocess.append(file)
            continue
        print(f'Manually Process: {reprocess}')

        # Calculate microns per pixel
        micron_per_pixel = scale_data[magnification]['ratio_um_per_pixel']
        print(f'Microns per pixel:{micron_per_pixel}')
        # Write the filename in CSV
        writer.writerow([os.path.basename(file)])


        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Focus on the bottom half of the image
        img_height, img_width = gray.shape[:2]
        bottom_half = gray[img_height // 2:, :]

        # Run OCR and store text regions (scale bar areas)
        data = pytesseract.image_to_data(bottom_half, output_type=pytesseract.Output.DATAFRAME)
        text = pytesseract.image_to_string(bottom_half)
        data = data.dropna(subset=['text'])  # Drop rows with no text
        print(f'Text Detected: {text}')


        # Store the text regions that will be painted white (scale bar areas)
        text_regions = []

        if not data.empty:
            # Group by line number (block_num + par_num + line_num ensures unique lines)
            for _, line_df in data.groupby(['block_num', 'par_num', 'line_num']):
                left = line_df['left'].min()
                top = line_df['top'].min()
                right = (line_df['left'] + line_df['width']).max()
                bottom = (line_df['top'] + line_df['height']).max()

                # Adjust coordinates to match the original image
                top += img_height // 2
                bottom += img_height // 2

                search_top = max(0, top - 50)  # Search above box for scale bar line
                line_found = False
                for y in range(search_top, top):
                    row = gray[y, left:right]
                    if np.all(row < 50):  # Check if the row is a continuous dark line
                        top = y  # Extend the box to include the line (masks scale bar)
                        line_found = True
                        break

                # Filter out overly large text regions
                if (right - left) > 0.3 * img_width or (bottom - top) > 0.3 * img_height:
                    continue  # Skip this region if it is too large/ detecting false text 

                # Store the text region coordinates
                text_regions.append((left, top, right, bottom))

                # Paint the line white (optional if you want to see)
                cv.rectangle(img, (left, top), (right, bottom), (115, 210, 255), 4)
        else:
            print("No text regions detected.")
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        # Step 1: NLM denoising
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        nlm = cv.fastNlMeansDenoising(gray_img, h=10, templateWindowSize=7,
                                    searchWindowSize=21)

        # Step 2: CLAHE on NLM output
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img_clahe = clahe.apply(nlm).astype(np.uint8)

        # Step 3: Compute entropy on CLAHE image
        entropy_img = entropy(img_as_ubyte(nlm), disk(2))

        # Step 4: Normalize entropy to 0-255 uint8
        norm_entropy = cv.normalize(entropy_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
        norm_entropy = cv.bitwise_not(norm_entropy)

        # Step 5: Fuse CLAHE image + normalized entropy map
        fused = cv.addWeighted(img_clahe, 0.6, norm_entropy, 0.4, 0)
        fused = cv.GaussianBlur(fused, (3, 3), 0)

        # Step 6: Threshold fused image using 2 thresholding methods
        threshold_value_yen = threshold_yen(fused)
        try:
            threshold_value_minimum = threshold_minimum(fused)
        except RuntimeError:
            print("Warning: threshold_minimum failed to find two maxima. Using a fallback value.")
            threshold_value_minimum = threshold_yen(fused)  # Fallback to yen intensity as a threshold

        # Weighted average of thresholds
        weighted_threshold = (.9 * threshold_value_yen) + (.1 * threshold_value_minimum)
        binary = (fused > weighted_threshold).astype(np.uint8)

        # Step 7: Invert binary so fibrils are white (foreground), background is black
        inverted_binary = np.logical_not(binary.astype(bool))
        filled_mask = binary_fill_holes(inverted_binary)
        filled_mask = np.logical_not(filled_mask) # Invert back

        # Convert back to uint8 (0, 255) for OpenCV
        filled_mask = (filled_mask * 255).astype(np.uint8)

        # Visualization
        fig, ax = plt.subplots(2, 3, figsize=(8, 8))
        ax[0,0].imshow(img, cmap='gray')
        ax[0,0].set_title("Original")
        ax[0,0].axis('off')
        ax[0,1].imshow(nlm, cmap='gray')
        ax[0,1].set_title("NLM Denoised")
        ax[0,1].axis('off')
        ax[0,2].imshow(img_clahe, cmap='gray')
        ax[0,2].set_title("CLAHE")
        ax[0,2].axis('off')
        ax[1,0].imshow(norm_entropy, cmap='gray')
        ax[1,0].set_title("Entropy Map")
        ax[1,0].axis('off')
        ax[1,1].imshow(fused, cmap='gray')
        ax[1,1].set_title("Fused (CLAHE + Entropy)")
        ax[1,1].axis('off')
        ax[1,2].imshow(binary, cmap='gray')
        ax[1,2].set_title("Final Binary")
        ax[1,2].axis('off')
        plt.show()
        inverted_filled_mask = cv.bitwise_not(filled_mask)

        # Run connected components

        num_labels, labels, stats, centroids = cv.connectedComponentsWithStats(inverted_filled_mask, connectivity=8)
        output = cv.cvtColor(filled_mask, cv.COLOR_GRAY2BGR)

        lengths_um = []  # Collect lengths of the longest lines in micrometers

        
        for i in range(1, num_labels):  # skip background
            x = stats[i, cv.CC_STAT_LEFT]
            y = stats[i, cv.CC_STAT_TOP]
            w = stats[i, cv.CC_STAT_WIDTH]
            h = stats[i, cv.CC_STAT_HEIGHT]
            area_pixels = stats[i, cv.CC_STAT_AREA]
            area_um2 = area_pixels * (micron_per_pixel ** 2)
            (cX, cY) = centroids[i]

            # Skip small components
            if area_pixels < 25 or w < 8 or h < 8:
                continue
            
            # Check if fibril overlaps with any text region (scale bar)
            fibril_in_text_region = False
            for text_left, text_top, text_right, text_bottom in text_regions:
                # Check if bounding boxes overlap
                if not (x + w < text_left or x > text_right or y + h < text_top or y > text_bottom):
                    fibril_in_text_region = True
                    break
            
            # Skip fibrils that are in text regions (scale bar areas)
            if fibril_in_text_region:
                continue

            # Get pixel coordinates of the current component
            fibril_pixels = np.column_stack(np.where(labels == i))

            # Find the longest pair of points
            max_dist = 0
            longest_pair = None
            for p1 in fibril_pixels:
                distances = np.sum((fibril_pixels - p1) ** 2, axis=1)  # squared distances
                idx = np.argmax(distances)
                dist = np.sqrt(distances[idx])  # Euclidean distance
                if dist > max_dist:
                    max_dist = dist
                    longest_pair = (tuple(p1), tuple(fibril_pixels[idx]))

            longest_line_px = max_dist
            longest_line_um = longest_line_px * micron_per_pixel
            lengths_um.append(longest_line_um)  # Append length to the list

            # Draw longest line in blue
            if longest_pair is not None:
                cv.line(output, (longest_pair[0][1], longest_pair[0][0]), (longest_pair[1][1], longest_pair[1][0]), (255, 0, 0), 2)
            # Draw bounding box
            cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv.circle(output, (int(cX), int(cY)), 3, (0, 255, 0), -1)
            cv.putText(output, f"ID {i}", (int(x), int(y) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 1, cv.LINE_AA)

            # Write to CSV
            
            writer.writerow([f'Fibril {i} detected at ({x},{y}). Width {w*micron_per_pixel}, Height {h*micron_per_pixel}, Area {area_um2}. Centroid at ({cX}, {cY}) Longest Line (pixels, microns): {longest_line_px}, {longest_line_um}'])
        
        # Plot histogram of lengths
        plt.imshow(cv.cvtColor(output, cv.COLOR_BGR2RGB))
        plt.title("Connected Components with Bounding Boxes and IDs")
        plt.show()

        plt.figure(figsize=(8, 6))
        plt.hist(lengths_um, bins=20, color='blue', edgecolor='black', alpha=0.7)
        plt.title("Histogram of Fibril Lengths")
        plt.xlabel("Length (µm)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()