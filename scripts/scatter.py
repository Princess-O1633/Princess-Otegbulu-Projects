import re
import os
import csv
import glob
import pandas as pd
import cv2 as cv
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
from skimage.filters import threshold_minimum, threshold_yen
from scipy.ndimage import binary_fill_holes
from matplotlib.widgets import RectangleSelector
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Path for batch images
path = os.path.join("2500X Batch 1&2", "", "*.*")

reprocess = []

def manual_scale_selection(image, file):
    print(f"Manual scale selection for {file}. Please select the scale bar.")

    # Display the image using matplotlib
    fig, ax = plt.subplots()
    ax.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    ax.set_title("Select the scale bar region")
    rect = plt.Rectangle((0, 0), 1, 1, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

    def on_select(eclick, erelease):
        # Update rectangle dimensions based on user selection
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        rect.set_xy((min(x1, x2), min(y1, y2)))
        rect.set_width(abs(x2 - x1))
        rect.set_height(abs(y2 - y1))
        fig.canvas.draw()

    # Connect the rectangle selector
    selector = RectangleSelector(ax, on_select, useblit=True, button=[1], interactive=True)
    plt.title("Select the scale bar region and press Enter")
    plt.axis('off')
    plt.show()

    # Get the rectangle coordinates
    x, y, w, h = rect.get_x(), rect.get_y(), rect.get_width(), rect.get_height()
    x, y, w, h = int(x), int(y), int(w), int(h)

    # Crop the selected region
    scale_bar_region = image[y:y+h, x:x+w]
    gray_scale_bar = cv.cvtColor(scale_bar_region, cv.COLOR_BGR2GRAY)

    # Find the longest dark line in the selected region
    longest_line_length = 0
    for row in gray_scale_bar:
        dark_pixels = np.where(row < 50)[0]  # Find dark pixels in the row
        if len(dark_pixels) > 1:
            line_length = dark_pixels[-1] - dark_pixels[0]
            if line_length > longest_line_length:
                longest_line_length = line_length

    while longest_line_length < 20:
        print("Error: Could not detect a scale bar line.")
        print("Re-select the scale bar region.")
        return manual_scale_selection(image, file)

    # Ask the user for the scale bar length in microns
    while True:
        try:
            scale_length_um = float(input("Enter the scale bar length in microns (e.g., 10, 50, 100): "))
            break
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

    # Calculate microns per pixel
    micron_per_pixel = scale_length_um / longest_line_length
    print(f"Scale bar length: {longest_line_length} pixels, {scale_length_um} microns")
    print(f"Calculated microns per pixel: {micron_per_pixel:2f}")
    return micron_per_pixel


# Scale calibration data
scale_data = {
    100: {"scale_length_um": 1000, "ratio_um_per_pixel": 1000 / 968},
    500: {"scale_length_um": 500, "ratio_um_per_pixel": 500 / 244},
    700: {"scale_length_um": 50, "ratio_um_per_pixel": 50 / 341},
    1000: {"scale_length_um": 10, "ratio_um_per_pixel": 10 / 98},
    2500: {"scale_length_um": 10, "ratio_um_per_pixel": 10 / 242}
}

# Initialize list to store all data for Excel output
all_data = []

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

    # Determine microns per pixel based on available scale data
    if magnification in scale_data:
        # Use predefined scale data
        micron_per_pixel = scale_data[magnification]['ratio_um_per_pixel']
        print(f"Using predefined scale data for {magnification}x: {micron_per_pixel} microns/pixel")
    else:
        # Use manual scale selection for unknown magnifications
        print(f"No scale data available for {magnification}x magnification. Using manual selection.")
        micron_per_pixel = manual_scale_selection(img, file)
        if micron_per_pixel is None:
            print(f"Skipping file {file} due to manual scale selection failure.")
            reprocess.append(file)
            continue

    print(f'Microns per pixel: {micron_per_pixel:2f}')
    
    # Add filename row to data
    all_data.append({
        'FIBRIL #': os.path.basename(file),
        'AREA': '',
        'LENGTH': ''
    })

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
    
    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title("Text Regions (Scale Bar Areas)")
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
    fused = cv.addWeighted(img_clahe, 0.7, norm_entropy, 0.3, 0)
    fused = cv.GaussianBlur(fused, (3, 3), 0)

    # Step 6: Threshold fused image using 2 thresholding methods
    threshold_value_yen = threshold_yen(fused)
    try:
        threshold_value_minimum = threshold_minimum(fused)
    except RuntimeError:
        print("Warning: threshold_minimum failed to find two maxima. Using a fallback value.")
        threshold_value_minimum = threshold_yen(fused)*.50  # Fallback to yen intensity as a threshold
    print(f"Threshold values - Yen: {threshold_value_yen}, Minimum: {threshold_value_minimum}")

    # Weighted average of thresholds
    weighted_threshold = (.9 * threshold_value_yen) + (.1 * threshold_value_minimum)
    binary = (fused > weighted_threshold).astype(np.uint8)
    print(f"Weighted threshold value: {weighted_threshold}")
    # Step 7: Invert binary so fibrils are white (foreground), background is black
    inverted_binary = np.logical_not(binary.astype(bool))
    filled_mask = binary_fill_holes(inverted_binary)
    filled_mask = np.logical_not(filled_mask) # Invert back

    # Convert back to uint8 (0, 255) for OpenCV
    filled_mask = (filled_mask * 255).astype(np.uint8)

    # Visualization
    fig, ax = plt.subplots(2, 3, figsize=(8, 8))
    ax[0,0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB), cmap='gray')
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
    areas_um2_list = []  # Collect areas (µm^2) for histogram and analysis
    number = 1
    for i in range(1, num_labels):  # skip background
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area_pixels = stats[i, cv.CC_STAT_AREA]
        area_um2 = area_pixels * (micron_per_pixel ** 2)
        (cX, cY) = centroids[i]

        # Skip small components
        if area_pixels < 20 or w < 5 or h < 5:
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
        areas_um2_list.append(area_um2)  # Append area (µm^2) for histogram/analysis
        

        # Draw longest line in blue
        if longest_pair is not None:
            cv.line(output, (longest_pair[0][1], longest_pair[0][0]), (longest_pair[1][1], longest_pair[1][0]), (255, 0, 0), 2)
        # Draw bounding box
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv.circle(output, (int(cX), int(cY)), 3, (0, 255, 0), -1)
        cv.putText(output, f"ID {i}", (int(x), int(y) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 0), 1, cv.LINE_AA)

        # Add fibril data to the list
        all_data.append({
            'FIBRIL #': f'Fibril {number}',
            'AREA': round(area_um2, 2),
            'LENGTH': round(longest_line_um, 2)
        })
        number += 1
    
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
    if areas_um2_list:
        plt.figure(figsize=(8, 6))
        plt.hist(areas_um2_list, bins=20, color='green', edgecolor='black', alpha=0.7)
        plt.title("Histogram of Area")
        plt.xlabel("Area (µm^2)")
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("No component areas to plot histogram.")
    if areas_um2_list and lengths_um:
        plt.figure(figsize=(8, 6))
        plt.scatter(areas_um2_list, lengths_um, c='purple', alpha=0.7, edgecolors='black')
        plt.title("Scatter Plot of Fibril Area vs Length")
        plt.xlabel("Area (µm^2)")
        plt.ylabel("Length (µm)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show()
    else:
        print("No data to plot scatter plot.")

# Create DataFrame and save to Excel
df = pd.DataFrame(all_data)
df.to_excel("fibril_analysis_output.xlsx", index=False)
print("Data saved to fibril_analysis_output.xlsx")

# Print files that need manual processing
print(f'\nFiles that need manual processing: {reprocess}')