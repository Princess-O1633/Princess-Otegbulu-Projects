import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import cv2
from math import gcd, sqrt
from matplotlib.widgets import RectangleSelector

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='File to be analyzed')
parser.add_argument('-s', '--scale', help='Scale in microns')
parser.add_argument('-a', '--area', help='Area threshold', type=int, default=3)  # fibril areas under this threshold are ignored
parser.add_argument('-t', '--threshold', help='Background threshold', type=int, default=120)  # threshold separating fibrils from background
args = parser.parse_args()
file_path = args.file
scale = args.scale
area_threshold = args.area
threshold = args.threshold

img_path = os.path.join("", file_path)
with Image.open(img_path) as img:
    arr = np.array(img)

# Convert to grayscale
arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
shape = arr.shape

fig, ax = plt.subplots()

scale_location = None

def onselect(eclick, erelease):
    global scale_location
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"Selected scale coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    scale_location = [(int(y1), int(x1)), (int(y2), int(x2))]

if scale is not None:
    ax.imshow(arr, cmap="gray")
    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], spancoords='pixels')
    plt.show()

plt.imshow(arr, cmap="gray")
plt.show()

# Binarize image: background=255, fibrils <255
_, binary = cv2.threshold(arr, threshold, 255, cv2.THRESH_BINARY_INV)

plt.imshow(binary, cmap="gray")
plt.show()

if scale_location is not None:
    (y1, x1), (y2, x2) = scale_location
    binary[y1:y2+1, x1:x2+1] = 0
    
if scale is not None:
    print("Scale location", scale_location)
    scale_length = 0
    for row in range(scale_location[0][0], scale_location[1][0] + 1):
        length = 0
        for col in range(scale_location[0][1], scale_location[1][1] + 1):
            if binary[row][col] > 0:
                length += 1
        scale_length = max(scale_length, length)
    print("Scale's length is", scale_length, "pixels")

# Find connected components (fibrils)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

print(f"Found {num_labels - 1} fibril candidates (excluding background)")

def get_direction(dy, dx):
    g = gcd(dy, dx)
    return (dy // g, dx // g)

def find_farthest_points(points):
    max_distance = 0
    farthest_pair = (None, None)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]
            distance = sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
            if distance > max_distance:
                max_distance = distance
                farthest_pair = (p1, p2)
    return farthest_pair, max_distance

def findLongestLine(component_mask):
    # Find all points in component
    points = np.column_stack(np.where(component_mask > 0))
    if len(points) == 0:
        return (None, None), 0

    max_length = 0
    best_pair = (None, None)

    # Brute force all pairs (can be slow for large fibrils)
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]
            dist = np.linalg.norm(p1 - p2)
            if dist > max_length:
                max_length = dist
                best_pair = (p1, p2)

    return best_pair, int(max_length)

use_pixels = True

plot_pixels = []

sizes = []
lengths = []

sizes_adjusted = []
lengths_adjusted = []

if scale is not None:
    use_pixels = False
    scale = int(scale)
    print("Scale is", scale, "micrometers")

scale = 1
scale_length = 1

for label in range(1, num_labels):  # skip label 0 (background)
    area = stats[label, cv2.CC_STAT_AREA]
    if area < area_threshold:
        continue

    left = stats[label, cv2.CC_STAT_LEFT]
    top = stats[label, cv2.CC_STAT_TOP]
    width = stats[label, cv2.CC_STAT_WIDTH]
    height = stats[label, cv2.CC_STAT_HEIGHT]

    component_mask = (labels == label).astype(np.uint8) * 255

    print(f"Detected fibril {label} at ({top}, {left}) with area {area}", end=" ")

    pair, length = findLongestLine(component_mask)

    if pair[0] is not None and pair[1] is not None:
        # Store points for plotting: note y=rows, x=cols
        plot_pixels.append([[pair[0][1], pair[1][1]], [pair[0][0], pair[1][0]]])
        print(f"and length {length}")
    else:
        print("and length 0")

    sizes.append(area)
    lengths.append(length)

    if not use_pixels:
        adjusted_area = area * (scale / scale_length) ** 2
        sizes_adjusted.append(adjusted_area)
        adjusted_length = length * scale / scale_length
        lengths_adjusted.append(adjusted_length)

print("Number of fibrils", len(sizes))

if use_pixels:
    print("Sizes of all fibrils in pixels:", sizes)
else:
    print("Sizes of all fibrils in micrometers:", sizes_adjusted)

if len(sizes) > 0:
    if use_pixels:
        print("Average pixel area:", sum(sizes) / len(sizes))
    else:
        print("Average micrometer area:", sum(sizes_adjusted) / len(sizes_adjusted))

if use_pixels:
    print("Lengths of all fibrils in pixels:", lengths)
else:
    print("Lengths of all fibrils in micrometers:", lengths_adjusted)

if len(lengths) > 0:
    if use_pixels:
        print("Average fibril length in pixels:", sum(lengths) / len(lengths))
    else:
        print("Average fibril length in micrometers:", sum(lengths_adjusted) / len(lengths_adjusted))


def displayHistogram(unit, start, end, lengths):
    bins = ((end - start) // unit)  # number of ranges
    plt.figure()
    plt.hist(lengths, bins=bins, range=[start, end], align="mid")
    plt.xlabel(f"Lengths of Fibrils in {'Pixels' if use_pixels else 'Microns'}")
    plt.ylabel("Occurrences")
    plt.title("Occurrences of each range of fibril lengths")


displayHistogram(1, 0, max(lengths), lengths)

plt.figure()
plt.imshow(arr, cmap="gray")
for point in plot_pixels:
    plt.plot(point[0], point[1], color="red")
plt.show()
