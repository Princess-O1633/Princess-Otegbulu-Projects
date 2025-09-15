# Load Libraries  python3 main.py -f "photos\Day 16 Alpha-NP 50x Partial Coaxial upper left.jpg" -s 50
from skimage import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import cv2
from math import sqrt, gcd
from matplotlib.widgets import RectangleSelector
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import skeletonize
from collections import defaultdict

# CLI Parser
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', required=True, help='File to be analyzed')
parser.add_argument('-s', '--scale', required=True, help='Scale in microns')
parser.add_argument('-a', '--area', help='Area threshold', type=int, default=3)
parser.add_argument('-t', '--threshold', help='Background threshold', type=int, default=80)

# Parse arguments
args = parser.parse_args()
file_path = args.file
scale = args.scale
area_threshold = args.area
threshold = args.threshold

# Load and preprocess image
img_path = io.imread(file_path)
with Image.open(img_path) as img:
    arr = np.array(img)

# Convert to grayscale
arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
shape = arr.shape

# Interactive Scale Selection
fig, ax = plt.subplots()
scale_location = None

def onselect(eclick, erelease):
    global scale_location
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    print(f"Selected scale coordinates: ({x1}, {y1}) to ({x2}, {y2})")
    scale_location = [(int(y1), int(x1)), (int(y2), int(x2))]

if scale != None:
    ax.imshow(arr, cmap="gray")
    rect_selector = RectangleSelector(ax, onselect, useblit=True, button=[1], spancoords='pixels')
    plt.title('Select Region')
    plt.show()

plt.title('Gray scale')
plt.imshow(arr, cmap="gray")
plt.show()

# Create binary image (invert so fibrils are white, background is black)
binary = np.zeros_like(arr)
binary[arr <= threshold] = 255  # Fibrils become white (255)
binary[arr > threshold] = 0     # Background becomes black (0)



if scale != None:
    print("Scale location", scale_location)
    scale_length = 0 # Initializes longest row length

    # scale_location[0][0] = Starting ROW | scale_location[1][0] + 1 = Ending Row
    for row in range(scale_location[0][0], scale_location[1][0] + 1): # Loops over each row in the selected scale bar region
        length = 0
        for col in range(scale_location[0][1], scale_location[1][1] + 1):
            if arr[row][col] < 255:
                length += 1
        scale_length = max(scale_length, length)
    print("Scale's length is", scale_length, "pixels")

if scale_location:
    # Coordinates
    y1, x1 = scale_location[0]
    y2, x2 = scale_location[1]
    arr[y1:y2 + 1, x1:x2 + 1] = 255 # Turns them white to prevent being detected as fibrils

alreadySeen = set() # Pixels alr counted

def findFibrilData(startingPoint):
    # helper function to find the area and boundaries of a fibril
    if startingPoint in alreadySeen: # If alr seen STOP
        return 0, [], [0, 0], [0, 0]
    if arr[startingPoint[0]][startingPoint[1]] == 255: # If it is part of bg STOP
        return 0, [], [0, 0], [0, 0]
    
    fibril = [[255 for i in range(shape[1])] for j in range(shape[0])] # creates blank img same size as arr

    top_left = [len(arr), len(arr[0])] # tuple of top-left corner of the rectangular region containining the fibril
    bottom_right = [0, 0] # tuple of the bottom-right corner of the region containing the fibril
    totalPixelCount = 0 # area of fibril

    queue = [startingPoint]
    while len(queue) > 0:
        # starting from initial point, enqueue surrounding points. dequeue the points and continue.
        # eventually, the boundaries of the fibril will be found
        point = queue.pop()

        if point in alreadySeen:
            continue
        alreadySeen.add(point)
        

        if (point[0] - 1 > 0 and (arr[point[0] - 1][point[1]] < 255)):
            queue.append((point[0] - 1, point[1]))
            fibril[point[0] - 1][point[1]] = 0
        if (point[0] - 1 > 0 and point[1] - 1 > 0 and (arr[point[0] - 1][point[1] - 1] < 255)):
            queue.append((point[0] - 1, point[1] - 1))
            fibril[point[0] - 1][point[1] - 1] = 0
        if (point[0] > 0 and point[1] - 1 > 0 and (arr[point[0]][point[1] - 1] < 255)):
            queue.append((point[0], point[1] - 1))
            fibril[point[0]][point[1] - 1] = 0
        if (point[0] + 1 < len(arr) and (arr[point[0] + 1][point[1]] < 255)):
            queue.append((point[0] + 1, point[1]))
            fibril[point[0] + 1][point[1]] = 0
        if (point[1] + 1 < len(arr[0]) and (arr[point[0]][point[1] + 1] < 255)):
            queue.append((point[0], point[1] + 1))
            fibril[point[0]][point[1] + 1] = 0
        if (point[0] + 1 < len(arr) and point[1] + 1 < len(arr[0]) and (arr[point[0] + 1][point[1] + 1] < 255)):
            queue.append((point[0] + 1, point[1] + 1))
            fibril[point[0] + 1][point[1] + 1] = 0
        if (point[0] + 1 < len(arr) and point[1] - 1 > 0 and (arr[point[0] + 1][point[1] - 1] < 255)):   
            queue.append((point[0] + 1, point[1] - 1))
            fibril[point[0] + 1][point[1] - 1] = 0
        if (point[0] - 1 > 0 and point[1] + 1 < len(arr[0]) and (arr[point[0] - 1][point[1] + 1] < 255)):
            queue.append((point[0] - 1, point[1] + 1))
            fibril[point[0] - 1][point[1] + 1] = 0

        if arr[point[0]][point[1]] < 255: # if the point dequeued belongs to the fibril
            top_left[0] = min(point[0], top_left[0]) # update top-left
            top_left[1] = min(point[1], top_left[1])

            bottom_right[0] = max(point[0], bottom_right[0]) # update bottom-right
            bottom_right[1] = max(point[1], bottom_right[1])

            totalPixelCount += 1 # increase area
    
    if totalPixelCount < area_threshold:
        return 0, [], [0, 0], [0, 0]

    if (top_left != [len(arr), len(arr[0])] and bottom_right != [0, 0]):        
        print("Detected fibril from (" + str(top_left[0])  + ", " + str(top_left[1]) + ") to (" + str(bottom_right[0]) + ", " + str(bottom_right[1]) + ") with area", totalPixelCount, end=" ")

    return totalPixelCount, fibril, top_left, bottom_right

def get_direction(dy, dx):
    # helper function to simplify the direction tuple by dividing by GCD of the two numbers
    g = gcd(dy, dx)
    return (dy // g, dx // g)

def find_farthest_points(points):
    max_distance = 0
    farthest_pair = (None, None)

    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            p1, p2 = points[i], points[j]
            distance = sqrt((p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)  # Euclidean distance
            if distance > max_distance:
                max_distance = distance
                farthest_pair = (p1, p2)

    return farthest_pair, max_distance

def findLongestLine(matrix, top_left, bottom_right):
    # helper function to find the longest line passing through a fibril
    if matrix == []:
        return 0

    pair = ()
    l = 0

    for i1 in range(top_left[0], bottom_right[0] + 1):
        for j1 in range(top_left[1], bottom_right[1] + 1):
            if matrix[i1][j1] < 255:
                # (i1, j1) is the base point
                directions = defaultdict(list)
                for i2 in range(top_left[0], bottom_right[0] + 1):
                    for j2 in range(top_left[1], bottom_right[1] + 1):
                        # (i2, j2) are the secondary points
                        if matrix[i2][j2] < 255 and [i1, j1] == [i2, j2]:
                            # if base point is secondary point
                            directions[(0, 0)] = [[i1, j1]]
                            # direction of (0,0) should be unique, since only one way this direction is computed, and that is if base point = secondary point
                        elif matrix[i2][j2] < 255 and [i1, j1] != [i2, j2]:

                            # Get the direction between (i1, j1) and (i2, j2)
                            dy, dx = i2 - i1, j2 - j1
                            direction = get_direction(dy, dx) # simplify direction
                            
                            if direction not in directions: # if not already in directions dictionary
                                directions[direction] = [[i1, j1]]  # initialize with length 1, the current secondary point

                            # Extend the line
                            directions[direction].append([i2, j2])

                pa = None
                length = 0

                for k in directions.keys():
                    p, distance = find_farthest_points(directions[k])
                    if pa == None or distance > length:
                        length = distance
                        pa = p

                if length > l:
                    l = length
                    pair = pa

    return pair, int(l)

use_pixels = True

def displayHistogram(unit, start, end, lengths):
    # helper function to display histogram
    bins = ((end - start) // unit)  # number of ranges
    plt.figure()
    plt.hist(lengths, bins=bins, range=[start, end], align="mid")
    plt.xlabel("Lengths of Fibrils in Pixels")
    plt.ylabel("Occurrences")
    plt.title("Occurrences of each range of fibril lengths")

plot_pixels = []

sizes = []
lengths = []

sizes_adjusted = []
lengths_adjusted = []
if scale != None:
    use_pixels = False
    scale = int(scale)
    print("Scale is", scale, "micrometers")

scale = 1
scale_length = 1
for row in range(len(arr)):
    for col in range(len(arr[0])):
        # for every point (row, col) in the frames
        if scale_location != None and scale_location[0][0] <= row <= scale_location[1][0] and scale_location[0][1] <= col <= scale_location[1][1]:
            # if point is contained inside the scale, skip it
            continue # pixel is in scale ROI
        fibrilArea, fibril, top_left, bottom_right = findFibrilData((row, col))
       
        if fibrilArea > 0:
            sizes.append(fibrilArea)
            if not use_pixels:
                fibrilArea = fibrilArea * (scale / scale_length)**2
                sizes_adjusted.append(fibrilArea)

        if fibril != []:
            pair, length = findLongestLine(fibril, top_left, bottom_right)

            plot_pixels.append([[pair[0][1], pair[1][1]], [pair[0][0], pair[1][0]]])
            print("and length", length) # debug line
            lengths.append(length)

            if not use_pixels:
                length = length * scale / scale_length
                lengths_adjusted.append(length)

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


displayHistogram(1, 0, max(lengths), lengths)

plt.figure()
plt.imshow(arr, cmap="gray")
for point in plot_pixels:
    plt.plot(point[0], point[1], color="red")
plt.show()