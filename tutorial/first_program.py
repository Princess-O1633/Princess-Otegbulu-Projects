from skimage import io
from skimage.util import img_as_float
import numpy as np
import matplotlib.pyplot as plt

# Load and display an image using skimage

my_image = io.imread('IP_img1.png')

#Print image array and its min/max values using 0-255
print(my_image)
print(f" My image {my_image.min()}, {my_image.max()}")
plt.imshow(my_image)
plt.show()

# Draw a rectangle on the image
my_image[10:500, 10:500, :] = [255, 0, 130, 255]
plt.imshow(my_image)
plt.show()

# Convert the image to float and display it
my_float_image = img_as_float(my_image)
print(f" My float image {my_float_image.min()}, {my_float_image.max()}")


random_image = np.random.random([500,500])
plt.imshow(random_image)
plt.show()
print(f" Random Image {random_image.min()}, {random_image.max()}")

dark_image = my_image * 0.5
plt.imshow(dark_image)
plt.show()