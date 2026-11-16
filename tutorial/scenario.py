import matplotlib.pyplot as plt 
from skimage import io, restoration
from skimage.morphology import disk
from skimage. filters.rank import entropy
import numpy as np
img = io.imread('D2-aNP 40x-Full coaxial.jpg', as_gray=True)
entropy_img = entropy(img, disk(3))
plt.imshow(entropy_img, cmap='gray')
plt.title('Entropy Image')
plt.show()

from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu
thresh = threshold_otsu(entropy_img)

binary = entropy_img <= thresh
plt.imshow(binary, cmap='gray')
plt.title('Binary Image')   
plt.show()
print("The percent white region is: ", (np.sum(binary == 1)*100)/(np.sum(binary == 0) + np.sum(binary == 1))) 
