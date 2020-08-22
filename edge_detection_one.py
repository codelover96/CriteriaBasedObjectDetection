import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# read image
filepath = "G:/Programming/Python/Edge Detection/"
filename = "L1C_T34SGJ_A017522_20200714T091738.tif"
img = cv2.imread(filepath+filename, 0)
# Find edge with Canny edge detection
edges = cv2.Canny(img, 50, 150)

# display results
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

destination_path = "G:/Programming/Python/Edge Detection/"
cv2.imwrite(os.path.join(destination_path, "edges_"+filename), edges)

plt.show()
