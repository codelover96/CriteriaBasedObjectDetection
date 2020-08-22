import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# get all filenames from 'images' folder
for root, dirs, files in os.walk("./images"):
    for filename in files:
        print(filename)
        # read image
        source_path = 'G:/Programming/python/Edge Detection/images/'
        img = cv2.imread(source_path + filename, 0)
        # Find edge with Canny edge detection
        edges = cv2.Canny(img, 100, 200)
        # display results
        # plt.subplot(121), plt.imshow(img, cmap='gray')
        # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        # plt.imshow(edges, cmap='gray')

        destination_path = "G:/Programming/Python/Edge Detection/images/edges/"
        cv2.imwrite(os.path.join(destination_path, filename), edges)
        # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        # plt.show()
