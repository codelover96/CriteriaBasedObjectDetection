import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
import os

image_path = Path("G:/Programming/Python/Edge Detection/")
image_name = "cr_00000011.jpg"
# read the image
image = cv2.imread(os.path.join(image_path, image_name))

# convert to grayscale
grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# perform edge detection
edges = cv2.Canny(grayscale, 20, 150)

# detect lines in the image using Hough lines technique
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 60, np.array([]), 50, 5)

# iterate over the output lines and draw them
for line in lines:
    for x1, y1, x2, y2 in line:
        cv2.line(image, (x1, y1), (x2, y2), (20, 220, 20), 3)


# show the image
plt.imshow(image)
plt.show()
