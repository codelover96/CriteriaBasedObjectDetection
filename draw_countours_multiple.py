import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# get all filenames from 'images' folder
for root, dirs, files in os.walk("./images/"):
    for filename in files:
        print(filename)
        source_path = 'G:/Programming/python/Edge Detection/images/'
        img = cv2.imread(source_path + filename)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        low = 50
        upper = 150
        edge = cv2.Canny(image, low, upper)

        contours, h = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for c in contours:
            hull = cv2.convexHull(c)
            cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)

        destination_path = "G:/Programming/Python/Edge Detection/edges-and-contours/"
        cv2.imwrite(os.path.join(destination_path, 'edg-' + str(low) + '-' + str(upper) + '-cont-' + filename),
                    edge)
