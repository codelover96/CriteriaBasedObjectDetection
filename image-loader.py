import os
import cv2
for root, dirs, files in os.walk('./images'):
    for filename in files:
        print(filename)
        # print(type(filename))
        print('G:/Programming/python/Edge Detection/images' + filename)
        img = cv2.imread('G:/Programming/python/Edge Detection/images/' + filename, 0)
        cv2.imshow('', img)
        cv2.waitKey(0)
