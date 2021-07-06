import cv2
from pathlib import Path
import os

image_path = Path("G:/Programming/Python/Edge Detection/")
image_name = "cropped_L1C_T34SGG_A026502_20200719T091703.tif"

originalImage = cv2.imread(os.path.join(image_path, image_name))
grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
grayImage = cv2.medianBlur(grayImage, 3)

(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY)

th2 = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

cv2.imshow('Black white image', blackAndWhiteImage)
cv2.imshow('Original image', originalImage)
cv2.imshow('Gray image', grayImage)
cv2.imshow('Mean thresholding', th2)
cv2.imshow('Gaussian thresholding', th3)

cv2.waitKey(0)
cv2.destroyAllWindows()
