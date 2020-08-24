import cv2
import numpy as np
import matplotlib.pyplot as plt

filename = '00fa68a59'
file_ext = '.jpg'
original_image = cv2.imread(filename + file_ext)
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
low = 100
upper = 200
edge = cv2.Canny(image, low, upper)

contours, h = cv2.findContours(edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(image, [hull], 0, (0, 255, 0), 2)


# plt.imshow(image)
# plt.show()
cv2.imwrite('edges-'+str(low)+'-'+str(upper)+'-contours-'+filename+file_ext, image)
