import numpy as np
import cv2
from pathlib import Path
import os

image_path = Path("G:/Programming/Python/Edge Detection/")
output_path = Path("G:/Programming/Python/Edge Detection/color_reduced/")
image_name = "cropped_L1C_T34SGG_A026502_20200719T091703.tif"
img = cv2.imread(os.path.join(image_path, image_name))
Z = img.reshape((-1, 3))

# convert to np.float32
Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
K = 8
attempts = 1000
ret, label, center = cv2.kmeans(Z, K, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(img.shape)

cv2.imshow('res2', res2)
# cv2.imwrite(os.path.join(output_path, "cr_"+image_name), res2)

cv2.waitKey(0)
cv2.destroyAllWindows()
