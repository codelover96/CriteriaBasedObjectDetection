"""
OBJECT DETECTION WITH TEMPLATES

Need a source image and a template image.
The template image T is slided over the source image (as in 2D convolution),
and the program tries to find matches using statistics.
Several comparison methods are implemented in OpenCV.
It returns a grayscale image, where each pixel denotes how much does the
neighbourhood of that pixel match with template.

Once you got the result, you can use cv2.minMaxLoc() function
to find where is the maximum/minimum value. Take it as the top-left corner of the
rectangle and take (w,h) as width and height of the rectangle.
That rectangle can be drawn on the region of matched template.
"""
### Template matching, single object in an image.
# Multiple methods to see which one works best.

import cv2
import numpy as np
from matplotlib import pyplot as plt

img_rgb = cv2.imread('cr_cropped_L1C_T34SGG_A026502_20200719T091703.tif')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
template = cv2.imread('1_cropped_L1C_T34SGG_A026502_20200719T091703.tif', 0)
h, w = template.shape[::]

# methods available: ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
#            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

res = cv2.matchTemplate(img_gray, template, cv2.TM_SQDIFF)
# For TM_SQDIFF, Good match yields minimum value; bad match yields large values
# For all others it is exactly opposite, max value = good fit.
plt.imshow(res, cmap='gray')

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

top_left = min_loc  # Change to max_loc for all except for TM_SQDIFF
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img_gray, top_left, bottom_right, 255, 2)  # White rectangle with thickness 2.

cv2.imshow("Matched image", img_gray)
cv2.waitKey()
cv2.destroyAllWindows()
