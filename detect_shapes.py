from pyimagesearch.shapedetector import ShapeDetector
import imutils
import cv2
import numpy as np

filepath = "G:/Programming/Python/Edge Detection/"
filename = "004_green_square.jpg"
original_image = cv2.imread(filepath + filename)
image = original_image.copy()
b, g, r = cv2.split(original_image)

# resized = imutils.resize(image, width=300)
# ratio = image.shape[0] / float(resized.shape[0])

resized = image
ratio = 1

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(gray, 3)
b = cv2.medianBlur(b, 3)
b_shapes = b.copy()

g = cv2.medianBlur(g, 3)
g_shapes = g.copy()

r = cv2.medianBlur(r, 3)
r_shapes = r.copy()


edges_b = cv2.Canny(b, 100, 150)
edges_g = cv2.Canny(g, 100, 150)
edges_r = cv2.Canny(r, 100, 150)
edges_bgr = cv2.Canny(gray, 100, 150)

# find contours in the thresholded image and initialize the shape detector
cnts_bgr = cv2.findContours(edges_bgr, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts_bgr = imutils.grab_contours(cnts_bgr)
# blue
cnts_b = cv2.findContours(edges_b, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts_b = imutils.grab_contours(cnts_b)
# green
cnts_g = cv2.findContours(edges_g, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts_g = imutils.grab_contours(cnts_g)
# red
cnts_r = cv2.findContours(edges_r, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cnts_r = imutils.grab_contours(cnts_r)

sd = ShapeDetector()
print("Available shapes : triangle, rectangle, square, pentagon, ellipse, circle, all." + " For all type 'all'")
requested = input("Give shape to detect: ")

if not(requested.isalpha()):
    print("Input contains digits. Exiting...")
    exit(1)

hull_list = []
# approx_image = image.copy()
counter = 0
# loop over the contours
# original and edges_bgr
for c in cnts_bgr:
    counter += 50
    # compute the center of the contour, then detect the name of the shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
    cY = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
    shape, vertices, approx = sd.detect(c)

    # multiply the contour (x, y)-coordinates by the resized ratio
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    area = cv2.contourArea(c, True)
    if abs(area) < 50:
        continue
    hull = cv2.convexHull(c)
    hull_list.append(hull)
    # print(len(hull))
    # draw a rotated bounding box for contour c
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # im = cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
    # cv2.circle(image, (cX, cY), 1, (0, 0, 255), 1)

    if requested == "all":
        # print all
        cv2.drawContours(image, [hull], 0, (counter, 255, 255 - counter), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if requested != "all" and requested == shape:
        cv2.drawContours(image, [hull], 0, (counter, 255, 255 - counter), 2)
        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


# loop through contours in blue channel
# print("contours of blue channel")
# print(cnts_b)
print("is type of " + str(type(cnts_b)))

for c in cnts_b:
    counter += 50
    # compute the center of the contour, then detect the name of the shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
    cY = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
    shape, vertices, approx = sd.detect(c)

    # multiply the contour (x, y)-coordinates by the resized ratio
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    area = cv2.contourArea(c, True)
    if abs(area) < 50:
        continue
    hull = cv2.convexHull(c)
    hull_list.append(hull)
    # print(len(hull))
    # draw a rotated bounding box for contour c
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # im = cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
    # cv2.circle(image, (cX, cY), 1, (0, 0, 255), 1)

    if requested == "all":
        # print all
        cv2.drawContours(b_shapes, [hull], 0, (counter, 255, 255 - counter), 2)
        cv2.putText(b_shapes, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if requested != "all" and requested == shape:
        cv2.drawContours(b_shapes, [hull], 0, (counter, 255, 255 - counter), 2)
        cv2.putText(b_shapes, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# loop through contours in green channel
for c in cnts_g:
    counter += 50
    # compute the center of the contour, then detect the name of the shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
    cY = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
    shape, vertices, approx = sd.detect(c)

    # multiply the contour (x, y)-coordinates by the resized ratio
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    area = cv2.contourArea(c, True)
    if abs(area) < 50:
        continue
    hull = cv2.convexHull(c)
    hull_list.append(hull)
    # print(len(hull))
    # draw a rotated bounding box for contour c
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # im = cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
    # cv2.circle(image, (cX, cY), 1, (0, 0, 255), 1)

    if requested == "all":
        # print all
        cv2.drawContours(g_shapes, [hull], 0, (counter, 255, 255 - counter), 2)
        cv2.putText(g_shapes, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if requested != "all" and requested == shape:
        cv2.drawContours(g_shapes, [hull], 0, (counter, 255, 255 - counter), 2)
        cv2.putText(g_shapes, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# loop through contours in red channel
for c in cnts_r:
    counter += 50
    # compute the center of the contour, then detect the name of the shape using only the contour
    M = cv2.moments(c)
    cX = int((M["m10"] / (M["m00"] + 1e-7)) * ratio)
    cY = int((M["m01"] / (M["m00"] + 1e-7)) * ratio)
    shape, vertices, approx = sd.detect(c)

    # multiply the contour (x, y)-coordinates by the resized ratio
    # then draw the contours and the name of the shape on the image
    c = c.astype("float")
    c *= ratio
    c = c.astype("int")
    area = cv2.contourArea(c, True)
    if abs(area) < 50:
        continue
    hull = cv2.convexHull(c)
    hull_list.append(hull)
    # print(len(hull))
    # draw a rotated bounding box for contour c
    # rect = cv2.minAreaRect(c)
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)
    # im = cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # cv2.drawContours(image, [c], -1, (0, 255, 0), 1)
    # cv2.circle(image, (cX, cY), 1, (0, 0, 255), 1)

    if requested == "all":
        # print all
        cv2.drawContours(r_shapes, [hull], 0, (counter, 255, 255 - counter), 2)
        cv2.putText(r_shapes, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    if requested != "all" and requested == shape:
        cv2.drawContours(r_shapes, [hull], 0, (counter, 255, 255 - counter), 2)
        cv2.putText(r_shapes, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


cv2.imshow("blue", b)
cv2.imshow("green", g)
cv2.imshow("red", r)
cv2.imshow("Original image", original_image)
cv2.imshow("Gray image", gray)
cv2.imshow("image with shapes", image)
cv2.imshow("blue with shapes", b_shapes)
cv2.imshow("green with shapes", g_shapes)
cv2.imshow("red with shapes", r_shapes)
cv2.imshow("blue edges", edges_b)
cv2.imshow("green edges", edges_g)
cv2.imshow("red edges", edges_r)
cv2.waitKey(0)
