import cv2


# maybe make some changes here, so shape detector is more accurate on detecting different shapes We basically
# determine the shape of an object based on the number of vertices the Polygon approximation has The approximation is
# need a precision called epsilon. more here :
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features
# /py_contour_features.html#contour-approximation


class ShapeDetector:

    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01 * peri, True)
        # if the shape is a triangle, it will have  3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is a square or a rectangle

        elif len(approx) == 4:
            # compute the bounding box of the contour and use
            # the bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ration that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if 0.95 <= ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"
        elif 6 < len(approx) < 15:
            shape = "eclipse"
        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of shape
        return shape, len(approx), approx
