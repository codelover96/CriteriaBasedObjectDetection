import cv2
import numpy as np


def ORB_detector(new_image, image_template):
    print("orb called")
    # Function that compares input image to template
    # It then returns the number of ORB matches between them

    image1 = new_image

    # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB_create(1000, 1.2)
    print("orb values is", orb)
    # Detect keypoints of original image
    print(0)
    (kp1, des1) = orb.detectAndCompute(image1, None)
    print(1)
    # Detect keypoints of rotated image
    (kp2, des2) = orb.detectAndCompute(image_template, None)
    print(2)
    # Create matcher
    # Note we're no longer using Flannbased matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    print(3)
    # Do matching
    matches = bf.match(des1, des2)
    print(4)
    # Sort the matches based on distance.  Least distance
    # is better
    matches = sorted(matches, key=lambda val: val.distance)
    print("orb returning")
    return len(matches)


# Load our image template, this is our reference image
image_template = cv2.imread('1_cropped_L1C_T34SGG_A026502_20200719T091703.tif')

# Get webcam images
# ret, frame = cap.read()
original_image = cv2.imread("cropped_L1C_T34SGG_A026502_20200719T091703.tif")
image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
# Get height and width of webcam frame
height, width = image.shape[:2]

# Define ROI Box Dimensions (Note some of these things should be outside the loop)
top_left_x = width / 3
top_left_y = (height / 2) + (height / 4)
bottom_right_x = (width / 3) * 2
bottom_right_y = (height / 2) - (height / 4)

# Draw rectangular window for our region of interest
cv2.rectangle(image, int(top_left_x), int(bottom_right_y), 255, 3)

# Crop window of observation we defined above
cropped = image[int(bottom_right_y):int(top_left_y), int(top_left_x):int(bottom_right_x)]

# Flip image orientation horizontally
image = cv2.flip(image, 1)

# Get number of ORB matches
matches = ORB_detector(cropped, image_template)
print("orb ended")
# Display status string showing the current no. of matches
output_string = "Matches = " + str(matches)
cv2.putText(image, output_string, (50, 450), cv2.FONT_HERSHEY_COMPLEX, 2, (250, 0, 150), 2)

# Our threshold to indicate object deteciton
# For new images or lightening conditions you may need to experiment a bit
# Note: The ORB detector to get the top 1000 matches, 350 is essentially a min 35% match
threshold = 350

# If matches exceed our threshold then object has been detected
if matches > threshold:
    cv2.rectangle(image, int(top_left_x), int(bottom_right_y), (0, 255, 0), 3)
    cv2.putText(image, 'Object Found', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

cv2.imshow('Object Detector using ORB', image)

cv2.waitKey(0)
