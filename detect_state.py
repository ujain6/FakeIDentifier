import cv2
import numpy as np
import glob
import os
from scipy import ndimage, misc

#empty list to store template images
template_dict = []

#make a list of all template images from a directory
files1= glob.glob('id_templates/*.jpeg')
for curr_file in files1:
    image = cv2.imread(curr_file, 0)
    # assert image != None
    if image is not None:
        template_dict.append({curr_file: image})

test_image = cv2.imread('driver1.jpg')
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)


# loop for matching
for tmp in template_dict:

    temp = list(tmp.values())[0]
    (tH, tW) = temp.shape[::-1]

    # cv2.imshow("Template", tmp)
    # cv2.waitKey(500)
    # cv2.destroyAllWindows()
#
    result = cv2.matchTemplate(test_image, temp, cv2.TM_CCOEFF)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    threshold = max_val > 80000000.0 and max_val < 90000000.0


    if threshold:
        cv2.imshow("Template", temp)
        print(tmp.keys())
        cv2.waitKey(500)

        top_left = max_loc
        bottom_right = (top_left[0] + tW, top_left[1] + tH)

        cv2.rectangle(test_image, top_left, bottom_right, 255, 2)
        cv2.imshow('Result', test_image)
        cv2.waitKey(500)




# Show result
cv2.imshow('Result',test_image)
cv2.waitKey(0)
quit(0)