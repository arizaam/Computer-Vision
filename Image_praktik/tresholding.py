import numpy as np
import os
import cv2

imagePath = os.path.join(".","gng.png")

image = cv2.imread(imagePath)
cv2.imshow("Original", image)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(image, (5,5), 0)
cv2.imshow("image",image)

(T, thresh) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold Binary", thresh)

(T, threshInv) = cv2.threshold(blurred, 155, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Threshold Binary Inverse", threshInv)

cv2.imshow("Mount", cv2.bitwise_and(image, image, mask =threshInv))
cv2.waitKey(0)