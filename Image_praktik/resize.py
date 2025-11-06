import numpy as np
import os
import imutils
import cv2

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required = True,

# help = "Path to the image")
# args = vars(ap.parse_args())
imagePath = os.path.join(".", "gng.png")

image = cv2.imread(imagePath)

cv2.imshow("Original", image)

r = 150.0 / image.shape[1]
dim = (150, int(image.shape[0] * r))

resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("Resized (Width)", resized)
cv2.waitKey(0)