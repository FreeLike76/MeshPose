import cv2
import numpy as np

from modules import utils
from modules.features import extractor, detectors, descriptors

image = cv2.imread("data/test.png")
image = utils.resize_image(image, 512)

print("detectors", detectors.DETECTORS)
print("descriptors", descriptors.DESCRIPTORS)

pipeline = extractor.ClassicalExtractor(detector="SIFT", descriptor="SIFT", verbose=True)

kp, des = pipeline(image)

img = cv2.drawKeypoints(image, kp, None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("features", img)
cv2.waitKey(0)