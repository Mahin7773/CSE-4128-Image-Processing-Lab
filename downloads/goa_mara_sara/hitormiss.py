import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

image = cv2.imread('hitmiss.jpeg', cv2.IMREAD_GRAYSCALE)
r, image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
image_inv = cv2.bitwise_not(image)
s1 = np.array([[0, 0, 0],
               [1, 1, 0],
               [1, 0, 0]], np.uint8)
s2 = np.array([[0, 1, 1],
               [0, 0, 1],
               [0, 0, 1]], np.uint8)
s3 = np.array([[1, 1, 1],
               [0, 1, 0],
               [0, 1, 0]], np.uint8)

w = np.ones((3, 3), np.uint8)
s1 = s1 * 255
print(s1)
s2 = s2 * 255
print(s2)
s3 = s3 * 255
print(s3)
w = w * 255
print(w)

s1 = cv2.resize(s1, (150, 150), interpolation=cv2.INTER_NEAREST)
s2 = cv2.resize(s2, (150, 150), interpolation=cv2.INTER_NEAREST)
s3 = cv2.resize(s3, (150, 150), interpolation=cv2.INTER_NEAREST)
w = cv2.resize(w, (150, 150), interpolation=cv2.INTER_NEAREST)
temp1 = cv2.erode(image, s1, iterations=1)
temp2 = cv2.erode(image_inv, (w - s1), iterations=1)
temp11 = cv2.bitwise_and(temp1, temp2)

temp3 = cv2.erode(image, s2, iterations=1)
temp4 = cv2.erode(image_inv, (w - s2), iterations=1)
temp22 = cv2.bitwise_and(temp3, temp4)

temp5 = cv2.erode(image, s3, iterations=1)
temp6 = cv2.erode(image_inv, (w - s3), iterations=1)
temp33 = cv2.bitwise_and(temp5, temp6)

out = cv2.bitwise_or(temp11, temp22)
out = cv2.bitwise_or(out, temp33)

cv2.imshow("input", image)
cv2.imshow("inv_input", image_inv)
cv2.imshow("temp11", temp11)
cv2.imshow("temp22", temp22)
cv2.imshow("temp33", temp33)
cv2.imshow("out", out)

cv2.waitKey(0)
cv2.destroyAllWindows()
