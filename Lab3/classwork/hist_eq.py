# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:04:54 2023

@author: mimigician
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('eye.png', 0)

cv2.imshow("Input", img)

plt.figure(figsize=(10, 4))

plt.subplot(1, 4, 1)
plt.hist(img.ravel(), 255, [0, 255])
plt.title("Input Image Histogram")
#plt.show()

equalized = np.zeros_like(img)

img_h = img.shape[0]
img_w = img.shape[1]

total_pixel = img_h * img_w

histogram = np.zeros(256)
for i in range(img_h):
    for j in range(img_w):
        histogram[img[i][j]] += 1

pdf = histogram / total_pixel

cdf = pdf
rounded_cdf = pdf

sum = 0.0
for i in range(256):
    sum += pdf[i]
    cdf[i] = sum
    rounded_cdf[i] = round(255 * cdf[i])

plt.subplot(1, 4, 2)
plt.title("image cdf")
plt.plot(cdf)
#plt.show()

# print(s)
for i in range(img_h):
    for j in range(img_w):
        equalized[i][j] = rounded_cdf[img[i][j]]

histogram2 = np.zeros(256)
for i in range(img_h):
    for j in range(img_w):
        histogram2[equalized[i][j]] += 1

pdf2 = histogram2 / total_pixel

cdf2 = pdf2
rounded_cdf2 = pdf2

sum = 0.0
for i in range(256):
    sum += pdf2[i]
    cdf2[i] = sum
    rounded_cdf2[i] = round(255 * cdf2[i])

plt.subplot(1, 4, 4)
plt.plot(cdf2)
plt.title("output cdf")
#plt.show()

plt.subplot(1, 4, 3)
plt.hist(equalized.ravel(), 255, [0, 255])
plt.title("Output Image Histogram")

cv2.imshow("Output", equalized)

plt.show()

#cv2.imshow("Output", equalized)

cv2.waitKey(0)
cv2.destroyAllWindows()
