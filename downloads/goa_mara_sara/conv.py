import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def min_max_normalization(image):
    h = image.shape[0]
    w = image.shape[1]
    mi = np.min(image)
    ma = np.max(image)
    out = np.zeros(image.shape, np.uint8)
    for i in range(0, h):
        for j in range(0, w):
            out[i][j] = ((image[i][j] - mi) / (ma - mi)) * 255
    return out


image = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)

filter = np.ones((5, 5), np.int8)
filter = filter / 25

image2 = cv2.copyMakeBorder(image, 2, 2, 2, 2, cv2.BORDER_CONSTANT)
a = filter.shape[0] // 2
b = filter.shape[1] // 2

print(image2.shape)

output = np.zeros(image.shape, np.float32)

for i in range(a, image2.shape[0] - a):
    for j in range(b, image2.shape[1] - b):
        temp = 0
        for k in range(-a, a + 1):
            for l in range(-b, b + 1):
                temp += filter[a - k][b - l] * image2[i + k][j + l]
        output[i - a][j - b] = temp

output = min_max_normalization(output)

cv2.imshow('input', image)
cv2.imshow('2nd image', image2)
cv2.imshow('output', output)

cv2.waitKey(0)
cv2.destroyAllWindows()
