import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def min_max_normalize(image):
    h = image.shape[0]
    w = image.shape[1]
    mi = np.min(image)
    ma = np.max(image)
    output = np.zeros(image.shape, np.uint8)
    for i in range(0, h):
        for j in range(0, w):
            output[i][j] = ((image[i][j] - mi) / (ma - mi)) * 255
    return output


image = cv2.imread("Lena.jpg", cv2.IMREAD_GRAYSCALE)
kernel = np.zeros((5, 5), np.float32)

a = kernel.shape[0] // 2
b = kernel.shape[1] // 2
sigma = int(input("Enter the value of sigma:"))
for i in range(-a, a + 1):
    for j in range(-b, b + 1):
        temp = (i * i) + (j * j)
        temp /= (2 * sigma ** 2)
        temp = np.exp(-temp)
        temp /= (2 * 3.1416 * sigma ** 2)
        kernel[a + i][b + j] = temp

center_x = int(input("Enter the value of center x:"))
center_y = int(input("Enter the value of center y:"))

top_pad = kernel.shape[0] - center_x - 1
bottom_pad = kernel.shape[0] - top_pad - 1
left_pad = kernel.shape[1] - center_y - 1
right_pad = kernel.shape[1] - left_pad - 1
print(top_pad, bottom_pad, left_pad, right_pad)
image2 = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT)
output = np.zeros(image.shape, np.float32)
for i in range(top_pad, image2.shape[0] - bottom_pad):
    for j in range(left_pad, image2.shape[1] - right_pad):
        temp = 0
        for k in range(-top_pad, bottom_pad + 1):
            for l in range(-left_pad, right_pad + 1):
                temp += (image2[i + k][j + l] * kernel[center_x - k][center_y - l])
        temp /= kernel.sum()
        output[i - top_pad][j - left_pad] = temp

output = min_max_normalize(output)

cv2.imshow("input", image)
cv2.imshow("padded", image2)
cv2.imshow("Out", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
