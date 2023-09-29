import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def min_max_normalization(image):
    h = image.shape[0]
    w = image.shape[1]
    mi = np.min(image)
    ma = np.max(image)
    out = np.zeros((image.shape), np.uint8)
    for i in range(0, h):
        for j in range(0, w):
            out[i][j] = ((image[i][j] - mi) / (ma - mi)) * 255
    return out


image = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)
sigma = int(input("Enter value of sigma of spatial domain:"))
sigma2 = int(input("Enter the value of sigma of range domain:"))
kernel = np.zeros((5 * sigma, 5 * sigma), np.float32)
a = kernel.shape[0] // 2
b = kernel.shape[1] // 2

for i in range(-a, a + 1):
    for j in range(-b, b + 1):
        temp = (i * i) + (j * j)
        temp /= (2 * sigma ** 2)
        temp = np.exp(-temp)
        temp /= (2 * 3.1416 * sigma ** 2)
        kernel[a - i][b - j] = temp

output = np.zeros((image.shape), np.float32)
image2 = cv2.copyMakeBorder(image, b, a, b, a, cv2.BORDER_CONSTANT)
image2 = image2 / 255
for i in range(a, image2.shape[0] - a):
    for j in range(b, image2.shape[1] - b):
        kernel_temp = np.zeros((kernel.shape), np.float32)
        for x in range(-a, a + 1):
            for y in range(-b, b + 1):
                temp = image2[i][j] - image2[i + x][j + y]
                temp = (temp * temp)
                temp /= (2 * sigma2 * sigma2)
                temp = np.exp(-temp)
                temp /= (2 * sigma2 * sigma2 * 3.1416)
                kernel_temp[a - x][b - y] = temp * kernel[a - x][b - y]
        temp = 0
        for k in range(-a, a + 1):
            for l in range(-b, b + 1):
                temp += image2[i + k][j + l] * kernel_temp[a - k][b - l]
        temp /= kernel_temp.sum()
        output[i - a][j - b] = temp

print(output)
# output = min_max_normalization(output)
# plt.imshow(output, 'gray')
# plt.show()

cv2.imshow("Input", image)
cv2.imshow("padded", image2)
cv2.imshow("Out", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
