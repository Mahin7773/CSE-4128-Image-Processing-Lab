import math
import numpy as np
import matplotlib.pyplot as plt
import cv2


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

sigma = int(input("Enter the value of sigma:"))
k_h = 5 * sigma
k_w = 5 * sigma
kernel = np.zeros((k_h, k_w), np.float32)
a = k_h // 2
b = k_w // 2
for i in range(-a, a + 1):
    for j in range(-b, b + 1):
        temp = 1 / (2 * 3.1416 * (sigma ** 2))
        temp = temp * np.exp(-(((i * i) + (j * j)) / sigma ** 2))
        kernel[a + i][b + j] = temp
print(kernel)

image2 = cv2.copyMakeBorder(image, a, a, b, b, cv2.BORDER_REPLICATE)

output = np.zeros((image.shape), np.float32)
kernel_sum = kernel.sum()
print(kernel_sum)
for i in range(a, image2.shape[0] - a):
    for j in range(b, image2.shape[1] - b):
        temp = 0
        for k in range(-a, a + 1):
            for l in range(-b, b + 1):
                temp += (kernel[a - k][b - l] * image2[i + k][j + l])
        output[i - a][j - b] = temp / kernel_sum

output = min_max_normalization(output)

cv2.imshow("input", image)
cv2.imshow("Padded", image2)
cv2.imshow("out", output)
plt.imshow(output, 'gray')

cv2.waitKey(0)
cv2.destroyAllWindows()
