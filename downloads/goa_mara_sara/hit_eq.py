import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def min_max_normalization(image):
    mi = np.min(image)
    ma = np.max(image)
    out = np.zeros((image.shape), np.uint8)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            out[i][j] = ((image[i][j] - mi) / (ma - mi)) * 255
    return out


image = cv2.imread('Lena.jpg', cv2.IMREAD_GRAYSCALE)

freq = np.zeros(256, np.float32)
for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
        freq[image[i][j]] += 1
total = image.shape[0] * image.shape[1]
print(freq)
pdf = np.zeros(256, np.float32)
for i in range(0, 256):
    pdf[i] = freq[i] / total
print(pdf)
cdf = np.zeros(256, np.float32)
cdf[0] = pdf[0]
for i in range(1, 256):
    cdf[i] = cdf[i - 1] + pdf[i]

s = np.zeros(256, np.float32)

for i in range(0, 256):
    s[i] = np.round(cdf[i] * 255)

print(s)

output = np.zeros((image.shape), np.float32)

for i in range(0, image.shape[0]):
    for j in range(0, image.shape[1]):
        output[i][j] = s[image[i][j]]

output = min_max_normalization(output)

freq2 = np.zeros(256, np.float32)
for i in range(0, output.shape[0]):
    for j in range(0, output.shape[1]):
        freq2[output[i][j]] += 1
total2 = output.shape[0] * output.shape[1]

pdf2 = np.zeros(256, np.float32)
for i in range(0, 256):
    pdf2[i] = freq2[i] / total2

cdf2 = np.zeros(256, np.float32)
cdf2[0] = pdf2[0]
for i in range(1, 256):
    cdf2[i] = cdf2[i - 1] + pdf2[i]

cv2.imshow("input", image)
cv2.imshow("output", output)
figure = plt.figure(figsize=(15, 7))
row = 2
col = 2
figure.add_subplot(row, col, 1)
plt.hist(output.ravel(), 255, [0, 255])
figure.add_subplot(row, col, 2)
plt.hist(image.ravel(), 255, [0, 225])
figure.add_subplot(row, col, 3)
plt.plot(cdf)
figure.add_subplot(row, col, 4)
plt.plot(cdf2)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
