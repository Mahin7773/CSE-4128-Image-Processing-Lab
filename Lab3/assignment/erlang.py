import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import erlang

inp = np.zeros(256, dtype=int)
er = np.zeros(256, dtype=int)
ipdf = np.zeros(256, dtype=float)
erpdf = np.zeros(256, dtype=float)
icdf = np.zeros(256, dtype=int)
ercdf = np.zeros(256, dtype=int)

img = cv2.imread('eye.png', 0)
cv2.imshow('image image', img)

plt.subplot(1, 3, 1)
plt.hist(img.ravel(), 256, [0, 255])
plt.title("Input Image Histogram")

z = img.shape[0] * img.shape[1]

erl = erlang.rvs(20, scale=2, size=(img.shape[0], img.shape[1]))
print(erl)
# round up and type cust to int from float
erl = np.round(erl).astype(int)
# make the range between 0-255
erl[erl > 255] = 255
erl[erl < 0] = 0

plt.subplot(1, 3, 2)
plt.hist(erl.ravel(), 256, [0, 255])
plt.title("Erlang Histogram")

x = 0

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        inp[img[i][j]] = inp[img[i][j]] + 1

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        er[erl[i][j]] = er[erl[i][j]] + 1

for i in range(0, 256):
    ipdf[i] = inp[i] / z
    x = x + ipdf[i]
    icdf[i] = round(255 * x)

x = 0
for i in range(0, 256):
    erpdf[i] = er[i] / (z)
    x = x + erpdf[i]
    ercdf[i] = round(255 * x)

# histogram matching
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        for k in range(256):
            if ercdf[k] >= icdf[img[i][j]]:
                img[i][j] = k
                break

plt.subplot(1, 3, 3)
plt.hist(img.ravel(), 256, [0, 255])
plt.title("Output Image Histogram")

cv2.imshow('output image', img)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
