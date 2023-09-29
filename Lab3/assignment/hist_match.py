import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import erlang


def cnt(arr, x, y):
    out = np.zeros(256, dtype=int)
    for i in range(x):
        for j in range(y):
            out[arr[i][j]] += 1
    return out


def prob(arr, size):
    pdf = np.zeros(256, dtype=float)
    cdf = np.zeros(256, dtype=int)
    x = 0
    for i in range(0, 256):
        pdf[i] = arr[i] / size
        x += pdf[i]
        cdf[i] = round(255 * x)
    return pdf, cdf


def find_nearest(a, a0):
    idx = np.abs(a - a0).argmin()
    return idx


inp = np.zeros(256, dtype=int)
match = np.zeros(256, dtype=int)
ipdf = np.zeros(256, dtype=float)
mpdf = np.zeros(256, dtype=float)
icdf = np.zeros(256, dtype=int)
mcdf = np.zeros(256, dtype=int)

img = cv2.imread('eye.png', 0)
cv2.imshow('image image', img)
# plt.subplot(1, 2, 1)
# plt.imshow(img, 'gray')
# plt.title('Input Image')

z = img.shape[0] * img.shape[1]

k = int(input('enter shape parameter k:'))
m = int(input('enter scale parameter miu:'))

erl = erlang.rvs(k, scale=m, size=(img.shape[0], img.shape[1]))
print(erl)

erl = np.round(erl).astype(int)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if erl[i][j] > 255:
            erl[i][j] = 255
        elif erl[i][j] < 0:
            erl[i][j] = 0

inp = cnt(img, img.shape[0], img.shape[1])
match = cnt(erl, img.shape[0], img.shape[1])

ipdf, icdf = prob(inp, z)
mpdf, mcdf = prob(match, z)

print(erl)
plt.subplot(1, 3, 1)
plt.hist(img.ravel(), 256, [0, 255])
plt.title("Input Image Histogram")
plt.subplot(1, 3, 2)
plt.hist(erl.ravel(), 256, [0, 255])
plt.title("Erlang Histogram")


# histogram matching
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i][j] = find_nearest(mcdf, icdf[img[i][j]])

plt.subplot(1, 3, 3)
plt.hist(img.ravel(), 256, [0, 255])
plt.title("Output Image Histogram")

cv2.imshow('output image', img)

# plt.subplot(1, 2, 2)
# plt.imshow(img, 'gray')
# plt.title('Output Image')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
