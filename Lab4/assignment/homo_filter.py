import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def min_max_normalization(image):
    h = image.shape[0]
    w = image.shape[1]
    min = np.min(image)
    max = np.max(image)
    output = np.zeros(image.shape, np.uint8)

    for i in range(0, h):
        for j in range(0, w):
            temp = ((image[i][j] - min) / (max - min)) * 255
            output[i][j] = temp
    return output


i_filter = np.ones((512, 512), np.uint8)

for i in range(0, i_filter.shape[0]):
    for j in range(0, i_filter.shape[1]):
        i_filter[i][j] = ((pow(j - 0, 2) + pow(i - 511, 2)) / (pow(511, 2) + pow(511, 2))) * 255

image = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('input image', image)

cv2.imshow('i_filter', i_filter)

i_filter = np.array(i_filter, np.uint32)  # type cast to uint32
image = np.array(image, np.uint32)

if_image = i_filter + image
if_image = min_max_normalization(if_image)
cv2.imshow('i_filtered image', if_image)

if_image = np.array(if_image, np.uint32)  # type cast to uint32

if_image = np.log1p(if_image)

if_fft = np.fft.fft2(if_image)
if_fft_shift = np.fft.fftshift(if_fft)
if_fft_mag = np.abs(if_fft_shift)
if_fft_phase = np.angle(if_fft_shift)
mag_plot = np.log1p(if_fft_mag)
mag_plot = min_max_normalization(mag_plot)
cv2.imshow('if_image magnitude', mag_plot)

yh = 1.2
yl = 0.5
c = 0.1
d0 = 20

homo_filter = np.zeros(if_image.shape, np.float32)
for i in range(0, homo_filter.shape[0]):
    for j in range(0, homo_filter.shape[1]):
        r = ((i - homo_filter.shape[0] // 2) ** 2 + (j - homo_filter.shape[1] // 2) ** 2) / (d0 ** 2)
        homo_filter[i][j] = (yh - yl) * (1 - np.exp(-c * r)) + yl
homo_filterr = min_max_normalization(homo_filter)
cv2.imshow("homo_filter", homo_filterr)

# homomorphic filtering
homo_out = if_fft_mag * homo_filter

homo_out = np.multiply(homo_out, np.exp(1j * if_fft_phase))
homo_image = (np.fft.ifft2(np.fft.ifftshift(homo_out)))
homo_image = np.expm1(np.abs(homo_image))
homo_image = min_max_normalization(homo_image)
cv2.imshow('output image', homo_image)

cv2.waitKey()
cv2.destroyAllWindows()
