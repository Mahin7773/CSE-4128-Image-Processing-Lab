# -*- coding: utf-8 -*-
"""
Created on Tue May 30 00:10:25 2023

@author: User
"""

import cv2
import matplotlib.pyplot as plt
import math
import numpy as np


def min_max_normalization(image):
    height = image.shape[0]
    width = image.shape[1]
    min = np.min(image)
    max = np.max(image)
    output = np.zeros(image.shape, np.uint8)

    for i in range(0, height):
        for j in range(0, width):
            temp = ((image[i][j] - min) / (max - min)) * 255
            output[i][j] = temp
    return output


illum = np.ones((512, 512), np.uint8)

for i in range(0, illum.shape[0]):
    for j in range(0, illum.shape[1]):
        temp = ((pow(j - 0, 2)) / (pow(511, 2))) * 255
        illum[i][j] = temp
illumination = cv2.imshow('Pattern', illum)

input = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('image', input)

illum = np.array(illum, np.uint32)
input = np.array(input, np.uint32)
print(illum.dtype, input.dtype)
corrupted = illum + input
corrupted_img = min_max_normalization(corrupted)
cv2.imshow('Corrupted image', corrupted_img)

corrupted_image = np.array(corrupted_img, np.uint32)
corrupted_image = np.log1p(corrupted_image)
cor_ft = np.fft.fft2(corrupted_image)
cor_ft_shift = np.fft.fftshift(cor_ft)
cor_ft_mag = np.abs(cor_ft_shift)
cor_ft_phase = np.angle(cor_ft_shift)
corrupted_image_mag_plot = np.log1p(cor_ft_mag)
cor_image_mag_plott = min_max_normalization(corrupted_image_mag_plot)
cv2.imshow('cor_img_mag_plot', cor_image_mag_plott)

yh = 1.2
yl = 0.5
c = 0.1
d0 = 50

homo_filter = np.zeros(corrupted_img.shape, np.float32)
for i in range(0, homo_filter.shape[0]):
    for j in range(0, homo_filter.shape[1]):
        r = ((i - homo_filter.shape[0] // 2) ** 2 + (j - homo_filter.shape[1] // 2) ** 2) / (d0 ** 2)
        homo_filter[i][j] = (yh - yl) * (1 - np.exp(-c * r)) + yl
homo_filterr = min_max_normalization(homo_filter)
cv2.imshow("Homo_filter", homo_filterr)

corrected_out = cor_ft_mag * homo_filter
corrected_out = np.multiply(corrected_out, np.exp(1j * cor_ft_phase))
corrected_image = (np.fft.ifft2(np.fft.ifftshift(corrected_out)))
corrected_image = np.expm1(np.abs(corrected_image))
corrected_image = min_max_normalization(corrected_image)
cv2.imshow('output', corrected_image)
cv2.waitKey()
cv2.destroyAllWindows()
