# Fourier transform - guassian lowpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc
import math


def min_max_normalize(img_inp):
    inp_min = np.min(img_inp)
    inp_max = np.max(img_inp)

    for i in range(img_inp.shape[0]):
        for j in range(img_inp.shape[1]):
            img_inp[i][j] = (((img_inp[i][j] - inp_min) / (inp_max - inp_min)) * 255)
    return np.array(img_inp, dtype='uint8')


# take image
img_input = cv2.imread('period_input4.jpeg', 0)

img = dpc(img_input)

image_size = img.shape[0] * img.shape[1]

# fourier transform
ft = np.fft.fft2(img)

ft_shift = np.fft.fftshift(ft)

magnitude_spectrum_ac = np.abs(ft_shift)
magnitude_spectrum = 1 * np.log(np.abs(ft_shift) + 1)
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)

r = int(input('enter r value:'))  # 5

x1 = int(input('\nenter x value:'))  # 52
y1 = int(input('enter y value:'))  # 52

x2 = int(input('\nenter x value:'))  # 14
y2 = int(input('enter y value:'))  # 116

x3 = int(input('\nenter x value:'))  # 116
y3 = int(input('enter y value:'))  # 14

x4 = int(input('\nenter x value:'))  # 182
y4 = int(input('enter y value:'))  # 52

kernel = np.ones((img.shape[0], img.shape[1]), dtype=int)
print(kernel)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        d1 = math.sqrt((i - x1) * (i - x1) + (j - y1) * (j - y1))
        d2 = math.sqrt((i - x2) * (i - x2) + (j - y2) * (j - y2))
        d3 = math.sqrt((i - x3) * (i - x3) + (j - y3) * (j - y3))
        d4 = math.sqrt((i - x4) * (i - x4) + (j - y4) * (j - y4))
        if d1 < r or d2 < r or d3 < r or d4 < r:
            kernel[i][j] = 0
            kernel[abs(img.shape[0] - i)][abs(img.shape[1] - j)] = 0

# mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

# notch
notched_magnitude_ac = magnitude_spectrum_ac * kernel
notched_magnitude = magnitude_spectrum * kernel
notched_magnitude_spectrum_scaled = min_max_normalize(notched_magnitude)

## phase add
final_result = np.multiply(magnitude_spectrum_ac, np.exp(1j * ang))
notch_result = np.multiply(notched_magnitude_ac, np.exp(1j * ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(final_result)))
img_back_scaled = min_max_normalize(img_back)

notch_back = np.real(np.fft.ifft2(np.fft.ifftshift(notch_result)))
notch_back_scaled = min_max_normalize(notch_back)

## plot
cv2.imshow("image", img_input)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum_scaled)
cv2.imshow("kernel", notched_magnitude)
cv2.imshow("notched", notched_magnitude_spectrum_scaled)

# cv2.imshow("Phase",ang)
cv2.imshow("Inverse transform", img_back_scaled)
cv2.imshow("notch inverse", notch_back_scaled)
# cv2.imshow("kernel",kernel)


cv2.waitKey(0)
cv2.destroyAllWindows()
