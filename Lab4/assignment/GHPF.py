# Fourier transform - guassian highpass filter

import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dpc


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
magnitude_spectrum = 20 * np.log(np.abs(ft_shift))

# mag, ang = cv2.cartToPolar(ft_shift[:,:,0],ft_shift[:,:,1])
ang = np.angle(ft_shift)

# guassian low pass filter
guass_sigma = int(input('Enter value of sigma: '))

guassian_filter = [[0 for i in range(img.shape[0])] for j in range(img.shape[1])]
half_size = int(img.shape[0] / 2)
normal = 1 / (2 * np.pi * (guass_sigma ** 2))
for i in range(-half_size, half_size + 1):
    for j in range(-half_size, half_size + 1):
        guassian_filter[i + half_size][j + half_size] = 1 - np.exp(
            -((i ** 2 + j ** 2) / (2 * guass_sigma ** 2))) * normal

# remove noise from magnitude with GLPF
magnitude_spectrum_scaled = min_max_normalize(magnitude_spectrum)
guass_magnitude_spectrum = np.multiply(guassian_filter, magnitude_spectrum)
guass_magnitude_spectrum_scaled = min_max_normalize(guass_magnitude_spectrum)

## phase add
guass_result = np.multiply(guass_magnitude_spectrum, np.exp(1j * ang))

# inverse fourier
img_back = np.real(np.fft.ifft2(np.fft.ifftshift(guass_result)))
img_back_scaled = min_max_normalize(img_back)

## plot
cv2.imshow("image", img_input)
cv2.imshow("Magnitude Spectrum", magnitude_spectrum_scaled)
cv2.imshow("Noise removed Magnitude Spectrum", guass_magnitude_spectrum_scaled)

cv2.imshow("Inverse transform", img_back_scaled)

# plt.subplot(121),plt.imshow(img, cmap = "gray")
# plt.title('Input image'), plt.xticks([]), plt.yticks([])

# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = "gray")
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# plt.subplot(221),plt.imshow(guass_magnitude_spectrum, cmap = "gray")
# plt.title('Noise removed Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()
#
# plt.subplot(321),plt.imshow(guass_magnitude_spectrum_scaled, cmap = "gray")
# plt.title('Noise removed Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()

# plt.subplot(221),plt.imshow(img_back, cmap = "gray")
# plt.title('Inverse transform'), plt.xticks([]), plt.yticks([])
# plt.show()

# plt.subplot(321),plt.imshow(img_back_scaled, cmap = "gray")
# plt.title('Inverse transform'), plt.xticks([]), plt.yticks([])
# plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
