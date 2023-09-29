# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 13:47:33 2023

@author: User
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)

print(np.max(img))


def min_max_normalization(image):
    mi = np.min(image)
    ma = np.max(image)
    out = np.zeros(image.shape, np.uint8)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            out[i][j] = ((image[i][j] - mi) / (ma - mi)) * 255
    return out


def gamma_correction():
    out = np.zeros(img.shape, np.float32)
    g = float(input('Gamma value: '))

    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            a = img.item(i, j)
            out[i][j] = 255 * (pow(a, g) / pow(255, g))
    plt.imshow(img, 'gray')
    plt.title('Input')
    plt.show()
    
    # cv2.imshow("input", img)
    # output = min_max_normalization(out)
    #
    # cv2.imshow("output", output)

    plt.imshow(out, 'gray')
    plt.title('Output Gamma %f' % g)
    plt.show()


def inverse_log():
    out = np.zeros(img.shape, dtype=np.float32)
    c = 255 / np.log(1 + np.max(img))
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            a = img.item(i, j)
            out[i][j] = np.exp((1 + a) / c)

    plt.imshow(img, 'gray')
    plt.title('Input')
    plt.show()

    plt.imshow(out, 'gray')
    plt.title('Output Inverse log')
    plt.show()


def contrast_stretching():
    out = np.zeros(img.shape, dtype=np.float32)
    c = np.max(img)
    d = np.min(img)
    print(c, d)
    # img2=np.zeros(img.shape,np.float32)
    low_val = int(input("Enter lower value: "))
    high_val = int(input("Enter higher value: "))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a = img[i][j]
            if low_val != high_val:
                out[i][j] = (a - low_val) * (255 - 0 / high_val - low_val)
            else:
                if a < low_val:
                    out[i][j] = 0
                else:
                    out[i][j] = 255

            # temp=255*((img[i][j]-d)/(c-d))

    # plt.imshow(img, 'gray')
    # plt.title('Input')
    # plt.show()
    cv2.imshow('in', img)
    out = min_max_normalization(out)
    # plt.imshow(out, 'gray')
    # plt.title('Output Contrast Strecthing')
    # plt.show()
    cv2.imshow('out', out)

    print(img)
    print(out)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


print('1. Gamma correction \n2. Inverse log \n3. Contrast stretching')

while 1:
    choice = int(input('Choice: '))

    if choice == 1:
        gamma_correction()
    elif choice == 2:
        inverse_log()
    elif choice == 3:
        contrast_stretching()
    elif choice == 0:
        break

cv2.waitKey(0)
cv2.destroyAllWindows()
