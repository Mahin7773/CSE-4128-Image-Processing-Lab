import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def min_max_normalization(image):
    mi = np.min(image)
    ma = np.max(image)
    out = np.zeros((image.shape), np.uint8)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            out[i][j] = ((image[i][j] - mi) / (ma - mi)) * 255
    return out


def clip(src):
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i, j] < 0:
                src[i, j] = 0
            elif src[i, j] > 255:
                src[i, j] = 255
    return src


def get_gauss_kernel(size, sigma):
    center = size // 2
    kernel = np.zeros((size, size))
    cons = 1 / (2 * 3.14 * pow(sigma, 2))
    exp_div = 2 * pow(sigma, 2)
    for i in range(size):
        for j in range(size):
            kernel[i][j] = cons * np.exp(-(pow(i - center, 2) + pow(j - center, 2)) / exp_div)
    return kernel


def convolution(src, kernel, des):
    pad_x = kernel.shape[0] // 2
    pad_y = kernel.shape[1] // 2
    image_padded = cv2.copyMakeBorder(src, pad_x, pad_x, pad_y, pad_y, cv2.BORDER_REPLICATE)
    m, n = image_padded.shape
    for i in range(pad_x, m - pad_x):
        for j in range(pad_y, n - pad_y):
            temp = 0
            for k in range(-pad_x, pad_x + 1):
                for l in range(-pad_y, pad_y + 1):
                    temp += image_padded[i + k, j + l] * kernel[pad_x - k, pad_y - l]
            des[i - pad_x, j - pad_y] = temp
    return des


def stretch(intensity, a, b):
    iI = intensity
    minI = a
    maxI = b
    minO = 0
    maxO = 255
    if a != b:
        iO = (iI - minI) * (((maxO - minO) / (maxI - minI)) + minO)
    else:  # thresholding
        if intensity < a:
            iO = minO
        else:
            iO = maxO
    return iO


def gamma():
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    value = float(input('Enter the gamma value: '))
    out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            a = (img.item(i, j) / 255) ** value  # normalised then gamma corrected
            out[i, j] = 255 * a  # scaled back to original
    plt.imshow(out, 'gray')
    plt.title('Fig:     Gamma Correction')
    # plt.savefig('gamma_5.png', dpi=300, bbox_inches='tight')
    plt.show()


def contrast_stretch():
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    # cv2.imshow('original', img)
    print('min pixel value in image: ', np.min(img))
    print('max pixel value in image: ', np.max(img))
    mn = int(input('enter min intensity for stretch: '))
    mx = int(input('enter max intensity for stretch: '))
    out = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out.itemset((i, j), stretch(img.item(i, j), mn, mx))
    # cv2.imshow('contrast stretching', out)
    out = min_max_normalization(out)
    plt.imshow(out, 'gray')
    plt.title('Fig:    Contrast Stretching')
    # plt.savefig('thresholding.png', dpi=300, bbox_inches='tight')
    plt.show()


def Log():
    src = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    c = 255 / (np.log(1 + np.max(src)))
    out = np.zeros((512, 512), dtype=np.uint8)

    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            a = c * math.log(1 + src.item(i, j))  # log transformation
            out.itemset((i, j), a)

    # cv2.imshow('log transformation', out)
    plt.imshow(out, 'gray')
    plt.title('Fig:     Log Transformation')
    # plt.savefig('Log.png', dpi=300, bbox_inches='tight')
    plt.show()


def ILog():
    src = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    c = 255 / (np.log(1 + np.max(src)))
    out = np.zeros((512, 512), dtype=np.uint8)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            a = math.exp(src.item(i, j)) ** (1 / c) - 1  # inverse log transformation
            out.itemset((i, j), a)

    # cv2.imshow('inverse log transformation', out)
    plt.imshow(out, 'gray')
    plt.title('Fig:     Inverse Log Transformation')
    # plt.savefig('ILog.png', dpi=300, bbox_inches='tight')
    plt.show()


def mean():
    mk = int(input('kernel height: '))
    nk = int(input('kernel width: '))
    mask = np.ones([mk, nk], dtype=int)  # matrix values of 1
    mask = mask / (mk * nk)
    img = cv2.imread('noise.png', 0)
    # cv2.imshow('original', img)
    m, n = img.shape
    img_new = np.zeros([m, n])
    img_new1 = convolution(img, mask, img_new)
    plt.imshow(img_new1, 'gray')
    plt.title('Fig:     Averaging Filter')
    # plt.savefig('mean15.png', dpi=300, bbox_inches='tight')
    plt.show()


def median():
    img = cv2.imread('noise.png', 0)
    m, n = img.shape
    img_new1 = np.zeros([m, n])
    # Median filter
    mk = int(input('filter height: '))
    nk = int(input('filter width: '))
    pad_x = mk // 2
    pad_y = nk // 2
    mask = np.ones([mk, nk], dtype=int)  # kernel with all value as 1
    image_padded = cv2.copyMakeBorder(img, pad_x, pad_x, pad_y, pad_y, cv2.BORDER_REPLICATE)
    m = image_padded.shape[0]
    n = image_padded.shape[1]

    for i in range(pad_x, m - pad_x):
        for j in range(pad_y, n - pad_y):
            temp = []
            for k in range(mk):
                for l in range(nk):
                    temp.append(image_padded[i - pad_x + k, j - pad_y + l] * mask[k, l])
            temp.sort()
            img_new1[i - pad_x, j - pad_y] = temp[len(temp) // 2]
    img_new1 = img_new1.astype(np.uint8)
    # compare = np.concatenate((img, img_new1), axis=1)
    # cv2.imshow('median filter', compare)
    plt.imshow(img_new1, 'gray')
    plt.title('Fig:     Median Filter')
    # plt.savefig('median5.png', dpi=300, bbox_inches='tight')
    plt.show()


def sobel():
    sobel_Xkernel = np.array(([-1, 0, 1], [-2, 0, 2], [-1, 0, 1]), np.float32)
    sobel_Ykernel = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]), np.float32)

    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    m, n = img.shape
    img_new1 = np.zeros((m, n), np.float32)
    img_new2 = np.zeros((m, n), np.float32)

    img_newX = convolution(img, sobel_Xkernel, img_new1)
    img_newY = convolution(img, sobel_Ykernel, img_new2)

    img_newX = clip(img_newX)
    img_newY = clip(img_newY)

    plt.imshow(img_newX, 'gray')
    plt.title('Fig:     Sobel Vertical Masking')
    # plt.savefig('sobelX.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.imshow(img_newY, 'gray')
    plt.title('Fig:     Sobel Horizontal Masking')
    # plt.savefig('sobelY.png', dpi=300, bbox_inches='tight')
    plt.show()


def laplacian():
    kernelPos = np.array(([1, 1, 1], [1, -8, 1], [1, 1, 1]), np.float32)
    kernelNeg = np.array(([-1, -1, -1], [-1, 8, -1], [-1, -1, -1]), np.float32)
    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    m, n = img.shape
    img_new1 = np.zeros((m, n), np.float32)
    img_new2 = np.zeros((m, n), np.float32)

    img_pos = convolution(img, kernelPos, img_new1)
    img_neg = convolution(img, kernelNeg, img_new2)

    img_pos = clip(img_pos)
    img_neg = clip(img_neg)

    plt.imshow(img_pos, 'gray')
    plt.title('Fig:     Laplacian Positive Mask (outward edges)')
    # plt.savefig('sobelX.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.imshow(img_neg, 'gray')
    plt.title('Fig:     Laplacian Negative Mask (inward edges)')
    # plt.savefig('sobelY.png', dpi=300, bbox_inches='tight')
    plt.show()


def gaussian():
    size = int(input('enter the size of kernel: '))
    sigma = float(input('enter sigma value: '))

    gauss_kernel = get_gauss_kernel(size, sigma)
    print(gauss_kernel)

    img = cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE)
    m, n = img.shape
    img_new = np.zeros([m, n])

    img_new = convolution(img, gauss_kernel, img_new)

    plt.imshow(img_new, 'gray')
    plt.title('Fig:     Gaussian Blur')
    # plt.savefig('Gauss 15x15.png', dpi=300, bbox_inches='tight')
    plt.show()


plt.imshow(cv2.imread('lena.jpg', cv2.IMREAD_GRAYSCALE), 'gray')
plt.title('input image')
plt.show()

print('1. Gamma Correction')
print('2. Contrast Stretching')
print('3. Inverse Log Transformation')
print('4. Mean Filtering')
print('5. Median Filtering')
print('6. Gaussian Filtering')
print('7. Laplacian Filter')
print('8. Sobel Filter')
print('9. Exit')

while 1:
    choice = int(input('\nenter choice: '))
    if choice == 0:
        break
    elif choice == 1:
        gamma()
    elif choice == 2:
        contrast_stretch()
    elif choice == 3:
        Log()
        ILog()
    elif choice == 4:
        mean()
    elif choice == 5:
        median()
    elif choice == 6:
        gaussian()
    elif choice == 7:
        laplacian()
    elif choice == 8:
        sobel()
    elif choice == 9:
        break
