from scipy.ndimage import rotate
import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

def min_max_normalization(image):
    h=image.shape[0]
    w=image.shape[1]
    min=np.min(image)
    max=np.max(image)
    output=np.zeros((image.shape),np.uint8)

    for i in range(0,h):
        for j in range(0,w):
            temp=((image[i][j]-min)/(max-min))*255
            output[i][j]=temp
    return output

pattern_x=np.ones((512,512),np.uint8)

for i in range(0,pattern_x.shape[0]):
    for j in range(0,pattern_x.shape[1]):
        temp=((pow(j-0,2))/(pow(511,2)))*255
        pattern_x[i][j]=temp


pattern_y=np.ones((512,512),np.uint8)

for i in range(0,pattern_y.shape[0]):
    for j in range(0,pattern_y.shape[1]):
        temp=((pow(i-0,2))/(pow(511,2)))*255
        pattern_y[i][j]=temp



deg=int(input("Enter the degree:"))
angle=np.deg2rad(deg)

pattern=np.cos(angle)*pattern_x+np.sin(angle)*pattern_y

pattern=min_max_normalization(pattern)

print(pattern)
cv2.imshow('pattern',pattern)


input=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('input',input)

pattern=np.array(pattern,np.uint32)
input=np.array(input,np.uint32)
print(pattern.dtype,input.dtype)
cor1=pattern+input
corrupted_image=min_max_normalization(cor1)
cv2.imshow('corrupted_image',corrupted_image)


corrupted_imagee=np.array(corrupted_image,np.uint32)
corrupted_imagee=np.log1p(corrupted_imagee)
cor_fft=np.fft.fft2(corrupted_imagee)
cor_fft_shift=np.fft.fftshift(cor_fft)
cor_fft_mag=np.abs(cor_fft_shift)
cor_fft_phase=np.angle(cor_fft_shift)
corrupted_image_mag_plot=np.log1p(cor_fft_mag)
cor_image_mag_plott=min_max_normalization(corrupted_image_mag_plot)
cv2.imshow('cor_img_mag_plot',cor_image_mag_plott)

yh=1.2
yl=0.5
c=0.1
d0=50

homo_filter=np.zeros((corrupted_image.shape),np.float32)
for i in range(0,homo_filter.shape[0]):
    for j in range(0,homo_filter.shape[1]):
        r=((i-homo_filter.shape[0]//2)**2+(j-homo_filter.shape[1]//2)**2)/(d0**2)
        homo_filter[i][j]=(yh-yl)*(1-np.exp(-c*r))+yl
homo_filterr=min_max_normalization(homo_filter)
cv2.imshow("Homo_filter",homo_filterr)


corrected_out=cor_fft_mag*homo_filter
corrected_out=np.multiply(corrected_out,np.exp(1j*cor_fft_phase))
corrected_image=(np.fft.ifft2(np.fft.ifftshift(corrected_out)))
corrected_image=np.expm1(np.abs(corrected_image))
corrected_image=min_max_normalization(corrected_image)
cv2.imshow('output',corrected_image)

cv2.waitKey()
cv2.destroyAllWindows()