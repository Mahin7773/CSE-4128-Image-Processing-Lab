import cv2
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')
def min_max_normalization(image):
    mi=np.min(image)
    ma=np.max(image)
    output=np.zeros((image.shape),np.uint8)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            output[i][j]=((image[i][j]-mi)/(ma-mi))*255
    return output

points=[]
def onclick(event):
    global x,y
    ax=event.inaxes
    if ax is not None :
        x,y=ax.transData.inverted().transform([event.x,event.y])
        x=int(round(x))
        y=int(round(y))
        points.append((x,y))
        

image=cv2.imread('period_input.jpg',cv2.IMREAD_GRAYSCALE)
image_fft=np.fft.fft2(image)
image_fft_shift=np.fft.fftshift(image_fft)
image_ffts_mag=np.abs(image_fft_shift)
image_ffts_angle=np.angle(image_fft_shift)
image_ffts_magg=np.log1p(image_ffts_mag)
image_ffts_magg=min_max_normalization(image_ffts_magg)
plt.title("Image in fourier domain:")
img=plt.imshow(image_ffts_magg,'gray')
img.figure.canvas.mpl_connect('button_press_event',onclick)
plt.show(block=True)
print(points)

notch_filter=np.ones((image.shape),np.float32)
h=notch_filter.shape[0]//2
w=notch_filter.shape[1]//2
d0=int(input("The value of d0:"))
for i in range(0,notch_filter.shape[0]):
    for j in range(0,notch_filter.shape[1]):
        for k in range(0,len(points)):
            x=points[k][0]
            y=points[k][1]
            x,y=y,x
            if(x<=h):
                x2=h+(h-x)
            else:
                x2=h-(x-h)
            if(y<=w):
                y2=w+(w-y)
            else:
                y2=w-(y-w)
            d=np.sqrt((i-x)**2+(j-y)**2)
            d2=np.sqrt((i-x2)**2+(j-y2)**2)
            if(d<d0):
                notch_filter[i][j]*=0
            else:
                notch_filter[i][j]*=1
            if(d2<d0):
                notch_filter[i][j]*=0
            else:
                notch_filter[i][j]*=1

notch_filterr=min_max_normalization(notch_filter)
output_fft=image_ffts_mag*notch_filter
output_fftt=np.log1p(output_fft)
output_fftt=min_max_normalization(output_fftt)

output=np.multiply(output_fft,np.exp(1j*image_ffts_angle))
output=np.fft.ifftshift(output)
output=np.fft.ifft2(output)
output=np.abs(output)
output=min_max_normalization(output)

cv2.imshow('Input',image)
cv2.imshow("notch",notch_filterr)
cv2.imshow("Output in fourier",output_fftt)
cv2.imshow("Output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()