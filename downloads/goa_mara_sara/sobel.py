import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

def min_max_normalization(image):
    h=image.shape[0]
    w=image.shape[1]
    mi=np.min(image)
    ma=np.max(image)
    out=np.zeros((image.shape),np.uint8)
    for i in range(0,h):
        for j in range(0,w):
            out[i][j]=((image[i][j]-mi)/(ma-mi))*255
    return out


image=cv2.imread('Lena.jpg',cv2.IMREAD_GRAYSCALE)

kernel=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
a=kernel.shape[0]//2
b=kernel.shape[1]//2

image2=cv2.copyMakeBorder(image,a,a,b,b,cv2.BORDER_REPLICATE)

output=np.zeros((image.shape),np.float32)
kernel_sum=kernel.sum()
print(kernel_sum)
for i in range(a,image2.shape[0]-a):
    for j in range(b,image2.shape[1]-b):
        temp=0
        for k in range(-a,a+1):
            for l in range(-b,b+1):
                temp+=(kernel[a-k][b-l]*image2[i+k][j+l])
        output[i-a][j-b]=temp
 
print(output)
plt.imshow(output,'gray')
plt.show()
output_norm=min_max_normalization(output)

kernel2=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

a=kernel2.shape[0]//2
b=kernel2.shape[1]//2

image2=cv2.copyMakeBorder(image,a,a,b,b,cv2.BORDER_REPLICATE)

output2=np.zeros((image.shape),np.float32)
kernel_sum=kernel.sum()
print(kernel_sum)
for i in range(a,image2.shape[0]-a):
    for j in range(b,image2.shape[1]-b):
        temp=0
        for k in range(-a,a+1):
            for l in range(-b,b+1):
                temp+=(kernel2[a-k][b-l]*image2[i+k][j+l])
        output2[i-a][j-b]=temp
 
print(output2)
plt.imshow(output2,'gray')
plt.show()
output2_norm=min_max_normalization(output2)


output3=np.sqrt(output**2+output2**2)
output3=min_max_normalization(output3)
plt.imshow(output3,'gray')
plt.show()











cv2.imshow("input",image)
cv2.imshow("Padded",image2)
cv2.imshow("out",output_norm)
cv2.imshow("out2",output2_norm)
cv2.imshow("out3",output3)
plt.imshow(output,'gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()