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

kernel=np.array([[0,1,0],[1,-4,1],[0,1,0]])
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
output=image-output
plt.imshow(output,'gray')
plt.show()
output=min_max_normalization(output)

















cv2.imshow("input",image)
cv2.imshow("Padded",image2)
cv2.imshow("out",output)
plt.imshow(output,'gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()