import cv2
import matplotlib.pyplot as plt
import numpy as np

def min_max_normalization(image):
    mi=np.min(image)
    ma=np.max(image)
    output=np.zeros((image.shape),np.uint8)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            output[i][j]=((image[i][j]-mi)/(ma-mi))*255
    return output

image=cv2.imread('ripple.png',cv2.IMREAD_GRAYSCALE)
output=np.zeros((image.shape),np.float32)
center_x=image.shape[0]//2
center_y=image.shape[1]//2

print(center_x,center_y)

a=int(input("Enter alpha:"))

taux=int(input("Enter taux:"))
tauy=int(input("Enter tauy:"))

for i in range(0,image.shape[0]):
    for j in range(0,image.shape[1]):
        xn=i+a*np.sin(((2*3.1416)/taux)*(i-center_x))
        yn=j+a*np.sin(((2*3.1416)/tauy)*(j-center_y))
        x=int(np.round(xn))
        y=int(np.round(yn))
        aa=xn-x
        bb=yn-y
        if(x<image.shape[0]-1 and y<image.shape[1]-1):
            image_array=[[image[x][y],image[x+1][y]],
                         [image[x][y+1],image[x+1][y+1]]]
            temp1=np.array([[1-aa],
                           [aa]])
            temp2=np.array([[1-bb,bb]])
            ans=np.matmul(image_array,temp1)
            ans=np.matmul(ans,temp2)
            z=ans[0][0]
        else:
            z=0
        output[i][j]=z
output=min_max_normalization(output)


cv2.imshow("input",image)
cv2.imshow("output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()