import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

image=cv2.imread("Lena.jpg")
def min_max_normalization(image):
    mi=np.min(image)
    ma=np.max(image)
    output=np.zeros((image.shape),np.uint8)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            output[i][j]=((image[i][j]-mi)/(ma-mi))*255
    return output

def his_eq(image):
    fre=np.zeros(256,np.float32)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            fre[image[i][j]]+=1
    pdf=np.zeros(256,np.float32)
    total=image.shape[0]*image.shape[1]
    for i in range(0,256):
        pdf[i]=fre[i]/total
    cdf=np.zeros(256,np.float32)
    cdf[0]=pdf[0]
    for i in range(1,256):
        cdf[i]=cdf[i-1]+pdf[i]
    s=np.zeros(256,np.float32)
    for i in range(0,256):
        s[i]=np.round(cdf[i]*255)
    output=np.zeros((image.shape),np.float32)
    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            output[i][j]=s[image[i][j]]
    output=min_max_normalization(output)
    return output
b,g,r=cv2.split(image)

fig=plt.figure(figsize=(15,8))
rows=1
col=3
fig.add_subplot(rows,col,1)
plt.hist(b.ravel(),255,[0,255])
fig.add_subplot(rows,col,2)
plt.hist(g.ravel(),255,[0,255])
fig.add_subplot(rows,col,3)
plt.hist(r.ravel(),255,[0,255])
plt.show()

b1=his_eq(b)
g1=his_eq(g)
r1=his_eq(r)

fig2=plt.figure(figsize=(15,8))
rows=1
col=3
fig2.add_subplot(rows,col,1)
plt.hist(b1.ravel(),255,[0,255])
fig2.add_subplot(rows,col,2)
plt.hist(g1.ravel(),255,[0,255])
fig2.add_subplot(rows,col,3)
plt.hist(r1.ravel(),255,[0,255])
plt.show()


output=cv2.merge((b1,g1,r1))











cv2.imshow("input",image)
cv2.imshow("output",output)
cv2.waitKey(0)
cv2.destroyAllWindows()