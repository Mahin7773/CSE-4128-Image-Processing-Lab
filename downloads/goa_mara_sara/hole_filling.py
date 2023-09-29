import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import matplotlib

matplotlib.use('TkAgg')

points = []


def onclick(event):
    global x, y
    ax = event.inaxes
    if ax is not None:
        x, y = ax.transData.inverted().transform([event.x, event.y])
        x = int(round(x))
        y = int(round(y))
        points.append((x, y))


def min_max_normalization(image):
    mi = np.min(image)
    ma = np.max(image)
    output = np.zeros((image.shape), np.uint8)
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            output[i][j] = ((image[i][j] - mi) / (ma - mi)) * 255
    return output


image = cv2.imread('input.jpg', 0)
r, image = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY)
image_inv = cv2.bitwise_not(image)

se = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
se = se * 255

fig = plt.imshow(image, 'gray')
fig.figure.canvas.mpl_connect('button_press_event', onclick)
plt.show(block=True)
print(points)

output = np.zeros((image.shape), np.uint8)
for i in points:
    x = i[0]
    y = i[1]
    x, y = y, x
    while (1):
        output[x, y] = 1 * 255
        o1 = cv2.dilate(output, se, iterations=1)
        o1 = cv2.bitwise_and(image_inv, o1)
        cv2.imshow("Temp", o1)
        cv2.waitKey(0)
        if (o1 == output).all():
            break
        output = o1

output = cv2.bitwise_or(image, output)
cv2.imshow('out', output)
cv2.imshow("input", image)
cv2.imshow("inverse", image_inv)
cv2.imshow("se", se)
cv2.waitKey(0)
cv2.destroyAllWindows()
