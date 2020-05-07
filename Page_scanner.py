import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

img = []
folder = "./PDFs/Pages"
for file in os.listdir(folder):

    image = cv2.imread(os.path.join(folder, file), 1)
    image = cv2.resize(image, (700, 700))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
    edge = cv2.Canny(gray, 300, 700, apertureSize=5)
    cv2.imshow("edge", gray)
    cv2.waitKey(0)

    items, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = 0

    max_area = cv2.contourArea(items[0])
    for i in range(len(items)):

        if max_area < cv2.contourArea(items[i]):
            max_area = cv2.contourArea(items[i])
            index = i

    print(max_area)
    cv2.drawContours(image, items, index, (255, 255, 0), 3)
    print(index)
    #img.append(cv2.imread(os.path.join(folder, file), 1))
    #plt.imshow(edge)
    #plt.show()
    cv2.imshow("img", image)
    cv2.waitKey(0)


'''for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
'''