import cv2
from matplotlib import pyplot as plt
import os
import numpy as np


def count_lines(page):
    page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    page = cv2.resize(page, (700, 700))
    thresh = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    
    cv2.imshow("win", thresh)
    cv2.waitKey(0)

folder = "./PDFs/Pages"
for file in os.listdir(folder):

    image = cv2.imread(os.path.join(folder, file), 1)

    (ydim, xdim, _) = image.shape
    # print(image.shape)
    if (xdim % 8 == 0) & (ydim % 8 == 0):
        nxdim = int(xdim / 8)
        nydim = int (ydim / 8)
        div_fact = 8
    elif (xdim % 4 == 0) & (ydim % 4 == 0):
        nxdim = int (xdim / 4)
        nydim = int (ydim / 4)
        div_fact = 4
    print(nxdim, nydim)
    img = image.copy()
    image = cv2.resize(image, (nxdim, nydim))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
    edge = cv2.Canny(gray, 300, 700, apertureSize=5)

    items, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    index = 0

    max_area = cv2.contourArea(items[0])
    for i in range(len(items)):

        if max_area < cv2.contourArea(items[i]):
            max_area = cv2.contourArea(items[i])
            index = i

    # print(max_area)
    # cv2.drawContours(image, items, index, (255, 255, 0), 3)
    page = np.empty_like(gray)
    page[:] = 0
    cv2.drawContours(page, items, index, (255, 255, 255), 1)
    peri = cv2.arcLength(items[index], True)
    approx = cv2.approxPolyDP(items[index], 0.04*peri, True)
    corners = cv2.convexHull(approx, clockwise=False)
    print("Corners:", corners)
    # corners = cv2.goodFeaturesToTrack(page, 4, 0.1, 100, True, useHarrisDetector=True)
    for i in range(len(corners)):

        x1, y1 = corners[i].ravel()

        if i >= 3:
            x2, y2 = corners[0].ravel()
        else:
            x2, y2 = corners[i+1].ravel()

        xslope = abs(x2 - x1)
        yslope = abs(y2 - y1)

        if xslope > yslope:
            # print("Horizontal Line")
            if x1 > x2:
                #Bottom Line:
                Bottom_Right = (x1, y1)
                br = (x1*div_fact, y1*div_fact)

                Bottom_Left = (x2, y2)
                bl = (x2 * div_fact, y2 * div_fact)

                print(Bottom_Right, Bottom_Left)

            elif x1 < x2:
                #Top_Line:
                Top_Right = (x2, y2)
                Top_Left = (x1, y1)

                tr = (x2 * div_fact, y2 * div_fact)
                tl = (x1 * div_fact, y1 * div_fact)

    # cv2.circle(image, Top_Left, 5, (255, 255, 0), -1)
    # cv2.circle(image, Top_Right, 5, (0, 255, 0), -1)
    # cv2.circle(image, Bottom_Right, 5, (255, 0, 0), -1)
    # cv2.circle(image, Bottom_Left, 5, (0, 0, 255), -1)

    # cv2.circle(img, tl, 50, (255, 255, 0), -1)
    # cv2.circle(img, tr, 50, (0, 255, 0), -1)
    # cv2.circle(img, br, 50, (255, 0, 0), -1)
    # cv2.circle(img, bl, 50, (0, 0, 255), -1)


    pt3 = np.float32([[tl], [tr], [br], [bl]])
    pt4 = np.float32([[0, 0], [xdim, 0], [xdim, ydim], [0, ydim]])

    M = cv2.getPerspectiveTransform(pt3, pt4)
    crp_page = cv2.warpPerspective(img, M, (xdim, ydim))


    '''fig = plt.figure()
    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.title("Grayscale")
    plt.imshow(gray, cmap='gray')
    plt.subplot(2, 2, 3)
    plt.title("Edge")
    plt.imshow(edge, cmap='binary')
    plt.subplot(2, 2, 4)
    plt.title("Original_Cropped")
    plt.imshow(crp_page)
    plt.show()
'''
    # cv2.imwrite(file, crp_page)
    count_lines(crp_page)
