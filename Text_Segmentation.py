import cv2
import numpy as np
import math
import os
from matplotlib import pyplot as plt
from scipy.signal import argrelmin
import tensorflow as tf

img = np.arange(16).reshape((4, 4))
window = ['flat', 'hanning', 'hamming', 'barlett', 'blackmann']

def applySumFunction(img):
    res = np.sum(img, axis=0)
    return res

def normalize(img):
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s>0 else img
    return img

def createKernel(kernelSize, sigma, theta):
    assert kernelSize % 2
    halfsize = kernelSize // 2
    kernel = np.zeros([kernelSize, kernelSize])

    sigmaX = sigma
    sigmaY = sigma * theta
    for i in range(kernelSize):
        for j in range(kernelSize):
            x = i - halfsize
            y = j - halfsize

            expTerm = np.exp(-x**2 / 2 * sigmaX - y**2 / 2 * sigmaY)
            xTerm = (x**2 - sigmaX**2) / (2*math.pi * sigmaX**5 * sigmaY)
            yTerm = (y**2 - sigmaY**2) / (2*math.pi * sigmaY**5 * sigmaX)

    kernel[i, j] = (xTerm + yTerm) * expTerm
    kernel = kernel / np.sum(kernel)
    return kernel


def smooth(x, windowLength = 11, window = 'hanning'):
    if x.ndim != 1:
        raise ValueError("Input vector needs to be 1 to smooth.")
    if x.size < windowLength:
        raise ValueError("Input vector needs to be bigger than window size.")
    if windowLength < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[ windowLength-1:0:-1 ], x, x[-2:windowLength-1:-1]]

    if window == 'flat':
        w = np.ones(windowLength, 'd')
    else:
        w = eval('np.' + window + '(windowLength)')

    y = np.convolve(w/w.sum(), s, mode='valid')
    return y

def crop_text_to_lines(text, blanks):
    x1 = 0
    y = 0
    lines = []

    for i, blank in enumerate(blanks):
        x2 = blank
        print('x1=', x1, 'x2=', x2, 'Diff=', x2-x1)
        line = text[:, x1:x2]
        lines.append(line)
        x1 = blank
    return lines

def display_lines(lines_arr, orient = 'vertical'):
    plt.figure(figsize=(30, 30))
    if not orient == 'vertical':
        for i, l in enumerate(lines_arr):
            line = l
            plt.subplot(2, 10, i+1)
            plt.axis('off')
            plt.title('Line #{0}'.format(i))
            _ = plt.imshow(line, cmap='gray', interpolation='bicubic')
            plt.xticks([]), plt.yticks([])
    else:
        for i, l in enumerate(lines_arr):
            line = l
            plt.subplot(40, 1, i+1)
            plt.axis('off')
            plt.title("Line#{0}".format(i))
            _ = plt.imshow(line, cmap='gray', interpolation='bicubic')
            plt.xticks([]), plt.yticks([])
    plt.show()

def segment(page_Path):
    for file in os.listdir(page_Path):
        page = (cv2.imread(os.path.join(page_Path, file), 0))
        page = np.transpose(page)
        page = cv2.resize(page, (1360, 768))

        kernelSize = 9
        sigma = 4
        theta = 1.5

        imgFiltered = cv2.filter2D(page, -1, createKernel(kernelSize, sigma, theta), borderType=cv2.BORDER_REPLICATE)

        normazied_img = normalize(imgFiltered)
        (m, s) = cv2.meanStdDev(imgFiltered)
        summ = applySumFunction(normazied_img)

        smoothed = smooth(summ, 35)

        mins = argrelmin(smoothed, order=2)
        arr_mins = np.array(mins)

        # plt.plot(smoothed)
        # plt.plot(arr_mins, smoothed[arr_mins], 'x')
        # plt.show()

        found_lines = crop_text_to_lines(page, arr_mins[0])

        sess = tf.Session()
        found_lines_arr = []
        with sess.as_default():
            for i in range(len(found_lines) - 1):
                found_lines_arr.append(tf.expand_dims(found_lines[i], -1).eval())

segment("./PDFs/Output/IoT 3")