import PDF_to_JPEG
import cv2
import os
import numpy
PDF_to_JPEG.cvt_to_pdf()
Pages = []


def load_from_folder():
    folder = "./PDFs/Output"
    for file in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file))
        print(img)
        if img is not None:
            Pages.append(img)
            cv2.imshow("Image", img)
            cv2.waitKey(0)

load_from_folder()
