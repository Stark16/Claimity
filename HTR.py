import PDF_to_JPEG
import cv2
import os
from matplotlib import pyplot as plt
import numpy
PDF_to_JPEG.cvt_to_pdf()



def load_from_folder():
    folder = "./PDFs/Output"
    for file in os.listdir(folder):

        if os.path.join(folder, file)[-4:] == "jpg":
            img = cv2.imread(os.path.join(folder, file), 0)


load_from_folder()
