from pdf2image import  convert_from_path
import os

def cvt_to_pdf():
    path = "./PDFs/Input/IoT 3.pdf"
    filename = path.split('/')[-1]
    print("./PDFs/Output/" + filename[:-4])
    try:
        output_path = "./PDFs/Output/" + filename[:-4]
        os.mkdir(output_path)
        pages = convert_from_path(path, 500)
        i = 1
        for page in pages:
            page_name = output_path + filename[:-4] + "_" + str(i) + ".jpg"
            page.save(page_name, "JPEG")
            i += 1
    except:
        print("The file is already converted to JPEG ")

