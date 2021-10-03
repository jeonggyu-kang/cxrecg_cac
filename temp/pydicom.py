import numpy as np
import pydicom
from PIL import image
import os

direc_dicom = '/media/jg/0a1f7cc9-1c9e-4e92-b4d8-d31e8b907842/home/compu/cac_sample'  # file path

file_list = os.listdir(direc_dicom)

'''
img = pydicom.dcmread('direc_dicom/'+file)
img = img.pixel_array.astype(float)

rescaled_img = (np.maximum(mg,0)/img.max())*255    # float pixels
final_img -= np.unit8(rescaled_img)     # integer pixels

final_img = Image.fromarray(final_img)
final_img.show()
final_img.save(new_image.png)
'''