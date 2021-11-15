import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np


# file containing the name and the object to analyze
os.chdir('archive/annotations/')
list_files = os.listdir()

#create the .info file to train the cascade classifier
info_draft = []
for img in list_files:
    info_line = []
    tree = ET.parse(img)
    root = tree.getroot()
    info_line.append("archive/images/"+root[1].text)
    info_line.append(str(len(root)-4))
    for i in range(4, len(root)):
        temp = []
        temp.append(root[i][5][0].text)
        temp.append(root[i][5][1].text)
        temp.append(str(int(root[i][5][2].text)-int(root[i][5][0].text)))
        temp.append(str(int(root[i][5][3].text) - int(root[i][5][1].text)))
        info_line += temp
    info_draft.append(" ".join(info_line))
"""
os.chdir("/Users/jean-baptiste/PycharmProjects/Face_Mask_detector/archive/")
file = "pos.dat"
pos = open(file, "w")
pos.writelines(info_draft)
pos.close()
"""
"""
#Creation of the negative images dataset file only runned once and commented after

os.chdir("/Users/jean-baptiste/PycharmProjects/Face_Mask_detector/archive/negative/")
list_files = os.listdir()
os.chdir("/Users/jean-baptiste/PycharmProjects/Face_Mask_detector/archive/")
file = open("neg.txt", "w")
for i in range(len(list_files)):
    file.write("negative/"+list_files[i]+"\n")
file.close()
"""





