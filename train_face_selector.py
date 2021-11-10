# First preparing the data to have all the image in a CSV
# file containing the name and the object to analyze
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import argparse
import glob
import random as rand
import os
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import numpy as np
pd.set_option('display.max_columns', None)

os.chdir('archive/annotations/')
list_files = os.listdir()

header = ['image', 'dimension']
for i in range(1, 116):
    header.append(f'Object {i}')


# create a clean dataframe containing the info in annotation folder
csv_draft = []
for img in list_files:
    csv_line = []
    tree = ET.parse(img)
    root = tree.getroot()
    csv_line.append(root[1].text)
    height, width = root[2][0].text, root[2][1].text
    csv_line.append([height, width])
    for i in range(4, len(root)):
        temp = []
        temp.append(root[i][0].text)
        for point in root[i][5]:
            temp.append(point.text)
        csv_line.append(temp)
    for i in range(len(csv_line), 117):
        csv_line.append(0)
    csv_draft.append(csv_line)

df = pd.DataFrame(csv_draft)

df.columns = header

classes = ["with_mask", "mask_weared_incorrect", "without_mask"]
labels = []
data = []

os.chdir('/Users/jean-baptiste/PycharmProjects/Face_Mask_detector/archive/images')

for index, row in df.iterrows():
    img = cv2.imread(df['image'][index])
    #scale to dimension
    X, Y = df["dimension"][index]
    X, Y = int(X), int(Y)
    img = cv2.resize(img, (X,Y))
    #find the face in each object
    collumn = 1
    info = df[f'Object {collumn}'][index]
    while collumn < 116 and df[f'Object {collumn}'][index] != 0:
        info = df[f'Object {collumn}'][index]
        label = info[0]
        info[0] = info[0].replace(str(label), str(classes.index(label)))
        info = [int(each) for each in info]
        face = img[info[2]:info[4],info[1]:info[3]]
        if((info[3]-info[1])>40 and (info[4]-info[2])>40):
            try:
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                data.append(face)
                labels.append(label)
            except:
                pass
        collumn += 1

print("Done!")
data = np.array(data, dtype="float32")
print(data)
labels = np.array(labels)
print(labels)
