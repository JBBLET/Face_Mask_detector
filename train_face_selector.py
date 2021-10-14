# First preparing the data to have all the image in a CSV
# file containing the name and the object to analyze

import os
import xml.etree.ElementTree as ET
import pandas as pd
pd.set_option('display.max_columns', None)

os.chdir('archive/annotations/')
list_files = os.listdir()

header = ['image', 'dimension']
for i in range(1, 116):
    header.append(f'Object {i}')

csv_draft = []
for img in list_files:
    csv_line = []
    tree = ET.parse(img)
    root = tree.getroot()
    csv_line.append(root[1].text)
    height, width = root[2][0].text,root[2][1].text
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

data = pd.DataFrame(csv_draft)

data.columns = header
print(data.head())
