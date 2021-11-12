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
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import pandas as pd
import cv2
import numpy as np
pd.set_option('display.max_columns', None)

# First preparing the data to have all the image in a CSV
# file containing the name and the object to analyze
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

#Preparing the data for the training of the model into lists of objects and labels
classes = ["with_mask", "mask_weared_incorrect", "without_mask"]
labels = []
data = []

with_incorrectly_worn_mask = True
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
                if label != "mask_weared_incorrect" or with_incorrectly_worn_mask:
                    data.append(face)
                    labels.append(label)
            except:
                pass
        collumn += 1

data = np.array(data, dtype="float32")
labels = np.array(labels)

lb = LabelEncoder()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#Deep learning parameters
INIT_LR = 1e-4
EPOCHS = 50
BS = 1

#Create the partition for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, stratify=labels, random_state=42)

# construct the training image generator for data augmentation ie create more images to train by slightly modifying
# images in the dataset
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(64, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
if with_incorrectly_worn_mask:
    headModel = Dense(3, activation="softmax")(headModel)
else:
    headModel = Dense(2,activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)

# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False


opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# train the head of the network
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

predIdxs = model.predict(testX, batch_size=32)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# serialize the model to disk
os.chdir('/Users/jean-baptiste/PycharmProjects/Face_Mask_detector/')
model.save("archive/model/model1.h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
