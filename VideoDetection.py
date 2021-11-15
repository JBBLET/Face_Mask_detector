import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow import expand_dims
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

face_cascade = cv2.CascadeClassifier('archive/model/haarcascade_frontalface_default.xml')
model = load_model("archive/model/model3.h5")

classes = ["with_mask","without_mask", "mask_weared_incorrect"]

def detect_and_classify(img):
    img_copy = img.copy()
    gray_img = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    faces_list = face_cascade.detectMultiScale(gray_img, 1.1,4,minSize=(30, 30))
    locations, proba, labels = [], [], []
    for faces in faces_list:
        x, y, w, h = faces
        face = img[y:y + h, x:x + w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)
        face_class = model.predict(face)[0]
        max_index = np.argmax(face_class)
        locations.append(faces)
        labels.append(classes[max_index])
        proba.append(face_class[max_index])

    return(locations,labels,proba)

cap = cv2.VideoCapture(0)


while True:

    _, img = cap.read()

    loc,labels,proba = detect_and_classify(img)
    if len(loc)>0:
        for i in range(len(labels)):
            x,y,w,h = loc[i]
            if labels[i] == "with_mask":
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                cv2.putText(img, labels[i] +" "+str(int(proba[i]*100))+"%", (x, y - 10),
                             0,0.9, (0, 255, 0), 2)
            elif labels[i] == "without_mask":
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(img, labels[i] + " " + str(int(proba[i] * 100)) + "%", (x, y - 10),
                             0,0.9, (0, 0, 255), 2)
            else:
                img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv2.putText(img, labels[i] + " " + str(int(proba[i] * 100)) + "%", (x, y - 10),
                            0,0.9, (255, 0, 0), 2)
    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
