import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

print("test")

DATADIR = "../Imagesfiles"
CATEGORIESTraining = ["CB55/img/training", "CS18/img/training","CS863/img/training"]
CATEGORIESResult = ["CB55/pixel-level-gt/training", "CS18/pixel-level-gt/training", "CS863/pixel-level-gt/training"]
training_data = []
def create_training_data():
    for category in CATEGORIESTraining:
        path = os.path.join(DATADIR, category) # path to img/training or pixel-level-gt/training dir
        secondPath = os.path.join(DATADIR,CATEGORIESResult[CATEGORIESTraining.index(category)])
        #class_num = CATEGORIES.index(category)
        newImg =  os.listdir(secondPath)
        i = 0;
        for img in os.listdir(path):
            try:
                imgTraining_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
        #        newTraining_array = cv2.resize(imgTraining_array, (50,50))
                imgResult_array = cv2.imread(os.path.join(secondPath,newImg[i]),cv2.IMREAD_COLOR)
                #imgResult_array = cv2.imread(os.path.join(secondPath,os.listdir(secondPath))[img.index],cv2.IMREAD_COLOR)
         #       newResult_array = cv2.resize(imgResult_array, (50,50))
                training_data.append([imgTraining_array,imgResult_array])
                i= i+1
            except Exception as e:
                pass

create_training_data()
plt.imshow(training_data[0][1], cmap="gray")
plt.show()
