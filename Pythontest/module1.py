import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

DATADIR = "../PetImages"
CATEGORIES = ["Dog", "Cat"]
training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) # path to cats or dogs dir
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (50,50))
                training_data.append([new_array, class_num])
                #print(img_array.shape)
                #plt.imshow(new_array, cmap="gray")
                #plt.show()
            except Exception as e:
                pass

#            break
#        break
create_training_data()
print(len(training_data))
import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

x = [] #trainingdata
y = [] #labels
for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1,50,50,1)

import pickle 

#pickle_out = open("x.pickle","wb")
#pickle.dump(x,pickle_out)
#pickle_out.close()

#pickle_out = open("y.pickle","wb")
#pickle.dump(y,pickle_out)
#pickle_out.close()

#pickle_in = open("x.pickle","rb")
#x = pickle.load(pickle_in)