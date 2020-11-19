import keras
import os
import numpy as np


print(keras.__version__)
print(tf.__version__)        

path_to_train = ''
path_to_model = ''
path_to_test = ''

model = keras.models.load_model(path_to_model)

labels = []
dirs = os.listdir(path)
for item in dirs:
    labels.append(item)
    np.sort(labels)

dirs = os.listdir(path_to_test)
for item in dirs:
    img = cv2.imread(path_to_test + item)
    img_array = img_to_array(img)
    img_array = img_array/255.0
    pred = model.predict(img_array)
    print(labels[np.argmax(pred)])
