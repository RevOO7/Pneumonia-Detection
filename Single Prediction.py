import numpy as np
import cv2
import os
from keras.preprocessing import image
from keras.models import load_model
model = load_model('pneumonia-model&weights.h5')
model.summary()

model.load_weights('pneumonia-weights.h5')

#model.get_weights()

#New predictions

img_path = 'data/NORMAL/IM-0022-0001.png'

img = cv2.imread(img_path)
img = image.img_to_array(img)
img = np.expand_dims(img,axis=0)
result = model.predict(img)

if result == 1.0:
    pred = 'Positive'
else:
    pred = 'Negative'  
print (pred)
    

    