import tensorflow as tf
import os
import cv2
import imghdr
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.metrics import Precision, Recall, BinaryAccuracy
import cv2


model = load_model(os.path.join("models", "happy_sad_model.h5"))
img = cv2.imread("men3.jpg")
img1 = cv2.imread("woman6.jpg")
plt.imshow(img)
plt.show()
resize = tf.image.resize(img, (256, 256))
resize1 = tf.image.resize(img1, (256, 256))
result = []
result.append(model.predict(np.expand_dims(resize/255.0, 0)))
result.append(model.predict(np.expand_dims(resize1/255.0, 0)))
print(result)
for yhat in result:
    if yhat < 0.5:
        print("Happy")
    else:
        print("Sad")
