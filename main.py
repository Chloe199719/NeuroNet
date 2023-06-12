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

gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = "data"
os.listdir(data_dir)

image_exts = ["jpeg", "jpg", "bmp", "png"]
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path)
            tip = imghdr.what(image_path)
            if (tip) not in image_exts:
                print('image not in ext  list {}'.format(image_path))
                os.remove(image_path)
        except Exception as e:
            print("issue with image {}".format(image_path))
            os.remove(image_path)

data = tf.keras.utils.image_dataset_from_directory("data")
data = data.map(lambda x, y: (x/255.0, y))
scalar = data.as_numpy_iterator().next()

train_size = int(0.85 * len(data))
val_size = int(0.15 * len(data))
test_size = int(0.1 * len(data))

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)


model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation='relu'))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile("adam", loss=tf.losses.BinaryCrossentropy(),
              metrics=["accuracy"])
print(model.summary())
logdir = "logs"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
hist = model.fit(train, epochs=5, validation_data=val,
                 callbacks=[tensorboard_callback])

pre = Precision()
rec = Recall()
acc = BinaryAccuracy()


for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    rec.update_state(y, yhat)
    acc.update_state(y, yhat)
print(
    f'Precision: {pre.result().numpy()} , Recall: {rec.result().numpy()} , Accuracy: {acc.result().numpy()}')

img = cv2.imread("chloe.jpg")
img1 = cv2.imread("sad.jpg")
plt.imshow(img)
plt.show()
resize = tf.image.resize(img, (256, 256))
resize1 = tf.image.resize(img1, (256, 256))
yhat = model.predict(np.expand_dims(resize/255.0, 0))
yhat1 = model.predict(np.expand_dims(resize1/255.0, 0))
print(yhat, yhat1)

model.save(os.path.join("models", "manorwomman.h5"))
# data_iterrator = data.as_numpy_iterator()
# batch = data_iterrator.next()

# fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
# for idx, img in enumerate(batch[0][:4]):
#     ax[idx].imshow(img.astype(int))
#     ax[idx].title.set_text(batch[1][idx])
