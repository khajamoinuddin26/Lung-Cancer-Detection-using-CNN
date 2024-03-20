import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from tensorflow import keras
from keras.models import save_model
from keras.layers import Input


# data process
def preprocess(pixels):
    strentch = np.std(pixels)
    pixels /= strentch
    mean = np.mean(pixels)
    pixels -= mean


Path = "Dataset/"
folders = ["Bengin cases/Bengin case ", "Malignant cases/Malignant case ", "Normal cases/Normal case "]
number = [120, 561, 416]
Pixels_train = []
Labels_train = []
Pixels_test = []
Labels_test = []


for i in range(3):
    folder = folders[i]
    for j in range(number[i]):

        if i == 1:
            label = 1
        else:
            label = 0

        picture_number = "(" + str(j + 1) + ").jpg"
        picture = Path + folder + picture_number
        image = plt.imread(picture)
        image = resize(image, output_shape=(256, 256, 3))
        if j <= number[i] / 10:
            Pixels_test.append(image)
            Labels_test.append(label)
        else:
            Pixels_train.append(image)
            Labels_train.append(label)


Pixels_train = np.asarray(Pixels_train, dtype=float)
Pixels_test = np.asarray(Pixels_test, dtype=float)
preprocess(Pixels_train)
preprocess(Pixels_test)

Labels_train = np.asarray(Labels_train, dtype=int)
Labels_test = np.asarray(Labels_test, dtype=int)


# Model
# inpt = Input(shape=(256, 256, 3))
dcnn = keras.models.Sequential()
dcnn.add(keras.layers.Conv2D(1, kernel_size=7, strides=3, activation='relu', kernel_initializer='uniform'))
dcnn.add(keras.layers.BatchNormalization())
dcnn.add(keras.layers.MaxPooling2D(pool_size = (2, 2), strides = 2))
dcnn.add(keras.layers.Conv2D(96, kernel_size=5, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
dcnn.add(keras.layers.BatchNormalization())
dcnn.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2))
dcnn.add(keras.layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
dcnn.add(keras.layers.Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
dcnn.add(keras.layers.Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initializer='uniform'))
dcnn.add(keras.layers.MaxPooling2D(pool_size=(3, 3), strides=3))
dcnn.add(keras.layers.Flatten())
dcnn.add(keras.layers.Dense(256, activation='relu'))
dcnn.add(keras.layers.Dropout(0.5))
dcnn.add(keras.layers.Dense(2048, activation='relu'))
dcnn.add(keras.layers.Dropout(0.5))
dcnn.add(keras.layers.Dense(1, activation='sigmoid'))
dcnn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0002, decay=1e-8), loss=['binary_crossentropy'], metrics=['accuracy'])

# model.summary()
r = dcnn.fit(Pixels_train, Labels_train, validation_data=(Pixels_test, Labels_test), epochs=10, verbose=1)
save_model(dcnn, 'Dcnn.h5')


print(r.history)

plt.plot(r.history['accuracy'], label='acc')
plt.plot(r.history['val_accuracy'], label='val_acc')
plt.legend()
plt.savefig('acc.png')
plt.clf()

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()
plt.savefig('loss.png')
plt.clf()
