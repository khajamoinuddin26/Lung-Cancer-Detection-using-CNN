import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from keras.models import Model, save_model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, concatenate
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D


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


# model
def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):
    # Bottleneck layers
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(bn_size * nb_filter, (1, 1), strides=(1, 1), padding='same')(x)

    # Composite function
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (3, 3), strides=(1, 1), padding='same')(x)

    if drop_rate: x = Dropout(drop_rate)(x)

    return x


def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):
    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=3)

    return x


def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):
    nb_filter = int(x.shape.as_list()[-1] * compression)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (1, 1), strides=(1, 1), padding='same')(x)
    if is_max != 0:
        x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    else:
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    return x


growth_rate = 12
inpt = Input(shape=(256, 256, 3))
x = Conv2D(growth_rate * 2, (3, 3), strides=1, padding='same')(inpt)
x = BatchNormalization(axis=3)(x)
x = LeakyReLU(alpha=0.1)(x)
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = TransitionLayer(x)
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = TransitionLayer(x)
x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)
x = BatchNormalization(axis=3)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation='sigmoid')(x)
model = Model(inpt, x)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# model.summary()
r = model.fit(Pixels_train, Labels_train, validation_data=(Pixels_test, Labels_test), batch_size=32, epochs=10, verbose=1)
save_model(model, 'Dense.h5')

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
