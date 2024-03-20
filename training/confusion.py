import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
from keras.models import load_model


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

model = load_model('vgg16.h5')

tp = 0
tn = 0
fp = 0
fn = 0
for i in range(112):
    picture = Pixels_test[i]
    picture = np.expand_dims(picture, axis=0)
    label = Labels_test[i]
    result = np.argmax(model.predict(picture))
    # result = float(model.predict(picture))

    if result == 1 and label == 1:
        tp += 1
    if result == 0 and label == 1:
        fn += 1
    if result == 1 and label == 0:
        fp += 1
    if result == 0 and label == 0:
        tn += 1

    # if result >= 0.5 and label == 1:
    #     tp += 1
    # if result >= 0.5 and label == 0:
    #     fn += 1
    # if result < 0.5 and label == 1:
    #     fp += 1
    # if result < 0.5 and label == 0:
    #     tn += 1


print(f"TP: {tp}\nFP: {fp}\nFN: {fn}\nTN: {tn}")

score = model.evaluate(Pixels_test, Labels_test, verbose=0)
print('Test loss =', score[0])
print('Test accuracy =', score[1])


