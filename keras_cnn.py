import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow_core.python.keras.layers import Conv2D, Flatten, Dense
from tensorflow.python.keras import Sequential

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def get_dataset(training=True):
    fashion_mnist = keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    # Expand Dims
    # print(np.shape(train_images))s
    train_images = np.expand_dims(train_images, axis=3)
    test_images = np.expand_dims(test_images, axis=3)
    if not training:
        return np.array(test_images), np.array(test_labels)
    else:
        return np.array(train_images), np.array(train_labels)
    # print(np.shape(train_images))


def build_model():
    model = keras.Sequential()
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


def train_model(model, train_img, train_lab, test_img, test_lab, T):
    train_lab = keras.utils.to_categorical(train_lab)
    test_lab = keras.utils.to_categorical(test_lab)
    model.fit(train_img, train_lab, validation_data=(test_img, test_lab), epochs=T)


def predict_label(model, images, index):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    list = model.predict(images)
    i = 0
    res = []
    for elem in list[index]:
        res.append([elem, i])
        i += 1
    res = sorted(res, key=lambda x: x[0], reverse=True)
    # print(res)
    for i in range(3):
        print('{}: {}%'.format(class_names[res[i][1]], format(round(res[i][0] * 100, 2), ".2f")))



# train_images, train_labels = get_dataset()
# # # print(train_images.shape)
# test_images, test_labels = get_dataset(False)
# model = build_model()
# #
# train_model(model, train_images, train_labels, test_images, test_labels, 1)
# predict_label(model, test_images, 0);
# keras.utils.plot_model(models, to_file='model.png')
