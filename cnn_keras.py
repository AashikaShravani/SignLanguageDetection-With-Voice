import numpy as np
import pickle
import cv2
import os
from keras import optimizers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image_size():
    img = cv2.imread('gestures/1/100.jpg', 0)
    return img.shape

def get_num_of_classes():
    return len(os.listdir('gestures/'))

image_x, image_y = get_image_size()

def cnn_model():
    num_of_classes = get_num_of_classes()
    model = Sequential()
    model.add(Conv2D(16, (2,2), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same'))
    model.add(Conv2D(64, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_of_classes, activation='softmax'))
    sgd = optimizers.SGD(lr=1e-2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    filepath="cnn_model_keras2.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint1]
    return model, callbacks_list

def train():
    with open("train_images", "rb") as f:
        train_images = np.array(pickle.load(f))
    with open("train_labels", "rb") as f:
        train_labels = np.array(pickle.load(f), dtype=np.int32)

    # Resize the images to match the input shape of the model
    train_images_resized = np.zeros((train_images.shape[0], image_x, image_y))
    for i in range(train_images.shape[0]):
        train_images_resized[i] = cv2.resize(train_images[i], (image_x, image_y))

    train_images_resized = np.reshape(train_images_resized, (-1, image_x, image_y, 1))

    # Ensure that the number of classes matches the expected output shape of the model
    num_of_classes = get_num_of_classes()
    if num_of_classes != train_labels.max() + 1:
        print("Number of classes in labels does not match the expected output shape of the model.")
        return

    train_labels = to_categorical(train_labels, num_classes=num_of_classes)

    model, callbacks_list = cnn_model()
    model.summary()
    model.fit(train_images_resized, train_labels, epochs=20, batch_size=500, callbacks=callbacks_list)

train()
