import numpy as np
import keras
import cv2
import os
import argparse

from keras.models import Sequential
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense

parser = argparse.ArgumentParser()

parser.add_argument('-f', dest='file', required=True, help='File Path')
parser.add_argument('-m', dest='model', required=False, help='Model Path')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

num_classes = 12

input_shape = (256, 256, 1)

breeds = ['Sphynx', 'Birman', 'Egyptian Mau', 'Ragdoll', 'Abyssinian', 'Siamese', 'Maine Coon', 'Bengal', 'British Shorthair', 'Bombay', 'Russian Blue', 'Persian']

args = parser.parse_args()

model = Sequential(
    [
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dropout(0.5),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ]
)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

if args.model is None:
    model.load_weights('models/save_at_50.h5')
else:
    model.load_weights(args.model)

# img = cv2.imread('photos/Test/Persian.jpg', cv2.IMREAD_GRAYSCALE)
try:
    img = cv2.imread(args.file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, dsize=input_shape[:2], interpolation=cv2.INTER_CUBIC)
    x = img.astype('float32') / 255
    x = np.expand_dims(x, -1)
    x = np.expand_dims(x, 0)
    h = np.argmax(model.predict(x))

    print(breeds[h])
except IndexError:
    print('Please provide file path')