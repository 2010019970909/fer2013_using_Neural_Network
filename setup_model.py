
import sys, os
import pandas as pd
# pip install keras tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
# from keras.losses import categorical_crossentropy
# from keras.optimizers import Adam
from keras.regularizers import l2

"""
batch_size = 64
epochs = 100

"""

# model from the tutorial: https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f
def tuto_model(num_features = 64, num_labels = 7, width=48, height=48, print_summary=False):
    #desinging the CNN
    model = Sequential()

    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*2*2*num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2*2*2*num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2*2*num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2*num_features, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, activation='softmax'))

    if print_summary:
        model.summary()

    return model

def ecption():
    pass

def custom_model():
    pass

def main(select_model=0, filename=None, print_summary=True):
    if select_model==0: # Tuto model
        if filename:
            tuto_model(num_features = 64, num_labels = 7, width=48, height=48, print_summary=print_summary)
        else:
            tuto_model(num_features = 64, num_labels = 7, width=48, height=48, print_summary=print_summary)
    elif select_model==1: # Xecption
        if filename:
            pass
        else:
            pass
    elif select_model==2: # custom_model
        if filename:
            pass
        else:
            pass

if __name__ == "__main__":
    main(select_model=3, filename=None, print_summary=True)
