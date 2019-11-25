
import sys, os
import pandas as pd
# pip install keras tensorflow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.regularizers import l2

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

    for layer in model.layers:
        layer.trainable = True

    if print_summary:
        model.summary()

    return model

def xception(width=48, height=48, print_summary=False):
    from keras.applications.xception import Xception, preprocess_input, decode_predictions #299*299
    from keras.layers import GlobalAveragePooling2D
    from keras.models import Model

    base_model = Xception(include_top=False, weights='imagenet', input_shape=(width,height,1))  # La pouvez tester différentes architectures

    # create a custom top classifier
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x)
    model = Model(inputs=base_model.inputs, outputs=predictions)
    
    for layer in model.layers:
        layer.trainable = True

    if print_summary:
        model.summary()

    return model

def custom_model():
    model = Sequential()
    for layer in model.layers:
        layer.trainable = True
    return model

def main(select_model=0, filename=None, print_summary=True, train=False, batch_size = 64, epochs = 100):
    import numpy as np
    if select_model==0: # Tuto model
        model = tuto_model(num_features = 64, num_labels = 7, width=48, height=48, print_summary=print_summary)
    elif select_model==1: # Xecption
        model=xception(width=48, height=48, print_summary=print_summary)
    elif select_model==2: # custom_model
        model=custom_model()

    if train:
        #model.compile(optimizer ='sgd', loss= 'mean_squared_error', metrics=['accuracy'])
        #Compliling the model with adam optimixer and categorical crossentropy loss
        model.compile(loss=categorical_crossentropy,
            optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
            metrics=['accuracy'])

        #training the model
        model.fit(np.load('modXtrain.npy'), np.array('modytrain.npy'),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(np.array('modXvalid.npy'), np.array('modyvalid.npy')),
        shuffle=True)

    if filename:
        #model.save_weights(filename+'_weights.h5')
        model.save(filename+'_complete.h5')

        #saving the  model to be used later
        fer_json = model.to_json()
        with open(filename+".json", "w") as json_file:
            json_file.write(fer_json)
        model.save_weights(filename+".h5")


if __name__ == "__main__":
    main(select_model=1, filename='fer', print_summary=True, train=True)
