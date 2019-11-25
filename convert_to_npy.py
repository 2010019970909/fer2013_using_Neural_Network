# Source: https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Load the data and convert them to numpy files fdataX.npy and flabels.npy 
def convert(datafile='./fer2013.csv', images_key='pixels', labels_key='emotion', width=48, height=48):
    data = pd.read_csv(datafile)
    # getting the images from the dataset
    datapoints = data[images_key].tolist()

    # getting features for training
    X = []
    for xseq in datapoints:
        X.append(np.asarray([int(xp) for xp in xseq.split(' ')]).reshape(width, height).astype('float32'))

    # adapt the dimension of the data for the CNN
    X = np.expand_dims(np.asarray(X), -1)

    # getting labels for training
    y = pd.get_dummies(data[labels_key]).as_matrix()

    # storing them using numpy
    np.save('fdataX', X)
    np.save('flabels', y)
    return X, y

if __name__ == "__main__":
    X, y = convert()
    print("Preprocessing Done")
    print("Number of Features: "+str(len(X[0])))
    print("Number of Labels: "+ str(len(y[0])))
    print("Number of examples in dataset:"+str(len(X)))
    print("X,y stored in fdataX.npy and flabels.npy respectively")