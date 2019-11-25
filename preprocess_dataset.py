# make sure to install sklearn and numpy to be able to use them
# for python 3.8: https://github.com/numpy/numpy/issues/12016#issuecomment-555354320
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess(fdataX='./fdataX.npy', flabels='./flabels.npy'):
    # load the data
    x = np.load(fdataX)
    y = np.load(flabels)

    # Normalise the data
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    """
    for xx in range(10):
        plt.figure(xx)
        plt.imshow(x[xx].reshape((48, 48)), interpolation='none', cmap='gray')
    plt.show()
    """

    # splitting into training, validation and testing data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

    # saving the test samples to be used later
    np.save('modXtrain', X_train)
    np.save('modytrain', y_train)
    np.save('modXtest', X_test)
    np.save('modytest', y_test)
    np.save('modXvalid', X_valid)
    np.save('modyvalid', y_valid)

if __name__ == "__main__":
    preprocess()