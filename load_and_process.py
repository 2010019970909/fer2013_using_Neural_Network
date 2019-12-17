# Source: https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier


# Load the data and convert them to numpy files fdataX.npy and flabels.npy 
def load_and_process(datafile='./fer2013.csv', images_key='pixels', labels_key='emotion', width=48, height=48):
    print('Open the file')
    data = pd.read_csv(datafile)
    print('File opened, procced to the data extraction')

    # getting the images from the dataset
    datapoints = data[images_key].tolist()
    usage = np.array(data['Usage'].tolist())
    y = np.array(data['emotion'].tolist()) # Extract the target

    # getting features for training
    X = []
    for xseq in datapoints:
        X.append([int(pixel) for pixel in xseq.split(' ')])# [int(xp) for xp in xseq.split(' ')]).reshape(width, height).astype('float32'))

    X = np.asarray(X)
    # adapt the dimension of the data for the CNN
    # X = np.expand_dims(np.asarray(X), -1)

    # getting labels for training
    # y = pd.get_dummies(data[labels_key]).as_matrix()

    print("Extract the training, validation and testing set from the data")
    # Extract the training, validation and testing set from the data
    train_in  = X[usage == 'Training']
    train_out = y[usage == 'Training']
    # print(len(train_in), len(train_out))

    valid_in  = X[usage == 'PrivateTest']
    print(valid_in.shape)
    valid_out = y[usage == 'PrivateTest']
    print(valid_out.shape)
    # print(len(valid_in), len(valid_out))

    test_in   = X[usage == 'PublicTest']
    test_out  = y[usage == 'PublicTest']
    # print(len(test_in), len(test_out))

    print('Shuffle the sets')
    # Shuffling the training database in order to improve the efficiency 
    train_in, train_out = shuffle(train_in, train_out)
    valid_in, valid_out = shuffle(valid_in, valid_out)
    test_in, test_out   = shuffle(test_in, test_out)

    # m_classes = len(np.unique(train_out))
    # print(m_classes)

    print('Normalise the sets')
    # Normalise the input variables
    scaler_train    = preprocessing.MinMaxScaler(feature_range=(-128,127)).fit(train_in)
    scaled_train_in = scaler_train.transform(train_in)
    
    scaler_valid    = preprocessing.MinMaxScaler(feature_range=(-128,127)).fit(train_in)
    scaled_valid_in = scaler_valid.transform(valid_in)
    # print(scaled_valid_in.shape)

    scaler_test     = preprocessing.MinMaxScaler(feature_range=(-128,127)).fit(train_in)
    scaled_test_in  = scaler_test.transform(test_in)
    # print(len(scaled_train_in), len(scaled_valid_in))

    print('PCA analysis')
    # PCA analysis
    pca = PCA(n_components = 103)
    model_pca = pca.fit(scaled_train_in) # Fit the PCA to the data
    pca_train = model_pca.transform(scaled_train_in)

    pca_valid = model_pca.transform(scaled_valid_in)
    print('PCA VALID SHAPE' , pca_valid.shape)

    pca_test  = model_pca.transform(scaled_test_in)

    # Scale the data
    scaler = StandardScaler()  # doctest: +SKIP
    # Don't cheat - fit only on training data
    scaler.fit(pca_train)  # doctest: +SKIP
    scaled_pca_train = scaler.transform(pca_train)  # doctest: +SKIP
    # apply same transformation to test data
    scaled_pca_valid = scaler.transform(pca_valid)  # doctest: +SKIP
    scaled_pca_test = scaler.transform(pca_test)  # doctest: +SKIP

    # CLASSES_NAME = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # Single Perceptron
    print('Train perceptron from PCA')
    clf = Perceptron(fit_intercept=False,
                     max_iter=1000,
                     tol=1e-3,
                     shuffle=False).fit(scaled_pca_train, train_out)
    print('End training')

    # print(clf.predict(scaled_pca_train))
    print(clf.score(scaled_pca_train, train_out))

    train_pred        = clf.predict(scaled_pca_train)
    conf_matrix_train = confusion_matrix(train_out, train_pred)
    print(conf_matrix_train)

    # print(clf.predict(scaled_pca_valid))
    # print(scaled_pca_valid.shape, valid_out.shape)
    print(clf.score(scaled_pca_valid, valid_out))
    valid_pred        = clf.predict(scaled_pca_valid)
    conf_matrix_valid = confusion_matrix(valid_out, valid_pred)
    print(conf_matrix_valid)

    # print(clf.predict(scaled_pca_test))
    print(clf.score(scaled_pca_test, test_out))
    test_pred        = clf.predict(scaled_pca_test)
    conf_matrix_test = confusion_matrix(test_out, test_pred)
    print(conf_matrix_test) 
    
    # MLP
    print('Train MLP from PCA')
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1).fit(scaled_pca_train, train_out)
    print('End training')

    # print(clf.predict(scaled_pca_train))
    print(clf.score(scaled_pca_train, train_out))

    train_pred        = clf.predict(scaled_pca_train)
    conf_matrix_train = confusion_matrix(train_out, train_pred)
    print(conf_matrix_train)

    # print(clf.predict(scaled_pca_valid))
    # print(scaled_pca_valid.shape, valid_out.shape)
    print(clf.score(scaled_pca_valid, valid_out))
    valid_pred        = clf.predict(scaled_pca_valid)
    conf_matrix_valid = confusion_matrix(valid_out, valid_pred)
    print(conf_matrix_valid)

    # print(clf.predict(scaled_pca_test))
    print(clf.score(scaled_pca_test, test_out))
    test_pred        = clf.predict(scaled_pca_test)
    conf_matrix_test = confusion_matrix(test_out, test_pred)
    print(conf_matrix_test)

    print('Train perceptron from normalised pixels')
    clf = Perceptron(fit_intercept=False,
                     max_iter=1000,
                     tol=1e-3,
                     shuffle=False).fit(scaled_train_in, train_out)
    print('End training')

    # print(clf.predict(scaled_pca_train))
    print(clf.score(scaled_train_in, train_out))

    train_pred        = clf.predict(scaled_train_in)
    conf_matrix_train = confusion_matrix(train_out, train_pred)
    print(conf_matrix_train)

    # print(clf.predict(scaled_pca_valid))
    # print(scaled_pca_valid.shape, valid_out.shape)
    print(clf.score(scaled_valid_in, valid_out))
    valid_pred        = clf.predict(scaled_valid_in)
    conf_matrix_valid = confusion_matrix(valid_out, valid_pred)
    print(conf_matrix_valid)

    # print(clf.predict(scaled_pca_test))
    print(clf.score(scaled_test_in, test_out))
    test_pred        = clf.predict(scaled_test_in)
    conf_matrix_test = confusion_matrix(test_out, test_pred)
    print(conf_matrix_test) 
    
    # MLP
    print('Train MLP from normalised pixels')
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(100,), random_state=1).fit(scaled_train_in, train_out)
    print('End training')

    # print(clf.predict(scaled_pca_train))
    print(clf.score(scaled_train_in, train_out))

    train_pred        = clf.predict(scaled_train_in)
    conf_matrix_train = confusion_matrix(train_out, train_pred)
    print(conf_matrix_train)

    # print(clf.predict(scaled_pca_valid))
    # print(scaled_pca_valid.shape, valid_out.shape)
    print(clf.score(scaled_valid_in, valid_out))
    valid_pred        = clf.predict(scaled_valid_in)
    conf_matrix_valid = confusion_matrix(valid_out, valid_pred)
    print(conf_matrix_valid)

    # print(clf.predict(scaled_pca_test))
    print(clf.score(scaled_test_in, test_out))
    test_pred        = clf.predict(scaled_test_in)
    conf_matrix_test = confusion_matrix(test_out, test_pred)
    print(conf_matrix_test)

    """
    Conclusion:

    Perceptron, PCA (accuracy of around 20%) was less efficient than MLP, PCA
    (accuracy of around 40%) but they both worked better than Perceptron, pixels
    (accuracy of around 20%) and MLP, pixels (accuracy of around 10%).
    Also, in the confusion matrix we can see that the emotion 1 (disgust)
    is the most difficult to classify (in any case).
    With 1000 hidden layers (against the 15 of before) and the adam solver
    (against the 'lbfgs') the NNet gets overfitted but we slightly improve
    the performances with the normalised pixels.
    The perceptron works better with the pixels with an accuracy of around 30%
    against 20% with the PCA. For the 1000 hidden layers MLP, PCA the accuracy is
    around 63% for the training and around 40% for the test (validation and test set).
    For the 1000 hidden layers MLP, pixels the accuracy is
    around 61% for the training and around 35% for the test (validation and test set).
    The state-of-the-art for the image classifier is an ImageNet network
    (https://paperswithcode.com/sota/image-classification-on-imagenet) which can reach
    above 70% accuracy values. Also the usage of CNN should improve the accuracy of
    the neural network, as implemented in the following article:
    https://medium.com/themlblog/how-to-do-facial-emotion-recognition-using-a-cnn-b7bbae79cd8f
    """

if __name__ == "__main__":
    load_and_process()
    print("Preprocessing Done")