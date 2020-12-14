#neural network that is for organizing reviews into a positive or negative class

import pandas as pd
import numpy as np
import pdb
import nltk

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from preprocessing  import gimme
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
import random
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
def main():

    nltkDownloads()
    set_random_seed(12)
    traindata, validdata, testdata, trainlabels, validlabels, testlabels = gimme()


    samples, inputs = traindata.shape
    #pdb.set_trace()
    #create a Neural Network
    model = Sequential()

    #this network technically work but the accuracy for the commented out network is terrible
    # model.add(Dense(units=750, activation='sigmoid', name='hidden1', input_shape=(inputs,)))
    # model.add(Dense(units=1000, activation='sigmoid', name='hidden2'))
    # model.add(Dense(units=1500, activation='sigmoid', name='hidden3'))
    # model.add(Dense(units=1000, activation='sigmoid', name='hidden4'))
    # model.add(Dense(units=1, activation='softmax', name='output'))
    # model.summary()



    #this network achieves a decent accuracy
    model.add(Dense(50, activation = "sigmoid", input_shape=(inputs, )))
    model.add(Dropout(0.3, noise_shape=None, seed=None))
    model.add(Dense(50, activation = "sigmoid"))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Dense(50, activation = "sigmoid"))
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Dense(50, activation = "sigmoid"))
    # Output- Layer
    model.add(Dense(1, activation = "sigmoid"))
    model.summary()

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

    # Train the network
    history = model.fit(traindata, trainlabels,
                        epochs=50,
                        batch_size=5,
                        validation_data=(validdata, validlabels),
                        verbose=1)

    # Compute the accuracy
    metrics = model.evaluate(testdata, testlabels, verbose=0)
    # predlabels = model.predict(testdata)
    # print(predlabels)
    # prelabels = np.argmax(predlabels, axis=1)
    # print(predlabels)
    print(f"loss on test data = {metrics[0]:0.4f}")
    print(f"accuracy on test data = {metrics[1]:0.4f}")
    #print(confusion_matrix(testlabels, predlabels))
    # Show the confusion matrix for test data
    pred = model.predict(testdata)
    pred = 1*(pred >= 0.5)
    cm = confusion_matrix(testlabels, pred)
    print("Confusion matrix:")
    print('\n'.join([''.join(['{:6}'.format(item) for item in row]) for row in cm]))

def nltkDownloads():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

def set_random_seed(seed):
    '''Set random seed for repeatability.'''
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    main()
