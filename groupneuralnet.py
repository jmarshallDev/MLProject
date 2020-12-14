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

def main():
    nltkDownloads()
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
    print(f"loss on test data = {metrics[0]:0.4f}")
    print(f"accuracy on test data = {metrics[1]:0.4f}")

def nltkDownloads():
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

if __name__ == '__main__':
    main()
