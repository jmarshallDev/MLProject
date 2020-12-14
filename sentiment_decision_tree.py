 # decisiontree.py
"""Predict Sentiment of a given sentence using a decision tree."""

import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from preprocessing import gimme

ROOT = os.path.dirname(os.path.abspath(__file__))  # root directory of this code


def main():
    # Loading and Splitting Data
    training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels = gimme()

    # Train a decision tree via information gain on the training data
    clf = DecisionTreeClassifier(
        criterion="entropy",
        splitter="best",
        max_depth=None,  # class_weight="balanced",
        random_state=0)
    clf.fit(training_data, training_labels)

    # Test the decision tree
    pred = clf.predict(testing_data)

    # Compare training and test accuracy
    print("train accuracy =", np.mean(training_labels == clf.predict(training_data)))
    print("test accuracy =", np.mean(testing_labels == pred))

    # Show the confusion matrix for test data
    cm = confusion_matrix(testing_labels, pred)
    print("Confusion matrix:")
    print('\n'.join([''.join(['{:4}'.format(item) for item in row]) for row in cm]))

    # Visualize the tree using matplotlib and plot_tree
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(11, 5), dpi=150)
    plot_tree(clf, class_names=["positive", "negative"], filled=True, rounded=True, fontsize=6)
    plt.show()


if __name__ == '__main__':
    main()
