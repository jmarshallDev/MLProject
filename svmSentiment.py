import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import preprocessing as pp
from sklearn.metrics import confusion_matrix
import pdb


def main():
    training_data, validation_data, testing_data, training_labels, validation_labels, testing_labels = pp.gimme()

    C=0.86
    clf = svm.SVC(kernel='poly', degree=2, C=C) #rbf is current best at 75.2508% and 93.4% for training, linear is second at
                                            #74.6% accuracy, and sigmoid is right behind with 73.9% accuracy
                                            #0.8193979933110368 at c=1 linear
                                            #0.8093645484949833 at c = 0.5 at linear
                                            # 0.8294314381270903 at c = 0.75 and 0.8
                                            # 0.8327759197324415 at C=0.
                                            # 0.8127090301003345 poly deg 2 c=0.75
                                            # 0.8260869565217391 poly deg 2 C=0.85
                                            # 0.8294314381270903 poly deg 2 C=0.9
                                            # 0.8327759197324415 poly deg 2 C=0.86 or C=0.865
                                            # 0.8160535117056856 poly deg 3 C>=0.58 <= 0.86
    clf.fit(training_data, training_labels)
    accuracy = np.mean(clf.predict(testing_data) == testing_labels)
    print("Quadratic kernel SVM with C=0.86")
    print("The testing accuracy is ", accuracy)
    print("Confusion matrix:")
    print(confusion_matrix(testing_labels, clf.predict(testing_data)))



if __name__ == '__main__':
    main()
