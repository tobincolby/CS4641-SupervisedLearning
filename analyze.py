import numpy as np
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit
import pickle
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import learning_curve
from sklearn import preprocessing

def main():
    learning()

def learning():
    car_tree_clf = pickle.load(open('cv_outputs/new_cardecision_tree.clf', 'rb'))
    car_nn_clf = pickle.load(open('cv_outputs/new_carnn.clf', 'rb'))
    car_boost_clf = pickle.load(open('cv_outputs/new_carboost.clf', 'rb'))
    car_svm1_clf = pickle.load(open('cv_outputs/new_carsvm_poly.clf', 'rb'))
    car_svm2_clf = pickle.load(open('cv_outputs/new_carsvm_rbf.clf', 'rb'))
    car_knn_clf = pickle.load(open('cv_outputs/new_carknn.clf', 'rb'))
    car_train_file = open("data/car_quality_train.csv")
    car_train_data = np.loadtxt(car_train_file, delimiter=",")
    car_test_file = open("data/car_quality_test.csv")
    car_test_data = np.loadtxt(car_test_file, delimiter=",")



    plot_learning_curve(car_tree_clf, "Car Decision Tree Learning Curve", car_train_data, car_test_data)
    plot_learning_curve(car_nn_clf, "Car Neural Net Learning Curve", car_train_data, car_test_data)
    plot_learning_curve(car_boost_clf, "Car Boosted Decision Tree Learning Curve", car_train_data, car_test_data)
    plot_learning_curve(car_svm1_clf, "Car SVM (Poly) Learning Curve", car_train_data, car_test_data)
    plot_learning_curve(car_svm2_clf, "Car SVM (RBF) Learning Curve", car_train_data, car_test_data)
    plot_learning_curve(car_knn_clf, "Car k-Nearest Neighbors Learning Curve", car_train_data, car_test_data)

    tic_tree_clf = pickle.load(open('cv_outputs/new_tic_decision_tree.clf', 'rb'))
    tic_nn_clf = pickle.load(open('cv_outputs/new_tic_nn.clf', 'rb'))
    tic_boost_clf = pickle.load(open('cv_outputs/new_tic_boost.clf', 'rb'))
    tic_svm1_clf = pickle.load(open('cv_outputs/new_tic_svm_poly.clf', 'rb'))
    tic_svm2_clf = pickle.load(open('cv_outputs/new_tic_svm_rbf.clf', 'rb'))
    tic_knn_clf = pickle.load(open('cv_outputs/new_tic_knn.clf', 'rb'))

    tic_train_file = open("data/tic_train.csv")
    tic_train_data = np.loadtxt(tic_train_file, delimiter=",")
    tic_test_file = open("data/tic_test.csv")
    tic_test_data = np.loadtxt(tic_test_file, delimiter=",")

    plot_learning_curve(tic_tree_clf, "Tic Tac Toe Decision Tree Learning Curve", tic_train_data, tic_test_data)
    plot_learning_curve(tic_nn_clf, "Tic Tac Toe Neural Net Learning Curve", tic_train_data, tic_test_data)
    plot_learning_curve(tic_boost_clf, "Tic Tac Toe Boosted Decision Tree Learning Curve", tic_train_data, tic_test_data)
    plot_learning_curve(tic_svm1_clf, "Tic Tac Toe SVM (Poly) Learning Curve", tic_train_data, tic_test_data)
    plot_learning_curve(tic_svm2_clf, "Tic Tac Toe SVM (RBF) Learning Curve", tic_train_data, tic_test_data)
    plot_learning_curve(tic_knn_clf, "Tic Tac Toe k-Nearest Neighbors Learning Curve", tic_train_data, tic_test_data)

def plot_learning_curve(estimator, title, train, test, ylim=None, train_sizes=np.linspace(.1, 1.0, 10), scale=False):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Error")

    train_scores = np.empty(train_sizes.size, dtype=tuple)
    test_scores = np.empty(train_sizes.size, dtype=tuple)

    if not scale:
        total_train_X = train[:, :-1]
    else:
        total_train_X = preprocessing.scale(train[:, :-1])

    total_train_Y = train[:, -1]

    if not scale:
        total_test_X = test[:, :-1]
    else:
        total_test_X = preprocessing.scale(test[:, :-1])

    total_test_Y = test[:, -1]

    for i in range(train_sizes.size):
        train_avg = np.empty(5)
        test_avg = np.empty(5)
        for j in range(0,5):

            np.random.shuffle(train)
            temp_train = train[:int(len(train) * train_sizes[i]), :]
            if not scale:
                data_X = temp_train[:, :-1]
            else:
                data_X = preprocessing.scale(temp_train[:, :-1])
            data_Y = temp_train[:, -1]
            estimator.fit(X=data_X, y=data_Y)

            train_avg[j] = estimator.score(X=total_train_X, y=total_train_Y)
            test_avg[j] = estimator.score(X=total_test_X, y=total_test_Y)


        train_scores[i] = tuple((np.average(train_avg), np.std(train_avg)))
        test_scores[i] = tuple((np.average(test_avg), np.std(test_avg)))

    plt.grid()

    plt.fill_between(train_sizes, [1-tu[0] - tu[1] for tu in train_scores],
                     [1-tu[0] + tu[1] for tu in train_scores], alpha=0.1,
                     color="r")

    plt.fill_between(train_sizes, [1-tu[0] - tu[1] for tu in test_scores],
                     [1-tu[0] + tu[1] for tu in test_scores], alpha=0.1, color="g")

    plt.plot(train_sizes, [1- tu[0] for tu in train_scores], 'o-', color="r",
             label="Training error")
    plt.plot(train_sizes, [1- tu[0] for tu in test_scores], 'o-', color="g",
             label="Testing error")

    plt.legend(loc="best")
    plt.savefig("%s.png" % title)
    plt.show()


if __name__ == "__main__":
        main()