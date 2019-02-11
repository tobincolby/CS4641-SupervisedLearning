import numpy as np
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import grid_search
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ShuffleSplit

import pickle

def main():
    #crossval(True)
    crossval(False)


def crossval(car_stuff):
    ########### load data ###########
    from sklearn import preprocessing
    if car_stuff:
        subst = "new_car"
        car_file = open("data/car_quality_train.csv")
        car_data = np.loadtxt(car_file, delimiter=",")
        data_X = car_data[:, :-1]
        data_Y = car_data[:, -1]
        data_X_scaled = data_X
        print("Loaded Car Data\n")
    else:


        subst = "new_tic_"
        student_file = open("data/tic_train.csv")
        student_data = np.loadtxt(student_file, delimiter=",")
        data_X = student_data[:, :-1]
        data_Y = student_data[:, -1]
        data_X_scaled = data_X
        print("Loaded Tic Tac Toe Data\n")


    ########### train a decision tree, 5-fold cross validation ###########
    print("Training Decision Trees\n")
    decision_tree = tree.DecisionTreeClassifier()
    decision_tree = decision_tree.fit(data_X, data_Y)
    depth_map = []
    worst_depth = decision_tree.tree_.max_depth


    parameters = {'max_depth': np.arange(2,worst_depth), 'min_samples_leaf': np.arange(1, 50)}
    clf = grid_search.GridSearchCV(tree.DecisionTreeClassifier(),
                                    parameters,
                                    cv=5,
                                    verbose=1,
                                    n_jobs=4)
    clf.fit(X=data_X, y=data_Y)
    best_tree = clf.best_estimator_

    pickle.dump(best_tree, open("cv_outputs/" + subst + "decision_tree.clf", "wb"))
    pickle.dump(clf.grid_scores_, open("cv_outputs/" + subst + "decision_tree.data", "wb"))
    print("Finished Decision Trees\n")

    ########### train a neural net, 5-fold cross validation ###########

    print("Training NNs\n")
    parameters = {'hidden_layer_sizes': tuple((element,) for element in range(2, 200, 10)),
                  'max_iter': np.arange(500, 1100, 50)}
    clf = grid_search.GridSearchCV(MLPClassifier(solver="lbfgs", random_state=1),
                                   parameters,
                                   cv=5,
                                   verbose=1,
                                   n_jobs=4)
    clf.fit(X=data_X_scaled, y=data_Y)
    best = clf.best_estimator_
    pickle.dump(best, open("cv_outputs/" + subst + "nn.clf", "wb"))
    pickle.dump(clf.grid_scores_, open("cv_outputs/" + subst + "nn.data", "wb"))
    print("Finished NNs\n")


    ########### train a boosted dt, 5-fold cross validation ###########
    print("Training Boosted\n")
    parameters = {'n_estimators': np.arange(1,100), 'learning_rate': np.arange(0.1, 2., 0.1)}
    clf = grid_search.GridSearchCV(AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=best_tree.max_depth, min_samples_leaf=best_tree.min_samples_leaf), random_state=1),
                                    parameters,
                                    cv=5,
                                    verbose=1,
                                    n_jobs=4)
    clf.fit(X=data_X, y=data_Y)
    best = clf.best_estimator_
    pickle.dump(best, open("cv_outputs/" + subst + "boost.clf", "wb"))
    pickle.dump(clf.grid_scores_, open("cv_outputs/" + subst + "boost.data", "wb"))
    print("Finished Boosted\n")


    ########### train an svm, 5-fold cross validation ###########
    from sklearn import preprocessing

    data_X_scaled = preprocessing.scale(data_X)

    print("Training SVM 1\n")
    parameters = {'C': np.arange(1,100), 'degree': np.arange(1,10), 'max_iter':np.arange(1100,1300, 50)}
    clf = grid_search.GridSearchCV(SVC(kernel="poly", random_state=1, decision_function_shape='ovo', max_iter=100),
                                    parameters,
                                    cv=5,
                                    verbose=1,
                                    n_jobs=4)
    clf.fit(X=data_X_scaled, y=data_Y)
    best = clf.best_estimator_
    pickle.dump(best, open("cv_outputs/" + subst + "svm_poly.clf", "wb"))
    pickle.dump(clf.grid_scores_, open("cv_outputs/" + subst + "svm_poly.data", "wb"))
    print("Finished SVM 1\n")

    print("Training SVM 2\n")
    parameters = {'C': np.arange(0.1, 1.5, 0.1), 'gamma': np.arange(0.01, 2, 0.1), 'max_iter': np.arange(1100, 1300, 50)}
    clf = grid_search.GridSearchCV(SVC(kernel="rbf", random_state=1, decision_function_shape='ovo'),
                                   parameters,
                                   cv=5,
                                   verbose=1,
                                   n_jobs=4)
    clf.fit(X=data_X_scaled, y=data_Y)
    best = clf.best_estimator_
    pickle.dump(best, open("cv_outputs/" + subst + "svm_rbf.clf", "wb"))
    pickle.dump(clf.grid_scores_, open("cv_outputs/" + subst + "svm_rbf.data", "wb"))
    print("Finished SVM 2\n")


    ########### train a knn, 5-fold cross validation ###########
    print("Training KNN\n")
    parameters = {'n_neighbors': np.arange(1,100), 'p': np.arange(1,5)}
    clf = grid_search.GridSearchCV(KNeighborsClassifier(),
                                    parameters,
                                    cv=5,
                                    verbose=1,
                                    n_jobs=4)
    clf.fit(X=data_X, y=data_Y)
    best = clf.best_estimator_
    pickle.dump(best, open("cv_outputs/" + subst + "knn.clf", "wb"))
    pickle.dump(clf.grid_scores_, open("cv_outputs/" + subst + "knn.data", "wb"))
    print("Finished KNN\n")



if __name__== "__main__":
  main()