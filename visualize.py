import pickle
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

def main():
    car()
    tic()

def car():
    car_tree_grid = pickle.load(open('cv_outputs/new_cardecision_tree.data', 'rb'))
    car_nn_grid = pickle.load(open('cv_outputs/new_carnn.data', 'rb'))
    car_boost_grid = pickle.load(open('cv_outputs/new_carboost.data', 'rb'))
    car_svm1_grid = pickle.load(open('cv_outputs/new_carsvm_poly.data', 'rb'))
    car_svm2_grid = pickle.load(open('cv_outputs/new_carsvm_rbf.data', 'rb'))
    car_knn_grid = pickle.load(open('cv_outputs/new_carknn.data', 'rb'))

    w_tree_df = pd.DataFrame({'x': [item.parameters['max_depth'] for item in car_tree_grid],
                              'y': [item.parameters['min_samples_leaf'] for item in car_tree_grid],
                              'z': [1- item.mean_validation_score for item in car_tree_grid]})

    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_tree_df.x, w_tree_df.y, w_tree_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Car Evaluation Decision Tree')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Min Samples')
    ax.set_zlabel('Error')
    plt.savefig('car-evaluation-decision-tree.png')
    plt.show()

    w_nn_df = pd.DataFrame({'x': [item.parameters['hidden_layer_sizes'][0] for item in car_nn_grid],
                            'y': [item.parameters['max_iter'] for item in car_nn_grid],
                            'z': [1- item.mean_validation_score for item in car_nn_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_nn_df.x, w_nn_df.y, w_nn_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Car Evaluation Neural Net')
    ax.set_xlabel('Hidden Layer Size')
    ax.set_ylabel('Max Iterations')
    ax.set_zlabel('Error')
    plt.savefig('car-evaluation-neural-net.png')
    plt.show()

    w_boost_df = pd.DataFrame({'x': [item.parameters['n_estimators'] for item in car_boost_grid],
                               'y': [item.parameters['learning_rate'] for item in car_boost_grid],
                               'z': [1- item.mean_validation_score for item in car_boost_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_boost_df.x, w_boost_df.y, w_boost_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel('N Estimators')
    plt.title('Car Evaluation Boosted')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel('Error')
    plt.savefig('car-evaluation-boosted.png')
    plt.show()

    w_svm1_df = pd.DataFrame({'x': [item.parameters['C'] for item in car_svm1_grid],
                              'y': [item.parameters['degree'] for item in car_svm1_grid],
                              'z': [1- item.mean_validation_score for item in car_svm1_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_svm1_df.x, w_svm1_df.y, w_svm1_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Car Evaluation SVM (Poly)')
    ax.set_xlabel('C')
    ax.set_ylabel('Degree')
    ax.set_zlabel('Error')
    plt.savefig('car-evaluation-svm-poly.png')
    plt.show()

    w_svm2_df = pd.DataFrame({'x': [item.parameters['C'] for item in car_svm2_grid],
                              'y': [item.parameters['gamma'] for item in car_svm2_grid],
                              'z': [1- item.mean_validation_score for item in car_svm2_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_svm2_df.x, w_svm2_df.y, w_svm2_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Car Evaluation SVM (RBF)')
    ax.set_xlabel('C')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Error')
    plt.savefig('car-evaluation-svm-rbf.png')
    plt.show()

    w_knn_df = pd.DataFrame({'x': [item.parameters['n_neighbors'] for item in car_knn_grid],
                             'y': [item.parameters['p'] for item in car_knn_grid],
                             'z': [1- item.mean_validation_score for item in car_knn_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_knn_df.x, w_knn_df.y, w_knn_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Car Evaluation k-Nearest Neighbors')
    ax.set_xlabel('N Neighbors')
    ax.set_ylabel('P')
    ax.set_zlabel('Error')
    plt.savefig('car-evaluation-nearest-neighbors.png')
    plt.show()


def tic():
    tic_tree_grid = pickle.load(open('cv_outputs/new_tic_decision_tree.data', 'rb'))
    tic_nn_grid = pickle.load(open('cv_outputs/new_tic_nn.data', 'rb'))
    tic_boost_grid = pickle.load(open('cv_outputs/new_tic_boost.data', 'rb'))
    tic_svm1_grid = pickle.load(open('cv_outputs/new_tic_svm_poly.data', 'rb'))
    tic_svm2_grid = pickle.load(open('cv_outputs/new_tic_svm_rbf.data', 'rb'))
    tic_knn_grid = pickle.load(open('cv_outputs/new_tic_knn.data', 'rb'))

    w_tree_df = pd.DataFrame({'x': [item.parameters['max_depth'] for item in tic_tree_grid],
                              'y': [item.parameters['min_samples_leaf'] for item in tic_tree_grid],
                              'z': [1- item.mean_validation_score for item in tic_tree_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_tree_df.x, w_tree_df.y, w_tree_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Tic Tac Toe Decision Tree')
    ax.set_xlabel('Max Depth')
    ax.set_ylabel('Min Samples')
    ax.set_zlabel('Error')
    plt.savefig('tic-tac-toe-decision-tree.png')
    plt.show()

    w_nn_df = pd.DataFrame({'x': [item.parameters['hidden_layer_sizes'][0] for item in tic_nn_grid],
                            'y': [item.parameters['max_iter'] for item in tic_nn_grid],
                            'z': [1- item.mean_validation_score for item in tic_nn_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_nn_df.x, w_nn_df.y, w_nn_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Tic Tac Toe Neural Net')
    ax.set_xlabel('Hidden Layer Size')
    ax.set_ylabel('Max Iterations')
    ax.set_zlabel('Error')
    plt.savefig('tic-tac-toe-neural-net.png')
    plt.show()

    w_boost_df = pd.DataFrame({'x': [item.parameters['n_estimators'] for item in tic_boost_grid],
                               'y': [item.parameters['learning_rate'] for item in tic_boost_grid],
                               'z': [1- item.mean_validation_score for item in tic_boost_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_boost_df.x, w_boost_df.y, w_boost_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Tic Tac Toe Boosted')
    ax.set_xlabel('N Estimators')
    ax.set_ylabel('Learning Rate')
    ax.set_zlabel('Error')
    plt.savefig('tic-tac-toe-boosted.png')
    plt.show()

    w_svm1_df = pd.DataFrame({'x': [item.parameters['C'] for item in tic_svm1_grid],
                              'y': [item.parameters['degree'] for item in tic_svm1_grid],
                              'z': [1- item.mean_validation_score for item in tic_svm1_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_svm1_df.x, w_svm1_df.y, w_svm1_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Tic Tac Toe SVM (Poly)')
    ax.set_xlabel('C')
    ax.set_ylabel('Degree')
    ax.set_zlabel('Error')
    plt.savefig('tic-tac-toe-svm-poly.png')
    plt.show()

    w_svm2_df = pd.DataFrame({'x': [item.parameters['C'] for item in tic_svm2_grid],
                              'y': [item.parameters['gamma'] for item in tic_svm2_grid],
                              'z': [1- item.mean_validation_score for item in tic_svm2_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_svm2_df.x, w_svm2_df.y, w_svm2_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Tic Tac Toe SVM (RBF)')
    ax.set_xlabel('C')
    ax.set_ylabel('Gamma')
    ax.set_zlabel('Error')
    plt.savefig('tic-tac-toe-svm-rbf.png')
    plt.show()

    w_knn_df = pd.DataFrame({'x': [item.parameters['n_neighbors'] for item in tic_knn_grid],
                             'y': [item.parameters['p'] for item in tic_knn_grid],
                             'z': [1- item.mean_validation_score for item in tic_knn_grid]})
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(w_knn_df.x, w_knn_df.y, w_knn_df.z, cmap=cm.jet, linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Tic Tac Toe k-Nearest Neighbors')
    ax.set_xlabel('N Neighbors')
    ax.set_ylabel('P')
    ax.set_zlabel('Error')
    plt.savefig('tic-tac-toe-nearest-neighbors.png')
    plt.show()

if __name__ == "__main__":
        main()