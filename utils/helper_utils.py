from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt


def prepare_data():

    data = np.genfromtxt('../data/data.csv', delimiter=',')

    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

    return X_train, X_test, y_train, y_test


def plot_mean_error_vs_num_features(num_feature_list, error_list):

    plt.plot(num_feature_list, error_list, '-*')
    plt.show()


def prepare_wine_quality_data():

    data = np.genfromtxt('../../data/uci-wine-quality-dataset/winequality-data.csv', delimiter=',', skip_header=1)

    X = data[:, :11]
    y = data[:, 11]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

    return X_train, X_test, y_train, y_test


def prepare_wine_quality_test_data():
    data = np.genfromtxt('../../data/uci-wine-quality-dataset/winequality-solution-input.csv', delimiter=',', skip_header=1)

    X_test = data[:, :11]
    ids = data[:, 11]

    return X_test, ids

