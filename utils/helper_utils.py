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

