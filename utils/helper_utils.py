from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt

BASE_PATH = '/Users/dhanendrasoni/PycharmProjects/feature_selection'


def prepare_data():

    data = np.genfromtxt(BASE_PATH + '/data/data.csv', delimiter=',')

    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

    return X_train, X_test, y_train, y_test


def plot_mean_error_vs_num_features(num_feature_list, error_list):
    fig = plt.figure()
    plt.plot(num_feature_list, error_list, '-*')
    plt.xlabel('Number of features')
    plt.ylabel('Mean Squared Error')
    plt.show()
    fig.savefig(BASE_PATH + '/figures/mse_vs_features.png')


def prepare_wine_quality_data():

    data = np.genfromtxt(BASE_PATH + '/data/uci-wine-quality-dataset/winequality-data.csv', delimiter=',', skip_header=1)

    X = data[:, :11]
    y = data[:, 11]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

    return X_train, X_test, y_train, y_test


def prepare_wine_quality_test_data():
    data = np.genfromtxt(BASE_PATH + '/data/uci-wine-quality-dataset/winequality-solution-input.csv', delimiter=',', skip_header=1)

    X_test = data[:, :11]
    ids = data[:, 11]

    return X_test, ids


def prepare_million_song_data():
    data = np.genfromtxt(BASE_PATH + '/data/million-song-dataset/YearPrediction_500_samples.csv', delimiter=',', skip_header=0)
    # data = np.genfromtxt(BASE_PATH + '/data/million-song-dataset/YearPredictionMSD.csv', delimiter=',', skip_header=0)

    X = data[:, 1:]
    y = data[:, 0]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

    """You should respect the following train / test split:
    train: first 463,715 examples
    test: last 51,630 examples
    It avoids the 'producer effect' by making sure no song
    from a given artist ends up in both the train and test set."""

    # X_train = X[0:463715, :]
    # y_train = y[0:463715]
    #
    # X_test = X[463715:, :]
    # y_test = y[463715:]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100)

    return X_train, X_test, y_train, y_test


def plot_time_error_bar_graph(mean_square_errors, time_deltas, n_feature1, n_feature2):
    index = np.arange(len(mean_square_errors))
    bar_width = 0.25
    fig = plt.figure()
    p1 = plt.bar(index, mean_square_errors, bar_width, label='L1 Error')
    p2 = plt.bar(index + bar_width, time_deltas, bar_width, label='training time (ms)')
    plt.xticks(index + bar_width, (n_feature1, n_feature2))
    plt.xlabel('number of features')
    plt.ylabel('l1e and time')
    plt.legend()
    plt.show()
    fig.savefig(BASE_PATH + '/figures/error_bar_plot.png')


