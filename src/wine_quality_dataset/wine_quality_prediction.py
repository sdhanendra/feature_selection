from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.forwardSelection import forward_feature_selection
from src.backwardSelection import backward_feature_selection_p_value
from utils.helper_utils import prepare_wine_quality_data, plot_time_error_bar_graph

import matplotlib.pyplot as plt
import time


def wine_quality_prediction(selection_method, num_features):

    X_train, X_test, y_train, y_test = prepare_wine_quality_data()

    if selection_method == 'forward':
        feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, num_features)
    elif selection_method == 'backward':
        feature_set = backward_feature_selection_p_value(X_train, X_test, y_train, y_test, num_features)

    # print(feature_set)

    model = LinearRegression()

    start = time.time()

    model.fit(X_train[:, feature_set], y_train)

    end = time.time()

    time_diff = end - start
    print(time_diff * 1000)

    y_hat = model.predict(X_test[:, feature_set])

    mean_square_error = mean_squared_error(y_test, y_hat)

    return mean_square_error, time_diff*1000


def main():
    mean_square_error1, time_diff1 = wine_quality_prediction(selection_method='backward', num_features=8)

    mean_square_error2, time_diff2 = wine_quality_prediction(selection_method='backward', num_features=11)

    mean_square_errors = [mean_square_error1, mean_square_error2]
    time_deltas = [time_diff1, time_diff2]

    plot_time_error_bar_graph(mean_square_errors, time_deltas, 8, 11)

    # prediction(selection_method='backward', num_features=4)
    # prediction(selection_method='backward', num_features=9)


if __name__ == '__main__':
    main()
