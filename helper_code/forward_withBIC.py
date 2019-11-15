from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import math


def evaluate_metric(model, x_cv, y_cv):
    return mean_squared_error(y_cv, model.predict(x_cv))


def forward_feature_selection(x_train, x_cv, y_train, y_cv, n):
    feature_set = []
    for num_features in range(n):
        # metric list to hold the metric score
        metric_list = []
        bic_list = []
        # linear regression model for regression
        model = LinearRegression()

        for feature in range(len(x_train[0])):
            if feature not in feature_set:
                f_set = feature_set.copy()
                f_set.append(feature)
                model.fit(x_train[:,f_set], y_train)
                mse = evaluate_metric(model, x_cv[:, f_set], y_cv)
                metric_list.append((mse, feature))

                # Calculation for BIC
                sse = mse * len(y_cv)
                bic = len(x_cv[0]) * math.log(len(y_cv)) - 2 * math.log(sse)
                bic_list.append((bic, feature))

        # sort the metric score in ascending order of euclidean distances.
        print('MSE: ', metric_list)
        print('BIC: ', bic_list)

        metric_list.sort(key=lambda x : x[0], reverse = False)
        feature_set.append(metric_list[0][1]) # add the element that has the least distance
    return feature_set


def main():
    # load data
    data = np.genfromtxt('../data/data.csv', delimiter=',')

    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)
    feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, 4)

    print(feature_set)


if __name__ == '__main__':
    main()
