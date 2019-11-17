from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from utils.helper_utils import prepare_data


def evaluate_metric(model, x_cv, y_cv):
    return mean_squared_error(y_cv, model.predict(x_cv))


def forward_feature_selection(x_train, x_cv, y_train, y_cv, n):
    feature_set = []
    for num_features in range(n):
        # metric list to hold the metric score
        metric_list = []
        # linear regression model for regression
        model = LinearRegression()
        for feature in range(len(x_train[0])):
            if feature not in feature_set:
                f_set = feature_set.copy()
                f_set.append(feature)
                model.fit(x_train[:,f_set], y_train)
                metric_list.append((evaluate_metric(model, x_cv[:,f_set], y_cv), feature))

        # sort the metric score in ascending order of euclidean distances.
        metric_list.sort(key=lambda x : x[0], reverse = False)
        feature_set.append(metric_list[0][1]) # add the element that has the least distance
    return feature_set


def main():

    X_train, X_test, y_train, y_test = prepare_data()
    feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, 4)

    print(feature_set)


if __name__ == '__main__':
    main()
