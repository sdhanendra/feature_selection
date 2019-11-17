from sklearn import linear_model
# import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np


lambda_list = [0.01, 0.05, 0.1, 0.2, 0.3]


def train(lambda_list, train_x, train_y, model_name):
    coeffiecient_list = []
    classifier_list = []
    for lambda_ in lambda_list:
        if model_name == "Lasso":
            classifier = linear_model.Lasso(alpha = lambda_)
        elif model_name == "Ridge":
            classifier = linear_model.Ridge(alpha = lambda_)
        classifier.fit(train_x, train_y)
        classifier_list.append(classifier)
        coefficients = classifier.coef_
        coeffiecient_list.append(coefficients)
    return classifier_list, coeffiecient_list


def predict(test_x, test_y, classifier_list):
    mean_squared_error_list = []
    for classifier in classifier_list:
        prediction = classifier.predict(test_x)
        mean_sq_err = mean_squared_error(test_y.flatten(), prediction)
        mean_squared_error_list.append(mean_sq_err)
    return mean_squared_error_list




def main():
    # load data
    data = np.genfromtxt('../data/data.csv', delimiter=',')

    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

    classifier_list, coeffiecient_list = train(lambda_list, X_train, y_train, "Lasso")
    for coefficients in coeffiecient_list:
        print("coefficients: ", coefficients)

    mean_squared_error_list = predict(X_test, y_test, classifier_list)

    print(mean_squared_error_list)


if __name__ == '__main__':
    main()