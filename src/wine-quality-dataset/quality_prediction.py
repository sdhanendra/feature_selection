from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import csv
from src.forwardSelection import forward_feature_selection
from src.backwardSelection import backward_feature_selection_p_value
from utils.helper_utils import prepare_wine_quality_data, prepare_wine_quality_test_data


def write_solution_to_file(ids, y_test):

    with open('../../data/uci-wine-quality-dataset/submission.csv', mode='w+') as my_file:
        my_file_writer = csv.writer(my_file)
        my_file_writer.writerow(['id', 'quality'])

        for id, y in zip(ids, y_test):
            print(int(id), y)
            my_file_writer.writerow([int(id), y])


def prediction(selection_method):

    X_train, X_test, y_train, y_test = prepare_wine_quality_data()

    if selection_method == 'forward':
        feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, 8)
    elif selection_method == 'backward':
        feature_set = backward_feature_selection_p_value(X_train, X_test, y_train, y_test, 8)

    X_test, ids = prepare_wine_quality_test_data();

    model = LinearRegression()
    model.fit(X_train[:, feature_set], y_train)

    y_test = model.predict(X_test[:, feature_set])

    # print(y_test)

    # print(feature_set)
    return ids, y_test


def main():
    # ids, y_test = prediction(selection_method='forward')
    # write_solution_to_file(ids, y_test)

    ids, y_test = prediction(selection_method='backward')
    write_solution_to_file(ids, y_test)


if __name__ == '__main__':
    main()

