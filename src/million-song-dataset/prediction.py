from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import csv
from src.forwardSelection import forward_feature_selection
from src.backwardSelection import backward_feature_selection_p_value
from utils.helper_utils import prepare_million_song_data


def write_solution_to_file(y_test):

    with open('../../data/million-song-dataset/submission.csv', mode='w+') as my_file:
        my_file_writer = csv.writer(my_file)
        # my_file_writer.writerow(['id', 'quality'])

        for y in y_test:
            print(int(y))
            # my_file_writer.writerow([y])


def calc_accuracy(y, y_hat):

    total = len(y)
    count = 0

    for yi, y_hati in zip(y, y_hat):
        if int(yi) == round(y_hati):
            count += 1

    accuracy = count/total
    return accuracy


def prediction(selection_method):

    X_train, X_test, y_train, y_test = prepare_million_song_data()

    if selection_method == 'forward':
        feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, 46)
    elif selection_method == 'backward':
        feature_set = backward_feature_selection_p_value(X_train, X_test, y_train, y_test, 46)


    model = LinearRegression()
    model.fit(X_train[:, feature_set], y_train)

    y_test_pred = model.predict(X_test[:, feature_set])


    accuracy = calc_accuracy(y_test, y_test_pred)

    print(accuracy)

    # print(feature_set)
    return y_test_pred


def main():
    y_test = prediction(selection_method='forward')
    # write_solution_to_file(y_test)

    # ids, y_test = prediction(selection_method='backward')
    # write_solution_to_file(ids, y_test)


if __name__ == '__main__':
    main()