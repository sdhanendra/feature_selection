from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import csv
from src.forwardSelection import forward_feature_selection
from src.backwardSelection import backward_feature_selection_p_value
from utils.helper_utils import prepare_million_song_data, plot_time_error_bar_graph
import time

def write_solution_to_file(y_test):

    with open('../../data/million-song-dataset/submission.csv', mode='w+') as my_file:
        my_file_writer = csv.writer(my_file)
        # my_file_writer.writerow(['id', 'quality'])

        for y in y_test:
            print(int(y))
            # my_file_writer.writerow([y])


def calc_accuracy(y, y_hat):

    total = len(y)
    error_sum = 0

    for yi, y_hati in zip(y, y_hat):
        print(yi, ' :: ', y_hati)
        error_sum = error_sum +  abs(yi - y_hati)

    print(error_sum)
    mean_error = error_sum/total
    return mean_error


def prediction(selection_method, num_features):

    X_train, X_test, y_train, y_test = prepare_million_song_data()

    if selection_method == 'forward':
        feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, num_features)
    elif selection_method == 'backward':
        feature_set = backward_feature_selection_p_value(X_train, X_test, y_train, y_test, num_features)




    model = LinearRegression()

    start = time.time()

    model.fit(X_train[:, feature_set], y_train)

    end = time.time()
    time_diff = end - start
    print(time_diff * 1000)

    y_test_pred = model.predict(X_test[:, feature_set])

    mean_error = calc_accuracy(y_test, y_test_pred)

    print(mean_error)

    # print(feature_set)
    return mean_error, time_diff*1000


def main():
    mean_square_error1, time_diff1 = prediction(selection_method='backward', num_features=40)

    mean_square_error2, time_diff2 = prediction(selection_method='backward', num_features=90)

    mean_square_errors = [mean_square_error1, mean_square_error2]
    time_deltas = [time_diff1, time_diff2]

    plot_time_error_bar_graph(mean_square_errors, time_deltas, 40, 90)


    # mean_error, y_test = prediction(selection_method='forward')
    # # write_solution_to_file(y_test)
    #
    # y_test = prediction(selection_method='backward')
    # # write_solution_to_file(ids, y_test)


if __name__ == '__main__':
    main()