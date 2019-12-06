from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.forwardSelection import forward_feature_selection
from src.backwardSelection import backward_feature_selection_p_value, backward_feature_selection
from utils.helper_utils import prepare_data, plot_mean_error_vs_num_features


def optimal_features(X_train, X_test, y_train, y_test, feature_selection_method):

    num_features = len(X_train[0])
    model = LinearRegression()

    num_feature_list = []
    error_list = []

    for num in range(2, num_features+1):

        if feature_selection_method == 'forward':
            feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, num)
        elif feature_selection_method == 'backward':
            feature_set = backward_feature_selection(X_train, X_test, y_train, y_test, num)

        # fit the model on training given the feature set
        model.fit(X_train[:, feature_set], y_train)

        mean_square_error = mean_squared_error(y_test, model.predict(X_test[:, feature_set]))

        # print(feature_set, ' :: ', mean_square_error)

        num_feature_list.append(len(feature_set))
        error_list.append(mean_square_error)

    return num_feature_list, error_list


def main():

    X_train, X_test, y_train, y_test = prepare_data()

    # forward feature selection
    num_feature_list, error_list = optimal_features(X_train, X_test, y_train, y_test, feature_selection_method='forward')
    plot_mean_error_vs_num_features(num_feature_list, error_list)

    # backward feature selection
    num_feature_list, error_list = optimal_features(X_train, X_test, y_train, y_test, feature_selection_method='backward')
    plot_mean_error_vs_num_features(num_feature_list, error_list)


if __name__ == '__main__':
    main()
