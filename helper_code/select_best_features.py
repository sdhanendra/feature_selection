from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.forwardSelection import forward_feature_selection
from src.backwardSelection import backward_feature_selection_p_value





def optimal_features(X_train, X_test, y_train, y_test):

    num_features = len(X_train[0])
    print(num_features)

    model = LinearRegression()

    num_feature_list = []
    error_list = []

    for num in range(2, num_features+1):
        feature_set = backward_feature_selection_p_value(X_train, X_test, y_train, y_test, num)

        # fit the model on training given the feature set
        model.fit(X_train[:, feature_set], y_train)

        mean_square_error = mean_squared_error(y_test, model.predict(X_test[:, feature_set]))

        print(feature_set, ' :: ', mean_square_error)

        num_feature_list.append(len(feature_set))
        error_list.append(mean_square_error)

    return num_feature_list, error_list





def main():

    X_train, X_test, y_train, y_test = prepare_data()
    num_feature_list, error_list = optimal_features(X_train, X_test, y_train, y_test)
    plot_mean_error_vs_num_features(num_feature_list, error_list)


if __name__ == '__main__':
    main()
