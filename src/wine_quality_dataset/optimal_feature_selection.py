from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from src.forwardSelection import forward_feature_selection
from src.backwardSelection import backward_feature_selection_p_value
from utils.helper_utils import prepare_wine_quality_data, plot_mean_error_vs_num_features
from src.optimum_feature_select import optimal_features


def main():

    X_train, X_test, y_train, y_test = prepare_wine_quality_data()

    # forward feature selection
    # num_feature_list, error_list = optimal_features(X_train, X_test, y_train, y_test, feature_selection_method='forward')
    # plot_mean_error_vs_num_features(num_feature_list, error_list)

    # backward feature selection
    num_feature_list, error_list = optimal_features(X_train, X_test, y_train, y_test, feature_selection_method='backward')
    plot_mean_error_vs_num_features(num_feature_list, error_list)


if __name__ == '__main__':
    main()
