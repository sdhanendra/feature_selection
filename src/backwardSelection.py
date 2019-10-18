from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy import stats


class LinearRegression(LinearRegression):
    """
    LinearRegression class after sklearn's, but calculate t-statistics
    and p-values for model coefficients (betas).
    Additional attributes available after .fit()
    are `t` and `p` which are of the shape (y.shape[1], X.shape[1])
    which is (n_features, n_coefs)
    This class sets the intercept to 0 by default, since usually we include it
    in X.
    """

    def __init__(self, *args, **kwargs):
        if not "fit_intercept" in kwargs:
            kwargs['fit_intercept'] = False
        super(LinearRegression, self)\
                .__init__(*args, **kwargs)

    def fit(self, X, y, n_jobs=1):
        self = super(LinearRegression, self).fit(X, y, n_jobs)

        sse = np.sum((self.predict(X) - y) ** 2, axis=0) / float(X.shape[0] - X.shape[1])
        # se = np.array([
        #     np.sqrt(np.diagonal(sse[i] * np.linalg.inv(np.dot(X.T, X))))
        #                                             for i in range(sse.shape[0])
        #             ])

        se = np.array([np.sqrt(np.diagonal(sse * np.linalg.inv(np.dot(X.T, X))))])

        self.t = self.coef_ / se
        self.p = 2 * (1 - stats.t.cdf(np.abs(self.t), y.shape[0] - X.shape[1]))
        return self



def evaluate_metric(model, x_cv, y_cv):
    return mean_squared_error(y_cv, model.predict(x_cv))


def evaluate_metric_p_value(model, x_cv, y_cv):
    res = model.fit(x_cv, y_cv)
    return res.p


def backward_feature_selection(x_train, x_cv, y_train, y_cv, n):
    feature_set = [x for x in range(len(x_train[0]))]
    while len(feature_set) > n:
        # metric list to hold the metric score
        metric_list = []
        # linear regression model for regression
        model = LinearRegression()
        for feature in range(len(x_train[0])):
            if feature in feature_set:
                f_set = feature_set.copy()
                f_set.remove(feature)
                model.fit(x_train[:,f_set], y_train)
                metric_list.append((evaluate_metric(model, x_cv[:,f_set], y_cv), feature))

        metric_list.sort(key=lambda x : x[0], reverse = False)
        feature_set.remove(metric_list[0][1]) # remove the element that makes the least difference in the score. (min distance)
    return feature_set


def backward_feature_selection_p_value(x_train, x_cv, y_train, y_cv, n):
    feature_set = [x for x in range(len(x_train[0]))]
    while len(feature_set) > n:
        # for num_features in range(n):
        metric_list = []  # Choose appropriate metric based on business problem
        model = LinearRegression()  # You can choose any model you like, this technique is model agnostic

        f_set = feature_set.copy()
        model.fit(x_train[:, f_set], y_train)
        eval = evaluate_metric_p_value(model, x_cv[:, f_set], y_cv)

        j = 0
        for i in range(x_train.shape[1]):
            if i in f_set:
                metric_list.append((eval[0][j], i))
                j += 1

        metric_list.sort(key=lambda x: x[0], reverse=True)
        feature_set.remove(metric_list[0][1])
    return feature_set


def main():
    # load data
    data = np.genfromtxt('../data/data.csv', delimiter=',')

    X = data[:, 1:]
    y = data[:, 0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

    feature_set = backward_feature_selection(X_train, X_test, y_train, y_test, 4)
    print(feature_set)

    feature_set_p_value = backward_feature_selection_p_value(X_train, X_test, y_train, y_test, 4)
    print(feature_set_p_value)


if __name__ == '__main__':
    main()
