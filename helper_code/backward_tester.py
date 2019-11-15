from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import chi2
import numpy as np
import pandas as pd
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
    res = model.fit(x_cv, y_cv)
    return res.p

# # #pandas
# def forward_feature_selection(x_train, x_cv, y_train, y_cv, n):
#     feature_set = []
#     for num_features in range(n):
#         metric_list = [] # Choose appropriate metric based on business problem
#         model = LinearRegression() # You can choose any model you like, this technique is model agnostic
#         for feature in x_train.columns:
#             if feature not in feature_set:
#                 f_set = feature_set.copy()
#                 f_set.append(feature)
#                 model.fit(x_train[f_set], y_train)
#                 metric_list.append((evaluate_metric(model, x_cv[f_set], y_cv), feature))
#
#         metric_list.sort(key=lambda x : x[0], reverse = False) # In case metric follows "the more, the merrier"
#         feature_set.append(metric_list[0][1])
#     return feature_set



# # numpy
# def forward_feature_selection(x_train, x_cv, y_train, y_cv, n):
#     feature_set = [x for x in range(len(x_train[0]))]
#     while(len(feature_set) > n):
#     # for num_features in range(n):
#         metric_list = [] # Choose appropriate metric based on business problem
#         model = LinearRegression() # You can choose any model you like, this technique is model agnostic
#         for feature in range(len(x_train[0])):
#             if feature in feature_set:
#                 f_set = feature_set.copy()
#                 f_set.remove(feature)
#                 model.fit(x_train[:,f_set], y_train)
#                 metric_list.append((evaluate_metric(model, x_cv[:,f_set], y_cv), feature))
#
#         metric_list.sort(key=lambda x : x[0], reverse = False) # In case metric follows "the more, the merrier"
#         feature_set.remove(metric_list[0][1])
#     return feature_set



# numpy p   value
def forward_feature_selection(x_train, x_cv, y_train, y_cv, n):
    feature_set = [x for x in range(len(x_train[0]))]
    while(len(feature_set) > n):
    # for num_features in range(n):
        metric_list = [] # Choose appropriate metric based on business problem
        model = LinearRegression() # You can choose any model you like, this technique is model agnostic
        # for feature in range(len(x_train[0])):
        # if feature in feature_set:
        f_set = feature_set.copy()
        #     f_set.remove(feature)
        model.fit(x_train[:,f_set], y_train)
        eval = evaluate_metric(model, x_cv[:, f_set], y_cv)

        j = 0
        for i in range(x_train.shape[1]):
            if i in f_set:
                metric_list.append((eval[0][j], i))
                j += 1


        # metric_list.sort(reverse = True) # In case metric follows "the more, the merrier"
        metric_list.sort(key=lambda x: x[0], reverse=True)
        feature_set.remove(metric_list[0][1])
    return feature_set

# data = np.genfromtxt('groud_truth.csv', delimiter=',')

data = np.genfromtxt('data.csv', delimiter=',')

# print(np.shape(data))

X = data[:,1:]

# print(X[0,:])

# X = X[:,4:] + X[:,:4]
# X = np.concatenate((X[:,4:],X[:,:4]),axis=1)

# print(X[0,:])

y = data[:,0]

# print(np.shape(X))
# print(np.shape(y))

X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

print(np.size(X_train[:,0]))
print(np.size(y_train))

# numpy way
feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, 4)




# data = pd.DataFrame(data=data)
#
# print(data.head())




# # PANDA way
# X_train = pd.DataFrame(X_train)
# X_test = pd.DataFrame(X_test)
# y_train = pd.DataFrame(y_train)
# y_test = pd.DataFrame(y_test)
#
# feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, 2) # pandas






# feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, 3)

print(feature_set)
