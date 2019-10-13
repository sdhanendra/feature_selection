from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd


def evaluate_metric(model, x_cv, y_cv):
    # print(x_cv.size)
    # print(y_cv.size)
    return mean_squared_error(y_cv, model.predict(x_cv))

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



# numpy
def forward_feature_selection(x_train, x_cv, y_train, y_cv, n):
    feature_set = []
    for num_features in range(n):
        metric_list = [] # Choose appropriate metric based on business problem
        model = LinearRegression() # You can choose any model you like, this technique is model agnostic
        for feature in range(len(x_train[0])):
            if feature not in feature_set:
                f_set = feature_set.copy()
                f_set.append(feature)
                model.fit(x_train[:,f_set], y_train)
                metric_list.append((evaluate_metric(model, x_cv[:,f_set], y_cv), feature))

        metric_list.sort(key=lambda x : x[0], reverse = False) # In case metric follows "the more, the merrier"
        feature_set.append(metric_list[0][1])
    return feature_set


# data = np.genfromtxt('groud_truth.csv', delimiter=',')

data = np.genfromtxt('data.csv', delimiter=',')

# print(np.shape(data))

X = data[:,1:]

y = data[:,0]

# print(np.shape(X))
# print(np.shape(y))

X_train, X_test,  y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

print(np.size(X_train[:,0]))
print(np.size(y_train))

# numpy way
feature_set = forward_feature_selection(X_train, X_test, y_train, y_test, 5)




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
