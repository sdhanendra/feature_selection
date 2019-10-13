import numpy as np
import csv
from random import randrange
import matplotlib.pyplot as plt

# mu, sigma = 0, 1
#
# s = np.random.normal(mu, sigma, 1000)
#
# count, bins, ignored = plt.hist(s, 30, density=True)
#
# # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
#
# plt.show()


def data_function(x1, x2, x3, random_point):
    y = 0.5*x1 - 0.25*x2*x2 + 10*x3 + random_point
    return y

def generate_groud_truth():

    num_data_points = 1000

    # sample 1000 x1
    mu1, sigma1 = 1, 2
    s_x1 = np.random.normal(mu1, sigma1, num_data_points)

    # sampel 1000 x2
    mu2, sigma2 = 3, 1
    s_x2 = np.random.normal(mu2, sigma2, num_data_points)

    # sampel 1000 x3
    mu3, sigma3 = -2, 3
    s_x3 = np.random.normal(mu3, sigma3, num_data_points)

    mu_r, sigma_r = 0, 5
    s_rp = np.random.normal(mu_r, sigma_r, num_data_points)

    s_y = []
    for i in range(num_data_points):
        y = data_function(s_x1[i], s_x2[i], s_x3[i], s_rp[i])
        s_y.append(y)
    s_y = np.array(s_y)

    data = np.column_stack((s_y, s_x1, s_x2, s_x3, s_rp))

    return data


def write_ground_truth_to_file(data, file_name = 'groud_truth.csv'):
    # write everything to a csv file
    myFile = open(file_name, 'w+')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)

    print("Writing groud truth data complete")


def generate_random_features(data, num_rand_features = 5):
    num_data_points = 1000

    random_features = []
    # generate N random features
    for i in range(num_rand_features):
        mu = randrange(-5, 5)
        sigma = randrange(0, 5)
        s_rand = np.random.normal(mu, sigma, num_data_points)
        random_features.append(s_rand)
    rand_features = np.array(random_features)
    rand_features = np.transpose(rand_features)

    data = np.column_stack((data, rand_features))

    return data


def write_data_file(data, file_name = 'data.csv'):
    # write everything to a csv file
    myFile = open(file_name, 'w+')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)

    print("Writing data complete")


def main():
    data = generate_groud_truth()
    write_ground_truth_to_file(data)
    data = generate_random_features(data)
    write_data_file(data)


if __name__ == '__main__':
    main()


