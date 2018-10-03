import os

import numpy as np
from matplotlib import pyplot as plt

class LinearRegression:

    """
    LinearRegression is defined as class with severl properties useful for data reading & scaling,
    fitting the parameters
    """

    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0
        self.data = {}

    def read_csv(self, path):
        #km=[]
        #price=[]
        #with open(path, 'r') as f:
        #    for idx, line in enumerate(f):
        #        if idx > 0:
        #            vals=line.rstrip('\n').split(',')
        #            km.append(int(vals[0]))
        #            price.append(int(vals[1]))
        return np.genfromtxt(path, delimiter=',', skip_header=1)
        #return {"km": km, "price": price, "km_norm": [(i-np.mean(km))/np.std(km) for i in km], "price_norm": [(i-np.mean(price))/np.std(price) for i in price]}

    def estimate_price(self, mileage):
        return self.theta0 + self.theta1 * mileage

    def train_model(self, path, learningRate, max_epochs=10000, precision=0.00001, reset_theta=True, plot=False):
        def reconstruct_regressor():
            return ((self.theta0 - self.theta1 * np.mean(self.data[:,0]) / np.std(self.data[:,0])) * np.std(self.data[:,1]) + np.mean(self.data[:,1]),
                    self.theta1 / np.std(self.data[:,0]) * np.std(self.data[:,1]))

        if reset_theta:
            self.theta0 = 0
            self.theta1 = 0

        self.data = self.read_csv(path)
        m = len(self.data)

        print(self.data)

        if plot:
            plt.plot(self.data[:,0], self.data[:,1], 'ro')
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())

<<<<<<< HEAD
        for i in range(max_epochs):
            tmp_theta0 = learningRate * sum(
                [self.estimate_price(self.data['km_norm'][i]) - self.data['price_norm'][i] for i in range(m)]) / m
            tmp_theta1 = learningRate * sum(
                [(self.estimate_price(self.data['km_norm'][i]) - self.data['price_norm'][i]) * self.data['km_norm'][i] for i in range(m)]) / m
=======
        km_norm = (self.data[:,0]-np.mean(self.data[:,0])) / np.std(self.data[:,0])
        price_norm = (self.data[:,1]-np.mean(self.data[:,1])) / np.std(self.data[:,1])
>>>>>>> 3a8538a6f625005aabffbb005f1beb0a4a0b4222

        for i in range(self.max_epochs):
            tmp_theta0 = learningRate * sum(self.estimate_price(km_norm) - price_norm) / m
            tmp_theta1 = learningRate * sum((self.estimate_price(km_norm) - price_norm) * km_norm) / m
            self.theta0 -= tmp_theta0
            self.theta1 -= tmp_theta1

            if plot and (i%10 == 0):
                y_vals = reconstruct_regressor()[0] + reconstruct_regressor()[1] * x_vals
                plt.plot(x_vals, y_vals, '--')

            if (abs(tmp_theta0) < precision) and (abs(tmp_theta1) < precision):
                print('epochs:' + str(i))
                break
        if plot: plt.show()
        self.theta0 = reconstruct_regressor()[0]
        self.theta1 = reconstruct_regressor()[1]

    def visualize(self):
        plt.plot(self.data[:,0], self.data[:,1], 'ro')
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = self.theta0 + self.theta1 * x_vals
        plt.plot(x_vals, y_vals, '--')
        plt.show()


