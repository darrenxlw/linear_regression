import os

import numpy as np
import time
from matplotlib import pyplot as plt

class LinearRegression:
    max_epochs = 10000
    precision= 0.00001

    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0
        self.data = {}

    def read_csv(self, path):
        km=[]
        price=[]
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 0:
                    vals=line.rstrip('\n').split(',')
                    km.append(int(vals[0]))
                    price.append(int(vals[1]))
        return {"km": km, "price": price, "km_norm": [(i-np.mean(km))/np.std(km) for i in km], "price_norm": [(i-np.mean(price))/np.std(price) for i in price]}

    def estimate_price(self, mileage):
        return self.theta0 + self.theta1 * mileage

    def train_model(self, path, learningRate, reset_theta=True, plot=True):
        def reconstruct_regressor():
            return ((self.theta0 - self.theta1 * np.mean(self.data['km']) / np.std(self.data['km'])) * np.std(self.data['price']) + np.mean(self.data['price']),
                    self.theta1 / np.std(self.data['km']) * np.std(self.data['price']))

        if reset_theta:
            self.theta0 = 0
            self.theta1 = 0
        self.data = self.read_csv(path)
        m = len(self.data['price_norm'])

        if plot:
            plt.plot(self.data['km'], self.data['price'], 'ro')
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())

        for i in range(self.max_epochs):
            tmp_theta0 = learningRate * sum(
                [self.estimate_price(self.data['km_norm'][i]) - self.data['price_norm'][i] for i in range(m)]) / m
            tmp_theta1 = learningRate * sum(
                [(self.estimate_price(self.data['km_norm'][i]) - self.data['price_norm'][i]) * self.data['km_norm'][i] for i in range(m)]) / m

            self.theta0 -= tmp_theta0
            self.theta1 -= tmp_theta1

            if plot and (i%10 == 0):
                y_vals = reconstruct_regressor()[0] + reconstruct_regressor()[1] * x_vals
                plt.plot(x_vals, y_vals, '--')
                time.sleep(0.5)

            if (abs(tmp_theta0) < self.precision) and (abs(tmp_theta1) < self.precision):
                print('epochs:' + str(i))
                break
        if plot: plt.show()
        self.theta0 = reconstruct_regressor()[0]
        self.theta1 = reconstruct_regressor()[1]

    def visualize(self):
        plt.plot(self.data['km'], self.data['price'], 'ro')
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = self.theta0 + self.theta1 * x_vals
        plt.plot(x_vals, y_vals, '--')
        plt.show()



