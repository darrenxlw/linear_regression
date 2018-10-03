import os

import numpy as np
from matplotlib import pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class LinearRegression:

    """
    LinearRegression is defined as class with several properties useful for data reading & scaling,
    fitting the parameters, and finally plot the results of the linear regression versus the training data.
    
    NOTA: Although an analytical solution exists for linear regressions, the exercise imposes to use
    gradient descent.     
    """

    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0
        self.theta0_norm = 0
        self.theta1_norm = 0
        self.log = []
        self.data = {}

    def read_csv(self, path):

        """
        This functions uses numpy's genfromtxt function in order to read the data
        :param path: an OS path representing the name of the file
        :return: an array in the dimensions of the csv file
        """
        return np.genfromtxt(path, delimiter=',', skip_header=1)

    def estimate_price(self, theta0, theta1, mileage):
        return theta0 + theta1 * mileage

    def train_model(self, path, learning_rate=0.001, max_epochs=10000, precision=0.00001, reset_theta=True, plot=False):
        """
        This function does the gradient descent to find the right parameters theta0 and theta1
        It first scales the data. Then applies a gradient descent with a number of steps limited by max_epochs.
        The algorithm stops it's search when the difference between to steps for theta0 and theta1 gets
        below 'precision'.

        :param path: an OS path representing the name of the file
        :param learning_rate: the size of the steps to take at each epoch. (float) A large step makes the convergence
        potentially faster but might also not reach convergence at all.
        :param max_epochs: the maximum number of iterations. (int)
        :param precision: the desired precision. (float)
        :param reset_theta: True or False. If True, the previous values calculated for theta0 and theta1 will be erased
        :param plot: True or False. If True, will plot the results.
        :return: theta0 & theta1
        """
        def reconstruct_regressor():

            return ((self.theta0_norm - self.theta1_norm * np.mean(self.data[:,0]) / np.std(self.data[:,0])) * np.std(self.data[:,1]) + np.mean(self.data[:,1]),
                    self.theta1_norm / np.std(self.data[:,0]) * np.std(self.data[:,1]))


        if reset_theta:
            self.log = []
            self.theta0_norm = 0
            self.theta1_norm = 0

        self.data = self.read_csv(path)
        m = len(self.data)

        #print(self.data)

        if plot:
            plt.plot(self.data[:, 0], self.data[:, 1], 'ro')
            axes = plt.gca()
            x_vals = np.array(axes.get_xlim())

        km_norm = (self.data[:, 0]-np.mean(self.data[:, 0])) / np.std(self.data[:, 0])
        price_norm = (self.data[:, 1]-np.mean(self.data[:, 1])) / np.std(self.data[:, 1])


        for i in range(max_epochs):
            tmp_theta0 = learning_rate * sum(self.estimate_price(self.theta0_norm,self.theta1_norm,km_norm) - price_norm) / m
            tmp_theta1 = learning_rate * sum((self.estimate_price(self.theta0_norm,self.theta1_norm,km_norm) - price_norm) * km_norm) / m

            self.theta0_norm -= tmp_theta0
            self.theta1_norm -= tmp_theta1

            self.log.append((self.theta0_norm,self.theta1_norm))


            if plot and (i % 10 == 0):
                y_vals = reconstruct_regressor()[0] + reconstruct_regressor()[1] * x_vals
                plt.plot(x_vals, y_vals, '--')

            if (abs(tmp_theta0) < precision) and (abs(tmp_theta1) < precision):
                print('epochs:' + str(i))
                break
        if plot: 
            plt.show()
        self.theta0 = reconstruct_regressor()[0]
        self.theta1 = reconstruct_regressor()[1]

    def visualize(self):

        def er_flat(theta0_tmp, theta1_tmp,x):
            m=len(self.data)
            km_norm = (x[:, 0] - np.mean(x[:, 0])) / np.std(x[:, 0])
            price_norm = (x[:, 1] - np.mean(x[:, 1])) / np.std(x[:, 1])

            return 1 / (2 * m) * sum((self.estimate_price(theta0_tmp, theta1_tmp, km_norm) - price_norm) ** 2)

        plt.plot(self.data[:,0], self.data[:,1], 'ro')
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = self.theta0 + self.theta1 * x_vals
        plt.plot(x_vals, y_vals, '--')
        plt.show()

        theta0_tmp = np.arange(-1, 1, 0.01)
        theta1_tmp = np.arange(-1, 1, 0.01)
        x, y = np.meshgrid(theta0_tmp, theta1_tmp)

        z = np.zeros(shape=x.shape)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                z[i, j] = er_flat(x[i, j], y[i, j],self.data)
        print(z)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(x, y, z, cmap=cm.inferno,
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.8)

        #log0 = [i[0] for i in self.log]
        #log1 = [i[1] for i in self.log]
        #z2 = np.zeros(len(self.log))
        #for i in range(len(z2)):
        #    z2[i] = er_flat(log0[i], log1[i])
        #ax.plot(log0, log1, z2, linestyle='--', marker='3', color='c', markersize=0.1)


        plt.show()






#tmp=LinearRegression()
#tmp.train_model('data/data.csv',0.01)
#tmp.visualize()
#
#x=tmp.data
#def er_flat(theta0_tmp, theta1_tmp):
#    m = len(tmp.data)
#    km_norm = (tmp.data[:, 0] - np.mean(tmp.data[:, 0])) / np.std(tmp.data[:, 0])
#    price_norm = (tmp.data[:, 1] - np.mean(tmp.data[:, 1])) / np.std(tmp.data[:, 1])
#    return 1/(2*m) * sum((estimate_price(km_norm,theta0_tmp, theta1_tmp) - price_norm)**2)
#
#theta0_tmp = np.arange(-1,1,0.01)
#theta1_tmp = np.arange(-1,1,0.01)
#x,y=np.meshgrid(theta0_tmp,theta1_tmp)
#
#z=np.zeros(shape=x.shape)
#for i in range(z.shape[0]):
#    for j in range(z.shape[1]):[i,j] = er_flat(x[i,j],y[i,j])
#
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#surf = ax.plot_surface(x, y, z, cmap=cm.inferno,
#                       linewidth=0, antialiased=False)
#
##ax.set_zlim(0, 3.01)
#ax.zaxis.set_major_locator(LinearLocator(10))
#ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#
## Add a color bar which maps values to colors.
#fig.colorbar(surf, aspect=5)
#
#plt.show()
#
