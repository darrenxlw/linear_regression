# import imp
# imp.reload(linear_regression)
from ft_linear_regression import linear_regression

tmp = linear_regression.LinearRegression()

tmp.train_model('data/data.csv', 0.1, plot=True, reset_theta=False)
print('{}, {}'.format(tmp.theta0, tmp.theta1))
print('{}, {}'.format(tmp.theta0_norm, tmp.theta1_norm))

tmp.visualize()

tmp.theta0_norm = -1
tmp.theta1_norm = 1
tmp.train_model('data/data.csv', 0.1, plot=True, reset_theta=False)
tmp.visualize()
