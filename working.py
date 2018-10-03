# import imp
# imp.reload(linear_regression)
from ft_linear_regression import linear_regression

tmp = linear_regression.LinearRegression()

tmp.train_model('data/data.csv',0.01,plot=True)
print('{}, {}'.format(tmp.theta0, tmp.theta1))
print('{}, {}'.format(tmp.theta0_norm, tmp.theta1_norm))

tmp.visualize()