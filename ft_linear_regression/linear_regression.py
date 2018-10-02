import os

from matplotlib import pyplot as plt

class LinearRegression:
    def read_csv(self, path):
        km=[]
        price=[]
        with open(path, 'r') as f:
            for idx, line in enumerate(f):
                if idx > 0:
                    vals=line.rstrip('\n').split(',')
                    print(vals)
                    km.append(int(vals[0]))
                    price.append(int(vals[1]))
        return {"km": km, "price": price}

tmp=LinearRegression()
data=tmp.read_csv('data/data.csv')
plt.plot(data['km'],data['price'],'ro')
plt.show()