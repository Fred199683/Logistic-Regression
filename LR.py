import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt
def sigmoid(z):
    return 1/(1+np.exp(-z))
def costfunction(X, y, w):
    cost = 0
    size = y.shape[0]
    for i in range(size):
        if y[i] == 1:
            cost -= np.log(sigmoid(X[i]*w))
        else:
            cost -= np.log(1 - sigmoid(X[i]*w))
    return cost / size
def gradAscent(traindata,label,iter,alpha,step,lamda=0.001):
    dataMat=np.mat(traindata)
    labelMat=np.mat(label)
    m,n=np.shape(dataMat)
    weights=np.ones((n,1))
    weights=np.mat(weights)
    for k in range(iter):
        temp=costfunction(dataMat,labelMat,weights)
        weights=weights-alpha*((dataMat.transpose())*(sigmoid(dataMat*weights)-labelMat)+lamda*weights)
        if k%200==0:
            print("Loss is: ",temp,weights.transpose())
        if (k/step==0 and k!=0):
            alpha=alpha/5
    return weights
def preprocessing(x_train,x_test):
    sc=StandardScaler()
    sc.fit(x_train)
    x_train_scaled=sc.transform(x_train)
    x_test_scaled=sc.transform(x_test)
    return x_train_scaled,x_test_scaled
def split(ratio):
    Data = datasets.load_iris()
    #Data = datasets.load_wine()  #for Dataset wine
    x = Data.data
    y=Data.target
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = ratio, random_state = 0)
    return x_train,x_test,y_train,y_test
def plot(X,Y):
    x_min, x_max = X[:, 0].min() - .2, X[:, 0].max() + .2
    y_min, y_max = X[:, 1].min() - .2, X[:, 1].max() + .2
    h = .02
    logreg =linear_model.LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial')
    logreg.fit(X,Y)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.show()
if __name__=='__main__':
    x_train,x_test,y_train,y_test=split(0.3)
    x_train_scaled,x_test_scaled=preprocessing(x_train,x_test)
    #logreg=linear_model.LogisticRegression(C=1e4) #for ovr
    logreg=linear_model.LogisticRegression(C=1e4,multi_class='multinomial',solver='lbfgs') #ovm
    logreg.fit(x_train_scaled,y_train)
    print("Accuracy:",logreg.score(x_test_scaled,y_test))
    plot(x_train_scaled[:,:2],y_train)

