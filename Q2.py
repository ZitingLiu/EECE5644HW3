import modulefinder
import os
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import PolynomialFeatures # Important new include
from sklearn.model_selection import KFold # Important new include


np.random.seed(7)



abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

kf = KFold(n_splits=5, shuffle=True) 
N=10
Ntrain=50
Ntest=1000
m=np.array([1,2,3,4,5,6,7,8,9,10])
temp = np.random.random((10, 1))
c = temp.dot(temp.T) + np.eye(10) * 0.1
a= np.diag(c).sum() / 10 * np.array([1e-4, 1e-2, 1e-1, 1, 1e2, 1e4])
w = np.array([0,0,0,0,0,0,0,0,0,0])
print(a)
random_a=np.arange(1,11).reshape(1,-1)
#print(c)
#det=int(np.linalg.det(c))
#print(det)
def generateData(N,m,c,w,noise):
    z= np.random.multivariate_normal(mean=np.zeros(10),cov=noise*np.eye(10),size=N)
    y_noise=np.random.random(N)
    data= np.random.multivariate_normal(mean=m, cov=c, size=N)
    y = (np.dot(data + z, w.reshape(-1, 1))).ravel() + y_noise
    return data , y

def ll(w,X,y,b):
    mu_pred = X.dot(w)
    
    # Compute log-likelihood function, setting mu=0 as we're estimating it
    log_lld = np.sum(multivariate_normal.pdf(y - mu_pred, 0, b))
    
    # Return NLL
    return -2*log_lld

def map(X,y,b):
    X=np.column_stack((np.ones(len(X)),X))
    w=np.linalg.inv((X.T).dot(X)+((1/(b))*np.identity(np.shape(X)[1]))).dot(X.T).dot(y)

    return w


def lin_reg_loss(theta, X, y):
    # Linear regression model X * theta
    predictions = X.dot(theta)
    # Residual error (X * theta) - y
    error = predictions - y
    # Loss function is MSE
    loss_f = np.mean(error**2)

    return loss_f

def HyperParameterOpt(X,y,a):
    #print(N)
    betas=np.zeros(0)
    scores=np.zeros(0)
    b=0.00001

    while b<100000:
        score=0
        for train_indices, valid_indices in kf.split(X):
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[valid_indices]
            y_test = y[valid_indices]
            temp = minimize(ll, w, args=(X_train, y_train, b), tol=1e-6)
            mle=temp.x
            #w=MAP(X_train)
            score+=ll(mle,X_train,y_train,b)
            #score += lin_reg_loss(mle,X_test,y_test)
            #print(mle)
        
        new_score=score/10
        scores = np.append(scores, new_score)
        betas = np.append(betas, b)

        b *= 2

    best_beta = betas[np.argmax(scores)]
    plt.rcParams.update({'font.size': 14})
    plt.semilogx(betas, scores, marker='.')
    plt.scatter(best_beta, scores.max(), s=40, marker='o', edgecolors='r', facecolors='none')
    plt.title(r'$\alpha =$ {:.3e}'.format(a))
    plt.xlabel('beta')
    plt.ylabel('-2 log-likelihood')
    plt.tight_layout()
    #plt.show()
    #plt.savefig('beta&NLL'+str(a) +'.jpg')
    #plt.close()

    return best_beta, betas.size

def linear_model_analytical(X, y, beta):
    '''return analytical solution of the linear weight vector'''

    N, d = X.shape
    Z = np.ones((N, d + 1))
    Z[:, 1::] = X

    w = np.linalg.inv(np.dot(Z.T, Z) + 1/beta * np.eye(d + 1)).dot(np.dot(Z.T, y.reshape(-1, 1)))
    return w

def test():
    X_train, y_train = generateData(Ntrain, m, c, w, a[0])
    HyperParameterOpt(X_train,y_train,a[0])

def main():
    test_score = np.zeros(6)
    best_beta = np.zeros(6)
    betas_size = np.zeros(6)
    for i in range(6):
        X_train, y_train = generateData(Ntrain, m, c, w, a[i])
        X_test, y_test = generateData(Ntest, m, c, w, a[i])
        best_beta[i], betas_size[i]=HyperParameterOpt(X_train,y_train,a[i])
        weight=linear_model_analytical(X_test,y_test,best_beta[i])
        Z = np.ones((Ntest, 11))
        Z[:, 1::] = X_test
        test_score[i] = ((y_test - Z.dot(weight.reshape(-1, 1)).ravel())**2).sum()
        print("score on test set for aplha #"+str(i)+" is "+str(test_score[i]))
    plt.savefig("beta&NLL.jpg")
    #plt.show()
    plt.close
    print(best_beta)
    plt.plot(a, best_beta, marker='.')
    plt.title('Hyper-parameter v.s. Input Noise')
    plt.xlabel(r'input noise parameter $\alpha$')
    plt.ylabel(r'hyper-parameter $\beta$')
    plt.tight_layout()
    #plt.show()
    plt.close()

    plt.rcParams.update({'font.size': 14})
    plt.loglog(a, test_score, marker='.')
    plt.title('Impact of Input Noise')
    plt.xlabel(r'input noise parameter $\alpha$')
    plt.ylabel(r'score')
    plt.tight_layout()
    plt.show()
    plt.savefig("score.jpg")
    plt.close()

main()