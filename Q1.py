import matplotlib.pyplot as plt # For general plotting
import matplotlib.colors as mcol
import os
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import torch
from sklearn.neural_network import MLPClassifier
from scipy.stats import multivariate_normal as mvn
from sklearn.preprocessing import PolynomialFeatures # Important new include
from sklearn.model_selection import KFold # Important new include
from modules import models, prob_utils

np.set_printoptions(suppress=True)

# Set seed to generate reproducible "pseudo-randomness" (handles scipy's "randomness" too)
np.random.seed(123)
torch.manual_seed(123)



np.random.seed(7)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
######################################################################################################
#Data distribution & Generate data
t1 = 100
t2 = 200
t3 = 500
t4 = 1000
t5 = 2000
t6 = 5000
v = 10000
m= np.array([[2, 1, 2],
                  [5, 1, 4],
                  [4, 4, 1],
                  [1, 5, 4]])
m=np.transpose(m)
#print(m)
c = np.zeros((4, 3, 3))
for i in range(4):
    temp = np.random.random(3) * 5
    c[i, :, :] = np.diag(temp)

classPrior=np.array([0.25,0.25,0.25,0.25])
C=4

gauss_params = prob_utils.GaussianMixturePDFParameters(classPrior,C,m,np.transpose(c))

gauss_params.print_pdf_params()

X_train=[]
y_train=[]

n=m.shape[0]
#print(n)
Xt1,yt1 = prob_utils.generate_mixture_samples(t1, n, gauss_params, False)
X_train.append(np.transpose(Xt1))
y_train.append(yt1)

Xt2,yt2 = prob_utils.generate_mixture_samples(t2, n, gauss_params, False)
X_train.append(np.transpose(Xt2))
y_train.append(yt2)

Xt3,yt3 = prob_utils.generate_mixture_samples(t3, n, gauss_params, False)
X_train.append(np.transpose(Xt3))
y_train.append(yt3)
#print(y_train[0])

Xt4,yt4 = prob_utils.generate_mixture_samples(t4, n, gauss_params, False)
X_train.append(np.transpose(Xt4))
y_train.append(yt4)
#print(y_train[0])

Xt5,yt5 = prob_utils.generate_mixture_samples(t5, n, gauss_params, False)
X_train.append(np.transpose(Xt5))
y_train.append(yt5)
#print(y_train[0])
Xt6,yt6 = prob_utils.generate_mixture_samples(t6, n, gauss_params, False)
X_train.append(np.transpose(Xt6))
y_train.append(yt6)


X_valid,y_valid = prob_utils.generate_mixture_samples(v, n, gauss_params, False)

#print("labels: ")
#print(y)
fig0 = plt.figure(figsize=(4, 4), dpi=200)
ax1 = fig0.add_subplot(projection='3d')
color=['blue', 'red','green','orange']
for i in range(C):
    ax1.scatter(X_valid[0, (y_valid==i)],
              X_valid[1, (y_valid==i)],
              X_valid[2, (y_valid==i)],
              marker='.', c=color[i],alpha=0.6)
ax1.set_xlim((-10, 10))
ax1.set_ylim((-10, 10))
ax1.set_title('Q1 Samples')
plt.tight_layout()
plt.savefig('Q1_Samples.jpg')
plt.close()
plt.cla()
#plt.show()

#######################################################################################################
#Theoretically Optimal Classifier, proabbility of error 10%-20%
class_cond_likelihoods=np.zeros((C,v))
m=np.transpose(m)
for i in range (v):
    for j in range(C):
        class_cond_likelihoods[j][i] = multivariate_normal.pdf(X_valid[:, i], mean=m[j], cov=c[j])
    
    

#print(class_cond_likelihoods)
class_priors = np.diag(classPrior)
class_posteriors = class_priors.dot(class_cond_likelihoods)

decisions = np.argmax(class_posteriors, axis=0)

print("Confusion Matrix (rows: Predicted class, columns: True class):")
conf_mat = confusion_matrix(decisions, y_valid)
print(conf_mat)
correct_class_samples = np.sum(np.diag(conf_mat))
print("Total Mumber of Misclassified Samples: {:d}".format(v - correct_class_samples))



prob_error = 1 - (correct_class_samples / v)
print("Empirically Estimated Probability of Error: {:.4f}".format(prob_error))

###############################################################################################
#K-fold
def perr(label_test, decis_test, label_train, NC, true_pdf=False):
    p_error = 0
    for c in range(NC):
        if true_pdf:
            class_prior = 1 / NC
        else:
            class_prior = (label_train == c + 1).sum() / label_train.size

        if (label_test == c+1).sum() == 0:
            continue
        else:
            p_error += ((label_test == c+1) & (decis_test != c+1)).sum() / (label_test == c+1).sum() * class_prior

    return p_error

K=10
kf = KFold(n_splits=K, shuffle=True) 
def ModelOrderSelection(X,y,K,NC):
    N, d = X.shape
    val_perror = np.zeros(0)
    val_p = np.zeros(0)
    p = 0

    while p<150:
        p += 5
        cum_perror = 0
        for train_indices, valid_indices in kf.split(X):
            X_train = X[train_indices]
            y_train = y[train_indices]
            X_test = X[valid_indices]
            y_test = y[valid_indices]
            model = MLPClassifier(hidden_layer_sizes=p,activation='relu',tol=0.001,
                                solver='adam',max_iter=1000).fit(X_train, y_train)
            decision = model.predict(X_test)
            cum_perror += y_test.size * perr(y_test, decision, y_train, NC)

        new_perror = cum_perror / y.size

        val_perror = np.append(val_perror, new_perror)
        val_p = np.append(val_p, p)


    best_p = val_p[np.argmin(val_perror)]

    plt.rcParams.update({'font.size': 14})
    plt.scatter(best_p, val_perror.min(), s=40, edgecolors='r', facecolors='none', label='selected model')
    plt.plot(val_p, val_perror, marker='.')
    plt.xlabel('hidden layer size')
    plt.ylabel(r'cumulative $P_{error}$')
    plt.title('N = {}'.format(N), fontsize=16, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Model'+str(N)+'.jpg')
    #plt.show()
    plt.close()
    plt.clf()
    return best_p

#print(X_train[0].shape)
#print(y_train[0].shape)

X_valid=np.transpose(X_valid)
best_p = np.zeros(6)
for i in range(6):
    best_p[i] = int(ModelOrderSelection(X_train[i], y_train[i], K, C))
    print("best num of perceptron for training set "+str(i)+"is"+str(best_p[i]))

def plot_decision_boundaries(X, y, model_class, **model_params):
    N, d = X.shape
    try:
        X = np.array(X)
        y = np.array(y).flatten()
    except:
        print("Coercing input data to NumPy arrays failed")
    # Reduces to the first two columns of data
    reduced_data = X[:, :2]
    # Instantiate the model object
    model = model_class(**model_params)
    # Fits the model with the reduced data
    model.fit(reduced_data, y)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, m_max]x[y_min, y_max].    

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    # Meshgrid creation
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh using the model.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])    

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predictions to obtain the classification results
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel("Feature-1",fontsize=15)
    plt.ylabel("Feature-2",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('boundry'+str(N)+'.jpg')
    plt.close()
    plt.cla()
    #plt.show()

for i in range(6):
    plot_decision_boundaries(X_train[i],y_train[i],MLPClassifier,hidden_layer_sizes=int(best_p[i]),activation='relu',
                        solver='adam',tol=0.001,max_iter=1000)

test_perror = np.zeros(6)
for i in range(6):
    model = MLPClassifier(hidden_layer_sizes=int(best_p[i]),activation='relu',tol=0.001,
                        solver='adam',max_iter=1000).fit(X_train[i], y_train[i])
    decision = model.predict(X_valid)
    test_perror[i] = perr(y_valid, decision, y_train[i], 4)
    print("Model "+str(i)+" achieved Perr: "+str(test_perror[i])+"on validation set")


