import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
from scipy.stats import norm as sp_norm
from sklearn.model_selection import RandomizedSearchCV
from dataManager import regression_plot
# import NN
import network3
from network3 import Network
from network3 import FullyConnectedLayer

#%%

# first we need to design a target function which the NN model will attempt to learn
def t(params):
    a, b, c, d, e = params.T
    return a + 3*b - 5*d

# after performing feature selection, we should find that parameters c and e
# are the least significant, followed by a, then b, then d, where d has a strong
# negative effect on the target variable

# generate training and testing data where rows are samples and columns are features
def f(Nfeatures):
    # sampling distribution for features
    return mvn(np.zeros(Nfeatures), np.eye(Nfeatures)).rvs()

def generate_data(t, f, Nfeatures, Nsamples):
    X = np.zeros([Nsamples, Nfeatures])
    Y = np.zeros(Nsamples)

    for i in range(Nsamples):
        X[i, :] = f(Nfeatures)
        # add some standard Gaussian noise to Y
        Y[i] = t(X[i, :]) + np.random.randn()

    return X, Y

NF = 5
X_train, Y_train = generate_data(t, f, NF, 150)
X_test, Y_test = generate_data(t, f, NF, 50)

# now that we have training and testing data, we can train and test the NN
net = Network(NF)

# optimize hyper-parameters
'''
param_dist = {'nodes': sp_randint(10, 50),
              'eta': sp_norm(.035, .005),
              'lmbda': sp_norm(0, 1),
              'patience': sp_norm(15, 5)}

net.set_params(verbose=False)
random_search = RandomizedSearchCV(net,
    scoring='neg_mean_squared_error',
    param_distributions = param_dist, cv=5, iid=False)
random_search.fit(X_train, Y_train)
params = random_search.best_params_
net = Network(NF,
    nodes = params['nodes'],
    eta = params['eta'],
    lmbda = params['lmbda'],
    patience = params['patience'])
print(params)
'''

# fit the NN model to the training data and make predictions
net.fit(X_train, Y_train)
Y_pred_train = net.predict(X_train)
Y_pred_test = net.predict(X_test)

#%% plot prediction results
regression_plot(Y_train, Y_pred_train, Y_test, Y_pred_test)

# calculate feature importance based on training data
feature_importance = net.getFeatureImportance(X_train, Y_train)
print(feature_importance)

#%% to get a better feature importance estimate, I recommend bootstrapping the data
bootstrap_iterations = 25
p_bootstrap = .75
feature_importances = np.zeros([bootstrap_iterations, NF])

for i in range(bootstrap_iterations):
    # keep p_bootstrap % of training sample set after shuffling
    rand_inds = np.random.permutation(len(Y_train))
    X_train_sample = X_train[rand_inds, :][:int(p_bootstrap*len(Y_train)), :]
    Y_train_sample = Y_train[rand_inds][:int(p_bootstrap*len(Y_train))]
    # fit RF model to bootstrap sample of training data
    net.fit(X_train_sample, Y_train_sample)
    feature_importances[i, :] = net.getFeatureImportance(X_train_sample, Y_train_sample)

feature_importance = np.mean(feature_importances, 0)
print(feature_importance)

#%% plot feature selection results
true_feature_importances = np.array([1, 3, 0, -5, 0]) / 5

N = 5
ind = np.arange(N)  # the x locations for the groups
width = 0.27        # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)

rects1 = ax.bar(ind, true_feature_importances, width)
rects2 = ax.bar(ind+width, feature_importance, width)

ax.set_ylabel('Feature Importance')
ax.set_xticks(ind+width)
ax.set_xticklabels(('a', 'b', 'c', 'd', 'e'))
ax.legend((rects1[0], rects2[0]), ('True', 'NN'))

plt.tight_layout()
plt.show()
