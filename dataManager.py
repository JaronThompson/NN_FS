import numpy as np
import theano
import theano.tensor as T
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import stats

def sharedArray(features, targets=None, validation_split=None):
    # input data must be [N samples, N features] matrix
    if targets is not None:
        if validation_split:
            # place data into shared variables
            N_val = int(np.ceil(validation_split*len(targets)))
            N_train = len(targets) - N_val

            T_x = theano.shared(
                np.asarray(features[:N_train,:], dtype=theano.config.floatX), borrow=True)
            T_y = theano.shared(
                np.asarray(np.vstack(targets[:N_train]), dtype=theano.config.floatX), borrow=True)

            # place data into shared variables
            V_x = theano.shared(
                np.asarray(features[N_train:,:], dtype=theano.config.floatX), borrow=True)
            V_y = theano.shared(
                np.asarray(np.vstack(targets[N_train:]), dtype=theano.config.floatX), borrow=True)

            return [T_x, T_y], [V_x, V_y], N_train, N_val
        else:
            # place data into shared variables
            shared_x = theano.shared(
                np.asarray(features, dtype=theano.config.floatX), borrow=True)
            shared_y = theano.shared(
                np.asarray(np.vstack(targets), dtype=theano.config.floatX), borrow=True)

            return [shared_x, shared_y]
    else:
        shared_x = theano.shared(
            np.asarray(features, dtype=theano.config.floatX), borrow=True)
        return shared_x

def standardize(X, Xtrain, axis=0):
    if axis==1:
        X = X.T
        Xtrain = Xtrain.T

    Xavg = np.mean(Xtrain,0)
    Xstd = np.std(Xtrain,0)

    return (X-Xavg) / Xstd

def center(X, Xtrain, axis=0):
    if axis==1:
        X = X.T
        Xtrain = Xtrain.T

    Xavg = np.mean(Xtrain,0)
    Xstd = np.std(Xtrain,0)

    return (X-Xavg)

def scale(X, Xtrain, axis=0):
    if axis==1:
        X = X.T
        Xtrain = Xtrain.T

    scaler = MinMaxScaler().fit(Xtrain)

    return scaler.transform(X)

def regression_plot(Y_train, Y_pred_train, Y_test, Y_pred_test, Yerr=None):
    plt.subplot(1,2,1)
    plt.scatter(Y_train, Y_pred_train, facecolors='none', edgecolors='b', label='Data')

    pad = .25
    xlim = [min(Y_train)-pad, max(Y_train)+pad]
    ylim = [min(Y_train)-pad, max(Y_train)+pad]
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_train, Y_pred_train)
    x = np.linspace(min(xlim), max(xlim), 100)
    line = np.multiply(slope,x) + np.array(intercept)
    plt.plot(x, line, 'r', label='Fit')
    title_string = 'Training: R = %.3f' % r_value
    ylabel = 'Output = {0:.2f}*Target + {1:.2f}'.format(slope, intercept)
    plt.title(title_string)
    plt.xlabel('Target')
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.legend()

    # plot test data results
    plt.subplot(1,2,2)
    slope, intercept, r_value, p_value, std_err = stats.linregress(Y_test, Y_pred_test)
    x = np.linspace(min(xlim), max(xlim), 100)
    line = np.multiply(slope, x) + np.array(intercept)
    if Yerr is None:
        plt.scatter(Y_test, Y_pred_test, facecolors='none', edgecolors='b', label='Data')
    else:
        plt.errorbar(Y_test, Y_pred_test, linestyle='none', marker='o', yerr = Yerr, label='Data')
    plt.plot(x, line, 'r', label='Fit')
    title_string = 'Test: R = %.3f' % r_value
    ylabel = 'Output = {0:.2f}*Target + {1:.2f}'.format(slope, intercept)
    plt.title(title_string)
    plt.xlabel('Target')
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    #plt.ylim([ymin-pad, ymax+pad])
    plt.legend()
    # save or show figures
    plt.tight_layout(rect=[0, .03, 1, .95])
    plt.show()
