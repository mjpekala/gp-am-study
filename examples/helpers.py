"""
Helper functions used in jupyter notebooks in this directory.
"""

import csv
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import axes3d, art3d


import GPy   


#-------------------------------------------------------------------------------
# Functions for working with raw measurement data
#-------------------------------------------------------------------------------
def cart2polar(x, y):
    "Cartesian to polar coordinates."
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r[:,np.newaxis], theta[:,np.newaxis]


def stem3c(ax, x, y, z):
    "Colored 3d stem plot (similar to matlab's stem3())"
    ax.scatter(x, y, z, 'k')
    
    for ii in range(len(x)):
        color = 'green' if z[ii] >= 0 else 'blue'
        line = art3d.Line3D((x[ii], x[ii]),
                            (y[ii], y[ii]),
                            (0, z[ii]),
                            color=color)
        ax.add_line(line)   

        
def plot_xy_err_2d(indVar, dx, dy, xLabel='', yLabel=''):
    "Generates a 2D plot of position error in x and y dimensions"
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    plt.plot(indVar, dx, 'bo')
    plt.plot([np.min(indVar), np.max(indVar)], [0,0], 'k--')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title('x error')

    ax = fig.add_subplot(122)
    ax.plot(indVar, dy, 'bo')
    ax.plot([np.min(indVar), np.max(indVar)], [0,0], 'k--')
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_title('y error')

    
#-------------------------------------------------------------------------------
# Functions for fitting Gaussian processes (GPs)
#-------------------------------------------------------------------------------

def loo_err_1d(kernel, x, y):
    """
    Generates an estimate of leave-one-out error for a 1d data set.
    
       kernel : a GPy kernel object
       x      : a (n x 1) numpy array containing points in the input domain where 
                the underlying function was observations.
       y      : a (n x 1) numpy array corresponding to f(x) (i.e. observations)
    """

    def hold_out(v, idx):
        """Creates an (n x 1) array which is a copy of v without element v[idx].
        """
        rv = np.copy(v)
        rv = np.reshape(v, (v.size,1))  # for GPy, which wants 2 dimensions
        np.delete(rv, idx, 0)
        return rv

    def to_col(v): return np.reshape(v, (v.size,1))

    errl2 = np.zeros((x.size,1), dtype=np.float32)
    ypred = np.zeros((y.size,1), dtype=y.dtype)
    for ii in range(x.size):
        xTrain, yTrain = hold_out(x, ii), hold_out(y, ii)

        model = GPy.models.GPRegression(xTrain, yTrain, kernel)
        # XXX: configure initial hyperparameter guesses?
        model.optimize(messages=False, max_f_eval=1000)
 
        [ymu, ys2] = model.predict(to_col(x[ii]))
        errl2[ii] = (y[ii] - ymu)**2
        ypred[ii] = ymu

    return np.sqrt(np.sum(errl2)), ypred
