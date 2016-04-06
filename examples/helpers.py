"""
Helper functions used in jupyter notebooks in this directory.
"""

import csv
import numpy as np
import pylab as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, art3d
from matplotlib.ticker import NullFormatter


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

    
def scatter_hist(x,y,xlab='',ylab=''):
    # Taken from matplotlib examples:
    # http://matplotlib.org/examples/pylab_examples/scatter_hist.html
    nullfmt   = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left+width+0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8,8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y)

    if len(xlab): axScatter.set_xlabel(xlab)
    if len(ylab): axScatter.set_ylabel(ylab)

    # now determine nice limits by hand:
    binwidth = 0.25
    xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
    lim = ( int(xymax/binwidth) + 1) * binwidth

    #axScatter.set_xlim( (-lim, lim) )
    #axScatter.set_ylim( (-lim, lim) )

    #bins = np.arange(-lim, lim + binwidth, binwidth)
    bins = 20 #  mjp
    axHistx.hist(x, bins=bins)
    axHisty.hist(y, bins=bins, orientation='horizontal')

    axHistx.set_xlim( axScatter.get_xlim() )
    axHisty.set_ylim( axScatter.get_ylim() )

    plt.show()

    
#-------------------------------------------------------------------------------
# Functions for fitting Gaussian processes (GPs)
#-------------------------------------------------------------------------------

def to_col(v): 
    return np.reshape(v, (v.size,1))


def hold_out(v, idx):
    """Creates an (n x 1) array which is a copy of v without element v[idx].
    """
    rv = np.copy(v)
    rv = np.reshape(v, (v.size,1))  # for GPy, which wants 2 dimensions
    np.delete(rv, idx, 0)
    return rv


def loo_err_1d(kernel, x, y):
    """
    Generates an estimate of leave-one-out error for a 1d data set.
    
       kernel : a GPy kernel object
       x      : a (n x 1) numpy array containing points in the input domain where 
                the underlying function was observations.
       y      : a (n x 1) numpy array corresponding to f(x) (i.e. observations)
    """
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



def plot_gp_mean_2d(gp, Xobs, yobs,
                    title=None, outfile=None):
    """Visualize the mean of a Gaussian process that has already been fit to data.
    """
    assert(Xobs.ndim == 2)
    assert(Xobs.shape[1] == 2)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #------------------------------
    # show observations and 95% CI
    #------------------------------
    if Xobs is not None and Xobs.size > 0:
        [ymu, ys2] = gp.predict(Xobs)
        [yq1, yq2] = gp.predict_quantiles(Xobs)   # ~\pm 2*sqrt(ys2)
        
        ax.scatter(Xobs[:,0], Xobs[:,1], yobs, c='r', marker='o')

        for ii in range(len(yq1)):
            line = art3d.Line3D((Xobs[ii,0], Xobs[ii,0]),
                                (Xobs[ii,1], Xobs[ii,1]),
                                (yq1[ii], yq2[ii]),
                                marker='', markevery=(1,1), color='r')
            ax.add_line(line)

        
    #------------------------------
    # show surrogate mean for domain of interest
    #------------------------------
    n = 100
    xMin = np.min(Xobs, axis=0)
    xMax = np.max(Xobs, axis=0)
    Xp1, Xp2 = np.meshgrid(np.linspace(xMin[0], xMax[0], n),
                           np.linspace(xMin[1], xMax[1], n))
    Xp = np.column_stack((to_col(Xp1), to_col(Xp2)))

    [ymu, ys2] = gp.predict(Xp)
    Mu = np.reshape(ymu, Xp1.shape)
    S2 = np.reshape(ys2, Xp1.shape)
    
    ax.plot_wireframe(Xp1, Xp2, Mu, alpha=0.3)
    
    ax.contour(Xp1, Xp2, Mu,
               zdir='z', offset=ax.get_zlim()[0], cmap=cm.coolwarm)

    #------------------------------
    # add annotations and extras
    #------------------------------
    ax.set_xlabel('x'); ax.set_ylabel('y');  ax.set_zlabel('error')

    if title:
        plt.title(title)

    if outfile:
        plt.savefig(outfile, bbox_inches='tight')


