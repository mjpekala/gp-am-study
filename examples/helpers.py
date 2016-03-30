"""
Helper functions used in jupyter notebooks in this directory.
"""

import csv
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import axes3d, art3d


def load_data(fn):
    """Loads data set from CSV file.
       fn : The .csv file to load.  Format is assumed to be:
            xNominal, xMeasured, yNominal, yMeasured, dNominal, dMeasured, zNominal, zMeasured
            
            where x, y, d, z are pin x-position, y-position, diameter and height.
    """
    Z = []

    cast = lambda row: [row[0].strip(),] + [float(x) for x in row[1:]]
    
    with open(fn, 'rU') as f:
        reader = csv.reader(f, delimiter=',')
        for rowIdx, row in enumerate(reader):
            if rowIdx == 0: continue # skip header
                
            pinId, xNom, xMeas, yNom, yMeas, dNom, dMeas, zNom, zMeas = cast(row)
            Z.append((xNom, xMeas, yNom, yMeas, dNom, dMeas, zNom, zMeas))

    return np.array(Z)


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

        
def plot_xy_err_2d(indVar, dx, dy, xLabel=''):
    "Generates a 2D plot of position error in x and y dimensions"
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(121)
    plt.plot(indVar, dx, 'bo')
    plt.plot([np.min(indVar), np.max(indVar)], [0,0], 'k--')
    ax.set_xlabel(xLabel)
    ax.set_title('x error')

    ax = fig.add_subplot(122)
    ax.plot(indVar, dy, 'bo')
    ax.plot([np.min(indVar), np.max(indVar)], [0,0], 'k--')
    ax.set_xlabel(xLabel)
    ax.set_title('y error')

