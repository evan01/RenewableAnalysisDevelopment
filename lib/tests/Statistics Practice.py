import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas import TimeGrouper
import seaborn as sns
import statsmodels.formula.api as sm
import scipy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as MLAB
from mpl_toolkits.mplot3d import Axes3D

'''
    This file ultimately is to test the different plots and regression techniques on standardized data
'''

def statsPractice():
    # First get your x and y data
    x = np.random.random((50, 1))
    y = np.random.random((50, 1))
    '''
                FAKE DATA AND EXAMPLES
            '''
    # Then, going through the ramp and capacity data, do some sort of regression...
    print("\nFAKE DATA!! JUST FOR DEMOING STATS LIBRARY")
    print("Linear REGRESSION USING statsmodels library")
    df = pd.DataFrame({"Z": [10, 20, 30, 40, 50], "X": [20, 30, 10, 40, 50], "Y": [32, 234, 23, 23, 42523]})
    result = sm.ols(formula="Z ~ X + Y",
                    data=df).fit()  # This formula determines the relationship beforehand..., need to find this out
    print(result.summary())

    # some 3-dim points
    mean = np.array([0.0, 0.0, 0.0])
    cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
    data = np.random.multivariate_normal(mean, cov, 50)

    # regular grid covering the domain of the data
    X, Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
    XX = X.flatten()
    YY = Y.flatten()
    A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
    C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])

    # evaluate it on a grid
    Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)

    # plot points and fitted surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    ax.axis('equal')
    ax.axis('tight')
    plt.show()
    print("done")


def getBivariateDistribution():
    # Parameters to set
    mu_x = 0
    sigma_x = np.sqrt(3)

    mu_y = 0
    sigma_y = np.sqrt(15)

    # Create grid and multivariate normal
    x = np.linspace(-10, 10, 500)
    y = np.linspace(-10, 10, 500)
    X, Y = np.meshgrid(x, y)
    Z = MLAB.bivariate_normal(X, Y, sigma_x, sigma_y, mu_x, mu_y)

    # Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()


if __name__ == '__main__':
    getBivariateDistribution()
