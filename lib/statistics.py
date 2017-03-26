import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.mlab as BIV_NORM
from mpl_toolkits.mplot3d import Axes3D
from pandas import TimeGrouper
import seaborn as sns
import statsmodels.formula.api as sm
import scipy

'''
Some useful tutorials for the time series statistics that we are calculating
http://machinelearningmastery.com/time-series-data-visualization-with-python/ USEFUL!!
http://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/
'''
defaultOptions = {
    'TimeSeries': False,
    'rampVCap': True,
    'Hist': False,
    'BvDist': False,
    'Heatmap': True,
    'valLabel': None,
    'dateLabel': None
}

statData = {
    'rawData': None,
    'data': None,
    'ramp1': None,
    'ramp2': None,
    'mean': None,
    'sig_X': None,
    'sig_Y': None
}


class StatGen:
    label = ""
    K = 3  # How we want to divide the standard deviation of our data set

    def __init__(self, pathToCsv, options=defaultOptions):
        self.label = options['valLabel']
        self.options = options
        self.stats = statData
        self.importCsv(pathToCsv, options)
        self.getStatistics()

    def importCsv(self, path, opts):
        '''
        This function will import either the default csv file in the example above or a specified file
        :param path:
        :return:
        '''
        print("Importing the csv")
        # Then read default path, will read all files in data directory, using the BP dataset

        # Parser for the date time fields
        parser = lambda date: pd.datetime.strptime(date, '%m/%d/%y %H:%M')
        data = pd.read_csv(
            path,
            low_memory=False,
            header=0,
            skip_blank_lines=True,
            infer_datetime_format=True,
            parse_dates=[opts['dateLabel']],
            usecols=[opts['dateLabel'], self.label],
            date_parser=parser,
            na_filter=True,
            verbose=True
        )

        # Then convert the data to a propper pandas time series
        data.dropna()
        data[opts['dateLabel']] = pd.to_datetime(data[opts['dateLabel']])
        data.index = data[opts['dateLabel']]
        del data[opts['dateLabel']]
        self.stats['rawData'] = data

    def getStatistics(self):
        data = self.stats['rawData']

        # First thing to do is to remove the biggest outliers from the data set
        # Keep everything within 4 standard deviations
        # Also keep the x values for some regression, (AKA the CAP values)
        data = data[~((data - data.mean()).abs() > 4 * data.std())]
        data.dropna()
        x = data[self.label]

        # Then get the ramp data, make 1 master pandas dataframe
        ramp = data.diff().dropna()
        y = ramp.as_matrix()
        ramp = ramp.rename(columns={self.options['valLabel']: 'ramp'})
        data['ramp'] = ramp

        # Run the clustering algorithm on the data, to get the grid distribution
        self.kdTree(data, self.K)

        self.stats['data'] = data
        self.stats['ramp'] = ramp

    def kdTree(self, data, k):
        """
        This is an implementation of the kd-tree algorithm, will aid in clustering our data into a better model
        :param data: Pandas Dataframe
        :return: 
        """

        x = data[self.label]
        y = data['ramp']
        tree_y = self.createKdTree(x, 3, 5)  # 3 standard devs, 5 partitions per standard dev
        tree_x = self.createKdTree(y, 3, 5)

    def createKdTree(self, dataset, stds, k):
        """
        This function will build a kd tree that partitions the data set passed in by 
        standard deviation
        :param dataset: 
        :param stds: 
        :param k: 
        :return: 
        """

        # Get the mean and the standard deviation of the dataset
        std = int(dataset.std())
        mean = int(dataset.mean())

        # Important to then get the grid number for each datapoint...
        grids = stds * k * 2

        class Node:
            def __init__(self, val, parent=None):
                self.val = val
                self.parent = parent
                self.children = []
                self.sigmaNum = -1
                self.binNum = -1
                self.gridNum = -1

        class kdTree:
            def __init__(self, Node):
                self.root = Node

        # Do the standard deviations first, 3 standard deviations is enough?
        root = Node(mean)
        for i in range(1, stds + 1):
            Lval = mean - i * std
            Rval = mean + i * std
            NodeL, NodeR = Node(Lval, root), Node(Rval, root)
            root.children.append(NodeL)
            root.children.append(NodeR)
        root.children = sorted(root.children, key=lambda x: x.val)

        # Then within each standard deviation, we have k leaf nodes
        for i in range(len(root.children)):
            child = root.children[i]
            for j in range(k):
                kVal = child.val + int(j * std / k)
                nd = Node(kVal, child)
                nd.binNum = j
                nd.sigmaNum = i
                child.children.append(nd)

        print("k")

    def plotTimeSeries(self, data):
        data.plot()
        plt.savefig("./plots/originalSeries.png")

    def plotHistogram(self, data):
        data.hist()
        plt.savefig("./plots/histogram.png")

    def plotRampVCapacity(self, data):
        """
        This function plots a number of figures showing the relationships between ramp and capacity
        :param data:
        :return:
        """
        # opts = self.options
        # First thing to do is to get rid of the na vals, they seem to pop up often
        data.dropna(inplace=True, how='any')

        x = data[self.label]
        y = data['ramp']

        x = x.as_matrix()
        y = y.as_matrix()

        # There are multiple different kinds of plots for ramp and capacity
        sns.jointplot(x=self.label, y='ramp', data=data)  # Standard scatter
        sns.jointplot(x=self.label, y='ramp', data=data, kind="kde", ylim={-80, 80}, xlim={0, 1500},
                      color='r')  # A kind of heatmap
        sns.jointplot(x=self.label, y='ramp', data=data, kind='hex', ylim={-80, 80}, xlim={0, 1500},
                      color='r')  # Hex bin plot

        # Try some parametrization
        parametrized = sns.jointplot(x=self.label, y='ramp', data=data)
        parametrized.plot_marginals(sns.distplot)

        # Try to draw hills
        g = sns.JointGrid(x=self.label, y='ramp', data=data, ylim=(-80, 80), xlim=(0, 1000), size=5, ratio=2)
        g = g.plot_joint(sns.kdeplot, cmap="Reds_d")
        g = g.plot_marginals(sns.kdeplot, color='r', shade=True)

        # Try to draw a simple kde plot...
        sns.kdeplot(x, y, ylim={-80, 80})  # A hill like contour plot

        sns.plt.show()
        print("done")

    def getBivariateDistribution(self, data, GRID):
        """
        This will get the bivariate distribution of the data set and plot the output
        :param data:
        :param GRID:
        :return:
        """
        # Might be worthwhile to remove outliers... hmm kmeans might help with this
        data.dropna(inplace=True, how='any')
        x = data[self.label].as_matrix()
        y = data['ramp'].as_matrix()

        # Params to find using data
        Expectation_x = x.mean()
        Expectation_y = y.mean()

        sig_x = int(x.var() ** .5)
        sig_y = int(y.var() ** .5)

        # This is to give to the pdf function
        print("Applying the binning, meshgrid function")
        X, Y = np.meshgrid(x, y)
        pos = np.empty(X.shape + (2,))
        pos[:, :, 0] = X;
        pos[:, :, 1] = Y

        print("Aquiring normal distribution")
        Z = BIV_NORM.bivariate_normal(X, Y, sig_x, sig_y, Expectation_x, Expectation_y)

        print("Plot the distribution")

        # Make a 3D plot
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Z axis')
        plt.show()


        # Let scipy.stats do the multivariate normal distribution heavy lifting, pass in covariance matrix

    def plotStatistics(self):
        """
        This function will take in a dict of statics and plot all of them
        Each kind of statistic has a different output
        :param data:
        :param options:
        :return:
        """
        '''
            For each kind of option we need to generate a diferent plot
            ALSO WHY NO SWITCH STATEMENTS in python... :/
            We're using a python dictionary to access the different stats and series information,
            it's the most mem. efficient
        '''
        options = self.options
        series = self.stats['rawData']
        data = self.stats['data']
        ramp1 = self.stats['ramp1']

        # self.plotTimeSeries(series)

        # self.plotRampVCapacity(data)

        self.getBivariateDistribution(data, 60)

        print("Done plotting all the statistics")


if __name__ == '__main__':
    '''
        The only thing that the front end needs to provide is the following information...
    '''
    pathToCSV = "./data/WindGenTotalLoadYTD_2011_1.csv"
    opts = defaultOptions
    opts['valLabel'] = 'TOTAL WIND GENERATION  IN BPA CONTROL AREA (MW; SCADA 79687)'
    opts['dateLabel'] = 'Date/Time'

    s = StatGen(pathToCSV, opts)
    s.plotStatistics()
