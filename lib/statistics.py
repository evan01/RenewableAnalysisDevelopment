import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.mlab as BIV_NORM
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pandas import TimeGrouper
import seaborn as sns
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import statsmodels.formula.api as sm
import scipy
import scipy.optimize as sopt
import argparse
import math
import random


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
    'dateLabel': None,
    'outDir': None
}

statData = {
    'rawData': None,
    'data': None,
    'cap_series': None,
    'ramp_series': None,
    'x': None,
    'y': None,
    'summary_x': None,
    'summary_y': None
}


class StatGen:
    label = ""
    K = 3  # How we want to divide the standard deviation of our data set

    def __init__(self, pathToCsv, options=defaultOptions):
        self.label = options['valLabel']
        self.timeLabel = options['dateLabel']
        self.outDir = options['outDir']
        self.options = options
        self.stats = statData
        self.importCsv(pathToCsv, options)
        self.getStatistics()
        self.statAnalysis()

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
            parse_dates=[self.timeLabel],
            usecols=[self.timeLabel, self.label],
            date_parser=parser,
            na_filter=True,
            verbose=True
        )

        # Then convert the data to a propper pandas time series
        # start by getting the ramp values and dropping bad values and the normal data
        data['ramp'] = data[self.label].diff()
        data = data[np.isfinite(data['ramp'])]
        data = data[np.isfinite(data[self.label])]
        self.stats['rawData'] = data.copy()

        # Then try converting this into a propper time series
        data.index = data[self.timeLabel]
        del data[self.timeLabel]

        self.stats['data'] = data

    def getStatistics(self):
        # Get the Cap, Ramp and Summary Stats we're interested in
        data = self.stats['rawData']
        cap = data[self.label]
        ramp = data['ramp']
        x = data[self.label].as_matrix()
        y = ramp.as_matrix()
        summaryCap = data[self.label].describe()
        summaryRamp = data['ramp'].describe()

        # Store this data to the object
        self.stats['data'] = data
        self.stats['ramp'] = ramp
        self.stats['cap'] = cap
        self.stats['x'] = x
        self.stats['y'] = y
        self.stats['summary_x'] = summaryCap
        self.stats['summary_y'] = summaryRamp

    def statAnalysis(self):
        """
        This function is a caller of various statistical analysis functions on the data already entered
        :return: 
        """
        # Store this data to the object
        data = self.stats['data']
        ramp = self.stats['ramp']
        cap = self.stats['cap']
        times = data['Date/Time']
        x = self.stats['x']
        y = self.stats['y']
        summaryCap = self.stats['summary_x']
        summaryRamp = self.stats['summary_y']

        print("\tplotting cap series")
        self.plotCapacitySeries(times, x)

        print("\tplotting ramp series")
        self.plotRampSeries(times, y)

        print("\tplotting cap stats")
        self.plotCapacityStatistics(x, summaryCap, data)

        print("\tplotting ramp series")
        self.plotRampStatistics(y, summaryRamp, data)

        print("\tplotting joint ramp/cap series")
        # self.plotJointRampCapStats(x, y, data)

        print("\tPlot the 2d histogram")
        # self.plot2dHistogram(x, y)

        print("\tplotting 3d hist")
        # self.plot3dHistogram(x, y)

        print("\tgaus reg")
        self.plotGaussianRegression(x, y)

        print("\tpoly reg")
        self.plotPolynomialRegression(x, y, data)

        print("\tlinear reg")
        self.plotLinearRegression(x, y)

    def plotCapacitySeries(self, times, x):
        fig = plt.figure()
        fig.suptitle('Power Generation Time Series')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_xlabel(self.timeLabel)
        ax.set_ylabel(self.label)
        plt.plot(times, x)
        plt.savefig(self.outDir + "capSeries.png")

    def plotRampSeries(self, times, y):
        fig = plt.figure()
        fig.suptitle('Ramp (Derivative) of Power Gen Time Series')
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_xlabel(self.timeLabel)
        ax.set_ylabel('Ramp')
        plt.plot(times, y)
        plt.savefig(self.outDir + "rampSeries.png")

    def plotCapacityStatistics(self, x, capSum, data):
        """
        This function takes in a list of capacity values, and generates relevant stats about it
        :param x: 
        :return: 
        """
        mu = capSum['mean']
        sigma = capSum['std']
        count = capSum['count']
        fig = plt.figure()
        fig.suptitle('Capacity Analysis and Statistics (mu=' + str(int(mu)) + " sig=" + str(int(sigma)) + ")")
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_xlabel(self.label)
        ax.set_ylabel('Frequency')
        n, bins, p = plt.hist(x, 80, normed=1, alpha=0.8)

        # Add a line of best fit, pdf approx
        y = matplotlib.mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, y, 'r--', linewidth=1)
        plt.savefig(self.outDir + "capacityStats.png")

    def plotRampStatistics(self, y, rampSum, data):
        """
        This function takes in a list of ramp values and generaties relevant stats about it
        :param y: 
        :return: 
        """
        mu = rampSum['mean']
        sigma = rampSum['std']
        count = rampSum['count']
        fig = plt.figure()
        fig.suptitle('Ramp Analysis and Statistics (mu=' + str(int(mu)) + " sig=" + str(int(sigma)) + ")")
        ax = fig.add_subplot(111)
        fig.subplots_adjust(top=0.85)
        ax.set_xlabel('Ramp')
        ax.set_ylabel('Frequency')
        n, bins, p = plt.hist(y, 80, normed=1, alpha=0.8)

        # Add a line of best fit, pdf approx
        a = matplotlib.mlab.normpdf(bins, mu, sigma)
        l = plt.plot(bins, a, 'r--', linewidth=1)
        plt.savefig(self.outDir + "rampStats.png")

    def plotJointRampCapStats(self, x, y, data):
        """
        Thi function takes both ramp and capacity values at instances in time, and then makes judgements about them
        :param x: 
        :param y: 
        :return: 
        """
        print("\t\tscater")
        a = sns.regplot(x=self.label, y='ramp', data=data).get_figure()
        a.savefig(self.outDir + "scatter.png")

        # The first plot is a joint gaussian plot
        print("\t\tjoint gaussian")
        g = sns.PairGrid(data, diag_sharey=False)
        g.map_upper(sns.kdeplot, cmap="Blues_d")
        g.map_lower(plt.scatter)
        g.map_diag(sns.kdeplot, lw=3)
        g.savefig(self.outDir + "jointGaus.png")

        # The next is a a set of kde (contour) plots
        # todo modify the constant values to be std and mean
        print("\t\tjointKde1")
        h = sns.jointplot(x=self.label, y='ramp', data=data, kind="kde", ylim={-80, 80}, xlim={0, 1500},
                          color='r')  # A kind of heatmap
        print("\t\tjointKde2")
        i = sns.jointplot(x=self.label, y='ramp', data=data, kind='hex', ylim={-10, 10}, xlim={0, 40},
                          color='r')
        print("\t\tjointGrid")
        j = sns.JointGrid(x=self.label, y='ramp', data=data, ylim=(-80, 80), xlim=(0, 1000), size=5, ratio=2)
        j = j.plot_joint(sns.kdeplot, cmap="Reds_d")
        j = j.plot_marginals(sns.kdeplot, color='r', shade=True)

        h.savefig(self.outDir + "jointKde1.png")
        i.savefig(self.outDir + "jointKde2.png")
        j.savefig(self.outDir + "jointKde3.png")

    def plot2dHistogram(self, x, y):
        # Plot data
        fig1 = plt.figure()
        plt.plot(x, y, '.r')
        plt.xlabel('x')
        plt.ylabel('y')

        # Plot 2D histogram using pcolor
        # todo, find a way to not the axis values be hard coded
        fig2 = plt.figure()
        a = plt.hexbin(x, y, cmap=plt.cm.YlOrRd_r, gridsize=1600, mincnt=4, marginals=True)
        fig2.suptitle('2D histogram of Ramp and Capacity')
        ax = fig2.add_subplot(111)
        ax.set_xlabel(self.label)
        ax.set_ylabel('Ramp')
        cb = plt.colorbar()
        cb.set_label('counts')
        plt.axis([0, 30, -10, 10])

        fig1.savefig(self.outDir + "2dHist_noheat.png")
        fig2.savefig(self.outDir + "2dHist.png")

    def plot3dHistogram(self, x, y):
        '''
        Please note that the majority of code below was taken from 
        http://matplotlib.org/examples/mplot3d/hist3d_demo.html
        :param x: 
        :param y: 
        :return: 
        '''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Just change binsOne and binsTwo to lists.
        hist, yedges, xedges = np.histogram2d(x, y, bins=[32, 32], normed=False)

        # The start of each bucket.
        xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1])

        xpos = xpos.flatten()
        ypos = ypos.flatten()
        zpos = np.zeros_like(xpos)

        # The width of each bucket.
        dx, dy = np.meshgrid(yedges[1:] - yedges[:-1], xedges[1:] - xedges[:-1])

        dx = dx.flatten()
        dy = dy.flatten()
        dz = hist.flatten()

        ax.bar3d(ypos, xpos, zpos, dx, dy, dz, color='b', )
        plt.xlabel(self.label)
        plt.ylabel('ramp')
        plt.savefig(self.outDir + "3dHist.png")

    def plotGaussianRegression(self, x, y):
        """
        This function takes in ramp and capacity values and then performs the relevant regressions
        :param x: 
        :param y: 
        :return: 
        """

        # first thing is to get the x, y, and z coordinates (z is the hist of x,y joint)
        binSize = 32
        x_vals = []
        y_vals = []
        z_vals = []

        hist, xedges, yedges = np.histogram2d(x, y, bins=[binSize, binSize], normed=False)

        for c_idx in range(binSize):
            cap = xedges[c_idx]
            for r_idx in range(binSize):
                ramp = yedges[r_idx]
                x_vals.append(cap)
                y_vals.append(ramp)
                z_vals.append(hist[c_idx][r_idx])

        self.x = x_vals
        self.y = y_vals
        self.z = z_vals
        X, Y, Z = x_vals, y_vals, z_vals

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111, projection='3d')
        surf = ax2.plot_trisurf(X, Y, Z, cmap=cm.OrRd, linewidth=3, shade=True)
        # surf = ax2.plot_wireframe(X, Y, Z, cmap=cm.OrRd, linewidth=1)
        fig2.savefig(self.outDir + "surfacePlot.png")

    def plotPolynomialRegression(self, x, y, data):
        """
        This function takes in ramp and capacity values in instances in time and then tries to find the relationship
        using polynomial regression
        
        As of now this doesn't quite work, not sure why not
        :param x: 
        :param y: 
        :return: 
        """
        pass
        # dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']
        #
        # for dist_name in dist_names:
        #     dist = getattr(scipy.stats, dist_name)
        #     param = dist.fit(x)
        #     pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * len(x)
        #     plt.plot(pdf_fitted, label=dist_name)
        #     plt.xlim(0, 47)
        # plt.legend(loc='upper right')
        # plt.show()

    def plotLinearRegression(self, x, y):
        """
        This function takes in ramp and capacity values in instances in time and then tries to find the relationship
        using linear regression
        :param x: 
        :param y: 
        :return: 
        """
        return
        a = sns.residplot(x, y).get_figure()
        a.savefig(self.outDir + "linReg.png")

        res = sm.OLS(y, x).fit()
        print(res.summary())

    def gausFunction(self, x, y, xM, xSig, yM, ySig):
        """
        We want to approximate and perform regression of a standard gaussian function on our data
        :param x: 
        :param y: 
        :param xM: 
        :param xSig: 
        :param yM: 
        :param ySig: 
        :return: 
        """
        return 3 * math.e ^ (-() / () * () / (2))

def main():
    '''
        The only thing that the front end needs to provide is the following information...
    '''
    # Command line program so not as much effort is needed
    # parser = argparse.ArgumentParser(
    #     description="Generate Renewable Energy Statistics",
    # )
    # parser.add_argument('-p',help='the path to the csv file')
    # parser.add_argument('-d',help='The header of the Date/Time field in the CSV File')
    # parser.add_argument('-v',help='The header of the Capacity field in the CSV File')
    # args = parser.parse_args()

    pathToCSV = "./data/WindGenTotalLoadYTD_2011_1.csv"
    opts = defaultOptions
    opts['valLabel'] = 'TOTAL WIND GENERATION'
    opts['dateLabel'] = 'Date/Time'
    opts['outDir'] = './plots/'

    s = StatGen(pathToCSV, opts)
    # s.plotStatistics()


if __name__ == '__main__':
    main()
