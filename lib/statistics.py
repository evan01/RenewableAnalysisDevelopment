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
import argparse

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
        data = self.stats['rawData']

        # First thing to do is to remove the biggest outliers from the data set
        # Keep everything within 4 standard deviations
        # Also keep the x values for some regression, (AKA the CAP values)
        # data = data[~((data - data.mean()).abs() > 4 * data.std())]
        # data = data.dropna()
        # x = data[self.label]
        # x = x.dropna()
        # x = x.as_matrix()
        #
        # # Then get the ramp data, make 1 master pandas dataframe
        # ramp = data.diff().dropna()
        # y = ramp.as_matrix().T[0]
        # ramp = ramp.rename(columns={self.options['valLabel']: 'ramp'})
        # data['ramp'] = ramp
        # data = data.dropna()
        ramp = data['ramp']
        x = data[self.label].as_matrix()
        y = ramp.as_matrix()

        #Append the cap and ramp data to the summary of all the data
        self.stats['data'] = data
        self.stats['ramp'] = ramp
        self.stats['x'] = x
        self.stats['y'] = y

        # Then get the summary stats
        summaryCap = data['ramp'].describe()
        summaryRamp = ramp.describe()
        self.stats['summary_x'] = summaryCap
        self.stats['summary_y'] = summaryRamp

    def statAnalysis(self):
        """
        This function is a caller of various statistical analysis functions on the data already entered
        :return: 
        """

        x = self.stats['x']
        y = self.stats['y']
        series = self.stats['series']
        rawData = self.stats['rawData']
        data = self.stats['data']

        self.plotTimeSeries(rawData)

        self.plotCapacityStatistics(x)

        self.plotRampStatistics(y)

        self.plotJointRampCapStats(x, y)

        self.plotGaussianRegression(x, y)

        self.plotPolynomialRegression(x, y)

        self.plotLinearRegression(x, y)

    def plotTimeSeries(self, data):
        """
        This function takes in a pandas time series dataframe and then plots the output
        :param data: 
        :return: 
        """
        # s = data.set_index('Date/Time')[self.label]
        # s.plot()
        # plt.show()
        sns.tsplot(data, time=self.timeLabel, unit=self.label, estimator=np.median)
        # sns.tsplot(data,time=self.timeLabel,value='ramp')
        # print("done")

    def plotCapacityStatistics(self,x):
        """
        This function takes in a list of capacity values, and generates relevant stats about it
        :param x: 
        :return: 
        """
        pass

    def plotRampStatistics(self,y):
        """
        This function takes in a list of ramp values and generaties relevant stats about it
        :param y: 
        :return: 
        """
        pass

    def plotJointRampCapStats(self, x, y):
        """
        Thi function takes both ramp and capacity values at instances in time, and then makes judgements about them
        :param x: 
        :param y: 
        :return: 
        """
        pass

    def plotGaussianRegression(self, x, y):
        """
        This function takes in ramp and capacity values and then performs the relevant regressions
        :param x: 
        :param y: 
        :return: 
        """
        pass

    def plotPolynomialRegression(self, x, y):
        """
        This function takes in ramp and capacity values in instances in time and then tries to find the relationship
        using polynomial regression
        :param x: 
        :param y: 
        :return: 
        """
        pass

    def plotLinearRegression(self, x, y):
        """
        This function takes in ramp and capacity values in instances in time and then tries to find the relationship
        using linear regression
        :param x: 
        :param y: 
        :return: 
        """
        pass


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
    opts['valLabel'] = 'TOTAL WIND GENERATION  IN BPA CONTROL AREA (MW; SCADA 79687)'
    opts['dateLabel'] = 'Date/Time'

    s = StatGen(pathToCSV, opts)
    # s.plotStatistics()


if __name__ == '__main__':
    main()
