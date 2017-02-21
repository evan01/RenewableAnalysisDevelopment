import lib.pandasImporter as pi
import numpy as np
import pandas as pd
import glob
import os
from tqdm import tqdm
import matplotlib as plt
from matplotlib import pyplot
from pandas import TimeGrouper

'''
Some useful tutorials for the time series statistics that we are calculating
http://machinelearningmastery.com/time-series-data-visualization-with-python/ USEFUL!!
http://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/
'''
defaultOptions = {
    'TimeSeries': False,
    'StdDev': False,
    'Ramp1': False,
    'Hist': False,
    'BvDist': False,
    'Heatmap': True
}

defaultPath = "./data/"
singlePath = "./data/WindGenTotalLoadYTD_2011_1.csv"


def importCsv(path=None):
    '''
    This function will import either the default csv file in the example above or a specified file
    :param path:
    :return:
    '''
    print("Importing the csv")
    data = None
    if path is None:
        # Then read default path, will read all files in data directory, using the BP dataset
        fields = ['Date/Time', 'TOTAL WIND GENERATION  IN BPA CONTROL AREA (MW; SCADA 79687)']
        # Parser for the date time fields
        parser = lambda date: pd.datetime.strptime(date, '%m/%d/%y %H:%M')
        data = pd.read_csv(
            singlePath,
            low_memory=False,
            header=0,
            skip_blank_lines=True,
            infer_datetime_format=True,
            parse_dates=['Date/Time'],
            usecols=fields,
            date_parser=parser,
            na_filter=True,
            verbose=True
        )
        # allCSVFiles = glob.glob(os.path.join(defaultPath,"*.csv"))
        # df_from_file = (pd.read_csv(f,parse_dates=['Date/Time']) for f in tqdm(allCSVFiles,desc="Reading the CSV files"))
        # data = pd.concat(df_from_file,ignore_index=False)

    else:
        print("read a single path")
    # Then dro
    data = convertDataToTimeSeies(data)
    return data


def convertDataToTimeSeies(data):
    '''
    This function will convert the data that we have to the time series
    :param data:
    :return:
    '''
    # Make sure that the dates are date time objects
    data['Date/Time'] = pd.to_datetime(data['Date/Time'])
    # Set the date time as the index, delete the old column
    data.index = data['Date/Time']
    del data['Date/Time']

    return data


def getStatistics(data, options=defaultOptions):
    print("Computing the standard deviation")
    stdDev = data.std()
    print("Getting Ramp1")
    ramp1 = data.diff()
    print("Getting Ramp2")
    ramp2 = ramp1.diff()
    print("")

    return data


def plotTimeSeries(data):
    data.plot()
    pyplot.savefig("./plots/originalSeries.png")

    # fig.saveFig('rawSeries.png')
    # data.savefig('pandasSeries.png')


def plotHistogram(data):
    data.hist()
    pyplot.savefig("./plots/histogram.png")


def plotRampVCapacity(data):
    """

    :param data:
    :return:
    """
    # First get the ramp data
    ramp = data.diff()

    # then join the capacity and ramp data together
    # rampCapData = pd.concat([data,ramp],axis=1)
    ramp, cap = ramp.as_matrix(), data.as_matrix()
    scat = pyplot.scatter(cap, ramp)
    pyplot.plot()
    pyplot.savefig("./plots/rampCap.png")
    print("done")


def getBivariateDistribution(data):
    pass


def plotHeatmap(data):
    # Drop the Nan Values
    data = data.dropna()

    # First get the ramp data
    ramp = data.diff()

    # Flatten matrices
    ramp = ramp.as_matrix()[1:].flatten()
    data = data.as_matrix()[1:].flatten()
    heatmap, xedges, yedges = np.histogram2d(data, ramp, bins=[32, 32])
    heatmap = heatmap.T
    pyplot.imshow(heatmap, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
    # pyplot.imshow(heatmap, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], 10, -10])
    pyplot.plot()
    pyplot.savefig("./plots/heatmap.png")


def plotStatistics(data, options=defaultOptions):
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
    '''
    if options['TimeSeries']:
        plotTimeSeries(data)
    if options['Hist']:
        plotHistogram(data)
    if options['StdDev']:
        print(data.std())  # Seems kinda redundant...
    if options['Ramp1']:
        plotRampVCapacity(data)
    if options['Heatmap']:
        plotHeatmap(data)
    if options['BvDist']:
        getBivariateDistribution(data)

    print("Done computing all the statistics")


def getStatisticalData(pathToCSVUpload):
    '''
    This function will return the JSON statistics of all of the statistical data that's required
    :param pathToCSVUpload: the string path of the file that was uploaded to the server for processing
    :return: a python dictionary that contains all of the desired statistical data
    '''


def main():
    # First import that data file as a time series
    timeSeries = importCsv()

    # Then get the main statistics
    stats = getStatistics(timeSeries)

    # Then plot the statistics
    plotStatistics(stats)


if __name__ == '__main__':
    main()
