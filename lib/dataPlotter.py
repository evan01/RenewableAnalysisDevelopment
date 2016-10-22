'''
    This module will have functions which take in data and then plot the results
'''
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import lib.csvImporter as csvImporter
import lib.rampCalculator as r


def plotRampData():
    print("Plotting ramp data")

def savePlot(plt):
    '''
    This function will save a plot in the plots folder for viewing later
    :param plt:
    :return:
    '''
    currentDate = time.strftime("%d-%m-%y")
    currentTime = time.strftime("%H:%M:%S")
    saveFileName = currentDate + "_" + currentTime
    plt.savefig("./plots/"+saveFileName+".png")


def plotCapacityRamp(data):

    '''
    class entry:
    Date = ""
    Time = ""
    WindGenBPAControl = ""
    firstDerivative = 0
    secondDerivative = 0
    '''

    #First make a list of the Ramp data to go on the y axis
    # Then make a list of the capacity data to go onto the x axis
    ramp = []
    capacity = []
    for entry in tqdm(data,desc="Converting data"):
        ramp.append(entry.firstDerivative)
        capacity.append(entry.WindGenBPAControl)

    maxRampVal = max(ramp)
    minRampVal = min(ramp)
    maxCapVal = max(capacity)
    minCapVal = min(capacity)


    plt.plot(capacity,ramp,"ro")
    # plt.axis([minCapVal,maxCapVal,minRampVal,maxRampVal])
    plt.axis([minCapVal, maxCapVal, -40, 40])

    savePlot(plt)

    #Then figure out the axis size
    print("Hey ")

def main():
    data = csvImporter.readCsv("./data/WindGenTotalLoadYTD_2016.csv")
    data = r.getFirstDerivative(data)
    plotCapacityRamp(data)
    print("e")

if __name__ == '__main__':
    main()