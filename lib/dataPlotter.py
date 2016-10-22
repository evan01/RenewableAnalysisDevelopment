'''
    This module will have functions which take in data and then plot the results
'''
import matplotlib.pyplot as plt
import time


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

def main():
    plt.plot([1,2,3,4],[1,2,3,4],'ro')
    plt.axis([0,6,0,20])
    plt.ylabel("Numbers")
    savePlot(plt)

if __name__ == '__main__':
    main()