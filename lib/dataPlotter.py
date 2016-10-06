'''
    This module will have functions which take in data and then plot the results
'''
import matplotlib.pyplot as plt
import time


def plotRampData():
    print("Plotting ramp data")

def savePlot(plt):
    currentDate = time.strftime("%d-%m-%y")
    currentTime = time.strftime("%H:%M:%S")
    saveFileName = currentDate + "_" + currentTime
    plt.savefig("./plots/"+saveFileName+".png")

def main():
    plt.plot([1,2,3,4])
    plt.ylabel("Numbers")
    savePlot(plt)

if __name__ == '__main__':
    main()