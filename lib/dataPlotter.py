'''
    This module will have functions which take in data and then plot the results
'''
import time

import matplotlib.pyplot as plt
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import lib.csvImporter as csvImporter
import lib.rampCalculator as r
import numpy as np
from matplotlib import pyplot as PLT
from matplotlib import cm as CM
from matplotlib import mlab as ML
import random
import numpy as NP

DEBUG = False

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
    plt.savefig("./plots/" + saveFileName + ".png")


def plotCapacityRamp(data):
    '''
    class entry:
    Date = ""
    Time = ""
    WindGenBPAControl = ""
    firstDerivative = 0
    secondDerivative = 0
    '''

    # First make a list of the Ramp data to go on the y axis
    # Then make a list of the capacity data to go onto the x axis
    ramp = []
    capacity = []
    for entry in tqdm(data, desc="Converting data"):
        ramp.append(entry.firstDerivative)
        capacity.append(entry.WindGenBPAControl)

    maxRampVal = max(ramp)
    minRampVal = min(ramp)
    maxCapVal = max(capacity)
    minCapVal = min(capacity)

    # Try gettin the histogram of Data
    plt.hist(ramp)
    plt.show()

    plt.plot(capacity, ramp, "ro")
    plt.axis([minCapVal, maxCapVal, minRampVal, maxRampVal])
    # plt.axis([minCapVal, maxCapVal, -40, 40])
    plt.ylabel("Ramp Value ")
    plt.xlabel("Capacity")

    savePlot(plt)

    # Then figure out the axis size
    print("Hey ")


def get3dData(data, n):
    print("Aquiring 3d data")
    #Get all of the partition Data
    ramp,capacity,maxRampVal, minRampVal, \
    maxCapVal, minCapVal, rampPartition, \
    genPartition = get3dPartitionData(data,n)

    # Then create an n X n array
    matrix = [[0 for x in range(n)] for y in range(n)]

    # Then go through the elements in the list and place them in the correct array
    for rp, cp in zip(ramp, capacity):
        x, y = get3DPosition(rp, cp, rampPartition, genPartition, n)
        if DEBUG:
            print("\nx: " + str(x) + " Cap: " + str(cp))
            print("y: " + str(y) + " Ramp: " + str(rp))

        # place this in the array
        matrix[x][y] += 1

    #Now we should have correctly classified each data point in the array
    return matrix,rampPartition, genPartition

def get3dPartitionData(data,n):
    # First get the maximum and min values from out data
    ramp = []
    capacity = []
    for entry in tqdm(data, desc="Converting data"):
        ramp.append(entry.firstDerivative)
        capacity.append(entry.WindGenBPAControl)

    # Then calculate the max, min values along with partitions to divide our dataset into
    maxRampVal = max(ramp)
    minRampVal = min(ramp)
    maxCapVal = max(capacity)
    minCapVal = min(capacity)
    genPartition = int((maxCapVal - minCapVal) / n)
    #Ramp Partition is hard! Choose the max of 1 or two things
    rampPartition = int((max(maxRampVal,abs(minRampVal)) / (n/2)))+1

    if DEBUG:
        print("MaxGenVal:" + str(maxCapVal))
        print("MinGenVal:" + str(minCapVal))
        print("MaxRampVal: " + str(maxRampVal))
        print("MinRampVal: " + str(minRampVal))
        print("X/GenPartition: " + str(genPartition))
        print("Y/RampPartition: " + str(rampPartition))

    return ramp, capacity, maxRampVal,minRampVal,maxCapVal,minCapVal,rampPartition,genPartition


def get3DPosition(ramp, cap, rampPartition, genPartition, n):
    '''
        This function is going to take in a ramp value and a capacity and return and x,y coordinate
        representing where this data point should be on a nxn grid
    '''
    # Cap part is easy because it's always positive
    x = int(cap / genPartition)

    # The ramp component is difficult because it can be negative
    if (ramp > 0):
        y = int(ramp / rampPartition) + int(n/2)
    else:
        #then it's a negative number
        y = int((n/2)-1-int(-1*ramp / rampPartition))


    #Some exeptions that happen, when the max val is discovered array[n] doesn't exist
    if(x ==n):
        x-=1
    if(y == n):
        x-=1

    #When the user inputs a non even division of numbers, makes -'ve and +'ve numbers difficult
    if n % 2 != 0:
        raise ValueError("N given must be an even number!")

    return x, y


def plot3DData(matrix,ramp,capacity,n):
    '''
    This function takes in a data set and plots a 3 dimensional bar graph
    :param matrix:
    :param ramp:
    :param capacity:
    :param n:
    :return:
    '''
    print("Plotting 3d Data")

    capRanges = [i*capacity for i in range(n)]
    # y is half positive and half negative
    rampRanges = [-i*ramp for i in range(int(n/2))][::-1]+[i*ramp for i in range(int(n/2))]

    xPos = []
    yPos = []
    for i in capRanges:
        for j in rampRanges:
            xPos.append(i)
            yPos.append(j)

    z = []
    for i in matrix:
        for j in i:
            z.append(j)
    zpos = [1 for i in range(n**2)]
    dx = [capacity for i in range(n**2)]
    dy = [ramp for i in range(n**2)]
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    colors = ["#68ee00","#00d07c","#ff4c00","#f00045","#4db88d","#e96188"]
    fig.suptitle("Ramp vs Capacity Data Segmented into "+str(n)+" componets")
    ax1.set_xlabel("Capacity Gen")
    ax1.set_ylabel("Ramp Value")
    ax1.bar3d(xPos, yPos, zpos, dx, dy, z, color="#e96188")

    plt.show()

def plotHeatMap(data):
    ramp = []
    capacity = []
    for entry in tqdm(data, desc="Converting data"):
        ramp.append(entry.firstDerivative)
        capacity.append(entry.WindGenBPAControl)

    n = 1e5
    print("Meshgrid")
    X,Y = np.meshgrid(capacity,ramp)
    Z1 = ML.bivariate_normal(X, Y, 2, 2, 0, 0)
    Z2 = ML.bivariate_normal(X, Y, 4, 1, 1, 1)
    ZD = Z2 - Z1
    x = X.ravel()
    y = Y.ravel()
    z = ZD.ravel()
    gridsize = 30
    PLT.subplot(111)
    print("Placing items in bins")

    # if 'bins=None', then color of each hexagon corresponds directly to its count
    # 'C' is optional--it maps values to x-y coordinates; if 'C' is None (default) then
    # the result is a pure 2D histogram

    PLT.hexbin(x, y, C=z, gridsize=gridsize, cmap=CM.jet, bins=None)
    PLT.axis([x.min(), x.max(), y.min(), y.max()])

    print("Plott")
    cb = PLT.colorbar()
    cb.set_label('mean value')
    PLT.show()


def main():
    data = csvImporter.readData()
    data = r.getFirstDerivative(data)


    # APPLy TRANSFORMATIONS ON THE DATA HERE
    # Shrink data by a lot??
    # data = data[:4000]
    data = random.sample(data,5000)
    #initially n = 4
    n = 8

    plotHeatMap(data)


    # matrix,rampPartition,capPartition = get3dData(data, n)
    # plot3DData(matrix,rampPartition,capPartition,n)
    # print("e")


if __name__ == '__main__':
    main()
