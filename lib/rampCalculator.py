'''
    this module will take as input a data set of numbers and return a set
    that computes the ramp (gradient) of each data point
'''
import lib.csvImporter as csvImporter

def getRamp():
    '''
    This function will analyze a large data set and return a list of the first derivatives
    for each point in the data set
    '''
    data = csvImporter.readCsv(".\data\WindGenTotalLoadYTD_2016.csv")

    # Create the list of derivatitves that we will return
    firstDerivatives = []
    ramppoint1 = (int(data[1].WindGenBPAControl) - int(data[0].WindGenBPAControl))/(5)

    firstDerivatives.append((data[0].Time,ramppoint1))

    for point in data:

        #write shit in here
        print(point.WindGenBPAControl)

    print(ramppoint1)

    return firstDerivatives


import numpy as np
def main():
    getRamp()

if __name__ == '__main__':
    main()

