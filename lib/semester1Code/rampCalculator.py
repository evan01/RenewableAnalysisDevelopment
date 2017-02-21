'''
    this module will take as input a data set of numbers and return a set
    that computes the ramp (gradient) of each data point
'''
from tqdm import tqdm

import lib.semester1Code.csvImporter as csvImporter
from lib.semester1Code.csvImporter import entry


def getRamp():
    '''
    This function will analyze a large data set and return a list of the first derivatives
    for each point in the data set
    '''
    data = csvImporter.readCsv("./data/WindGenTotalLoadYTD_2016.csv")

    # Create the list of deriva titves that we will return
    firstDerivatives = []
    ramppoint1 = (int(data[1].WindGenBPAControl) - int(data[0].WindGenBPAControl))/(5)

    firstDerivatives.append((data[0].Time,ramppoint1))

    for point in data:

        #write shit in here
        print(point.WindGenBPAControl)

    print(ramppoint1)

    return firstDerivatives

#This function will take in a list of data with (time , value) pairs and output a second list
# Of the same size as the first list of the first derivatives
def getFirstDerivative(data):
    derivatives = []

    dataWithoutFirstPoint = data[1:]
    bar = tqdm(total=len(dataWithoutFirstPoint),desc="Calculating the first derivative")
    for p1,p2 in zip(data,dataWithoutFirstPoint):
        #todo Ask the professor whether or not we should handle varying times and not just assume time
        ramppoint = (float(p2.WindGenBPAControl) - float(p1.WindGenBPAControl)) / 5

        #Create an entry to put into a list of first derivatives
        datap = entry()
        datap.Date = p1.Date
        datap.Time = p1.Time
        datap.WindGenBPAControl = p1.WindGenBPAControl
        datap.firstDerivative = ramppoint

        #Append this to our list of first derivatives
        derivatives.append(datap)
        bar.update(1)

    return derivatives


def getSecondDerivatives(firstDerivatives):
    secondDerivatives = []

    firstDerwithoutFirstPoint = firstDerivatives[1:]
    bar2 = tqdm(total=len(firstDerwithoutFirstPoint), desc="Calculating the second derivative")
    for p1,p2 in tqdm (zip(firstDerivatives,firstDerwithoutFirstPoint)):
        ramppoint = (float(p2.firstDerivative) - float(p1.firstDerivative))/(5)

        datap = entry()
        datap.Date = p1.Date
        datap.Time = p1.Time
        datap.WindGenBPAControl = p1.WindGenBPAControl
        datap.firstDerivative = p1.firstDerivative
        datap.secondDerivative = ramppoint

        secondDerivatives.append(datap)
        bar2.update(1)

    return secondDerivatives


def main():
    data = csvImporter.readCsv("./data/WindGenTotalLoadYTD_2016.csv")
    firstDerivatives = getFirstDerivative(data)
    secondDerivatives = getSecondDerivatives(firstDerivatives)
    print("e")

if __name__ == '__main__':
    main()

