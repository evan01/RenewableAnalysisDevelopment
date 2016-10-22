'''
    this module will take as input a data set of numbers and return a set
    that computes the ramp (gradient) of each data point
'''
import lib.csvImporter as csvImporter
from tqdm import tqdm
from lib.csvImporter import entry

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
    for p1,p2 in tqdm(zip(data,dataWithoutFirstPoint)):
        #todo Ask the professor whether or not we should handle varying times and not just assume time
        ramppoint = (int(p2.WindGenBPAControl) - int(p1.WindGenBPAControl)) / 5

        #Create an entry to put into a list of first derivatives
        datap = entry()
        datap.Date = p1.Date
        datap.Time = p1.Time
        datap.WindGenBPAControl = p1.WindGenBPAControl
        datap.firstDerivative = ramppoint


        derivatives.append(datap)

    return derivatives


def getSecondDerivatives(firstDerivatives):
    secondDerivatives = []

    firstDerwithoutFirstPoint = firstDerivatives[1:]
    for p1,p2 in tqdm (zip(firstDerivatives,firstDerwithoutFirstPoint)):
        ramppoint = (int(p2.firstDerivative) - int(p1.firstDerivative)) / (5)

        datap = entry()
        datap.Date = p1.Date
        datap.Time = p1.Time
        datap.firstDerivative = p1.firstDerivative
        datap.secondDerivative = ramppoint

        secondDerivatives.append(datap)

    return secondDerivatives



def main():
    data = csvImporter.readCsv("./data/WindGenTotalLoadYTD_2016.csv")
    firstDerivatives = getFirstDerivative(data)
    secondDerivatives = getSecondDerivatives(firstDerivatives)


if __name__ == '__main__':
    main()

