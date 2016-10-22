'''
    This module will read a csv file and return the data inside of the file
'''
import csv
import re
from tqdm import tqdm


def readCsv(filename):
    entries = []

    with open(filename, 'r') as csvfile:
        # Open up the file and read it
        reader = csv.reader(csvfile, skipinitialspace=True)
        numEntries = sum(1 for line in open(filename))

        # Create a progress bar
        bar = tqdm(total=numEntries, desc="Reading the CSV file")

        # Loop through CSV and add classes to entries
        for row in reader:
            ent = entry()
            ent.Date, ent.Time = parseDateTime(row[0])
            # Parse the time and date
            if ent.Date == None or ent.Time == None:
                continue
            # Add the wind speed to the entry
            ent.WindGenBPAControl = float(row[2])
            entries.append(ent)
            bar.update(1)


    return entries


def parseDateTime(dateTimeString):
    date = re.findall("\d\d/\d\d/\d\d", dateTimeString)
    time = re.findall("\d\d:\d\d", dateTimeString)
    if len(date) == 0 or len(time) == 0:
        return None, None
    return date, time


class entry:
    Date = ""
    Time = ""
    WindGenBPAControl = ""
    firstDerivative = 0
    secondDerivative = 0


def main():
    readCsv('data/WindGenTotalLoadYTD_2016.csv')
    list = [1,3,78,2,3]
    for i in list:
        print(i)


if __name__ == '__main__':
    main()
