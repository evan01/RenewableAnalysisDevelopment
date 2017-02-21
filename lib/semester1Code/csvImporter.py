'''
    This module will read a csv file and return the data inside of the file
'''
import csv
import re
from tqdm import tqdm
import os

filenames = ['./data/WindGenTotalLoadYTD_2011.xls']

def readCsv(filename,debug = False):
    entries = []

    with open(filename, 'r') as csvfile:
        # Open up the file and read it
        reader = csv.reader(csvfile, skipinitialspace=True)
        numEntries = sum(1 for line in open(filename))

        # Create a progress bar
        bar = tqdm(total=numEntries, desc="Reading the CSV file")

        # Loop through CSV and add classes to entries
        count = 0
        for row in reader:
            try:
                ent = entry()
                ent.Date, ent.Time = parseDateTime(row[0])
                # Parse the time and date
                if ent.Date == None or ent.Time == None:
                    continue
                # Add the wind speed to the entry

                ent.WindGenBPAControl = float(row[2])
            except ValueError:
                continue
            entries.append(ent)
            bar.update(1)
            if(debug and count>1000):
                break
            count+=1

    return entries

def readData():
    data = []
    for i,j,k in os.walk("./data"):
        print("Reading a total of "+str(len(k)) + " files")
        for file in k:
            if file[-3:] == "csv":
                d = readCsv("./data/"+file,False)
                for entry in d:
                    data.append(entry)

    return data

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
    readData()


if __name__ == '__main__':
    main()
