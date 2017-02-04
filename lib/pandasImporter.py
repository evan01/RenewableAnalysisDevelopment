'''
This file will enable users to import csv time series data as pandas data structures
'''

import numpy as np
import pandas as pd
import glob
import os
import tqdm as tqdm

defaultPath = "./data/"
def importCsv(path=None):
    if path is None:
        #Then read default path, will read all files in data directory
        # data = pd.read_csv(defaultPath,header=0,skip_blank_lines=True,infer_datetime_format=True)
        allCSVFiles = glob.glob(os.path.join(defaultPath,"*.csv"))
        df_from_file = (pd.read_csv(f,parse_dates=['Date/Time']) for f in allCSVFiles)
        df_from_file
        data = pd.concat(df_from_file,ignore_index=False)
        print("done")
    else:
        print("read a single path")

if __name__ == '__main__':
    importCsv()