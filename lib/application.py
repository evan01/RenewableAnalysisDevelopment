import time

import matplotlib.pyplot as plt
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
import pandas as pd



def main():
    '''
    This is the main starting point for the statGen application
    :return:
    '''

    data = csvImporter.readData() #First read the data
    data = r.getFirstDerivative() #Then calculate the ramp data

    #With this ramp data we need to find a parametric curve that describes the data's fit

if __name__ == '__main__':
    main()