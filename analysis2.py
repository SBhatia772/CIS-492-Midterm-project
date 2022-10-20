# Last modified October 17, 2022
# Purpose: Data analysis of input training data.
# Want to see which features are most inter-dependent.

from cmath import exp
import csv
import pandas as pd
import numpy as nm
import matplotlib.pyplot as plot
import itertools as it
import os

def main():
    filestr = 'dataset/dataset/train/training-data.csv'
    outfile = 'outfile.csv'
    with open(filestr, 'r+', newline='') as data:
        #datasheet is a pandas dataframe of the singular input CSV:
        datasheet = pd.read_csv(data)
        outdf = pd.DataFrame(columns = datasheet.columns, index=datasheet.columns)
        # Loop through each column:
        for item in datasheet.items():
            # Loop through each column again to compare:
            for item2 in datasheet.items():
                if (item == item2):
                    continue
                # Find the linear correlation between the two columns:
                #concatstr = (item[0] + " X " + str(item2[0]) + ": " + str(item[1].corr(item2[1])) + "\n\n")
                #outstr += concatstr
                # More sophisticated: Use the dataframe location here:
                outdf.loc[item[0],item2[0]] = item[1].corr(item2[1])
                

                
        print(outdf)

    with open(outfile, 'w+', newline='') as outf:
        outdf.to_csv(path_or_buf = outf)
        
       
main()
