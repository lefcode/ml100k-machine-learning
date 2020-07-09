import numpy as np
import pandas as pd

def pre_process(file):#'u.data'
    #import all values in a matrix, the missing values are Nans
    full_mat = np.empty((944,1683)) #table for values
    full_mat[:] = np.nan
    data = pd.read_csv(file, sep='\t',names=['user_id', 'movie_id', 'rating', 'timestamp'])
    data.drop("timestamp", inplace=True, axis=1)
    for i in range(1,len(data)):
        full_mat[int(data.loc[i][0])][int(data.loc[i][1])]=int(data.loc[i][2]) #import values from u.daa
    
    if file=="/data/ml-100k/u.data":
      #for local installation remove the cooment of the next 3 lines
      #import os
      #path =os.getcwd()
      #pd.DataFrame(full_mat).to_csv(path+'\\data\\data.csv') 
      pd.DataFrame(full_mat).to_csv("C:\\Users\\lefteris\\.spyder-py3\\Genetic_Algorithm\\data\\data.csv")
    elif file=='/data/ml-100k/ua.base':
       #for local installation remove the cooment of the next 3 lines
       #import os
       #path =os.getcwd()
       #pd.DataFrame(full_mat).to_csv(path+'\\data\\data_base.csv')
       pd.DataFrame(full_mat).to_csv("C:\\Users\\lefteris\\.spyder-py3\\Genetic_Algorithm\\data\\data_base.csv")

pre_process('/data/ml-100k/u.data')
pre_process('/data/ml-100k/ua.base')