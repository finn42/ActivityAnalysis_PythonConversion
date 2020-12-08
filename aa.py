# function file for activity analysis 

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def activityCount(Data,FrameSize,HopSize,Thresh,actType):
    ''' def activityCount(Data,FrameSize,HopSize,Thresh,actType)
    (Time,Series,FrameSize,HopSize,Thresh,option)

     function to identify the occurances and popularity of activity 
     events (specified by 'option' and 'Thresh') across the columns
     of Data time series over frames of size FrameSize and 
     intervals of Hopsize. Columns are treated as signals to be 
     translated into point processes.
     
     Inputs: Data - an pandas dataframe with index of time in the same units
                    as FrameSize and Hopsize and columns of signal measurements
             FrameSize - the interval of signal actions are to be evaluated over
             HopSize - the spacing between these intervals of signal (can be smaller than FrameSize)
             Thresh - a minimum value (scalar) for events measured in frames
             actType - a string specifying the type of signal event to detect
                     one of {'Inc','Dec','Change'} (so far)
     Outputs: Acts - a pandas dataframe with index 'Time' of time points (centres of frames)
                     columns for every column of signal in Data (carries names)
                     final column ['Total'] that reports the ratio of signals active per frame
     Version 5.0, first in Python 3
     Finn Upham 2020 12 07
     '''

    Time = Data.index
    cols = Data.columns
    
    newTime = np.arange(FrameSize/2+Time[0],Time[-1],HopSize)
    Acts = pd.DataFrame()
    Acts['Time']=newTime
    Acts=Acts.set_index('Time')

    # for increases
    if actType == 'Inc':
        for col in cols:
            f = interp1d(Time,Data[col], kind='nearest',fill_value='extrapolate')
            a=f(newTime+FrameSize/2)-f(newTime-FrameSize/2)
            a[a>=Thresh] = 1
            a[a<Thresh] = 0
            Acts[col]=a
    if actType == 'Dec':
        for col in cols:
            f = interp1d(Time,Data[col], kind='nearest',fill_value='extrapolate')
            a=f(newTime+FrameSize/2)-f(newTime-FrameSize/2)
            a[a>-Thresh] = 0
            a[a<=-Thresh] = 1
            Acts[col]=a 
    if actType == 'Change':
        for col in cols:
            f = interp1d(Time,Data[col], kind='nearest',fill_value='extrapolate')
            a=np.abs(f(newTime+FrameSize/2)-f(newTime-FrameSize/2))
            a[a>=Thresh] = 1
            a[a<Thresh] = 0
            Acts[col]=a    

    Acts['Total'] = Acts.sum(1)/len(cols) # ratio of signals active per frame
    return Acts


