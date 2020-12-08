# function file for activity analysis 

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binom, poisson, chisquare

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

def equisplit(counts,nBins):
    ''' function to split discrete distribution into maximally-even sized nBins.
    If the distribution does not split cleanly (bin with less than 5 samples), 
    repeat evaluation with one less bin, to a min of 3 bins.
    
     Input: Counts - a panda series of counts (model), indexed sequentially. 
     Output: Index lists of cuts for bins 
     
     Function used by simpleActivityTest
 	'''
    minim = 5
    n = nBins

    cumV = counts.cumsum()
    eTot = cumV.iloc[-1]
    
    cuts = [counts.index[0]]
    for i in range(1,n):
        crit = (cumV-eTot*(i/n))
        crit[crit<0] = eTot
        #crit = abs((cumV-cumV.iloc[-1]*(i/n)))
        cuts.append(crit.idxmin())
    cuts.append(counts.index[-1])
    
    dists = []
    for i in range(len(cuts)-1):
        dists.append(counts.loc[cuts[i]:cuts[i+1]-1].sum())

    if np.min(dists)<5:
        print('Too many bins')
        if nBins>2:
            cuts = equisplit(counts,nBins-1)
        else:
            print('Not enough samples to cut for GoF test.')
    return cuts

def simpleActivityTest(A,Np,Nbins):
    ''' function to run a goodness of fit test on a random activity model
    against the measure distrubution of activity levels

    stest = {'Chi2':st,'pvalue':p,'Counts':Counts,'Bins':dists}
    '''
    ac = np.round(A.values*Np) # get these back to count values, integers
    meas = pd.Series(ac).value_counts()
    L=len(ac)
    aL = np.arange(Np)
    Counts = pd.DataFrame(index = aL)
    Counts['Measured'] = 0
    Counts.loc[meas.index,'Measured'] = meas.values
    if Np <100:
        Counts['Model'] = binom.pmf(aL, Np, A.sum()/L)*L
    else:
        Counts['Model'] = poisson.pmf(aL, Np*A.sum()/L)*L

    cuts = equisplit(Counts['Model'],Nbins)    
    dists = pd.DataFrame()
    for i in range(len(cuts)-1):
        dists = dists.append(Counts.loc[cuts[i]:cuts[i+1]-1,:].sum(), ignore_index=True)

    st,p = chisquare(dists['Measured'], f_exp=dists['Model'])

    stest = {'Chi2':st,'pvalue':p,'Counts':Counts,'Bins':dists}
    
    return stest


def score_C(pvals):
    # pval must be an np.array
    C =  -np.log10(pvals+10.0**(-16)).mean() 
    return C

def coordScoreSimple(Data,FrameSize,Thresh,actType,Nbins):
    t = pd.Series(Data.index)
    sF = 1/t.diff().median()
    winSize = int(FrameSize/sF)
    N = Data.shape
    CS=[]
    for i in range(winSize):
        Acts = activityCount(Data.loc[i:],FrameSize,FrameSize,Thresh,actType)
        stest = simpleActivityTest(Acts['Total'],N[1],Nbins)
        CS.append(stest['pvalue'])
    C=score_C(np.array(CS))
    return C

