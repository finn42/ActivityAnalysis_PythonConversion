# function file for activity analysis 

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import binom, poisson, chisquare,chi2

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

def equisplit(counts,nBins,minim = 5):
    ''' function to split discrete distribution into maximally-even sized nBins.
    If the distribution does not split cleanly (bin with less than 5 samples), 
    repeat evaluation with one less bin, to a min of 3 bins.
    
     Input: Counts - a panda series of counts (model), indexed sequentially. 
     Output: Index lists of cuts for bins 
     
     Function used by simpleActivityTest
 	'''
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

    if np.min(dists)<minim:
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

# [Chi,p,DAct,Bins,v1,v2] = alternatingActivitiesTest(AllC1,AllC2,k)
def alternatingActivitiesTest(Acts1,Acts2,nBins = 3):
    
    N = Acts1.shape
    Np = N[1]-1
    L = N[0]
    AC1 =  Acts1['Total']*Np
    Acts1 = Acts1.drop(columns=['Total'])
    AC2 =  Acts2['Total']*Np
    Acts2 = Acts2.drop(columns=['Total'])

    if L<50:
        print('Too few frames for analysis.')
        return
    if (Acts1+Acts2).max().max()>1:
        print('These forms of activity are not exclusive, use the biact function instead.')
        return

    Model = pd.DataFrame()
    altModel = pd.DataFrame()

    aL = np.arange(Np)
    Counts = pd.DataFrame(index = aL)
    meas = pd.Series(AC1).value_counts()
    Counts['Measured1'] = 0
    Counts.loc[meas.index,'Measured1'] = meas.values
    meas = pd.Series(AC2).value_counts()
    Counts['Measured2'] = 0
    Counts.loc[meas.index,'Measured2'] = meas.values

    Measured = pd.DataFrame(index = aL)
    for i in range(Np):
        Measured[i] = np.zeros(Np)
        if Counts.loc[i,'Measured2']>0:
            sub=AC1.loc[AC2==i].value_counts()
            Measured.loc[sub.index,i] = sub.values

    Model = pd.DataFrame(index = aL)
    pM2 = Counts['Measured2']/L
    for i in range(Np):
        Model[i] = pM2*Counts.loc[i,'Measured1']

    cuts1 = equisplit(Counts['Measured1'],nBins ,nBins*6);
    cuts2 = equisplit(Counts['Measured2'],nBins ,nBins*6);

    tabMod = pd.DataFrame()
    tabMea = pd.DataFrame()
    for j in range(len(cuts2)-1):
        a = []
        b = []
        for i in range(len(cuts1)-1):   
            A = Model.loc[cuts1[i]:cuts1[i+1]-1,cuts2[j]:cuts2[j+1]-1]
            a.append(A.sum().sum())
            B = Measured.loc[cuts1[i]:cuts1[i+1]-1,cuts2[j]:cuts2[j+1]-1]
            b.append(B.sum().sum())
        tabMod[j]=np.array(a)
        tabMea[j]=np.array(b)

    st = ((tabMod-tabMea)**2/tabMod).sum().sum()
    p = 1-chi2.cdf(st, sum(tabMod.shape)-2)

    stest = {'Chi2':st,'pvalue':p,'Model':Model,'Measured':Measured,'BinsModel':tabMod,'BinsMeasured':tabMea}
    return stest

def coordScoreAlternating(Data,FrameSize,Thresh1,actType1,Thresh2,actType2,Nbins=3):
	t = pd.Series(Data.index)
	sF = 1/t.diff().median()
	winSize = int(FrameSize/sF)
	N = Data.shape
	CS=[]
	for i in range(winSize):
	    Acts1 = activityCount(Data.loc[i:],FrameSize,FrameSize,Thresh1,actType1)
	    Acts2 = activityCount(Data.loc[i:],FrameSize,FrameSize,Thresh2,actType2)
	    stest = alternatingActivitiesTest(Acts1,Acts2,nBins = Nbins)
	    CS.append(stest['pvalue'])
	C=score_C(np.array(CS))
	return C