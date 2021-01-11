# function file for activity analysis 

import pandas as pd
import numpy as np
import math
import scipy as sc 
# from scipy.interpolate import interp1d
# from scipy.stats import binom, poisson, chisquare,chi2

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
                     one of {'Inc','Dec','Change','UBound','LBound'} (so far)
     Outputs: Acts - a pandas dataframe with index 'Time' of time points (centres of frames)
                     columns for every column of signal in Data (carries names)
                     final column ['Total'] that reports the ratio of signals active per frame
     Version 5.0, first in Python 3
     Finn Upham 2020 12 07
     '''

    Time = Data.index
    cols = Data.columns
    
    newTime = np.arange(FrameSize/2+Time[0],Time[-1],HopSize)
    Acts = pd.DataFrame(columns = cols)


    # for increases
    if actType == 'Inc':
        Acts['Time']=newTime
        Acts=Acts.set_index('Time')
        for col in cols:
            f = sc.interpolate.interp1d(Time,Data[col], kind='nearest',fill_value='extrapolate')
            a=f(newTime+FrameSize/2)-f(newTime-FrameSize/2)
            a[a>=Thresh] = 1
            a[a<Thresh] = 0
            Acts[col]=a
    if actType == 'Dec':
        Acts['Time']=newTime
        Acts=Acts.set_index('Time')
        for col in cols:
            f = sc.interpolate.interp1d(Time,Data[col], kind='nearest',fill_value='extrapolate')
            a=f(newTime+FrameSize/2)-f(newTime-FrameSize/2)
            a[a>-Thresh] = 0
            a[a<=-Thresh] = 1
            Acts[col]=a 
    if actType == 'Change':
        Acts['Time']=newTime
        Acts=Acts.set_index('Time')
        for col in cols:
            f = sc.interpolate.interp1d(Time,Data[col], kind='nearest',fill_value='extrapolate')
            a=np.abs(f(newTime+FrameSize/2)-f(newTime-FrameSize/2))
            a[a>=Thresh] = 1
            a[a<Thresh] = 0
            Acts[col]=a  
            
    if actType == 'UBound':
        # prime the Acts dataframe to have teh same columns as Data
        # interate over rows of 
        for i in range(len(newTime)):
            t = newTime[i]
            frame = Data.loc[t-(FrameSize*0.5):t+(FrameSize*0.5),:].copy()
            UB = frame.max(0)
            UB[UB>=Thresh] = Thresh
            UB[UB<Thresh] = 0
            UB[UB==Thresh] = 1
            df2 = pd.DataFrame(columns = cols)
            df2.loc[t] = UB
            Acts = Acts.append(df2)
    
    if actType == 'LBound':
        # prime the Acts dataframe to have teh same columns as Data
        # interate over rows of 
        for i in range(len(newTime)):
            t = newTime[i]
            frame = Data.loc[t-(FrameSize*0.5):t+(FrameSize*0.5),:].copy()
            UB = frame.min(0)
            UB[UB<=Thresh] = Thresh
            UB[UB>Thresh] = 0
            UB[UB==Thresh] = 1
            df2 = pd.DataFrame(columns = cols)
            df2.loc[t] = UB
            Acts =Acts.append(df2)

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
        Counts['Model'] = sc.stats.binom.pmf(aL, Np, A.sum()/L)*L
    else:
        Counts['Model'] = sc.stats.poisson.pmf(aL, Np*A.sum()/L)*L

    cuts = equisplit(Counts['Model'],Nbins)    
    dists = pd.DataFrame()
    for i in range(len(cuts)-1):
        dists = dists.append(Counts.loc[cuts[i]:cuts[i+1]-1,:].sum(), ignore_index=True)

    st,p = sc.stats.chisquare(dists['Measured'], f_exp=dists['Model'])

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


def alternatingActivitiesTest(Acts1,Acts2,nBins = 3): 
    N = Acts1.shape
    Np = N[1]-1
    L = N[0]
    AC1 =  np.round(Acts1['Total']*Np)
    Acts1 = Acts1.drop(columns=['Total'])
    AC2 =  np.round(Acts2['Total']*Np)
    Acts2 = Acts2.drop(columns=['Total'])

    if L<50:
        print('Too few frames for analysis.')
        #return
    if (Acts1+Acts2).max().max()>1:
        print('These forms of activity are not exclusive, use the biact function instead.')
        #return

    # two criteria for evaluating distribution: Total activity and ratio of one or the other
    ACall =  AC1 + AC2
    p = ACall.sum()/(Np*L) # probability of any activity
    p1 = AC1.sum()/(Np*L) 
    p2 = AC2.sum()/(Np*L)
    p12 = p2/p

    aL = np.arange(Np+1)
    MeasuredAll = pd.DataFrame(0,index = aL,columns = ['Measured'])
    meas = pd.Series(ACall).value_counts()
    MeasuredAll.loc[meas.index,'Measured'] = meas.values

    Model = pd.DataFrame(index = aL)
    for s in range(Np+1):
        subcount = pd.Series(0.0,index = aL)
        for r in range(s+1):
            if s == 0:
                idx = int(0.5*Np)
            else:
                idx = int(Np*(r)/(s+1))
            if s <= 50:
                subcount[idx] = math.comb(s,r)*((1-p12)**(s-r))*(p12**r)
            else:
                subcount[idx]=math.exp(-s*p12)*((s*p12)**r)/math.factorial(r)
        Model[s] = subcount*MeasuredAll.loc[s].values
    Independent = Model.sum(1)

    Measured = pd.DataFrame(index = aL)
    for s in range(Np+1):
        sub=AC2[ACall==s].value_counts()
        subcount = pd.Series(0.0,index = aL)
        if len(sub)>0:
            subcount = pd.Series(0.0,index = aL)
            for r in sub.index:
                if s == 0:
                    idx = int(0.5*Np)
                else:
                    idx = int(Np*(r)/(s+1))
                subcount[idx] = sub.loc[r]
        Measured[s] = subcount

    cuts1 = equisplit(Independent,nBins ,nBins*6);
    cuts2 = equisplit(MeasuredAll['Measured'],nBins ,nBins*6);

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
    p = 1-sc.stats.chi2.cdf(st, sum(tabMod.shape)-2)

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


# [Chi,p,DAct,Bins,v1,v2] = alternatingActivitiesTest(AllC1,AllC2,k)
def relatedActivitiesTest(Acts1,Acts2,nBins = 3):
    
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
    p = 1-sc.stats.chi2.cdf(st, sum(tabMod.shape)-2)

    stest = {'Chi2':st,'pvalue':p,'Model':Model,'Measured':Measured,'BinsModel':tabMod,'BinsMeasured':tabMea}
    return stest

def coordScoreRelated(Data1,Data2,FrameSize,Thresh1,actType1,Thresh2,actType2,Nbins=3):
	t = pd.Series(Data1.index)
	sF = 1/t.diff().median()
	winSize = int(FrameSize/sF)
	CS=[]
	for i in range(winSize):
	    Acts1 = activityCount(Data1.loc[i:],FrameSize,FrameSize,Thresh1,actType1)
	    Acts2 = activityCount(Data2.loc[i:],FrameSize,FrameSize,Thresh2,actType2)
	    stest = relatedActivitiesTest(Acts1,Acts2,nBins = Nbins)
	    CS.append(stest['pvalue'])
	C=score_C(np.array(CS))
	return C

def framedSum(AC,FrameSize):
    # AC is the output of actionCount, 
    # framesize is in the units of the AC index (time, s)
    # output is a counting of activity over frames of duration FrameSize. Basically convolution. 
    if 'Total' in AC.columns:
        AllC = AC.drop(columns=['Total'])
    else:
        AllC = AC.copy()
    sF=np.round(1/pd.Series(AllC.index).diff().median())
    frameN = int(FrameSize*sF)
    AllC = AllC.append(pd.DataFrame(0,index=AllC.index[-1]+((1+np.arange(frameN))/sF), columns=AllC.columns))
    AllC = AllC.append(pd.DataFrame(0,index=AllC.index[0]-((1+np.arange(frameN))/sF), columns=AllC.columns))
    AllC = AllC.sort_index()
    AllBlur = AllC.copy()
    for i in range(frameN-1):
        AllBlur += AllC.shift(i+1)
    Framed = AllBlur.loc[AC.index]
    Framed[Framed>0] = 1
    V = Framed.sum(1)#/len(AllC.columns)
    return V

def localActivityTest(AllC,FrameSize,ShuffleRange,Iter):
    # function to evaluate nonparametricly the distribution of coincidences of events in AllC
    # measured in frames FrameSize, with alignment shuffling range of ShufflrRange over Iter iterations
    # TODO evaluate distributions on local rank rather than absolute counts
    #function uses framedSum and score_c and AllC is optimally set up by actionCount, with Total removed

    sF=np.round(1/pd.Series(AllC.index).diff().median()) # assumes sample rate is >=1 Hz
    S = AllC.shape
    Np = S[1] # number of responses in AllC
    L = S[0] # number of samples in which activity has been assessed
    ShuffS = ShuffleRange*sF # the number of samples overwhich shuffling is performed
    Results = pd.DataFrame(index = AllC.index)
    Results['Activity-levels'] = framedSum(AllC,FrameSize) 

    # generate alternative distributions with shuffle range
    AlternativeCoincs = pd.DataFrame(0, index=AllC.index, columns=np.arange(Iter))
    for i in range(Iter):
        shifts = np.round(np.random.random_sample(Np)*ShuffS - ShuffS/2)
        # option = Loop
        SAllC = pd.DataFrame(index=AllC.index)
        for rn in range(S[1]):
            offs = int(shifts[rn])
            A = AllC[rn].shift(offs)
            if offs<0:
                A.iloc[offs:] = AllC[rn].iloc[:-offs].values
            else:
                if offs>0:
                    A.iloc[:offs] = AllC[rn].iloc[-offs:].values
                else:
                    A = AllC[rn]
            SAllC[rn]=A
        AlternativeCoincs[i] = framedSum(SAllC,FrameSize) 

    # now evaluate distributions over complete excerpt
    # TODO: trim the ends by half a shuffle range before evaluating statistics
    aL = np.arange(Np+1)
    TrueCounts = pd.DataFrame(0,index = aL,columns = ['Measured'])
    meas = pd.Series(Results['Activity-levels']).value_counts()
    TrueCounts.loc[meas.index,'Measured'] = meas.values/L

    AltCounts = pd.DataFrame(0,index = aL,columns=np.arange(Iter))
    for i in range(Iter): 
        meas = pd.Series(AlternativeCoincs[i]).value_counts()
        AltCounts.loc[meas.index,i] = meas.values/L

    # empDist
    cdistf = TrueCounts.cumsum()
    ACdistf = AltCounts.cumsum()
    meanAltCDistf = ACdistf.mean(1)
    distanceTrueMean = (cdistf.add(-meanAltCDistf,0)**2).sum()**(0.5)
    distanceMAlt = (ACdistf.add(-meanAltCDistf,0)**2).sum()**(0.5)
    p = len(distanceMAlt[distanceMAlt>distanceTrueMean.values[0]])/Iter
    CS = score_C(p)

    #capture distibutions per frame for local activity stats
    AltFrameCounts = pd.DataFrame(0,index=AlternativeCoincs.index,columns=aL)
    Results['Local_p']= 0.5
    for t in AlternativeCoincs.index:
        alt_lvls = AlternativeCoincs.loc[t].values
        actlvl_Counts = pd.DataFrame(0,index = aL,columns=[t])
        meas = pd.Series(alt_lvls).value_counts()
        actlvl_Counts.loc[meas.index,t] = meas.values
        AltFrameCounts.loc[t] = actlvl_Counts.transpose().values.cumsum()/Iter
        Results.loc[t,'Local_p'] = AltFrameCounts.loc[t,Results.loc[t,'Activity-levels']]
    AltFrameCounts = AltFrameCounts

    pVal = Results['Local_p']
    pVal[pVal<10.0**(-5)]= 10.0**(-5)
    surp = np.log10((1-pVal)/pVal)
    surp[surp>3] = 3
    surp[surp<-3] = -3
    Results['Surprise'] = surp
        
#    [pVal,Coinc,CoincRank,CoincSurprise,AlternativeCoincs,AltP,altpVal,NPCscore,altNPCscore]
    stest = {'pvalue':p,'MeasuredResults':Results,'Models':AlternativeCoincs,'CoordScore':CS}
    return stest

def activityLevelRanks(AC):
    # take in activity level time series and convert to rank values from cdf
    Np = AC.max()
    aL = np.arange(Np+1)
    TrueCounts = pd.DataFrame(0,index = aL,columns = ['Measured'])
    meas = pd.Series(AC).value_counts()
    TrueCounts.loc[meas.index,'Measured'] = meas.values/L
    TrueCounts['cdf'] = TrueCounts.cumsum()

    T = AC.copy()
    for index, row in TrueCounts.iterrows():
        T[T==index] = row['cdf']
    return T

def framedSum(AC,FrameSize):
    # AC is the output of actionCount, 
    # framesize is in the units of the AC index (time, s)
    # output is a counting of activity over frames of duration FrameSize. Basically convolution. 
    if 'Total' in AC.columns:
        AllC = AC.drop(columns=['Total'])
    else:
        AllC = AC.copy()
    sF=np.round(1/pd.Series(AllC.index).diff().median())
    frameN = int(FrameSize*sF)
    AllC = AllC.append(pd.DataFrame(0,index=AllC.index[-1]+((1+np.arange(frameN))/sF), columns=AllC.columns))
    AllC = AllC.append(pd.DataFrame(0,index=AllC.index[0]-((1+np.arange(frameN))/sF), columns=AllC.columns))
    AllC = AllC.sort_index()
    AllBlur = AllC.copy()
    for i in range(frameN-1):
        AllBlur += AllC.shift(i+1)
    Framed = AllBlur.loc[AC.index]
    Framed[Framed>0] = 1
    V = Framed.sum(1)/len(AllC.columns)
    return V

def Uniprob_Shuffle(AC,ShuffleRange,FrameSize):
    # AC is the output of actionCount, point process of action per response time series 
    # framesize is in the units of the AC index (time, s)
    # ShuffleRange is in the units of the AC index (time, s)
    # output is a counting of activity over frames of duration FrameSize. Basically convolution. 
    if 'Total' in AC.columns:
        AllC = AC.drop(columns=['Total'])
    else:
        AllC = AC.copy()
    sF=np.round(1/pd.Series(AllC.index).diff().median())
    frameN = int(FrameSize*sF)
    ShuffleN = int(ShuffleRange*sF)
    AllC = AllC.append(pd.DataFrame(0,index=AllC.index[-1]+((1+np.arange(ShuffleN))/sF), columns=AllC.columns))
    AllC = AllC.append(pd.DataFrame(0,index=AllC.index[0]-((1+np.arange(ShuffleN))/sF), columns=AllC.columns))
    AllC = AllC.sort_index()
    AllBlur = AllC.copy()
    for i in range(ShuffleN-1):
        AllBlur += AllC.shift(i+1)
    Uni_Spread = AllBlur.loc[AC.index]*frameN/ShuffleN
#     lastN = len(Uni_Spread)
#     HalfShuffN = int(ShuffleN/2) 
#     for i in range(HalfShuffN):
#         Uni_Spread.iloc[i] = Uni_Spread.iloc[i]*ShuffleN/(i+HalfShuffN)
#         Uni_Spread.iloc[lastN-i] = Uni_Spread.iloc[lastN-i]*ShuffleN/(i+HalfShuffN)
    Uni_Spread[Uni_Spread>1] = 1
    
    return Uni_Spread

def Local_Act_Dist_nk(p):
    # p is an ordered list of independent probabilities for a given activity
    # produce dataframe of the cumulative distribution of at least N elements of p being active
    # reduce the list number of unique values
    unique, counts = np.unique(p, return_counts=True)
    combos = dict(zip(unique, counts))
    
    C = pd.DataFrame(columns = ['Count','Probability'])
    C = C.append({'Count':len(p),'Probability': np.prod(p)},  ignore_index=True)
    for i in range(len(combos)):
        q = 1-unique[i]
        K = len(C)
        for k in range(1,counts[i]+1):
            for j in range(K):
                prob = C['Probability'].iloc[j]*((q/unique[i])**k)*math.comb(counts[i],k)
                C=C.append({'Count':C['Count'].iloc[j]-k,'Probability':prob},  ignore_index=True)
    #print(C['Probability'].sum())

    C = C.sort_values(by=['Count']).reset_index()
    C['cdf'] = C['Probability'].cumsum()
    D = C['Count'].diff().shift(-1)
    D.iloc[len(D)-1] = 1
    #print(C)

    ActLvl=C.loc[D==1,['Count','cdf']].copy().set_index('Count')
    return ActLvl

def Analytic_local_cdfs(AC,ShuffleRange,FrameSize):
    Uni_Spread = Uniprob_Shuffle(AC,ShuffleRange,FrameSize)
    Ordered_Spread = pd.DataFrame(np.sort(Uni_Spread.values, axis=1), index=Uni_Spread.index, columns=Uni_Spread.columns)
    PastDists = {}
    tic = time.time()
    V = Ordered_Spread.copy()
    aL = np.arange(len(Uni_Spread.columns)+1)
    AltFrameCounts = pd.DataFrame(0,index=V.index,columns=aL)

    for t,row in V.iterrows():
        r = row.copy()
        MinLvl = np.sum(r==1)
        r[r==1]=0
        p = r[r>0]
        p_name = str(p.values)
        if p_name in PastDists:
            ActLvl = PastDists[p_name].copy()
        else:
            ActLvl=Local_Act_Dist_nk(p)
            PastDists[p_name]=ActLvl.copy()
        ActLvl.index = ActLvl.index+int(MinLvl)
        actlvl_dist = pd.DataFrame(0,index = aL,columns=[t])
        actlvl_dist.loc[ActLvl.index,t] = ActLvl.values
        actlvl_dist.loc[ActLvl.index[-1]:,t] = 1.0
        AltFrameCounts.loc[t] = actlvl_dist.T.loc[t]
    print('total eval time:' + str(time.time()-tic))        
    print('dictionary of distributions:' + str(len(PastDists)))
    return AltFrameCounts