# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:07:13 2020

@author: Stefano
"""
from Higgs_functions import *

import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib as mat
import matplotlib.pyplot as plt

#%%
"""
1. GENERATE A SIMULATED DATA-SET AND MAKE AND PLOT A HISTOGRAM
"""
vals = generate_data() # A python list. 
 # Each list entry represents the rest mass reconstructed from a collision.
bin_heights, bin_edges, patches = plt.hist(vals, range = [104, 155], bins = 30)

dm=bin_edges[2]-bin_edges[1] #the width of each bin
bin_centres=bin_edges[:(len(bin_edges)-1)]+dm/2 
plt.errorbar(bin_centres,bin_heights, yerr=sp.sqrt(bin_heights),ls="None",
             fmt="|r",barsabove=True, label="statistical uncertainty")
plt.vlines(119.3, 0, bin_heights[8], ls="dashed", color="Orange")
plt.vlines(131.2, 0, bin_heights [16], ls="dashed", color="Orange")
plt.legend()
plt.xlabel("Mass (GeV)")
plt.ylabel("Number of Events")
plt.show()
"""
DEFINE USEFUL QUANTITIES:
    the number NB of bins below 120 and their upper limit LB
    the number NA of bins above 130 and their lower limit LA
    the sides arrays: the union of the below and above arrays 
"""
NB=(120-104)//dm #the number of bins below 120
LB=104+NB*dm  #the edge of the last bin before 120
NA=(155-130)//dm #the number of bins above 130
LA=155-NA*dm  #the edge of the first bin after 130
bin_centres_sides=np.append(bin_centres[:int(NB)],bin_centres[(int(NA)+1):])
bin_heights_sides=np.append(bin_heights[:int(NB)],bin_heights[(int(NA)+1):])
#%%
"""
2. BACKGROUND PARAMETERISATION: 
    It takes in consideration not only the masses below 120 GeV, but also the 
    ones above 130 GeV.
    First define the exponential function (with lambda l and the scaling factor
    A independent from each other)
    Then define the guess vaues based on the given heights and some theoretical
    considerations.
    Finally fit the curve and plot it compared to the histogram
    As we are able of finding very good guesses starting from theoretical 
    considerations, the 2D looping is entrusted to scipy.optimize.curve_fit.
"""
#DEFINE THE EXPONENTIAL
def expo (x,A,l):
    return A*sp.exp(-x/l)

#ANALYTICALLY FOUND GOOD ESTIMATIONS FOR LAMBDA (l) AND FOR THE AMPLITUDE (A)
Num=bin_centres_sides[-1]-bin_centres_sides[0] 
Den=sp.log(bin_heights_sides[0]/bin_heights_sides[-1]) 
l0=Num/Den #the estimate for lambda: the ratio N/D 

#equaling the area of the bins and the integral
A0=(1/l0)*dm*sum(bin_heights_sides)/(sp.exp(-104/l0)-sp.exp(-LB/l0)+\
                                     sp.exp(-LA/l0)-sp.exp(-155/l0)) 

#FIT USING CURVE_FIT
fit_param, fit_cov=curve_fit(expo,bin_centres_sides,bin_heights_sides,
                             p0=(A0,l0),sigma=sp.sqrt(bin_heights_sides))
A=fit_param[0]
l=fit_param[1]

#PLOT
x=sp.linspace(104,155,200)
plt.plot(x,expo(x,A,l),c="Red", label="Background-only Hypothesis")
plt.errorbar(bin_centres,bin_heights,yerr=sp.sqrt(bin_heights), xerr=dm/2,
             ls="None",fmt="b",barsabove=True, label="Observed Data")
plt.xlabel("masses (GeV)")
plt.ylabel("Number of Events")
plt.legend()
plt.show
#%%
"""
3. GOODNESS OF FIT:
    find the Chi Square (Chi2), the reduced Chi2 and the p_value 
    without taking into considerations the hypothethical Higgs peak region yet
"""
expected=expo((bin_centres_sides),A,l) #the background-only expectation
ddof=(len(bin_centres_sides)-len(fit_param)) #the degrees of freedom
GoodnessSides=stats.chisquare(bin_heights_sides,expected,ddof=ddof) #a size2 array
Chi2Exp=GoodnessSides[0] #the chi2
Chi2ExpReduced=Chi2Exp/ddof #the reduced chi2
p_value=stats.chi2.sf(Chi2Exp,ddof) #the p_value
#%%
"""
4. HYPOTHESIS TESTING: CAN WE SAY THE BACKGROUND IS SUFFICIENT?
   PART A:
    Now calculate the Chi2, the reduced Chi2 and the p_value also including the
    Higgs Boson peak region, not only the ones at the side.
"""
expected=expo(bin_centres,A,l)
ddof=(len(bin_centres)-len(fit_param))

GoodnessH=stats.chisquare(bin_heights,expected,ddof=ddof)
Chi2H=GoodnessH[0]
Chi2HReduced=Chi2H/ddof
p_value=stats.chi2.sf(Chi2H,ddof)
print("The Background-only Hypothesis Reduced Chi Square is "+ \
      str(round(Chi2HReduced,7)) +"\nThe Background-only Hypothesis p-value is "\
          + str(round(p_value,10)))
#%%
"""
4. PART B: ANLYSIS OF THE CHI2 PDF RANDOM FLUCTUATIONS:
    Execute a loop which, for different random datasets, calculates the best 
    background-only fit, together with the Chi2, the reduced Chi2 and the 
    p_value. As doing so it stores such results in arrays.
    It generates 0 Higgs Events and 10^5 background events and iterates N times
    (1000 iterations on my laptop take approximately 3 minutes)
"""
CHI2_REDUCED=[]
p_list=[]
np.random.seed(1)
NH=0
N=10000 #number of iterations

#DEFINE THE EXPONENTIAL
def expo (x,A,l):
    return A*sp.exp(-x/l)
    
for i in range(0,N): #begin the loop
#GENERATE DATA
    vals = generate_data(NH)
    
#CREATE THE HISTOGRAM    
    bin_heights, bin_edges, patches = plt.hist(vals, range = [104, 155], 
                                               bins = 30)
    plt.clf()#avoids the histograms being plotted
    
#DEFINE USEFUL QUANTITIES
    dm=dm=bin_edges[2]-bin_edges[1] #the width of each bin
    NB=(120-104)//dm #the number of bins below 120
    LB=104+NB*dm  #the edge of the last bin before 120
    NA=(155-130)//dm #the number of bins above 130
    LA=155-NA*dm  #the edge of the first bin after 130
    bin_centres=bin_edges[:(len(bin_edges)-1)]+dm/2
    
#FITTING PARAMETERS FOR BACKGROUND HYPOTHESIS ONLY
    Num=bin_centres[-1]-bin_centres[0] 
    Den=sp.log(bin_heights[0]/bin_heights[-1]) 
    l0=Num/Den #the estimate for lambda: the ratio N/D for each pair of bins
    A0=(1/l0)*dm*sum(bin_heights)/(sp.exp(-104/l0)-sp.exp(-LB/l0)+\
                                   sp.exp(-LA/l0)-sp.exp(-155/l0)) 
    fit_param, fit_cov=curve_fit(expo,bin_centres,bin_heights,p0=(A0,l0))
    A=fit_param[0]
    l=fit_param[1]
    
#CALCULATE CHI2 ETC...    
    expected=expo((bin_centres),A,l)
    ddof=(len(bin_centres)-len(fit_param))
    Result=stats.chisquare(bin_heights,expected,ddof=ddof)
    Chi2=Result[0]
    Chi2Reduced=Chi2/ddof
    p_value=stats.chi2.sf(Chi2,ddof)
    
#CREATE THE ARRAYS
    CHI2_REDUCED.append(Chi2Reduced)
    p_list.append(p_value)
    
CHI2_REDUCED=CHI2_REDUCED
p_list=p_list

#CALCULATE AND PRINT AVERAGES
CHI2_red_average=sum(CHI2_REDUCED)/len(CHI2_REDUCED)
p_average=sum(p_list)/len(p_list)

"""
PLOTS THE HISTOGRAM OF ALL THE (NON REDUCED) CHI2 SQUARED VALUES 
COMPARED TO THE CHI2(ddof) PDF
"""
CHI2=sp.array(CHI2_REDUCED)*ddof
chi2_heights, chi2_edges, patches = plt.hist(CHI2, 
                                             range = [min(CHI2), max(CHI2)], 
                                             bins =30)

x=sp.linspace(0, 100,200)
n=N*(chi2_edges[2]-chi2_edges[1]) 

plt.plot(x,n*stats.chi2.pdf(x,ddof),c="Red",label="Chi Square Scaled PDF")
plt.vlines(Chi2H,0,max(chi2_heights),linestyles="dashed",color="Purple", 
           label="Background-only Chi Square") 
plt.legend()
plt.xlabel("Chi Square Values")
plt.ylabel("Number of Events")
plt.show()

print("As it can be seen, random fluctuations are not enough to explain the \
 way to large Chi Square previously obtained by studying the Background-only \
 Hypothesis")
#%%
"""
4. PART C: HINTS OF HIGGS BOSON:
    Execute a loop which generates i Higgs Events and 10^5 background events
    and iterates N times for each i.
    For each different random dataset it calculates the best background-only 
    fit, together with the Chi2,the reduced Chi2 and the p_value. 
    As doing so it stores such results in arrays.
    (Before running, consider the number of iterations is NH*N/delta)
"""
CHI2_Final=[]
p_Final=[]
np.random.seed(1)
N_b = 10e5 # Number of background events, used in generation and in fit.
b_tau = 30. # Spoiler.
N=50 #number of iterations
NH=400 #maximum number of tested Higgs signals, should be left 400
delta=4 #MUST BE A NUMBER WHICH PERFECTLY DIVIDES NH:
R=int(NH/delta+1) #limit of j-iteration


for j in range(1,R):  #the loop over the numbers of Higgs Signals
    CHI2_REDUCED2=[]
    p_list2=[]
    CHI2_Averages_red=[]
    p_Averages=[]
    for i in range(0,N): #for each number of signals another loop to 
                         #attenuate the effect of random fluctautions
    #GENERATE DATA
        vals = generate_data(j*delta)
        
    #CREATE THE HISTOGRAM    
        bin_heights, bin_edges, patches = plt.hist(vals, range = [104, 155], 
                                                   bins = 30)
        plt.clf()#avoids the histograms being plotted
        
    #DEFINE USEFUL QUANTITIES
        dm=dm=bin_edges[2]-bin_edges[1] #the width of each bin
        NB=(120-104)//dm #the number of bins below 120
        LB=104+NB*dm  #the edge of the last bin before 120
        NA=(155-130)//dm #the number of bins above 130
        LA=155-NA*dm  #the edge of the first bin after 130
        bin_centres=bin_edges[:(len(bin_edges)-1)]+dm/2
        bin_centres_sides=np.append(bin_centres[:int(NB)],
                                    bin_centres[(int(NA)+1):])
        bin_heights_sides=np.append(bin_heights[:int(NB)],
                                    bin_heights[(int(NA)+1):])
        
    #FITTING PARAMETERS FOR BACKGROUND HYPOTHESIS ONLY
        Num=bin_centres[-1]-bin_centres[0] 
        Den=sp.log(bin_heights[0]/bin_heights[-1]) 
        l0=Num/Den #the estimate for lambda: the ratio N/D for each pair of bins
        A0=(1/l0)*dm*sum(bin_heights)/(sp.exp(-104/l0)-sp.exp(-LB/l0)+\
                                       sp.exp(-LA/l0)-sp.exp(-155/l0)) 
        fit_param, fit_cov=curve_fit(expo,bin_centres_sides,bin_heights_sides,
                                     p0=(A0,l0))
        A=fit_param[0]
        l=fit_param[1]
        
    #CALCULATE CHI2 ETC...    
        expected=expo((bin_centres),A,l)
        ddof=(len(bin_centres)-len(fit_param))
        Result=stats.chisquare(bin_heights,expected,ddof=ddof)
        Chi2=Result[0]
        Chi2Reduced=Chi2/ddof
        p_value=stats.chi2.sf(Chi2,ddof)
        
    #CREATE THE ARRAYS
        CHI2_REDUCED2.append(Chi2Reduced)
        p_list2.append(p_value)
    
    CHI2_REDUCED2=CHI2_REDUCED2
    p_list2=p_list2
    
    #CALCULATE AND PRINT AVERAGES FOR EACH i (for each number of Higgs Signals)
    CHI2_red_average=sum(CHI2_REDUCED2)/len(CHI2_REDUCED2)
    p_average=sum(p_list2)/len(p_list2)
    
    #CREATE THE FINAL ARRAY
    CHI2_Final.append(CHI2_red_average)
    p_Final.append(p_average)
    
#PLOT THE RESULTS TO HAVE A QUALITATIVE IDEA
plt.plot(sp.linspace(delta,NH,int(NH/delta)),p_Final)
plt.hlines(0.05,0,NH,linestyles="dashed", label="0.05 p_value threshold",
           color="r")
plt.legend()
plt.xlabel("Number of Higgs Signals")
plt.ylabel("p_value")
plt.show()
#%%
"""
5.  SIGNAL ESTIMATION
    PART A:
    INITIAL ESTIMATIONS OF THE GAUSSIAN PARAMETERS ARE GIVEN: 
        G0=700, m0=125 GeV, s0=1.5
    HOWEVER SUCH PARAMETERS ARE ANYWAYS FITTED 
    (TOGETHER WITH BACKGROUND PARAMETERS, WHICH HAVE BEEN PREVIOUSLY
    FITTED ALONE) TO ACHIEVE A BETTER ACCURACY
"""
np.random.seed(1)
vals = generate_data() # A python list. 
 # Each list entry represents the rest mass reconstructed from a collision.
bin_heights, bin_edges, patches = plt.hist(vals, range = [104, 155], bins = 30)

dm=bin_edges[2]-bin_edges[1] #the width of each bin
bin_centres=bin_edges[:(len(bin_edges)-1)]+dm/2 


G0=700
m0=125
s0=1.5
N=bin_centres_sides[-1]-bin_centres_sides[0] 
D=sp.log(bin_heights_sides[0]/bin_heights_sides[-1]) 

Fit_param, Fit_cov=curve_fit(signal,bin_centres,bin_heights,p0=(A,l,G0,m0,s0))
A=Fit_param[0]
l=Fit_param[1]
G=Fit_param[2]
m=Fit_param[3]
s=Fit_param[4]

expected=signal(bin_centres,A,l,G,m,s)
ddof=(len(bin_centres)-len(Fit_param))
Goodness=stats.chisquare(bin_heights,expected,ddof=ddof)
Chi2Sig=Goodness[0]
Chi2SigReduced=Chi2Sig/ddof
p_valueSig=stats.chi2.sf(Chi2Sig,ddof)
print("First Method:\nThe Background + Higgs Hypothesis Reduced Chi Square is "\
      +str(round(Chi2SigReduced,7)) +\
          "\nThe Background + Higgs Hypothesis p-value is "+\
              str(round(p_valueSig,10)))

x=sp.linspace(104,155,200)
plt.plot(x,signal(x,A,l,G,m,s),c="Red", label="Background + Higgs Hypothesis")
plt.hist(vals, range = [104, 155], bins = 30,color="b")
plt.xlabel("Mass (GeV)")
plt.ylabel("Number of Events")
plt.legend()
plt.show()

plt.plot(x,signal(x,A,l,G,m,s),c="Red", label="Background + Higgs Hypothesis")
plt.errorbar(bin_centres,bin_heights,yerr=sp.sqrt(bin_heights), xerr=dm/2,
             ls="None",fmt="b",barsabove=True, label="Observed Data")
plt.xlabel("Mass (GeV)")
plt.ylabel("Number of Events")
plt.legend()
plt.show()
#%%
"""
5. PART B: THE INITIAL ESTIMATIONS FOR THE GAUSSIAN ARE NOT GIVEN
     BY ANALYSIING THE DIFFERENCE BETWEEN THE OBSERVED DATA AND THE 
     BACKGROUND-ONLY CURVE AN ESTIMATE OF THE MASS IS FOUND
     THEN, BY THEORETICAL CONSIDERATIONS, ESTIMATES FOR THE OTHER PARAMTERES 
     ARE FOUND AS WELL
     THEN THE CURVE IS FITTED, THE CHI2 CALCULATED, TOGETHER WITH THE P_VALUE
"""
#EXAMINE THE DIFFERENCES BETWEEN THE OBSERVED DATA AND THE EXPONENTIAL CURVE 
# TO FIND GOOD GUESSES FOR THE GAUSSIAN
def diff (x,heights):
    return heights-expo(x,A,l)
Differences=diff(bin_centres,bin_heights) 

maxDiff = max(Differences)  #the maximum of such differences
ic = Differences.tolist().index(maxDiff ) #the index of such maximum
m0 = bin_centres[ic]  #the corresponding value of the mass
k=Differences[ic-1]/maxDiff  #the difference for the (ic-1)th bin
s0=sp.sqrt(abs(-1/2/sp.log(k)))*dm #the estimate for the width of the gaussian
G0=maxDiff *sp.sqrt(2*sp.pi)*s0   #the estimate for the amplitude 

#FINALLY FIT
Fit_param, Fit_cov=curve_fit(signal,bin_centres,bin_heights,p0=(A0,l0,G0,m0,s0))
A=Fit_param[0]
l=Fit_param[1]
G=Fit_param[2]
m=Fit_param[3]
s=Fit_param[4]

#THEN CALCULATE CHI2 AND P_VALUE
expected=signal(bin_centres,A,l,G,m,s)
ddof=(len(bin_centres)-len(fit_param))
Goodness=stats.chisquare(bin_heights,expected,ddof=ddof)
Chi2Sig=Goodness[0]
Chi2SigReduced=Chi2Sig/ddof
p_valueSig=stats.chi2.sf(Chi2Sig,ddof)
print("Second Method:\nThe Background + Higgs Hypothesis Reduced Chi Square is "+\
      str(round(Chi2SigReduced,7)) +\
          "\nThe Background + Higgs Hypothesis p-value is "+\
          str(round(p_valueSig,10)))
print("The estimated mass value for the Higgs Boson is "+ str(round(m,2))+\
      u"\u00B1"+str(round(sp.sqrt(Fit_cov[3,3]),2))+ " GeV")
#FINALLY PLOT
x=sp.linspace(104,155,200)
plt.hist(vals, range = [104, 155], bins = 30,color="b")
plt.plot(x,signal(x,A,l,G,m,s),c="Red",label="Background + Higgs Hypothesis")
plt.xlabel("Mass (GeV)")
plt.ylabel("Number of Events")
plt.legend()
plt.show()

plt.errorbar(bin_centres,bin_heights,yerr=sp.sqrt(bin_heights), xerr=dm/2,
             ls="None",fmt="b",barsabove=True, label="Observed Data")
plt.plot(x,signal(x,A,l,G,m,s),c="Red",label="Background + Higgs Hypothesis")
plt.xlabel("Mass (GeV)")
plt.ylabel("Number of Events")
plt.legend()
plt.show()
#%%
"""
5. PART C:LOOPING OVER THE WHOLE RANGE (104,155) OF MASSES IN ORDER TO FIND THE
   VALUE OF CHI2 FOR EACH OF THEM
    (WHILE KEEPING FIXED THE AMPLITUDE G0=700 AND THE WIDTH s=1.5 GeV)
    PLOTTING THE RESULTS ND FINDING THE BEST ESTIMATE OF THE MASS
    (The number of iterations is 51*n and leads to an IDEAL precision of 1/n;
     this loop is much faster: 16320 iterations in less than 5 seconds)
"""        
G0=700
s0=1.5
ddof=(len(bin_centres)-2)
n=320 # it's better if it's only a multiple of 2 and 5
masses=[]
Chi2Sig=[]
Chi2SigReduced=[]
p_valueSig=[]
for i in range(0,(155-104)*n):
    expected=signal(bin_centres,A,l,G,104+i/n,s0)
    masses.append(104+i/n)
    Goodness=stats.chisquare(bin_heights,expected,ddof=ddof)
    Chi2Sig.append(Goodness[0])
    Chi2SigReduced.append(Goodness[0]/ddof)
    p_valueSig.append(stats.chi2.sf(Goodness[0],ddof))
plt.plot(masses,Chi2SigReduced)
Chi2Min = min(Chi2Sig)  #the minimum among the chi2 
Chi2MinReduced=Chi2Min/ddof #the reduced chi2
p_valueMax=stats.chi2.sf(Chi2Min,ddof)
ic = Chi2Sig.index(Chi2Min) #the index of such minimum
m0 = masses[ic] 

plt.plot(masses,Chi2SigReduced,c="b")
plt.xlabel("Mass (GeV)")
plt.ylabel("Reduced Chi Square")
plt.vlines(m0,0,Chi2MinReduced,linestyles="dashed",color="Red",
           label="Best Fit:\n M="+str(m0)+"GeV\n Reduced Chi2="+\
               str(round(Chi2MinReduced,3)))
plt.hlines(Chi2MinReduced,104,m0,linestyles="dashed",color="Red")
plt.legend()
plt.show()
print("Third Method: \nThe best fit for Higgs Boson's mass is " +\
      str(round(m0,3)) + u"\u00B1"+ str(round(1/n,3))+\
          " GeV, to which correspond")
print("a Reduced Chi Square " +str(round(Chi2MinReduced,7)) + "\na p-value "+\
      str(round(p_valueMax,10)))
print("NOTE: additional considerations are needed about the estimation\
 of the uncertainty")