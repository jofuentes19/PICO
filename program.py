# -*- coding: utf-8 -*-
"""
GUNTER ANALYSIS R-134A

@author: joel fuentes
"""
##############################################################################
# INITIALIZE VARIABLES AND IMPORT NECESSARY PACKAGES 
from libc.math cimport exp 
from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf
import numpy
import pandas as pd
import random
import time 
import itertools
cimport numpy
cimport cython
cdef numpy.ndarray MC, MF, MH, MC1, MF1, MH1
cdef numpy.ndarray pvals, yvals, hvals, svals, scvals, sfvals, shvals
cdef numpy.ndarray tc, tf, th, SEITZ, stopC, stopF, stopH, SEIT
cdef float t1,t2,t,RLETMI, criten0, criten, p, h, element, k, x
cdef float DEDX, stop, grw, j, l, b, c, g, d,a,s,sim_time, conv, ll, aa, bb
cdef float thrEC, thrEF, thrEH,thr0EC,thr0EF,thr0EH 
cdef float nucEC, nucEF, nucEH, nuc_EC, nuc_EF, nuc_EH
cdef float EC, EF, EH, sft_thr, sft_thrC,sft_thrF,sft_thrH
cdef float sft_thr0C, sft_thr0F, sft_thr0H
cdef float rateC, rateF, rateH, rateCF, rateCFH, countC, countF, countH
cdef float mtc, mtf, mth, mc, mf, mh, WS, WR, WE, ETA, mc0, mf0, mh0
cdef float T, factor, rmolec, tcriti, pcriti, vcriti, corden, surfac
cdef float vapre, rhfg, pvapor, difst, beta, E_ion, A, B, rho_l, nuc_prob
##############################################################################
#CONSTANTS
#start timer
t1 = time.time()
#temperature in C
T=35.0
#FACTOR IS THE NUM. FACTOR IN DENOMINATOR FOR Ec
factor=0.0274
#RMOLEC IS MOLAR MASS IN KG/KMOL
rmolec=102.03
#THIS FROM REFPROP DATAFILE (CRITICAL T(C) AND P(ATM))
tcriti=101.06
pcriti=40.062
#CRITICAL VOLUME IN M^3/MOL = RMOLEC/CRITICAL DENSITY
#(CRITICAL DENSITY IS 511.9 KG/M^3 FROM REFPROP)
vcriti=1.9931E-4
#calculate the density correction factor for the dE/dx
corden = 1.1896 - (0.003792*T)
#CALCULATE THE SURFACE TENSION IN DYNE/CM
surfac = (52.949-(0.10762*(273.15+T)) 
          - ((3.4752E-4)*((273.15+T)**2.)) 
          + ((6.8232E-07)*((273.15+T)**3.)))
#CALCULATE THE VAPOUR PRESSURE IN ATM
vapre = (-155.35 + (1.8986*(273.15+T)) 
         - ((7.9265E-3)*((273.15+T)**2.)) 
         + ((1.1337E-05)*((273.15+T)**3.)))
#THIS IS THE LATENT HEAT OF VAPORIZATION IN KJ/KG FROM DIFFERENCE
#BETWEEN SATURATED VAPOR ENTHALPY AND LIQUID ENTHALPY
rhfg=(915.44-(6.8389*(273.15+T)) + (0.024056*((273.15+T)**2.))
      -(3.1581E-05*((273.15+T)**3.)))
#THIS IS THE VAPOR DENSITY IN G/CC
pvapor=(-2.6719+((2.9023E-2)*(273.15+T)) - ((1.0666E-4)*((273.15+T)**2.))
        +((1.3327E-07)*((273.15+T)**3.)))
#THIS IS THE DERIVATIVE OF SURFACE TENSION VS T
difst = (-0.79585+((6.4604E-3)*(273.15+T)) - (2.2514E-05*((273.15+T)**2.))
         +(2.7822E-08*((273.15+T)**3.)))
##############################################################################
#FUNCTIONS 
#atm to psig conversion function
conv=0.06804595706430118
cdef float psig_atm(float x):
    return x*(conv) + 1
#nucleation eff weighted coin toss
cdef float nuc_eff(float prob):
    return int(random.random() < prob)

#function for calculating RLETMI and criten using pressure and harper factor as inputs 
cdef calc(float p, float h):
    #keV
    criten0 = (((16.755/factor)*((surfac**3.) / 
                                 (((vapre-p)**2.)*1.0261E12)))*6.242E8)
    #CALCULATE THE CRITICAL RADIUS IN CM*1E-6
    critra = (2.*surfac)/((vapre-p)*1.013)
    #WORK OF BUBBLE SURFACE FORMATION
    WS = 4.*3.141592*(critra**2.)*(surfac-((273.15+T)*difst))*6.242E-4
    
    #WORK OF EVAPORATION
    WR=(4./3.)*3.141592*(critra**3.)*pvapor*rhfg*6.242E-3
    
    #WORK OF EXPANSION AGAINST PRESSURE
    WE=(4./3.)*3.141592*(critra**3.)*p*6.242E-4
    
#c *** THIS IS ETA AS IN EQUATION (30 OF NEW JOURNAL OF PHYSICS
    ETA=(criten0*factor)/(WS+WR+WE)
    
#C ***** RECALCULATE EXACT CRITICAL ENERGY (WS+WR+WE)
    criten=(criten0*factor)/ETA
    
# c**** CALCULATE THE MINIMUM LET IN keV/µm
    RLETMI=(100.*criten)/(h*critra)
    
    return RLETMI, criten

#%%
# dataframe refprop nist
columns= ['col1','col2','col3','col4','col5','col6']
C = pd.read_csv('C134A.txt', skipinitialspace=True, sep=" \s+",engine='python',names=columns)
H = pd.read_csv('H134A.txt', skipinitialspace=True, sep=" \s+",engine='python',names=columns)
F = pd.read_csv('F134A.txt', skipinitialspace=True, sep=" \s+",engine='python',names=columns)

CX1,CX2,CX3,CX4,CX5,CX6 = C.loc[:,"col1"], C.loc[:,"col2"], C.loc[:,"col3"], C.loc[:,"col4"], C.loc[:,"col5"], C.loc[:,"col6"] 

HX1,HX2,HX3,HX4,HX5,HX6 = H.loc[:,"col1"], H.loc[:,"col2"], H.loc[:,"col3"], H.loc[:,"col4"], H.loc[:,"col5"], H.loc[:,"col6"] 

FX1,FX2,FX3,FX4,FX5,FX6 = F.loc[:,"col1"], F.loc[:,"col2"], F.loc[:,"col3"], F.loc[:,"col4"], F.loc[:,"col5"], F.loc[:,"col6"] 

CX1=CX1.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})
CX4=CX4.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})
CX5=CX5.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})
CX6=CX6.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})

HX1=HX1.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})
HX4=HX4.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})
HX5=HX5.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})
HX6=HX6.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})

FX1=FX1.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})
FX4=FX4.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})
FX5=FX5.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})
FX6=FX6.str.extract(r'([-]?[\d.]*)([\w\D]*)').rename(columns={0:'VALUE', 1:'UNIT'})

#conversions
CON1=[CX1,HX1,FX1]
CON4=[CX4,HX4,FX4]
CON5=[CX5,HX5,FX5]
CON6=[CX6,HX6,FX6]

for o in CON1:   
    for i in range(len(CX2)):
        if o['UNIT'][i] == ' eV':
            o['VALUE'][i] = float(o['VALUE'][i]) / 1000
        else:
            if o['UNIT'][i] == ' MeV':
                o['VALUE'][i] = float(o['VALUE'][i]) * 1000
for o in CON4:
    for i in range(len(CX2)):
        if o['UNIT'][i] == ' A':
            o['VALUE'][i] = float(o['VALUE'][i]) / 10000
        else:
            if o['UNIT'][i] == ' cm':
                o['VALUE'][i] = float(o['VALUE'][i]) * 10000                
for o in CON5:
    for i in range(len(CX2)):
        if o['UNIT'][i] == ' A':
            o['VALUE'][i] = float(o['VALUE'][i]) / 10000
        else:
            if o['UNIT'][i] == ' cm':
                o['VALUE'][i] = float(o['VALUE'][i]) * 10000
for o in CON6:
    for i in range(len(CX2)):
        if o['UNIT'][i] == ' A':
            o['VALUE'][i] = float(o['VALUE'][i]) / 10000
        else:
            if o['UNIT'][i] == ' cm':
                o['VALUE'][i] = float(o['VALUE'][i]) * 10000
                
stopC0,stopF0,stopH0 = [],[],[]
SEIT0 = []
for d in range(len(CX2)):
    stopC0.append((CX2[d]+CX3[d]))
    stopF0.append((FX2[d]+FX3[d]))
    stopH0.append((HX2[d]+HX3[d]))
    SEIT0.append(float(CX1['VALUE'][d]))

stopC=numpy.array(stopC0)
stopF=numpy.array(stopF0)
stopH=numpy.array(stopH0)
SEIT=numpy.array(SEIT0)

#%%

sim_time=1.49E7
f = numpy.loadtxt('2E8n = (13.4ns).dat')
df = pd.DataFrame(f, columns = 
                  ['NPS','NPAR','IPT','NTYN','ZAID','NCL','EnReCo','TME',
                    'x','y','z','next3','next4','num_scat','next6','next7'])

#%%
# *** FILTER FOR ALL NON ZERO ENERGY EVENTS, IDENTIFY, AND CONVERT TO KEV
df = df[df['EnReCo'] != 0.0]
MC0 = df['EnReCo'][df[(df.ZAID == 6000.0) | (df.ZAID == 6)].index] * 1000
MF0 = df['EnReCo'][df[(df.ZAID == 9019.0) | (df.ZAID == 9)].index] * 1000
MH0 = df['EnReCo'][df[(df.ZAID == 1001.0) | (df.ZAID == 1)].index] * 1000

# *** CONVERT FROM SERIES TO ARRAY 
MC = numpy.asarray(MC0)
MF = numpy.asarray(MF0)
MH = numpy.asarray(MH0)

#%%

# *** AVERAGE PRESSURE FROM DATA (CUTS)
pvals=numpy.asarray([9.54,9.54,14.42,17.3,24.98,30.46,33.88,39.42,44.76,49.98,55.19,59.29])

# *** DEFINE PARAMETER SPACE AND STEPS FOR HARPER FACTOR AND SOFTNESS OF THRESHOLD(S) 
#HARPER FACTOR
hvals=numpy.linspace(4,5,10)
#SOFTNESS OF THRESHOLD
svals=numpy.linspace(0.1,10,10)
# scvals=numpy.linspace(0.1,10,5)
# sfvals=numpy.linspace(0.1,10,5)
# shvals=numpy.linspace(0.1,10,5)
# grvals=numpy.linspace(1.0,10.0,50)
# scvals=numpy.linspace(3,5,2)
# sc0vals=numpy.linspace(0.1,1,2)
# sfvals=numpy.linspace(3,5,2)
# sf0vals=numpy.linspace(0.1,1,2)
# shvals=numpy.linspace(3,5,2)
# sh0vals=numpy.linspace(0.1,1,2)
# *** FIND MAXIMUM STOPPING POWER VALUES 
mtc=numpy.max(stopC)
mtf=numpy.max(stopF)
mth=numpy.max(stopH)
#%%
# *** INITIALIZE OUTPUT FILE AND EXECUTE NESTED LOOP 
#CHANGE THIS TO LOCAL DIRECTORY
filepath = '/Users/joelfuentes/Downloads/sim.txt'
with open(filepath, 'a') as fout:
    # *** first mode for two free parameters: harper factor and uniform softness of threshold
    for j,s,g in itertools.product(hvals,svals,pvals):
        
    # *** second mode for four free parameters: 
        #harper factor and varying softness of threshold for C,F,H
    # for j,l,a,b,g in itertools.product(hvals,scvals, sfvals, shvals, pvals):
    # for a fixed harper factor 
    # for l,a,b,g in itertools.product(scvals, sfvals, shvals, pvals):
        
    # *** 2 harper factors and varying softness of threshold for C,F,H
    # for j,s,ss,g in itertools.product(hvals,svals,s0vals,pvals):
    # for j,l,a,b,ll,aa,bb,g in itertools.product(hvals,scvals,sfvals,shvals,sc0vals,sf0vals,sh0vals,pvals):    
        h=j
        sft_thr=s
        # sft_thr0=ss
        # sft_thrC=l
        # sft_thrF=a
        # sft_thrH=b
        # sft_thr0C=ll
        # sft_thr0F=aa
        # sft_thr0H=bb
        p = psig_atm(g)
        RLETMI, criten = calc(p, h)
        # *** THRESHOLD ENERY CALCULATION IMPOSING DOUBLE CONDITION 
        ######################################################### 
        if mtc >= RLETMI:
            for d, stop in zip(SEIT, stopC):
                DEDX=stop*corden
                if DEDX >= RLETMI and d >= criten:
                    EC=d
                    break
                else:
                    if DEDX >= RLETMI:
                        EC=criten
        else:
            EC=criten
        #########################################################      
        if mtf >= RLETMI:
            for d, stop in zip(SEIT, stopF):
                DEDX=stop*corden
                if DEDX >= RLETMI and d >= criten:
                    EF=d
                    break
                else:
                    if DEDX >= RLETMI:
                        EF=criten                        
        else:
            EF=criten
        #########################################################
        # *** EH SET TO 0 AS DEFAULT 
        # SINCE FOR MANY COMBINATIONS OF P, T AND H NO BUBBLES ARE PRODUCED 
        # *** (dE/dx threshold will not be surpassed for any recoil energies; 
        # Seitz threshold -> infinity)
        EH=0.0
        if mth >= RLETMI:
            for d, stop in zip(SEIT, stopH):
                DEDX=stop*corden
                if DEDX >= RLETMI and d >= criten:
                    EH=d
                    break
                else:
                    if DEDX >= RLETMI:
                        EH=criten

        #########################################################
        # *** THRESHOLD + SOFTNESS OF THRESHOLD
        thrEC, thrEF, thrEH = (EC+sft_thr),(EF+sft_thr),(EH+sft_thr)
        # thrEC, thrEF, thrEH = (EC+sft_thrC),(EF+sft_thrF),(EH+sft_thrH)
        # thrEC0, thrEF0, thrEH0 = (EC-sft_thrC),(EF-sft_thrF),(EH-sft_thrH)
        # thrEC, thrEF, thrEH = (EC+sft_thrC), (EF+sft_thrF), (EH+sft_thrH)
        # thr0EC, thr0EF, thr0EH = (EC-sft_thr0C), (EF-sft_thr0F), (EH-sft_thr0H)
        
        # *** CALCULATE SLOPE (LINEAR THRESHOLD MODEL)
        # mc, mf, mh = sft_thrC**(-1), sft_thrF**(-1), sft_thrH**(-1)
        # mc0, mf0, mh0 = (sft_thrC+sft_thr0C)**(-1), (sft_thrF+sft_thr0F)**(-1), (sft_thrH+sft_thr0H)**(-1)
        mc, mf, mh = sft_thr**(-1), sft_thr**(-1), sft_thr**(-1)
        # mc0, mf0, mh0 = (sft_thr+sft_thr0)**(-1), (sft_thr+sft_thr0)**(-1), (sft_thr+sft_thr0)**(-1)
        
        
        # *** INITIALIZE COUNTERS FOR TALLYING NUCLEATIONS 
        countC, countF, countH  = 0,0,0
        
        # *** BUBBLE NUCLEATION TALLY
        #########################################################
        for j in MC:
            if j >= thrEC:
                countC += 1.0
            if j >= EC and j < thrEC:
                nuc_EC = (nuc_eff(mc*(j - EC)))
                countC += nuc_EC
            # if j >= thr0EC and j < EC:
            #     nuc_EC = (nuc_eff(mc0*(EC-j)))
            #     countC += nuc_EC
        nucEC = countC
        #########################################################
        for j in MF:
            if j >= thrEF:
                countF += 1.0
            if j >= EF and j < thrEF:
                nuc_EF = nuc_eff(mf*(j - EF))
                countF += nuc_EF
            # if j >= thr0EF and j < EF:
            #     nuc_EF = (nuc_eff(mf0*(EF-j)))
            #     countF += nuc_EF
        nucEF = countF
        #########################################################
        if EH != 0:
            for j in MH:
                if j >= thrEH:
                    countH += 1.0
                if j >= EH and j < thrEH:
                    nuc_EH = nuc_eff(mh*(j - EH))
                    countH += nuc_EH
                # if j >= thr0EH and j < EH:
                #     nuc_EH = (nuc_eff(mh0*(EH-j)))
                #     countH += nuc_EH
        nucEH = countH
        #########################################################
        # *** CALCULATE INDIVIDUAL RATES 
        rateC, rateF, rateH = (nucEC/sim_time), (nucEF/sim_time), (nucEH/sim_time)
        # *** CALCULATE C+F AND C+F+H RATES 
        rateCF, rateCFH  = (rateC + rateF), (rateC + rateF + rateH)
        # *** WRITE TO FILE BELOW
        output = [sft_thr,h,g,rateCF,rateCFH,criten,EC,EF,EH,
                  rateC,rateF,rateH,RLETMI,E_ion,critra]
        # output = [sft_thrC,sft_thrF,sft_thrH,h,g,rateCF,rateCFH,criten,
        #EC,EF,EH,rateC,rateF,rateH,RLETMI]
        # output = [sft_thr,sft_thr0,h,g,rateCF,rateCFH,criten,EC,EF,EH,rateC,
        #rateF,rateH,RLETMI]
        # output = [sft_thrC,sft_thrF,sft_thrH,sft_thr0C,sft_thr0F,sft_thr0H,
        #h,g,rateCF,rateCFH,criten,EC,EF,EH,rateC,rateF,rateH,RLETMI]
        
        fout.write(str(output).lstrip('[').rstrip(']') + '\n')
                 
# *** END TIMER AND PRINT TIME 
t2 = time.time()
t = t2-t1
print(t)

        
        
     
        
            
            
            
            
            
            
            
            
            
            
