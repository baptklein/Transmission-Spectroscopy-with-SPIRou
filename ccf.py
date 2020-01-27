#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday 25/01/2020
@author:
Baptiste Klein
Institut de Recherche en Astrophysique et Planétologie
14 Avenue Édouard Belin
31400 Toulouse
France
Email: baptiste.klein@irap.omp.eu
"""



import numpy as np
from numpy import ma

import termcolor
import sys
import os
import time

from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib









def get_cc(Yd,Ym):
    C0 = 0.
    for n in range(len(Yd)):
        #c = np.ma.corrcoef(Yd[n],Ym[n]).data[0,1]
        c = np.corrcoef(Yd[n],Ym[n])[0,1]
        if np.isfinite(c): C0 += c
    return C0




class CCF:

    
    def __init__(self):
        
        
        self.planet = ""   ## Planet object
        self.model  = ""   ## Model object
        self.data   = ""   ## Observations object
        
        self.K_vec  = []   ## Values of semi-amplitude of the planet orbit for parameter search (1D vector)
        self.V0_vec = []   ## Values of radial velocity at mid-transit for parameter search (1D vector)
        self.V0_ccf = []   ## Values of RV used to cross-correl the template spectrum to each of spectrum of the reduced sequence
        self.ddv    = []   ## Vector of velocity used for the integration of the model when binned into the data sampling scheme
                           ## We generally center it on 0 with half width of 1 SPIRou pixel (i.e. ~2 km/s)        



    def bin_model(self,k,v,V_data):

        DVP   = self.planet.RVP(k,v)

        I_ret = np.zeros((len(self.data.date),len(V_data[0])))
        for nit in range(len(self.data.date)):
            I_tmp = np.zeros(len(V_data[0]))
            for dd in self.ddv:
                I_tmp += self.model.Fm[nit](V_data[nit]+dd-DVP[nit])
            I_ret[nit] = I_tmp/len(self.ddv)
        return I_ret




    def make_corr_map(self,V_data,I_data):


        corr  = []

        time1 = time.time()
        
        for kk in self.K_vec:
            C_line     = []
            for vv in self.V0_vec:
                Im = self.bin_model(kk,vv,V_data)
                C_line.append(get_cc(I_data,Im))
            corr.append(C_line)
        corr = np.array(corr,dtype=float)

        time2 = time.time()
        tx    = "Duration of the cross-correlation: " + str((time2-time1)/60.) + " min"
        print(termcolor.colored(tx,"blue"))
        
        return corr



            

    def make_ccf(self,V_data,I_data):


        print(termcolor.colored("Cross-correl template transmission spectrum to the sequence of spectra","blue"))  
        
        CCF_fin = []
        
        ntrace = 0
        
        for vv in self.V0_ccf:
            
            ### Bin model assuming Kp = 0 km/s
            Im = self.bin_model(0.0,vv,V_data)
            CCF_line = []
            
            for nn in range(len(self.data.date)):
                c = np.ma.corrcoef(I_data[nn],Im[nn]).data[0,1]
                if np.isfinite(c): C0 = c
                CCF_line.append(c)
            CCF_fin.append(CCF_line)
        CCF_fin = np.array(CCF_fin,dtype=float)

        print(termcolor.colored("DONE","green"))  

        return CCF_fin


























            


