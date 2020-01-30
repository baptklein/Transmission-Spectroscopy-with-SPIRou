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

import sys
import os
import time

from scipy import interpolate

import matplotlib.pyplot as plt
import matplotlib





def get_cc(Yd,Ym):
    """
    Compute the correlation coefficient between the sequence of spectra and the modelled sequence of the spectra
    If a spectrum in the modelled sequence of spectra is 0, np.corrcoef returns NaN. This displays a warning message,
    but we account for this in the process.
    Inputs:
    - Yd: 2D sequence of spectra
    - Ym: Modelled sequence of spectra (binned at the resolution of data - same shape as Yd)

    Outputs:
    - Correlation coefficient between the 2 spectra
    """

    C0 = 0.
    for n in range(len(Yd)):
        #c = np.ma.corrcoef(Yd[n],Ym[n]).data[0,1]
        c = np.corrcoef(Yd[n],Ym[n])[0,1]
        if np.isfinite(c): C0 += c  ### Avoid NaNs (modelled spectrum is 0 (no planet in out-of-transit periods)
    return C0




class CCF:
    
    """
    Class CCF that handles the Matching template filter and the cross-correlation between the modelled sequence of spectra and the data
    """

    
    def __init__(self):
        
        
        self.planet = ""   ## Planet object
        self.model  = ""   ## Model object
        self.data   = ""   ## Observations object
        
        self.K_vec  = []   ## Values of semi-amplitude of the planet orbit for parameter search (1D vector)
        self.V0_vec = []   ## Values of radial velocity at mid-transit for parameter search (1D vector)
        self.V0_ccf = []   ## Values of RV used to cross-correl the template spectrum to each of spectrum of the reduced sequence
        self.ddv    = []   ## Vector of velocity used for the integration of the model when binned into the data sampling scheme
                           ## We generally center it on 0 with half width of 1 SPIRou pixel (i.e. ~2 km/s)        



    def bin_model(self,k,v,V_data,ccf=0,temp=[]):

        """
        Bin the model at the resolution of the data accounting for the shifts in velocity for (k,v) values
        
        Inputs:
        - k,v:    semi-amplitude and mid-transit velocity (floats)
        - V_data: Velocity matrix of the sequence to bin (returned after applying OBS.shift_rv method.
                  Each line is the velocity vector of the spectrum shifted in the stellar rest frame

        Outputs:
        - I_ret: Binned model at the resolution of the data where spectra are shifted at (k,v)
        """

        ### Compute the Radial Velocity of the planet in the stellar rest frame
        DVP   = self.planet.RVP(k,v)

        I_ret = np.zeros((len(self.data.date),len(V_data[0])))  ### Init binned sequence of spectra

        if ccf == 0:

            for nit in range(len(self.data.date)):
                I_tmp = np.zeros(len(V_data[0]))

                ### For dd in the window centered on 0 and of 1 px width (here 2 km/s)
                for dd in self.ddv:
                    I_tmp += self.model.Fm[nit](V_data[nit]+dd-DVP[nit]) 
                I_ret[nit] = I_tmp/len(self.ddv) ### Average values to be closer to measured values
                                                 ### Note: if we take only 1 pt,

        else:

            ### In case of computing the CCF only, use the template instead of the model
            for nit in range(len(self.data.date)):
                I_tmp = np.zeros(len(V_data[0]))
                for dd in self.ddv:
                    I_tmp += temp(V_data[nit]+dd-DVP[nit]) 
                I_ret[nit] = I_tmp/len(self.ddv) 

        return I_ret ### Binned modelled sequence shifted at (kp,v0)




    def make_corr_map(self,V_data,I_data):

        """ 
        Explore (Kp,V0) parameter space by correlating the modelled sequence of spectra with the reduced sequence
        for all couple of parameters in the grid. 

        Inputs:
        - V_data: Velocity matrix where each line is the velocity vector of the spectrum shifted in the stellar rest frame
        - I_data: reduced sequence of spectra

        Outputs:
        - corr: Matrix of correlation coefficients (shape: (len(self.K_vec),len(self.V0_vec)))
        """

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
        print(tx)
        
        return corr



            

    def make_ccf(self,V_data,I_data,template_name):

        """
        Cross-correlate te template transmission spectrum with each spectrum of the sequence

        Inputs:
        - V_data: 2D velocity matrix where each line is in the stellar rest frame
        - I_data: 2D sequence of spectra
        - template_name: name of the used template

        Outputs:
        - Sequence of cross-correlations between template and each spectrum (shape: (N_obs,len(self.V0_ccf)))
        """


        print("Cross-correl template transmission spectrum to the sequence of spectra")
        CCF_fin = []        
        for vv in self.V0_ccf:
            
            ### Interpolate the templace
            temp = self.model.template[template_name]
            F = interpolate.interp1d(self.model.Vm,temp,kind='linear')

            ### Bin model assuming Kp = 0 km/s
            Im = self.bin_model(0.0,vv,V_data,1,F)
            CCF_line = []
            
            ### Compute cross-correlation
            for nn in range(len(self.data.date)):
                c = np.ma.corrcoef(I_data[nn],Im[nn]).data[0,1]
                CCF_line.append(c)
            CCF_fin.append(CCF_line)
        CCF_fin = np.array(CCF_fin,dtype=float)

        print("DONE")

        return CCF_fin


























            


