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

from scipy import signal
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.stats as stats
from scipy.optimize import minimize

from sklearn.decomposition import PCA

import astropy
from astropy.io import fits
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.modeling import models, fitting, polynomial

import matplotlib.pyplot as plt
import matplotlib



font = {'size'   : 18,
        'weight': 'light'}
axes = {'labelsize': 18,
        'labelweight': 'light'}

matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)



##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################




############################################################
# Master functions for data reduction                      #
############################################################

def plot_2d(x,y,z,LIM,LAB,title,**kwargs):
    """
    Use pcolor to display sequence of spectra
    
    Inputs:
    - x:        x array of the 2D map (if x is 1D vector, then meshgrid; else: creation of Y)
    - y:        y 1D vector of the map
    - z:        2D array (sequence of spectra; shape: (len(x),len(y)))
    - LIM:      list containing: [[lim_inf(x),lim_sup(x)],[lim_inf(y),lim_sup(y)]]
    - LAB:      list containing: [label(x),label(y),label(z)] - label(z) -> colorbar
    - title:    title of the map
    - **kwargs: **kwargs of the matplolib function pcolor
    
    Outputs:
    - Display 2D map of the sequence of spectra z
    
    """

    if len(np.shape(x))==1: X,Y  = np.meshgrid(x,y)
    else:
        X = x
        Y = []
        for n in range(len(x)):
            Y.append(y[n] * np.ones(len(x[n])))
        Y = np.array(Y,dtype=float)
    Z    = z

    fig  = plt.figure()
    plt.rcParams["figure.figsize"] = (10,7)
    ax   = plt.subplot(111)
    cc   = ax.pcolor(X, Y, Z,**kwargs)
    cb   = plt.colorbar(cc,ax=ax)
    
    ax.set_xlim(LIM[0][0],LIM[0][1])
    ax.set_ylim(LIM[1][0],LIM[1][1])
    
    ax.set_xlabel(LAB[0])
    ax.set_ylabel(LAB[1],labelpad=15)
    cb.set_label(LAB[2],rotation=270,labelpad=30)

    ax.set_title(title,pad=35)

    plt.show()
    
    
#################################################################################################    


def make_pca(N_comp,I_spec):
    
    """
    Apply principal component analysis (pca) to remove the first N_comp components from I_spec
    
    Inputs:
    - N_comp: Number of components to remove (integer)
    - I_spec: 2D spectrum on which we apply PCA (np.array (N_exp,N_wavelengths))
    
    Outputs:
    - I_pca: Cleaned matrix (after the first N_comp components are removed - np.array, same shape as I_spec)
    - I_del: list of components removed with pca (each component -> np.array, shape of I_spec)
    - e_val: vector of all eigenvalues computed with PCA  

    """

    ### Number of components
    NC = N_comp
    M  = I_spec 
    N  = len(M[0,:])
    K  = len(M[:,0])
    
    ### Apply PCA assuming centered matrix
    pca    = PCA(n_components=K)         ### Number of phases in our case
    M_proj = pca.fit_transform(M)        ### Project in the basis of eigenvectors
    comp   = pca.components_             ### All components of the PCA
    e_val  = pca.singular_values_        ### Eigenvalues of the PCA
    
    comp_r = np.array(comp,dtype=float)  ### Init components to reconstruct matrix
    ### Remove the first N_comp components
    for k in range(NC):
        comp_r[k,:] = np.zeros(N)
     
    ### Store the removed components
    I_del = []
    for k in range(NC):
        comp_del = np.zeros(np.shape(comp))
        comp_del[k,:] = comp[k,:]
        I_del.append(np.dot(M_proj,comp_del))
        
    ### Project matrix back into init basis using without the first N_comp components
    M_fin = np.dot(M_proj,comp_r)
    I_pca = M_fin
    I_del = np.array(I_del,dtype=float)
    e_val = np.array(e_val,dtype=float)
    
    
    return I_pca,I_del,e_val


#################################################################################################


def norm_pol(Wm,Im,order=2,n_iter=4,sig_clip=5.0,plot=False,title=""):
    
    """
    Iterative polynomial fit to Im rejecting outliers deviating from the model by more than sig_clip
    sigmas
    
    Inputs:
    - Wm: 1D wavelength vector
    - Im: 1D exposure vector (same size as Wm)
    - order: order of the polynomial fit
    - n_iter: Number of iterations of the iterative fit
    - sig_clip: Threshold for the outlier remove (in sigmas)
    - plot: if True, plot normalized spectrum (call plot_norm fct)
    - title: title of the plot if plot == True
    
    Outputs:
    - I_nor: Normalised spectrum
    
    """
    
    Wmean  = 0.5*(np.min(Wm)+np.max(Wm)) # Mean wavelength to center the fit
    
    ### Init model and estimator
    pol_0  = polynomial.Polynomial1D(order)
    fit    = fitting.LinearLSQFitter()
    or_fit = fitting.FittingWithOutlierRemoval(fit,sigma_clip,niter=n_iter,sigma=5.0) 
    
    ### Make fit
    or_fitted_model, mask = or_fit(pol_0,Wm-Wmean,Im)
    
    ### Make prediction
    filtered_data = np.ma.masked_array(Im, mask=mask)
    fitted_model  = fit(pol_0,Wm-Wmean,Im)
    I_pred        = fitted_model(Wm-Wmean)
    
    ### Normalize exposure
    I_nor         = Im/I_pred
    
    ### Plot data if plot == True
    if plot: plot_norm(Wm,Im,I_nor,I_pred,filtered_data,title)
    
    return I_nor



#################################################################################################


def mav(Wm,Im,N_best,N_bor):
    
    """
    Compute the moving median of Im and divide Im by the resulting moving median computing within
    [W[N_bor],W[-N_bor]] with N_best points
    
    Inputs:
    - Wm:     Wavelength vector
    - Im:     1D exposure to normalize (size: len(Wm)) 
    - N_best: Number of points using to compute the average
    - N_bor:  Number of points removed at each edge of each spectrum to avoid border effects
    
    Outputs:
    - W_tm: Wavelength vector within [W[N_bor],W[-N_bor]]
    - I_tm: 1D spectrum within [W[N_bor],W[-N_bor]]
    - I_nor: Normalised exposure (size: len(I_tm))
    - I_bin: Resulting moving median used to normalize the Im   
    """
    
    ### Init the moving median
    W_bin = []
    I_bin = []
    
    ### Make moving median
    for k in range(N_bor,len(Wm)-N_bor):
        
        ### if N_best < N_bor, need to adapt the size on the moving average on each edge
        if k < N_best:
            N_inf = 0
            N_sup = int(k+N_best)
        elif k + N_best > len(Wm):
            N_inf = int(k-N_best)
            N_sup = -1
            
        ### Else: standard moving average
        else:
            N_inf = int(k-N_best)
            N_sup = int(k+N_best)        
            
        W_bin.append(np.median(Wm[N_inf:N_sup]))
        r,cl,cm = stats.sigmaclip(Im[N_inf:N_sup],3,3) #Apply sigma-clipping
        I_bin.append(np.median(r))  #Take median

    W_bin,I_bin = np.array(W_bin,dtype=float),np.array(I_bin,dtype=float)

    ### Remove N_bor points at each edge of the data
    W_tm = Wm[N_bor:-N_bor]
    I_tm = Im[N_bor:-N_bor]
    
    ### Normalize spectrum    
    I_nor = I_tm/I_bin
    
    
    return W_tm,I_tm,I_nor,I_bin



#################################################################################################


def moving_average(Wm,Im,N_best,N_bor,plot=False,title=""):
    """
    Call the global function 'mav' to compute a moving median of Im between W[N_bor] and W[-N_bor]
    Each point Ik is obtained by taking the median on I[k-N_best:k+N_best] pts, using a 3 sigma-clipping
    process to get read of outliers. Points outside [W[N_bor],W[-N_bor]] are removed.
    
    Inputs:
    - Wm:     Wavelength vector
    - Im:     1D exposure to normalize (size: len(Wm)) 
    - N_best: Number of points using to compute the average
    - N_bor:  Number of points removed at each edge of each spectrum to avoid border effects
    - plot:   if True, plot each normalized spectrum (may take time)

    Outputs:
    - I_nor: Normalized 1D spectrum (size: len(Wm)-2*N_bor)
    - W_tmp: Normalized 1D wavelength vector (size: len(W_tmp)-2*N_bor)    
    """
    
    W_tmp, I_tmp, I_nor, I_bin = mav(Wm,Im,N_best,N_bor)

    ### Plot result
    if plot:
        xlab = -0.1 #Lad between yaxis and its label
        fig = plt.Figure(figsize=(10,5))
        plt.title(title)
        
        ax1 = plt.subplot(211)
        plt.plot(Wm,Im,color="black",linewidth=0.5,linestyle="-",zorder=1) #Raw data
        plt.plot(W_tmp,I_bin,"-",color="magenta",zorder=2) #Best prediction
        
        plt.ylabel(r"I/I$_{\rm{med}}$")
        ax1.get_yaxis().set_label_coords(xlab,0.5)
        
        ### Tune axes
        yinf = np.median(Im) - 5*np.std(Im)
        ysup = np.median(Im) + 5*np.std(Im)
        plt.ylim(yinf,ysup)
        plt.xlim(Wm[0],Wm[-1])
        plt.xticks([])

        
        ax2 = plt.subplot(212)
        plt.plot(W_tmp,I_nor,"-",color="black",linewidth=0.7,zorder=1) #Normalized spectrum
        plt.axhline(1.0,linestyle="--",color="magenta",zorder=2,linewidth=0.5) # Unity line
        
        ### Tune axes
        yinf = np.median(I_nor) - 5*np.std(Im)
        ysup = np.median(I_nor) + 5*np.std(Im)
        plt.ylim(yinf,ysup)
        plt.xlim(Wm[0],Wm[-1])
        
        plt.ylabel("Normalized")
        ax2.get_yaxis().set_label_coords(xlab,0.5)
        
        plt.subplots_adjust(wspace=0.5,hspace = 0.)
        plt.show()
        

    return I_nor,W_tmp   
    
        


#################################################################################################


def plot_norm(Wm,Im,I_nor,I_pred,filtered_data,title=""):
    """
    Plot a given exposured normalized using the iterative polynomial fit (norm_pol fct)
    
    Inputs
    - Wm:            1D wavelength vector
    - Im:            1D exposure vector (same size as Wm)
    - I_nor:         Normalised exposure
    - I_pred:        Best prediction from the model
    - filtered_data: Modeled data after outlier removal
    
    Outputs:
    - Display figure with 2 panels showing respectively the best polynomial fit to the data and
      the resulting normalized spectrum
    """
    
    plt.rcParams["figure.figsize"] = (10,7)
    
    xlab = -0.1  #pad between ylabel and yaxis
    fig = plt.Figure()
    plt.title(title)

    ax1 = plt.subplot(211)
    ax1.plot(Wm,Im,color="black",linewidth=1.0,zorder=1) #Raw data
    ax1.plot(Wm,I_pred,color="magenta",linewidth=2.0,zorder=3,linestyle="--") #Best prediction
    ax1.plot(Wm,filtered_data,color="#495CFF",linewidth=0.5,linestyle="-",zorder=2) #Data after outlier removal
    
    ax1.set_ylabel(r"I/I$_{\rm{med}}$")
    ax1.get_yaxis().set_label_coords(xlab,0.5)
    ax1.set_xticks([])
    
    ### Tune limits of Yaxis
    yinf = np.median(Im) - 5*np.std(Im)
    ysup = np.median(Im) + 5*np.std(Im)
    ax1.set_ylim(yinf,ysup)
    
    
    ax2 = plt.subplot(212)
    ax2.plot(Wm,I_nor,color="black",linestyle="-",linewidth=0.7) #Normalized spectrum
    ax2.axhline(1.0,linestyle="--",color="magenta",zorder=2,linewidth=1.0) #Horizontal lien at 1

    ax2.set_ylabel("Normalized flux")
    ax2.get_yaxis().set_label_coords(xlab,0.5)    
    ax2.set_xlabel(r"$\lambda$ [nm]")
    
    ### Tune limits of Yaxis
    yinf = np.median(I_nor) - 5*np.std(I_nor)
    ysup = np.median(I_nor) + 5*np.std(I_nor)
    ax2.set_ylim(yinf,ysup)   
    
    plt.subplots_adjust(wspace=0.5,hspace = 0.)
    plt.show()
    

#################################################################################################    
    
    
def poly_fit(x,y,deg,sig_clip,n_iter=3):
    """
    Make iterative polynomial fit with outlier removal
    
    Inputs:
    - x:        xaxis (1D np.array vector)
    - y:        yaxis, data to be modelled (1D np.array vector, size: len(x))
    - deg:      degree of the polynomial fit
    - sig_clip: Pts deviation by more than sig_clip sigmas from the models are removed at each iteration
    - n_iter:   number of iteration of the iterative fit
    
    Outputs:
    - fitted_model:  best prediction of the polynomial fit
    - filtered_data: remaining data after outlier removal
    """
    
    pol_f     = polynomial.Polynomial1D(deg) ### Init polynom
    fit       = fitting.LinearLSQFitter()   ### Init optim method
    or_fit    = fitting.FittingWithOutlierRemoval(fit,sigma_clip,niter=n_iter,sigma=sig_clip) 
    
    or_fitted_model,mask = or_fit(pol_f,x,y)  ### Fit data
    
    ### Prediction
    filtered_data        = np.ma.masked_array(y,mask=mask)
    fitted_model         = fit(pol_f,x,y)
    
    return fitted_model,filtered_data







##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################





############################################################
# CLASS Observations                                       #
############################################################







class Observations:
    
    """
    Class Observations
    Store all data regarding the data reduction process
    """
    
    
    def __init__(self,name=""):
        self.name      = name
        
        self.list_name = []      ## List of the names of the observation
        
        self.date      = []      ## Date of each observation [BJD UTC]        

        self.W_raw     = []      ## 1D vector (N_wavelengths) of wavelengths from SPIRou DRS
        self.I_raw     = []      ## Matrix (N_obs,N_wavelengths) with Blaze-corrected exposures

        self.W_raw_pl  = []      ## 1D vector (N_wavelengths) of wavelengths from SPIRou DRS + synthetic planet
        self.I_raw_pl  = []      ## Matrix (N_obs,N_wavelengths) with Blaze-corrected exposures + synthetic planet
        
        self.airmass   = []      ## Airmass measurement for all exposures
        self.berv      = []      ## Barycentric Earth Radial Velocity [km/s]
        self.rv_s      = []      ## Stellar radial velocity [km/s]
        self.snr       = []      ## SNR from SPIRou Data Reduction Software (DRS)

        
    #######################################################################
        
    def read_exp(self,nam,lim=5.0):
        """
        Inputs:
        - nam: Name of the file containing the Stokes I exposures for the considered order
               Format: date[BJD] \n berv[km/s] \n rv_s[km/s] \n snr[DRS] \n airmass \n
                       Wavelength[nm] Itensity[Blaze-corrected]
        - lim: if intensity larger than lim, point not read
        
        Outputs:
        - date, berv, rv_s, snr, airmass stored as attributes of the class
        - return W,I -> read Wavelength[nm] Itensity[Blaze-corrected] (1D-arrays)
        """
        
        ### Read all content of the file
        Table = open(nam,'r')
        lines = Table.readlines()
        Table.close()
        Final_table = []
        for row in lines:
            split = row.rstrip('\r\n').split(' ')
            N = len(split)
            store = []
            for k in range(N):
                if split[k]=='':
                    store.append(k)
            for l in reversed(store):
                del(split[l])
            Final_table.append(split)
            
        ### Store useful info for each exposure in class attributes
        self.date.append(float(Final_table[0][1]))
        self.berv.append(float(Final_table[1][1]))
        self.rv_s.append(float(Final_table[2][1]))
        self.snr.append(float(Final_table[3][1]))
        self.airmass.append(float(Final_table[4][1]))
        
        ### Store flux and wavelength   
        ### lim: condition for outliers removal
        w,i = [],[]
        for m in range(5,len(Final_table)):
            if np.isfinite(float(Final_table[m][1])) and float(Final_table[m][1]) < lim:
                w.append(float(Final_table[m][0]))
                i.append(float(Final_table[m][1]))
                
        return w,i
    
    
    #######################################################################
        
    def read_data(self,rep,lim,planet=False):
        """
        Master function to read data
        1. Identify the exposures to read (in rep)
        2. Read each exposure using self.read_exp fct
        3. Store Wavelength and intensity vectors as attributes
        Inputs:
        - rep: directory of the exposures to read
        - lim: if an intensity is larger than this threshold, the point is not stored
        - planet: if True, file with synthetic planet signature. Store I in self.I_raw_pl
                  else, Store I in self.I_raw
        """
        
        print(termcolor.colored("Read data","blue"))
        
        ### Store the name of all the files to read
        while True:
            try:
                list_nam = os.listdir(rep)
                list_nam = sorted(list_nam)
                break
            except TypeError:
                tx = "Error - Path " + str(rep) + " does not exist"
                print(termcolor.colored(tx,"red"))

        ### Initialize all values before reading each exposure
        self.W_raw = []
        if planet:
            self.I_raw_pl = []
            print(termcolor.colored("Read data with signature of synthetic planet","yellow"))
        else:
            self.I_raw = []
            print(termcolor.colored("Read data - no planet","yellow"))
        self.airmass,self.date,self.rv_s,self.berv,self.snr = [],[],[],[],[]
        compt = 0
        
        ### Read all exposures in rep and store their content as attributes
        for nam in list_nam:
            
            nm = rep + "/" + nam
            w,i = self.read_exp(nm,lim) #Read exposure
            
            # Store data
            if planet: self.I_raw_pl.append(i)
            else: self.I_raw.append(i)
                
            if compt == 0: self.W_raw = w  ## same wavelength vector for all exposures from DRS
            compt += 1

        ### Convert as arrays - needed for further operations
        self.W_raw = np.array(self.W_raw,dtype=float)
        if planet: self.I_raw_pl = np.array(self.I_raw_pl,dtype=float)
        else: self.I_raw = np.array(self.I_raw,dtype=float)
        self.airmass = np.array(self.airmass,dtype=float)
        self.date    = np.array(self.date,dtype=float)
        self.berv    = np.array(self.berv,dtype=float)
        self.rv_s    = np.array(self.rv_s,dtype=float)
        self.snr     = np.array(self.snr,dtype=float)

        print(termcolor.colored("DONE","green"))


    #######################################################################

        
    def plot_infos(self):
        """
        Plot main information relative to the exposures read using self.read_data
        
        Return a Figure with 3 panels showing the airmass, snr (from DRS) and the velocity shift
        that need to be applied to move into the stellar rest frame (km/s) as a function of time
        """
        
        xlab = -0.1 # pad between label and yaxis
        
        fig  = plt.figure()
        
        ax1 = plt.subplot(311)
        plt.plot(self.date,self.airmass,"-",color="magenta")
        plt.ylabel("Airmass")
        plt.xticks([])
        ax1.get_yaxis().set_label_coords(xlab,0.5) 
        
        ax2 = plt.subplot(312)
        plt.plot(self.date,self.snr,".",color="black")
        plt.ylabel("SNR (DRS)")
        plt.xticks([])
        ax2.get_yaxis().set_label_coords(xlab,0.5) 
        
        ax3 = plt.subplot(313)
        plt.plot(self.date,self.rv_s-self.berv,"*",color="black")
        plt.ylabel(r"V$_{\rm{S}}$-V$_{\rm{BE}}$ [km/s]")
        plt.xlabel("Observation date")
        ax3.get_yaxis().set_label_coords(xlab,0.5) 
        
        plt.subplots_adjust(wspace=0.5,hspace = 0.)
        plt.show()
        
        
        
    #######################################################################
        
        
        
    def subtract_median(self,I,ind_ini=0,ind_end=0):
        """
        Compute the median spectrum along the time axis
        Divide each exposure by the median
        
        Inputs:
        - I: 2D matrix (N_exposures,N_wavelengths) from which median is computed
        - ind_ini, ind_end: if both == 0 or ind_end<=ind_ini: compute median on all spectra
          else: ind_ini,ind_end stand for the beginning and the end of the planetary transit respectively
                then median computed on the out-of-transit spectra only
                
        Outputs:
        - I_med: Median spectrum computed from I
        - I_sub: Matrix obtained by dividing each exposure in I by I_med (residual spectra)
        """
        
        if (ind_ini == 0 and ind_end == 0) or (ind_end >= ind_ini):
            # Compute median on all spectra along the time axis
            I_med = np.median(I,axis=0)
        else:
            # Compute median on out-of-transit spectra only
            I_out = np.concatenate((I[:ind_ini],I[ind_end:]),axis=0)
            I_med = np.median(I_out,axis=0)
            
        # Divide each spectrum in I by I_med
        I_sub = I/I_med
        
        return I_med,I_sub
    
 

    #######################################################################

  
    
    def norm_pol(self,W,I_sub,order=2,n_iter=4,sig_clip=5.0,plot=False,**kwargs):
        
        """
        Normalize each residual spectrum (median subtracted) using a polynomial fit
        The polynomial fit is computed using a sig_clip-sigma clipping to get rid of outliers
        
        Inputs:
        - W:        Wavelength vector (1D np.array)
        - I_sub:    Matrix of the residual exposures (2D: (N_exposures,len(W)))
        
        **kwargs
        - order:    Order of the polynomial fit (WARNING: to high values may intrduced unwanted structures)
        - n_iter:   Number of iteration of the polynomial fit with sigm-clipping
        - sig_clip: For each iteration of the polynomial fit, if a point deviates by more than sig_clip,
                    it is removed from the fit
        - plot:     if True: plot each normalised spectrum (make take time if many spectra)
        
        Outputs:
        - I_norm: Normalised spectrum
        
        """
    
        ### read kwargs
        for arg in kwargs.keys():
            if arg == "plot":     plot     = kwargs[arg]
            if arg == "sig_clip": sig_clip = kwargs[arg]
            if arg == "n_iter":   n_iter   = kwargs[arg]
            if arg == "order":    order    = kwargs[arg]

        
        I_norm_fin = []
        
        ### Apply norm_pol global function to normalize each exposure in I_sub
        for nn in range(len(I_sub)):
            title   = "Observation number " + str(nn) # Title of the plot, if plot==True
            I_norm  = norm_pol(W,I_sub[nn],order,n_iter,sig_clip,plot,title)
            I_norm_fin.append(I_norm)
        return np.array(I_norm_fin,dtype=float)
    
 
    #######################################################################   
    
    
    def norm_mav(self,W,I_sub,N_best,N_bor,plot=False):
        
        """
        Normalize each residual spectrum using a moving average process (calling the function 'moving_average')
        Act as a low-pass filter more efficient than polynomial fit for low frequency structures
        But some risk to introduce undesired variations in spectra
        
        Inputs:
        - W Wavelength vector
        - I_sub: Matrix of the residual exposures after median-subtraction; shape: (N_exp,len(W))
        - N_best: Number of points using to compute the average
        - N_bor:  Number of points removed at each edge of each spectrum to avoid border effects
                  Points on each edge of each ordre tend to be highly polluted by noise
        - plot: if True, plot each normalized spectrum (may take time)
        
        Outputs:
        - I_norm_fin: Normalised np.array; shape: (N_exp, len(W)-2*N_bor)
        - W_tmp: 1D wavelength vector obtained by removing N_bor pts at each edge of W (len: len(W)-2*N_bor)
        """
        
        I_norm_fin = []
        for nn in range(len(I_sub)):
            title          = "Observation number " + str(nn)
            I_norm, W_tmp  = moving_average(W,I_sub[nn],N_best,N_bor,plot,title)
            I_norm_fin.append(I_norm)
        return np.array(I_norm_fin,dtype=float),W_tmp       
    
    
    
    #######################################################################    
    
    
    def detrend_airmass(self,W,I_norm,deg=2,log=False,plot=False):
        
        """
        Detrend normalized spectra with airmass
        Goal: remove residuals of tellurics not subracted when dividing by median spectrum
        Use least square estimator (LSE) to estimate the components of a linear (or log) model of airmass
        
        Inputs:
        - W:      1D wavelength vector
        - I_norm: 2D matrix of normalised spectra (N_exposures,len(W))
        - deg:    degree of the linear model (e.g. I(t) = a0 + a1*airmass + ... + an*airmass^(n))
        - log:    if 1 fit log(I_norm) instead of I(t)
        - plot:   if true, plot the components removed with this model
        
        Outputs:
        - I_m_tot: Sequence of spectra without the airmass detrended component        
        """
        
        if log: 
            I_tmp = np.log(I_norm)
        else:
            I_tmp = I_norm - 1
                    
        ### Covariance matrix of the noise from DRS SNRs
        COV_inv = np.diag(self.snr**(2))
            
        ### Apply least-square estimator
        X = []
        X.append(np.ones(len(I_tmp)))
        for k in range(deg):
            X.append(self.airmass**(k+1))        
        X        = np.array(X,dtype=float).T
        A        = np.dot(X.T,np.dot(COV_inv,X))
        b        = np.dot(X.T,np.dot(COV_inv,I_tmp))
        I_best   = np.dot(np.linalg.inv(A),b)
        
        ### Plot each component estimated with LSE
        if plot:
            fig = plt.figure()
            ax  = plt.subplot(111)
            c   = 0
            col = ["red","green","magenta","cyan","blue","black","yellow"]
            for ii in I_best:
                lab = "Order " + str(c)
                alp = 1 - c/len(I_best)
                plt.plot(W,ii,label=lab,color=col[c],zorder=c+1,alpha=alp)
                c += 1
                if c == 7: break
            plt.legend()
            plt.title("Components removed - airmass detrending")
            plt.xlabel(r"$\lambda$ [nm]")
            plt.ylabel("Residuals removed")
            plt.show()
        
            
        if log:
            I_m_tot  = I_tmp - np.dot(X,I_best)
            return np.exp(I_m_tot)
        else:
            I_m_tot  = I_tmp + 1 - np.dot(X,I_best)
            return I_m_tot
        
        
    #######################################################################        
        
        
    def px_std_distrib(self,Wm,I,sig_clip=5.0,deg=2,n_iter=4,plot=False):
        
        """
        - Compute the standard deviation of each pixel (using a sigma clipping process)
        - Make iterative polynomial fit with outlier removal of the resulting distribution (fct make_polyfit)
        - Keep only pixels not rejected in the previous step
        - Plot the resulting distribution
        
        Inputs:
        - Wm:       1D vector of wavelengths
        - I:        2D array of all the spectra within the sequence
        - sig_clip: pts deviating by more than sig_clip from model are rejected
        - deg:      degree of the polynomial fit
        - n_iter:   Number of iteration for the iterative polynomial fit
        - plot:     if True, plot the resulting distribution of std for each px and best fit
        
        Outputs:
        - W_filt: resulting wavelength vector (without outliers)
        - I_filt: resulting array of spectra without pixels rejected by the polynomial fit
        """
        
        ### Compute the standard deviation of each pixel along the time axis
        std_px = np.zeros(len(Wm))
        for k in range(len(Wm)):
            r = sigma_clip(I[:,k],sigma=sig_clip,\
                           maxiters=None,cenfunc='median', masked=True, copy=True)
            std_px[k] = r.std()
            
        ### Make iterative polynomial fit with outlier removal
        mod_px,filt_px = poly_fit(Wm,std_px,deg,sig_clip,n_iter)
        pred_px        = mod_px(Wm)
        
        ### Remove pixels deviation by more than sig_clip from the best model
        ind_px = []
        for n in range(len(filt_px)):
            if filt_px[n] != "--": ind_px.append(n)
            elif std_px[n] < pred_px[n]: ind_px.append(n)
        W_filt,I_filt  = Wm[ind_px],I[:,ind_px]
        
        ### If plot == True: display px/px std distrib and best fit
        if plot:
            fig = plt.figure()
            ax  = plt.subplot(111)
            plt.plot(Wm,std_px,".",color="blue") #Distribution of the std for each pixel
            plt.plot(Wm,pred_px,color="green",linestyle="--",linewidth=2,zorder=3) #Best prediction from the model
            plt.plot(Wm[ind_px],std_px[ind_px],"*",color="red",zorder=2,markersize=2.5)#Std of each px without outliers
            
            title = "rms per pixel along the time axis"
            plt.title(title)
            plt.xlabel(r"$\lambda$ [nm]")
            plt.ylabel(r"$\sigma$")
            plt.show()
            
            
        return W_filt,I_filt

    
    #######################################################################   

    def spec_std_distrib(self,Wm,I,N_mid=400,sig_clip=5):
        """
        - Compute the standard deviation of each pixel (using a sigma clipping process)
        - Make iterative polynomial fit with outlier removal of the resulting distribution (fct make_polyfit)
        - Keep only pixels not rejected in the previous step
        - Plot the resulting distribution
        
        Inputs:
        - Wm:       1D vector of wavelengths
        - I:        2D array of all the spectra within the sequence
        - N_mid:    Number of points taken from the mean wavelength to compute the std of the spectrum
                    WARNING: computed on I[id_mid-N_mid:id_mid+N_mid]
        - sig_clip: pts deviating by more than sig_clip from model are rejected
        
        Outputs:
        - Display figure showing the standard deviation of each spectrum (compared with standard deviation from DRS)
        - Display the median standard deviation on the spectra (and compare it with DRS)
        """

        ### compute index of mean wavelength
        W_mean = 0.5*(Wm[0]+Wm[-1])
        id_mid = np.argmin(np.abs(Wm-W_mean))
        
        ### Compute standard deviation on the center of each exposure I[id_mid-N_mid:id_mid+N_mid]
        std_sp = np.zeros(len(I))
        for n in range(len(I)):
            r = sigma_clip(I[n,id_mid-N_mid:id_mid+N_mid],sigma=3.0,\
                           maxiters=None,cenfunc='median', masked=True, copy=True)
            std_sp[n] = r.std()
            
        ### Compute and print median standard deviation of the sequence
        tx  = "Median STD DRS: " + str(np.median(1./self.snr)) + "\n"
        tx += "Median STD reduced sequence: " + str(np.median(std_sp))
        print(termcolor.colored(tx,"yellow"))

        ### Plot resulting distribution of SNR for each spectrum
        fig = plt.figure()
        ax  = plt.subplot(111)
        plt.plot(self.date,std_sp,"*",linewidth=0.,color="black",zorder=3) #STD of each spectrum
        plt.plot(self.date,1./np.array(self.snr),".",color="#556627") #STD from DRS
        
        title = "STD per spectrum of the sequence"
        plt.title(title)
        ax.set_xlabel(r"Orbital phase")
        
        ### Plot airmass for comparison
        ax2 = ax.twinx()
        ax2.plot(self.date,self.airmass,linewidth=1.,color='magenta',label="Airmass",zorder=1)
        ax2.tick_params('y', colors='magenta')
        ax2.set_ylabel('Airmass', color='magenta')
        plt.show()
        
        

                    






##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################




























