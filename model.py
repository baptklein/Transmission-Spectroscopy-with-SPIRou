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
import batman

import matplotlib.pyplot as plt
import matplotlib



font = {'size'   : 18,
        'weight': 'light'}
axes = {'labelsize': 18,
        'labelweight': 'light'}

matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)






class Model:
    
    """
    Class model
    Store the templates of transmission spectrum of the planet atmosphere
    """
    
    
    def __init__(self):
        
        
        self.template = {} #Dictonary of templates: key = name of the model (e.g. 'template_T1100')
                           #Value = 1D template of planet transmission spectrum
        
        self.Vm = []       #1D velocity vector of the model
        self.Fm = []
        
        
        
        
        
    def read(self,nam):
        """
        Inputs:
        - nam: Name of the template of planet transmission spectrum to read
        
        Outputs:
        - v: velocity vector (1D array)
        - i: template (1D array)
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
            
        ### Store flux and wavelength   
        v,i = [],[]
        for m in range(4,len(Final_table)):
            v.append(float(Final_table[m][0]))
            i.append(float(Final_table[m][1]))  
        return np.array(v,dtype=float),np.array(i,dtype=float)
    
    
        
        
        
        
    def read_models(self,rep_mod):

        """
        Read all the templates of planet transmission spectra stored in rep_mod

        Inputs:
        - rep_mod: Directory containing the template spectra
        
        Outputs:
        Make attributes of the class:
        - self.Vm:       velocity vector of the templates (densely sampled, R~10^6) 
        - self.template: Dictionary of all the templates read from rep_mod directory. The key to access a template is the name
                         of the file without extension (e.g. 'template_T800')

        """
        
        
        print(termcolor.colored("Read models","blue"))
        
        ### Store the name of all the files to read
        while True:
            try:
                list_nam = os.listdir(rep_mod)
                list_nam = sorted(list_nam)
                break
            except TypeError:
                tx = "Error - Path " + str(rep_mod) + " does not exist"
                print(termcolor.colored(tx,"red"))

        ### Initialize all values before reading each exposure
        self.Vm       = []
        self.templace = {}
        compt = 0
        
        ### Read all templates in rep and store their content as attributes
        for nam in list_nam:
            
            nm = rep_mod + "/" + nam
            v,i = self.read(nm)    #Read template

            if compt == 0: self.Vm = v  ## same velocity vector for all templates
            compt += 1
            
            ### Build dictionary
            key = nam[:-4]
            self.template[key] = i
            print(key)
            
        print(termcolor.colored("DONE","green"))


    def plot_template(self,template_name):


        I_t   = self.template[template_name]

        print(I_t)
        
        fig   = plt.figure()
        plt.title(template_name)
        plt.plot(self.Vm,np.array(I_t,dtype=float),linewidth=0.5,color="black")
        plt.xlabel(r"Velocity [km/s]") 
        plt.ylabel("Template")
        plt.show()
        
    
    
    def build_model(self,template_name,window):
        """
        Create a 2D sequence template
        Interpolate each template         
        """
        
        I_t   = self.template[template_name]
        I_mod = []
        for nn in range(len(window)):
            I_line = I_t*window[nn]
            I_mod.append(I_line)
        I_mod = np.array(I_mod,dtype=float)
                
        # Compute interpolation of each line of the model
        FM = []
        for n in range(len(window)):
            f_mod = interpolate.interp1d(self.Vm,I_mod[n],kind='linear')
            FM.append(f_mod)
        self.Fm = np.array(FM)
        return self.Fm
        
class Planet:
    
        
    def __init__(self,name):
        
        self.name  = name
        
        self.rp    = 0.0  ## Planet radius in stellar radius unit [R_s]
        self.inc   = 0.0  ## Inclination of the planetary transit [deg]
        self.t0    = 0.0  ## Mid-transit time [phase]
        self.a     = 0.0  ## SMA [R_s]
        self.per   = 0.0  ## Orbital period [d]
        self.ecc   = 0.0  ## eccentricity
        self.w     = 0.0  ## long. of periastrion [deg]
        self.ld    = ""   ## limb darkening model -- batman
        self.u     = []   ## factors for limb darkening
        
        self.date   = []   ## Phase vector
        self.batman = ""   ## Batman transit model
        self.flux   = []   ## Normalized light curve
        self.window = []   ## Window function from transit curve
        
        
    def make_batman(self):
        
        ### See https://www.cfa.harvard.edu/~lkreidberg/batman/
        ### Init
        params  = batman.TransitParams()
        params.rp        = self.rp                       
        params.inc       = self.inc
        params.t0        = self.t0   
        params.a         = self.a    
        params.per       = self.per  
        params.ecc       = self.ecc 
        params.w         = self.w          
        params.limb_dark = self.ld
        params.u         = self.u
        
        ### Make
        self.batman = batman.TransitModel(params,self.date)
        self.flux   = self.batman.light_curve(params)
  
    def make_window(self,plot=True):
        
        ### Build window
        FF          = 1.-self.flux
        self.window = FF/np.max(FF)
        
        ### Plot window
        if plot:
            plt.plot(self.date,self.window,color="black",linestyle="-",linewidth=1.0)
            plt.title("Weighting window for the transmission spectrum")
            plt.xlabel("Time [d]")
            plt.ylabel("Window")
            plt.show()
            
            
    def RVP(self,kp,v0):
        V1 = kp*np.sin(2*np.pi*(self.date-self.t0)/(self.per))+v0     
        return V1
        
