'''
Read in OH dust opacity tables :)
'''
import numpy as np
import astropy
from astropy import units as u
import astropy.constants as const
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate

class OH_1994_dust(object):

    def __init__(self,dust=None):
        self.path='OH_tables/'
        files={'mrn8':'OH_mrn8.dat',
                'mrn7':'OH_mrn7.dat',
                'mrn6':'OH_mrn6.dat',
                'mrn5':'OH_mrn5.dat',
                'mrn':'OH_mrn.dat',
                'thin_mrn8':'OH_thin_mrn8.dat',
                'thin_mrn7':'OH_thin_mrn7.dat',
                'thin_mrn6':'OH_thin_mrn6.dat',
                'thin_mrn5':'OH_thin_mrn5.dat',
                'thin_mrn':'OH_thin_mrn.dat',
                'thick_mrn8':'OH_thick_mrn8.dat',
                'thick_mrn7':'OH_thick_mrn7.dat',
                'thick_mrn6':'OH_thick_mrn6.dat',
                'thick_mrn5':'OH_thick_mrn5.dat',
                'thick_mrn':'OH_thick_mrn.dat'} #dictionary containign keys and filenames of dust types
        self.files=files
        if dust is not None:
            self.calculate_model(dust)

    def calculate_model(self,dust):
        df=pd.read_csv(self.path+self.files[dust],sep='\s+',names=['wavelength','absorption cross section'])
        wl=df['wavelength'].to_numpy(dtype=float)*u.micrometer
        freq=(const.c / wl).to('Hz')
        kappa=df['absorption cross section'].to_numpy(dtype=float)*u.cm*u.cm / u.g
        kappa_SI=kappa.si
        self.dust_opacities=kappa_SI
        self.freq=freq
        self.wavelength=wl
        return freq,kappa_SI

    #add function here to show what dust models are available?
    #def dust_available(self):



    def plot_opacities(self):
        x=self.freq
        y=self.dust_opacities
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot()
        ax.scatter(x,y)
        ax.set_title('Dust opacities')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\nu$ [Hz]')
        ax.set_ylabel(r'$m^2 kg^{-1}$',labelpad=10.0,rotation='horizontal')

    def interpolate(self):
        #by interpolating the data in the specified range,
        #get kv as fucntion of v.
        #fucntion to plot interpolation function
        x = self.freq
        y = self.dust_opacities
        x=np.append(x,1.0e9*u.Hz)
        y=np.append(y,(((0.0)*u.cm*u.cm / u.g)).si)
        f = interpolate.interp1d(x, y)
        self.interp_func=f

    def plot_interp(self):
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot()
        ax.set_title('Interpolated function')
        x = self.freq
        y = self.dust_opacities
        x=np.append(x,1.0e9*u.Hz)
        y=np.append(y,(((0.0)*u.cm*u.cm / u.g)).si)
        ax.scatter(x,y)
        new_x=np.linspace(np.amin(x),np.amax(x),100000)
        ax.plot(new_x,self.interp_func(new_x))
        ax.set_xscale('log')
        ax.set_xlabel(r'$\nu$ [Hz]')
        ax.set_ylabel(r'$m^2 kg^{-1}$',labelpad=10.0,rotation='horizontal')
    
    def calculate(self,nu):
        kappa_nu=self.interp_func(nu)
        return kappa_nu
    
    def plot_all(self,keys=None):
        if keys is None:
            keys=self.files
        fig = plt.figure(figsize=(10,8))
        ax = plt.subplot()
        ax.set_title('All models')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\nu$ [Hz]')
        ax.set_ylabel(r'$m^2 kg^{-1}$',labelpad=10.0,rotation='horizontal')
        for key in keys:
            freq,kappa=self.calculate_model(key)
            self.interpolate()
            ax.scatter(freq,kappa)
            new_x=np.linspace(np.amin(freq),np.amax(freq),100000)
            ax.plot(new_x,self.interp_func(new_x),label=key)
        ax.legend()



        
    