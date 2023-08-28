import numpy as np 
import astropy.constants as const
import matplotlib.pyplot as plt
import lmfit
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from regions import Regions
from radio_beam import Beam
#from sgrb2_functions import plot_hifi
import pandas as pd
from reproject import reproject_interp
from reproject import reproject_exact
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
import matplotlib.colors as colors
from matplotlib.backend_bases import MouseButton
import multiprocessing as mp
from regions import RectanglePixelRegion
from regions import Regions
from regions import CircleSkyRegion
import emcee
from lmfit import Parameter
#class for fitting

'''
To do:

1. Proper treatment of units for frequencies and units (astropy quantities)

2. Add proper descriptions of each function

3. Add wavelength plotting option
'''
class SED_fitting(object):

    def __init__(self,init_T,init_beta,init_N,df=pd.DataFrame(),dust='power',T_range=[2.73,2000], beta_range=[0.0,3.0], N_range=[1e24,1e30],T_fixed=False,beta_fixed=False,N_fixed=False):
        '''
        Initialise a modified black body object with initial guesses of Temp, column density and Beta
        add Hydrogen mass,mean molecular weight,gas-to-dust mass ratio and reference dust opacity as fixed 
        everything is SI
        '''
        #non-free parameters
        mH=1.6735575e-27 #Hydrogen mass in Kg
        mu=2.8     #mean molecular weight of hydrogen.
        chi_d=100  #gas to dust mass ratio
        kappa0=0.1773 #converted to m^2/kg #reference dust oppacity in cm^2 g^-1 ***need to choose better value***
        nu0=600e9      #reference frequency for reference dust opacity (Hz)

        self.init_T=init_T
        self.init_N=init_N
        self.init_beta=init_beta
        import lmfit
        variables=[init_T, init_beta, init_N,mH,mu,chi_d,kappa0,nu0]
        names=['T','beta','N','mH','mu','chi_d','kappa0','nu0']
        parameters = lmfit.Parameters()
        for i, v in enumerate(variables):
            parameters.add(names[i],value=v)


        parameters['beta'].vary = not beta_fixed
        parameters['beta'].min = beta_range[0]
        parameters['beta'].max = beta_range[1]

        parameters['T'].vary = not T_fixed
        parameters['T'].min = T_range[0]
        parameters['T'].max = T_range[1]

        parameters['N'].vary = not N_fixed
        parameters['N'].min = N_range[0]
        parameters['N'].max = N_range[1]

        parameters['mH'].vary = False
        parameters['mu'].vary = False
        parameters['chi_d'].vary = False
        parameters['kappa0'].vary = False
        parameters['nu0'].vary = False

        self.parameters = parameters

        #which dust function to use
        self.dust = dust

        self.toy_models=pd.DataFrame(columns=['T','beta','N','label'])
        self.df=df #data frame containing the files,paths,colors,labels and whether or not to fit.

    
    def fit_mod_blackbody(self,method,region_file=None,fluxes=None,frequencies=None,yerr=None):
        if method=='chi2':
            self.fit_mod_blackbody_chi(region_file,fluxes,frequencies,yerr,method='lm')
        elif method=='MCMC':
            self.fit_mod_blackbody_MCMC(region_file,fluxes,frequencies,yerr)
        elif method=='UltraNest':
            self.fit_mod_blackbody_UN(region_file,fluxes,frequencies,yerr)

    
    def fit_mod_blackbody_chi(self,region_file=None,fluxes=None,frequencies=None,yerr=None,method='lm'): 
        import lmfit
        #fit modified blackbody to a single pixel, i.e a 1D array of fluxes and frequencies.
        #make self.vals/self.errors attributes with T,NH,Beta values and errors to be returned.
        #How will we calculate errors?
        if fluxes is not None:
            self.df['frequencies']=frequencies
            self.df['fluxes']=fluxes
            self.df['yerr']=yerr
        if fluxes is None:
            self.calculate(region_file)
            fluxes=self.df.loc[self.df['fit?'] == True, 'fluxes']
            frequencies=self.df.loc[self.df['fit?'] == True, 'frequencies']
            yerr=self.df.loc[self.df['fit?'] == True, 'yerr']
            #yerr=fluxes*0.1
        #print(fluxes)
        parameters=self.parameters
        to_fit=self.other_mod_blackbody
        lm=lmfit.minimize(to_fit,self.parameters,args=(frequencies,fluxes,yerr),method=method,nan_policy='omit')
        self.lm = lm
        self.vals = (lm.params['T'].value, lm.params['beta'].value,
                     lm.params['N'].value)
        self.errors = (lm.params['T'].stderr, lm.params['beta'].stderr,
                     lm.params['N'].stderr)
        self.parameters['T']=lm.params['T']
        self.parameters['beta']=lm.params['beta']
        self.parameters['N']=lm.params['N']
        return self.vals,self.errors
    
    def fit_mod_blackbody_MCMC(self,region_file=None,fluxes=None,frequencies=None,yerr=None):
        if fluxes is not None:
            self.df['frequencies']=frequencies
            self.df['fluxes']=fluxes
            self.df['yerr']=yerr
        if fluxes is None:
            self.calculate(region_file)
            fluxes=self.df.loc[self.df['fit?'] == True, 'fluxes']
            frequencies=self.df.loc[self.df['fit?'] == True, 'frequencies']
            yerr=self.df.loc[self.df['fit?'] == True, 'yerr']

        solnx=np.array([self.init_T,np.log10(self.init_N),self.init_beta])
        pos = solnx +  1.0e-3 * np.random.randn(70,3)
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_uniform, args=(self.df['frequencies'], self.df['fluxes'], self.df['yerr']) )
        sampler.run_mcmc(pos, 10000, progress=True);    
        flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
        T = np.percentile(flat_samples[:, 0], [16, 50, 84])
        logN = np.percentile(flat_samples[:, 1], [16, 50, 84])
        beta = np.percentile(flat_samples[:, 2], [16, 50, 84])
        print(T,logN,beta)

        self.parameters['T']=Parameter(name='T',value=T[1])
        self.parameters['beta']=Parameter(name='beta',value=beta[1])
        self.parameters['N']=Parameter(name='N',value=logN[1])

        self.vals=T[1],logN[1],beta[1]
        self.errors_lo=T[0],logN[0],beta[0]
        self.errors_hi=T[2],logN[2],beta[2]
        return self.vals,self.errors_lo,self.errors_hi
    
    def fit_mod_blackbody_UN(self,region_file=None,fluxes=None,frequencies=None,yerr=None):
        return self.vals,self.errors
    
    def calculate(self,region_file):
        my_paths=self.df['path']
        my_files=self.df['file']
        maps = my_paths+my_files
        maps=maps.values
        #print(maps)
        fluxes=[]
        freqs=[]
        for j, map in enumerate(maps):
                #call calculate intensity function
                int,freq=calculate_intensity(map,region_file)
                fluxes.append(int)      #in K
                freqs.append(freq)      #in Hz
        fluxes=np.asarray(fluxes)
        freqs=np.asarray(freqs)
        yerr=fluxes*0.1

        #print(fluxes)
        self.df['fluxes']=fluxes
        self.df['frequencies']=freqs
        self.df['yerr']=yerr
        return

    def plot_fit(self,savefile=None,frequencies=None,fluxes=None,yerr=None,toy_models=False):
        if fluxes is not None:
            self.df['frequencies']=frequencies
            self.df['fluxes']=fluxes
            self.df['yerr']=np.abs(yerr)
        else:
            fluxes=self.df['fluxes']
            frequencies=self.df['frequencies']
            yerr=np.abs(self.df['yerr'])
        
        
        fake_frequencies=np.linspace(1.0e14,1.0e10,1000000,dtype=np.float128)
        fig=plt.figure(figsize=(15,10))
        ax=fig.add_subplot(111)
        if 'label' in self.df.columns:
            groups = self.df.groupby(['label', 'color','fit?'])
            for name,group in groups:
                label,color,fit=name
                if fit == True:
                    marker = 'o'
                else:
                    marker = '*'
                ax.errorbar(group['frequencies'],group['fluxes'],group['yerr'],xerr=None,fmt=marker,color=color,label=label)
        else:
            ax.errorbar(frequencies,fluxes,yerr,xerr=None,fmt='bo',label='data')
        #ax.errorbar(other_freqs,other_fluxes,other_yerr,fmt='ro')
        ax.plot(fake_frequencies,self.mod_blackbody(fake_frequencies))
        if toy_models == True:
            print(self.toy_models)
            for i,row in self.toy_models.iterrows():
                print(row['T'],row['beta'],row['N'],row['label'])
                T,beta,N=float(row['T']),float(row['beta']),float(row['N'])
                if row['label'] is None:
                    label='T='+str(row['T'])+',beta='+str(row['beta'])+',N='+str(row['N'])
                else:
                    label=row['label']
                print(row['T'],row['beta'],row['N'])
                ax.plot(fake_frequencies,self.toy_mod_blackbody(fake_frequencies,T,beta,N),label=label,ls='--')
                #print(self.toy_mod_blackbody(fake_frequencies,row['T'],row['beta'],row['N']))
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e6)
        ax.set_xlabel(r'$\nu$ [Hz]')
        ax.set_ylabel(r'$S_{\nu}$ [Jy/sr]')
        #plot_hifi(fig,ax=ax)
        #add text to plot with T,Beta, and N
        self.add_text_to_plot(ax)
        if savefile is not None:
            plt.savefig(savefile,dpi=300)
        return
    

    def add_text_to_plot(self,ax):
        #Maybe make this a bit sexier, its fine for now
        T=str(np.around(self.parameters['T'].value,1))+' '+r'K'
        B=str(np.around(self.parameters['beta'].value,1))
        N=str(np.format_float_scientific(((self.parameters['N'].value)/u.m/u.m).to(1/u.cm/u.cm).value,2))+' '+r'cm^{-2}'
        text = fr"$T = {T}$" + "\n" + fr"$\beta = {B}$" + "\n" + fr"$N = {N}$"
        ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=12, va='top', ha='left')
    
    def add_toy_model(self,T,N,beta,label=None):
        arr=np.array([[T,beta,N,label]])
        df=pd.DataFrame(arr,columns=['T','beta','N','label'])
        #self.toy_models=self.toy_models.append(df)
        self.toy_models=pd.concat([self.toy_models,df],ignore_index=True)

    '''Function to fit blackbody to each set of pixels
        Take N maps as input. Maps should have same pixel scales and be smoothed to a common resolution. Also same number of pixels/or find overlap region. 
        self is a sed object for initial guesses. Could adapt to a map of initial guesses'''
    
    def mapfit2(self,maps,yerr):
        #non parallel func
        hdul=fits.open(maps[0])
        data=hdul[0].data
        wcs=WCS(hdul[0].header)
        Tmap=np.empty_like(data)
        Nmap=np.empty_like(data)
        betamap=np.empty_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ints=[]
                freqs=[]
                for k,map in enumerate(maps):
                    int,freq=pixel_intensity(map,i,j)
                    ints.append(int)
                    freqs.append(freq)
                ints=np.array(ints)
                freqs=np.array(freqs)
                yerr=yerr
                self.fit_mod_blackbody(ints,freqs,yerr)
                print('....fitting_pixel.......',i,j)
                Tmap[i,j]=self.parameters['T'].value
                Nmap[i,j]=self.parameters['N'].value  
                betamap[i,j]=self.parameters['beta'].value
        results=[Tmap,Nmap,betamap]
        #need to add wcs and save as fits? 
        for l,result in enumerate(results):
            fig=plt.figure(figsize=(10,10))
            ax=plt.subplot(projection=wcs)
            im=ax.imshow(result,cmap='inferno')
            fig.colorbar(im,ax=ax)
            #ax.title(str(l))
            plt.savefig('plots/mapfit'+str(l)+'.png')

    def process_pixel(self,maps,yerr, i, j):
        if hasattr(SED_fitting, "mask"):
            if self.mask[i,j] == 0:
                output=np.nan,np.nan,np.nan
            else:
                ints = []
                freqs = []
                for k, map in enumerate(maps):
                    int, freq = pixel_intensity(map, i, j)
                    ints.append(int)
                    freqs.append(freq)
                ints = np.array(ints)
                freqs = np.array(freqs)
                yerr = yerr
                self.fit_mod_blackbody(fluxes=ints, frequencies=freqs, yerr=yerr)
                print('....fitting_pixel.......', i, j)
                output=self.parameters['T'].value, self.parameters['N'].value, self.parameters['beta'].value
        else:
                ints = []
                freqs = []
                for k, map in enumerate(maps):
                    int, freq = pixel_intensity(map, i, j)
                    ints.append(int)
                    freqs.append(freq)
                ints = np.array(ints)
                freqs = np.array(freqs)
                yerr = ints * 0.1
                self.fit_mod_blackbody(fluxes=ints, frequencies=freqs, yerr=yerr)
                print('....fitting_pixel.......', i, j)
                output=self.parameters['T'].value, self.parameters['N'].value, self.parameters['beta'].value
        return output

    def mapfit(self,maps,yerr,cores,outpath=''):
        hdul = fits.open(maps[0])
        data = hdul[0].data
        wcs = WCS(hdul[0].header).celestial
        Tmap = np.empty_like(data)
        Nmap = np.empty_like(data)
        betamap = np.empty_like(data)
        num_cores = np.minimum(mp.cpu_count()-1,cores)  # Number of CPU cores
        pool = mp.Pool(num_cores)  # Create a pool of worker processes
        
        results = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                    results.append(pool.apply_async(self.process_pixel, args=(maps,yerr, i, j)))
        
        # Get the results from the worker processes
        for i, res in enumerate(results):
            Tmap[i % data.shape[0], i // data.shape[0]], Nmap[i % data.shape[0], i // data.shape[0]], betamap[i % data.shape[0], i // data.shape[0]] = res.get()

        
        # Close the pool of worker processes
        pool.close()
        pool.join()
        
        results = [Tmap, Nmap, betamap]
        for l,result in enumerate(results):
            if hasattr(self, 'mask'):
                result = np.where(self.mask == 0, np.nan, result)
            fig=plt.figure(figsize=(10,10))
            ax=plt.subplot(projection=wcs)
            im=ax.imshow(result,cmap='inferno')
            fig.colorbar(im,ax=ax)
            #ax.title(str(l))
            plt.savefig('plots/mapfit'+outpath+str(l)+'.png')
            new_hdu=fits.PrimaryHDU(result,wcs.to_header())
            new_hdu.writeto('plots/'+outpath+str(l)+'.fits',overwrite=True)

    
    def interacive_plot(self,ref_image,maps):
        self.maps=maps
        image=fits.open(ref_image)
        image_data=extract_dimensions(image[0].data)
        wcs=WCS(image[0].header).celestial
        fig = plt.figure(figsize=(10,10))
        ax = plt.subplot(projection=wcs)#,slices=('x','y',0,0))
        im=ax.imshow(image_data, origin='lower', cmap='viridis', norm=colors.PowerNorm(gamma=0.5,vmin=0.005)) 
        ax.set_xlabel(r'RA')
        ax.set_ylabel(r'Dec')
        ax.set_title('reference image')
        #self.fig=fig

        self.cid=fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show()
        
        

    def on_click(self, event):
        #also make onclick close the first image.
        if event.button == 1:  # Left mouse button
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            print('Coordinates selected: ', x, ' ',y)
            #plt.close(self.fig)
            self.handle_click(x, y)
    
    def handle_click(self,x,y):
        savefile='interactive_Result.png'
        ints=[]
        freqs=[]
        for map in self.maps:
            #print(map)
            int,freq=pixel_intensity(map,x,y)
            ints.append(int)
            freqs.append(freq)
        ints=np.asarray(ints)
        freqs=np.asarray(freqs)
        yerr=ints*0.1
        #print(ints)
        #print(freqs)
        #print(yerr)

        
        self.fit_mod_blackbody(fluxes=ints,frequencies=freqs,yerr=yerr)
        self.plot_fit(savefile,frequencies=freqs,fluxes=ints,yerr=yerr)
        plt.show()



    def mod_blackbody(self,nu):
        parameters=self.parameters
        h=const.h.value
        c=const.c.value
        k_B=const.k_B.value
        T=parameters['T'].value
        tau=self.optical_depth(nu)
        B_nu=((2.*h)*(nu**3.) / (c**2.)) / (np.exp((h*nu) / (k_B*T))-1.0)
        mod_bb=B_nu*(1.0-np.exp(-1.0*tau))*(10.0**26) 
        return mod_bb

    def other_mod_blackbody(self,parameters,nu,fluxes,yerr):
        self.parameters=parameters
        mod_bb=self.mod_blackbody(nu)
        y=(fluxes-mod_bb)/yerr
        return y

    def dust_opacity(self,nu,beta=None):
        if self.dust == 'power':
            parameters=self.parameters
            if beta is None:
                beta=parameters['beta']
            else:
                beta=beta
            kappa0=parameters['kappa0']
            chi_d=parameters['chi_d']
            nu0=parameters['nu0']
            #print(beta,type(beta))
            kappa_nu=(kappa0/chi_d)*((nu/nu0)**beta)
        else:
            from OH_tables import OH_1994_dust
            dust=OH_1994_dust(self.dust)
            dust.interpolate()
            kappa_nu=dust.calculate(nu)
        return kappa_nu

    def optical_depth(self,nu,beta=None,N=None):
        parameters=self.parameters
        if beta is None:
            kappa_nu=self.dust_opacity(nu)
        else:
            kappa_nu=self.dust_opacity(nu,beta)
        mu=parameters['mu'].value
        m_H=parameters['mH'].value
        if N is None:
            N=parameters['N'].value
        else:
            N=N
        tau=mu*m_H*kappa_nu*N
        return tau

    def toy_mod_blackbody(self,nu,T,beta,N):
        parameters=self.parameters
        h=const.h.value
        c=const.c.value
        k_B=const.k_B.value
        #kappa0=parameters['kappa0'].value
        #chi_d=parameters['chi_d'].value
        #nu0=parameters['nu0'].value
        #mu=parameters['mu'].value
        #m_H=parameters['mH'].value
        #nu_frac=np.divide(nu,nu0)
        #print(nu_frac)
        #nu_frac_power=np.power(nu_frac,beta)
        #kappa_nu=(kappa0/chi_d)*nu_frac_power
        tau=self.optical_depth(nu,beta,N)
        B_nu=((2.*h)*(nu**3.) / (c**2.)) / (np.exp((h*nu) / (k_B*T))-1.0)
        mod_bb=B_nu*(1.0-np.exp(-1.0*tau))*(10.0**26) 
        return mod_bb
    
    def add_mask(self,mask):
        self.mask=mask
        return
    
##### EMCEE functions

def mod_bb_emcee(nu,T,N,beta):
        h=const.h.value
        c=const.c.value
        k_B=const.k_B.value
        tau=optical_depth_emcee(nu,N,beta)
        B_nu=((2.*h)*(nu**3.) / (c**2.)) / (np.exp((h*nu) / (k_B*T))-1.0)
        mod_bb=B_nu*(1.0-np.exp(-1.0*tau))*(10.0**26) 
        return mod_bb

def optical_depth_emcee(nu,N,beta):
        m_H=1.6735575e-27 #Hydrogen mass in Kg
        mu=2.8     #mean molecular weight of hydrogen.
        chi_d=100  #gas to dust mass ratio
        kappa0=0.1773 #converted to m^2/kg #reference dust oppacity in cm^2 g^-1 ***need to choose better value***
        nu0=600e9      #reference frequency for reference dust opacity (Hz)
        kappa_nu=(kappa0/chi_d)*((nu/nu0)**beta)
        tau=mu*m_H*kappa_nu*N
        return tau

def log_likelihood(theta, freq, flux, yerr):
    T, logN, beta = theta
    N=np.power(10,logN)
    model = mod_bb_emcee(freq, T, N, beta)
    sigma2 = yerr ** 2 + model ** 2 
    return -0.5 * np.sum((flux - model) ** 2 / sigma2 + np.log(sigma2))

def log_prior_uniform(theta):
    T, logN, beta = theta
    prior = 0.0
    if 2.73 < T < 1000 and 10 < logN < 40 and 0.0 < beta < 5.0:
        return prior
    return -np.inf

def log_probability_uniform(theta, x, y, yerr):
    lp = log_prior_uniform(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)


def calculate_intensity(fitsfile,region_file):
        #assuming all images in Jy/beam for now.
        #convert to K
        hdul=fits.open(fitsfile)
        image_data=extract_dimensions(hdul[0].data)
        #print(image_data.shape)
        head=hdul[0].header
        #print(head)
        #print(fitsfile)
        try:
            restfreq=head['RESTFRQ']
        except:
            try:
                restfreq=head['RESTFREQ']
            except:
                raise Exception("No frequency information found in header for KEYS: RESTFREQ/RESTFRQ")
        #print('Freq',restfreq)
        restfreq=(restfreq*u.Hz)
        my_beam = Beam.from_fits_header(head)
        #print(my_beam.sr)
        #print('beam',my_beam)
        #print('beam sr',my_beam.sr)
        wcs = WCS(head).celestial
        try:
            sky_region = Regions.read(region_file, format='ds9')[0]
        except: 
            sky_region=region_file
        pixel_region = sky_region.to_pixel(wcs)
        mask=pixel_region.to_mask(mode='exact')
        avg_intensity=np.mean(mask.get_values(image_data))
        #avg_intensity=Jy_per_beam_to_K(avg_intensity*u.Jy,my_beam.sr,restfreq)
        #print(avg_intensity, 'Jy/beam')
        avg_intensity = avg_intensity / my_beam.sr
        #print(avg_intensity, 'Jy/sr')
        return avg_intensity.value,restfreq.value

def rms(fitsfile,region_file):
        #assuming all images in Jy/beam for now.
        #convert to K
        hdul=fits.open(fitsfile)
        image_data=extract_dimensions(hdul[0].data)
        #print(image_data.shape)
        head=hdul[0].header
        #print(head)
        #print(fitsfile)
        try:
            restfreq=head['RESTFRQ']
        except:
            try:
                restfreq=head['RESTFREQ']
            except:
                raise Exception("No frequency information found in header for KEYS: RESTFREQ/RESTFRQ")
        #print('Freq',restfreq)
        restfreq=(restfreq*u.Hz)
        my_beam = Beam.from_fits_header(head)
        #print(my_beam.sr)
        #print('beam',my_beam)
        #print('beam sr',my_beam.sr)
        wcs = WCS(head).celestial
        sky_region = Regions.read(region_file, format='ds9')[0]
        pixel_region = sky_region.to_pixel(wcs)
        mask=pixel_region.to_mask(mode='exact')
        rms=np.sqrt(np.mean(mask.get_values(image_data)**2))
        #avg_intensity=Jy_per_beam_to_K(avg_intensity*u.Jy,my_beam.sr,restfreq)
        #print(avg_intensity, 'Jy/beam')
        rms = rms / my_beam.sr
        #print(avg_intensity, 'Jy/sr')
        return rms.value,restfreq.value

def pixel_intensity(fitsfile,x,y):
        #assuming all images in Jy/beam for now.
        #convert to K
        with fits.open(fitsfile) as hdul:
            image_data=extract_dimensions(hdul[0].data)
            #print(image_data.shape)
            head=hdul[0].header
            #print(head)
            #print(fitsfile)
            try:
                restfreq=head['RESTFRQ']
            except:
                try:
                    restfreq=head['RESTFREQ']
                except:
                    raise Exception("No frequency information found in header for KEYS: RESTFREQ/RESTFRQ")
            restfreq=(restfreq*u.Hz)
            my_beam = Beam.from_fits_header(head)
            avg_intensity=image_data[y,x] #stupid fucking coordinates
            #print('pixel intensity: ',avg_intensity)
            avg_intensity = avg_intensity / my_beam.sr

        return avg_intensity.value,restfreq.value


#function which returns the two largest dimensions of an array
def extract_dimensions(array):
    if array.ndim <= 2:
        return array
    else:
        dimensions_to_remove = np.where(np.array(array.shape) < 2)[0]
        modified_array = np.squeeze(array, axis=tuple(dimensions_to_remove))
        return modified_array

