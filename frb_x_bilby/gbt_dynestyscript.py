import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging
from numpy import convolve
from matplotlib.gridspec import GridSpec
import bilby
import time as timer
from scipy import signal
#global fch1,bwchan,nchan,ftop,fcentre
from astropy import units as u
MIN_FLOAT = sys.float_info[3]
fch1=919.951172 ## Fch1  MHz
bwchan=-0.097656## bwchan 1 MHz
nchan=1024## nchan=336
ftop=fch1/1000 ##GHz
fcentre=(fch1+bwchan*nchan/2)/1000 ##GHz
print("fch1,bwchan(MHz),nchan,ftop,fcentre")
print(fch1,bwchan,nchan,ftop,fcentre,fch1+bwchan*nchan)
tsamp=81.92000*4/1000 ##us ___> ms

#### get a frequency splitter
def freq_splitter_idx(n,skip,end):
    dw=(end-skip)/n
    print(dw,bwchan)
    vi=np.arange(n)*dw*bwchan+0.5*dw*bwchan
    base=fch1+skip*bwchan
    vi=base+vi
    chan_idx=np.arange(n)*dw+skip
    chan_idx=np.append(chan_idx,end)
    chan_idx=chan_idx.astype(np.int)
    return vi,chan_idx
#### load data with this

def dataloader(name,chan_idx,head=10,tail=40):
    data=np.load(name)
    data=(data.T-np.median(data,1)).T
    #stokesI_rms=np.loadtxt("FRB_HTR_xpol-imageplane-rms.stokesI.txt")[1:].T
    #print("bandwidth GHz/MHz",bwchan,bwchan*1000)
    #print(data.shape)
    #print(ftop,vi,chan_idx)
    time=np.arange(data.shape[1])*tsamp
    #ytot=data.std()
    #print(ytot)
    sigma=[]
    ydata=[]
    for i in range(len(chan_idx)-1):
        ytot=data[chan_idx[i]:chan_idx[i+1]].mean(axis=0).std()
        ydata.append(data[chan_idx[i]:chan_idx[i+1]].mean(axis=0)/ytot)
        sigma.append(ytot)
    ydata=np.array(ydata)
    print (ydata.shape)
    x0=np.argmax(data.mean(axis=0)/ytot)
    print(time[x0],'peak data')
    ydata=ydata[:,x0-head:x0+tail]
    time=np.arange(ydata.shape[1])*tsamp
    return time,ydata,sigma

def scat_pulse_smear(t,t0,tau1,dm,dmerr,sigma,alpha,a,vi):
    ### vi is GHz
    dmerr=dmerr
    dm_0=dm+dmerr
    ti=tidm(dmerr,vi)##ms
    smear=delta_t(dm_0,vi,bwchan) ##ms
    width=np.sqrt(sigma**2+smear**2)
    gt0=np.mean(t)
    pulse=gaus_func(width,gt0,t,ti) ## create pulse
    scat_corr=scat(t,t0,tau1,alpha,vi) ## create scatter kernel
    flux=convolve(scat_corr,pulse,'same')
    flux/=np.max(flux) ### normalise
    return a*flux

def single_pulse_smear(t,t0,dm,dmerr,sigma,a,vi):
    ### vi is GHz
    dmerr=dmerr
    dm_0=dm+dmerr
    ti=tidm(dmerr,vi) ##ms
    smear=delta_t(dm_0,vi,bwchan) ##ms
    width=np.sqrt(sigma**2+smear**2)
    pulse=gaus_func(width,t0,t,ti) ## create pulse
    flux=pulse
    flux/=np.max(flux) ### normalise
    return a*flux

def dgaus_smear_4band(t,t1,t2,dmerr,sigi,sigi2,a1,a2,a3,a4,b1,b2,b3,b4):
    amp_list=np.array([a1,a2,a3,a4])
    b_list=np.array([b1,b2,b3,b4])
    model=[]
    sigma=sigi
    sigma2=sigi2
    for vi,am,b in zip(freq,amp_list,b_list):
        #print (vi,am)
        flux1=single_pulse_smear(t,t1,dm,dmerr,sigma,am,vi)
        flux2=single_pulse_smear(t,t2,dm,dmerr,sigma2,b,vi)
        model.append(flux1+flux2)
    return np.array(model)

def scat_smear_4band(t,t0,tau1,a1,a2,a3,a4,dmerr,sigi):
    amp_list=np.array([a1,a2,a3,a4])
    alpha=4
    model=[]
    sigma=sigi
    for vi,am in zip(freq,amp_list):
        #print (vi,am)
        model.append(scat_pulse_smear(t,t0,tau1,dm,dmerr,sigma,alpha,am,vi))
    return np.array(model)

### basic functions here

### gaussian
def gaus_func(sigi,t0,t,ti):
    sit=1/np.sqrt(np.pi*2*(sigi**2))*np.exp(-(t-t0-ti)**2/sigi**2) ### model 0 in ravi 2018
    return sit

### adjust dm
def tidm(dmerr,vi):
    beta=2
    #ftop=1464/1000 ## MHz--->GHz
    #fbot=1128/1000 ## MHz--->GHz
    ### ftop GHz
    ### 4.15 ms
    ti=4.15*dmerr*(ftop**(-beta)-vi**(-beta)) ### ms
    return ti

### scattering
def scat(t,t0,tau1,alpha,v):
    ###tau=tau1/1000 ## ms
    flux=np.zeros(len(t)) + MIN_FLOAT
    flux[t>=t0]=np.exp(-(t[t>=t0]-t0)/(tau1*(v/fcentre)**(-alpha)))
    return flux

### dm smearing

def delta_t(dm,v,bwchan):
    ### calculate dm smearing
    v=v ###GHz
    B=bwchan ###1MHz channels
    inverse_v=1/v #### 1/GHz
    #print(v)
    dt=8.3*dm*(inverse_v**3)*B/2 #### unit:us, 2 sigma---> 1 sigma
    return dt/1000 ###us ---> ms


vi,chan_idx=freq_splitter_idx(n=4,skip=0,end=nchan)## full band used
print(vi,chan_idx)
global freq,dm
freq=np.array(vi)/1000 ## GHz
dm=458.2 ###dm value

time,ydata,err=dataloader("r1_full_t4.npy",chan_idx,head=100,tail=100)

plt.imshow(ydata,aspect='auto')
plt.show()
print(tsamp)
plt.plot(time,ydata.mean(0))
plt.show()
