import os # to list files, directories
from os.path import isfile, join # to list files
from os import system
from fnmatch import fnmatch # to match .fits .txt .. extenctions
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from numpy import sqrt, pi, exp, linspace
from scipy import optimize
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from astropy.modeling import models, fitting
from astropy.table import Table, Column


def readdata(name):
    """
    this function reads the fits file
    """
    global source, freq, bw, el, time, rtsys, dec, ra, rp, lp
    header_table0 = fits.getheader(name,0)
    source = header_table0['source']
    freq = header_table0['freq']
    bw = header_table0['bw']
    az = header_table0['az']
    el = header_table0['el']
    time = header_table0['time']
    rtsys = header_table0['rtsys']
    data_table, header_table = fits.getdata(name, 1, header=True)
    dec = data_table.field('DDEC')
    ra = data_table.field('DRA')
    rp = data_table.field('RP')
    lp = data_table.field('LP')

def gauss_fit(a,b):
    """
    this function fits the gauss
    """
    global r
    g_init = models.Gaussian1D(amplitude=1., mean=0, stddev=1.)
    fit_g = fitting.LevMarLSQFitter()
    r = fit_g(g_init, a, b)


def parameters(r,y,el,N):
    """
    calculate parameters of the gauss fit, write results to the file
    """
    global maximum, sigma, mean, fwhm
    maximum = r.amplitude.value # Amplitude Y-coordinate
    fwhm = 2.3548*r.stddev.value # HPBW
    mean = r.mean.value #offset value X-coordinate
    off_corr = np.exp(-4.0 * np.log(2.0) * (mean/fwhm)**2.0)
    T = maximum / off_corr

    savename = name.split('.')[0]
    if N==2:
        # calculate uncertainty of gauss fit
        sigma = np.sqrt(sum((r(x) - sbtr)**2)/len(r(x)))
        data_rows = [(name, source, el, time, rtsys, maximum, T, sigma, fwhm, mean)]
        t = Table(rows=data_rows, names=('File', 'Name', 'Elevat', 'Time(MJD)', 'Tsys(K)', 'Max(K)', 'T_oc', 'sigma', 'FWHM', 'Offset'),
                  dtype=('S7', 'S8', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
        t['Max(K)'].format = '.3f'
        t['T_oc'].format = '.3f'
        t['Elevat'].format = '.3f'
        t['Time(MJD)'].format = '.3f'
        t['Tsys(K)'].format = '.3f'
        t['FWHM'].format = '.2f'
        t['sigma'].format = '.3f'
        t['Offset'].format = '.2f'
        t.write(str(savename)+'_'+pol+'.txt', format='ascii', overwrite=True)



def baseline_calc():
    ''' calculating baseline
    xx1, xx2 - left and right borders of the gauss fit
    bb1, bb2 - left and right borders for the baseline calc
    b1, b2 - INDEX of the value closets to border (bb1,bb2)
    xx,yy - sets of two points to use in interpolation
    '''
    global bb1, bb2, f, sbtr

    xx1 = mean - fwhm
    if xx1 > 0:
        bb1 = int(fwhm/2 + xx1)
    else:
        bb1 = int(xx1 - fwhm/2)
    xx2 = mean + fwhm
    if xx2 > 0:
        bb2 = int(fwhm/2 + xx2)
    else:
        bb2 = int(xx2 - fwhm/2)

    ''' to know INDEX of the value closets to baseline border in X-array'''
    b1 = min(range(len(x)), key=lambda i: abs(x[i]-bb1))
    b2 = min(range(len(x)), key=lambda i: abs(x[i]-bb2))

    ''' linear interpolation between 2 points (baseline)'''
    yy1 = y[b1]
    yy2 = y[b2]
    xx = (bb1,bb2)
    yy = (yy1,yy2)
    f = interp1d(xx, yy, "linear", bounds_error=False)
    sbtr = y - f(x)
    inds = np.where(np.isnan(sbtr))
    sbtr[inds] = 0




def rms_calc():
    ''' calculating rms at different parts of scan
    '''
    global b1, b2, b3, b4, rms_first, rms_second

    b1 = int(0.6 * len(x))
    b2 = int(0.8 * len(x))
    b3 = -int(0.6 * len(x))
    b4 = -int(0.8 * len(x))
    first = list(range(b1,b2,1))
    second = list(range(b4,b3,1))    
    rms_first = np.std(rp[first])
    rms_second = np.std(rp[second])


def plot_source(x,y):
    """
    this function plots the figures
    """
    plt.title('File=%s, Source=%s, Freq(GHz)=%.2f BW=%.1f \n Ampl(K)=%.2f Sigma=%.2f, Offset(")=%.2f'
     % (name, source, freq, bw, maximum, sigma, mean), fontsize=9)
    plt.plot(x, y, 'r-', label='Data')
    plt.plot(x, f(x), 'y-', lw=1, label='baseline')
    plt.axvline(x=bb1, color='y', lw=.75, label='0+1.5*HPBW')
    plt.axvline(x=bb2, color='y', lw=.75)
    plt.plot(x, r(x), '.', label='Gaussian')
    plt.ylabel(r'$T_{ant}, K$')
    plt.xlabel(scan+' , arcsec')
    plt.legend(prop={'size':10}, labelspacing=0.1)
    savename = name.split('.')[0]
    plt.savefig(str(savename)+'_'+pol+'.png', bbox_inches='tight')
    plt.clf()


def catfiles():
    """
    compound resulting files using bash 'cat' procedure
    """
    system("cat *_RP.txt> RP_result.txt")
    system("cat *_LP.txt> LP_result.txt")

def rmrows():
    """
    remove extra headings after compounding files
    """
    f=open('RP_result.txt').readlines()

    if len(f) > 3:
        rows = (len(f) + 2) / 2
        row_rem = np.arange(2,rows)
        for i in row_rem:
            f.pop(i)
        with open('RP_result.txt','w') as F:
            F.writelines(f)


    g=open('LP_result.txt').readlines()

    if len(g) > 3:
        rows = (len(g) + 2) / 2
        row_rem = np.arange(2,rows)
        for i in row_rem:
            g.pop(i)
        with open('LP_result.txt','w') as F:
            F.writelines(g)


def rmfiles():
    """
    remove temporary files
    """
    system("rm *_RP.txt")
    system("rm *_LP.txt")


def list_files():
    """
    List files in a directory
    """
    global names
    pattern = "*.fits"
    files = [f for f in os.listdir('.') if os.path.isfile(f)] #current path
    names = []
    for name in files:
        if fnmatch(name, pattern):
            names.append(name) # write fits file to the list


def choose_scan():
    """
    (RA/DEC)
    """
    global scan, x
    if name[0] == "R":
     scan = "RA"
     x = ra
    else:
     scan = "Dec"
     x = dec



list_files()
for name in names:
 ''' # RP '''
 pol = 'RP'
 readdata(name)
 y = rp
 choose_scan()
 gauss_fit(x,y)
 N=1
 parameters(r,y,el,N)
 baseline_calc()
 gauss_fit(x,sbtr)
 N=2
 parameters(r,y,el,N)
 rms_calc()
 plot_source(x,y)
 ''' # LP '''
 pol = 'LP'
 y = lp
 choose_scan()
 gauss_fit(x,y)
 N=1
 parameters(r,y,el,N)
 baseline_calc()
 gauss_fit(x,sbtr)
 N=2
 parameters(r,y,el,N)
 rms_calc()
 plot_source(x,y)


catfiles()
rmrows()
rmfiles()

