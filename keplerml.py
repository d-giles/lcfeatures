%%timeit
# L.M. Walkowicz
# Rewrite of Revant's feature calculations, plus additional functions for vetting outliers


import random
import numpy as np
import scipy as sp
from scipy import stats
import pyfits
import math
import pylab as pl
import matplotlib.pyplot as plt
import heapq
from operator import xor
import scipy.signal
from numpy import float64
#import astroML.time_series
#import astroML_addons.periodogram
#import cython
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN,Ward
from numpy.random import RandomState
#rng = RandomState(42)
import itertools
import commands
# import utils
import itertools
#from astropy.io import fits

def read_revant_pars(parfile):
    pars = [line.strip() for line in open(parfile)]
    pararr = np.zeros((len(pars), 60))
    for i in range(len(pars)):
        pararr[i] = np.fromstring(pars[i], dtype=float, sep=' ')
    return pararr

def read_kepler_curve(file):
    lc = pyfits.getdata(file)
    t = lc.field('TIME')
    f = lc.field('PDCSAP_FLUX')
    err = lc.field('PDCSAP_FLUX_ERR')
    nf = f / np.median(f)
 
    nf = nf[np.isfinite(t)]
    t = t[np.isfinite(t)]
    t = t[np.isfinite(nf)]
    nf = nf[np.isfinite(nf)]

    return t, nf, err

def plot_kepler_curve(t, nf):
    fig, ax = plt.subplots(figsize=(5, 3.75))
    ax.set_xlim(t.min(), t.max())
    ax.set_xlabel(r'${\rm Time (Days)}$', fontsize=20)
    ax.set_ylabel(r'${\rm \Delta F/F}$', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.plot(t, nf, 'o',markeredgecolor='none', color='blue', alpha=0.2)
    plt.plot(t, nf, '-',markeredgecolor='none', color='blue', alpha=1.0)
    plt.show()

def calc_outliers_pts(t, nf):
    # Is t really a necessary input?
    posthreshold = np.mean(nf)+4*np.std(nf)
    negthreshold = np.mean(nf)-4*np.std(nf)
    
    numposoutliers,numnegoutliers,numout1s=0,0,0
    for j in range(len(nf)):
        # First checks if nf[j] is outside of 1 sigma
        if abs(np.mean(nf)-nf[j])>np.std(nf):
            numout1s += 1
            if nf[j]>posthreshold:
                numposoutliers += 1
            elif nf[j]<negthreshold:
                numnegoutliers += 1
    numoutliers=numposoutliers+numnegoutliers
    
    return numoutliers, numposoutliers, numnegoutliers, numout1s


def calc_slopes(t, nf, corrnf):

    slope_array = np.zeros(20)

    #slopes looks at point j and the next point to find the slope. Delta f/ Delta t
    slopes=[(nf[j+1]-nf[j])/(t[j+1]-t[j]) for j in range (len(nf)-1)]
    #corrslopes removes the longterm linear trend (if any) and then looks at the slope
    corrslopes=[(corrnf[j+1]-corrnf[j])/(t[j+1]-t[j]) for j in range (len(corrnf)-1)]
    meanslope = np.mean(slopes)
    # by looking at where the 99th percentile is instead of just the largest number,
    # I think it avoids the extremes which might not be relevant (might be unreliable data)
    # Is the miniumum slope the most negative one, or the flattest one?
    maxslope=np.percentile(slopes,99)
    minslope=np.percentile(slopes,1)
    # Separating positive slopes and negative slopes
    # Should both include the 0 slope? I'd guess there wouldn't be a ton, but still...
    pslope=[slopes[j] for j in range(len(slopes)) if slopes[j]>=0]
    nslope=[slopes[j] for j in range(len(slopes)) if slopes[j]<=0]
    # Looking at the average (mean) positive and negative slopes
    meanpslope=np.mean(pslope)
    meannslope=np.mean(nslope)
    # Quantifying the difference in shape.
    g_asymm=meanpslope / meannslope
    # Won't this be skewed by the fact that both pslope and nslope have all the 0's? Eh
    rough_g_asymm=len(pslope) / len(nslope)
    # meannslope is inherently negative, so this is the difference btw the 2
    diff_asymm=meanpslope + meannslope
    skewslope = scipy.stats.skew(slopes)
    absslopes=[abs(slopes[j]) for j in range(len(slopes))]
    meanabsslope=np.mean(absslopes)
    varabsslope=np.var(absslopes)
    varslope=np.var(slopes)
    #secder = Second Derivative
    # Reminder for self: the slope is "located" halfway between the flux and time points, 
    # so the delta t in the denominator is accounting for that.
    #secder=[(slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2) for j in range(1, len(nf)-1)]
    #algebraic simplification:
    secder=[2*(slopes[j]-slopes[j-1])/(t[j+1]-t[j-1]) for j in range(1, len(nf)-1)]
    meansecder=np.mean(secder)
    #abssecder=[abs((slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2)) for j in range (1, len(slopes)-1)]
    # simplification:
    abssecder=[abs(secder[j]) for j in range(1, len(secder))]
    absmeansecder=np.mean(abssecder)

    pslopestds=np.std(pslope)
    nslopestds=np.std(nslope)
    sdstds=np.std(secder)
    meanstds=np.mean(secder)
    stdratio=pslopestds/nslopestds

    pspikes =[slopes[j] for j in range(len(slopes)) if slopes[j]>=meanpslope+3*pslopestds] 
    nspikes=[slopes[j] for j in range(len(slopes)) if slopes[j]<=meannslope-3*nslopestds]
    psdspikes=[secder[j] for j in range(len(secder)) if secder[j]>=4*sdstds] 
    nsdspikes=[secder[j] for j in range(len(secder)) if secder[j]<=-4*sdstds]

    num_pspikes = len(pspikes)
    num_nspikes = len(nspikes)
    num_psdspikes = len(psdspikes)
    num_nsdspikes = len(nsdspikes)
    
    stdratio = pslopestds / nslopestds
    # The ratio of postive slopes with a following postive slope to the total number of points.

    pstrend=len([slopes[j] for j in range(len(slopes)-1) if (slopes[j]>0) & (slopes[j+1]>0)])/len(slopes)

    slope_array = [meanslope, maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes, stdratio, pstrend]

    return slopes, corrslopes, secder, slope_array

def calc_maxmin_periodics(t, nf, err):
#look up this heapq.nlargest crap
    #This looks up the local maximums. Adds a peak if it's the largest within 10 points on either side.

    naivemax,nmax_times = [],[]
    naivemins = []
    for j in range(len(nf)):
        if nf[j] == max(nf[max(j-10,0):min(j+10,len(nf-1))]):
            naivemax.append(nf[j])
            nmax_times.append(t[j])
        elif nf[j] == min(nf[max(j-10,0):min(j+10,len(nf-1))]):
            naivemins.append(nf[j])
    len_nmax=len(naivemax) #F33
    len_nmin=len(naivemins) #F34
    

    #wtf is this?
    #D: shifts everything to the left for some reason.
    #autopdcmax = [naivemax[j+1] for j in range(len(naivemax)-1)] = naivemax[1:]
    
    #naivemax[:-1:] is naivemax without the last value and autopdcmax is naivemax without the first value. why do this?
    #np.corrcoef(array) returns a correlation coefficient matrix. I.e. a normalized covariance matrix
    """
    It looks like it compares each point to it's next neighbor, hence why they're offset, 
    then determines if there's a correlation between the two. If the coefficient is closer
    to 1, then there's a strong correlation, if 0 then no correlation, if -1 (possible?) then anti-correlated.
    """
    mautocorrcoef = np.corrcoef(naivemax[:-1], naivemax[1:])[0][1] #F35
    mautocovs = np.cov(naivemax[:-1],naivemax[1:])[0][1] # Not a feature, not used elsewhere

    """peak to peak slopes"""
    ppslopes = [abs((naivemax[j+1]-naivemax[j])/(nmax_times[j+1]-nmax_times[j])) for j in range(len(naivemax)-1)]

    ptpslopes=np.mean(ppslopes) #F36

    maxdiff=[nmax_times[j+1]-nmax_times[j] for j in range(len(naivemax)-1)]

    periodicity=np.std(maxdiff)/np.mean(maxdiff) #F37
    periodicityr=np.sum(abs(maxdiff-np.mean(maxdiff)))/np.mean(maxdiff) #F38

    naiveperiod=np.mean(maxdiff) #F39
    # why not maxvars = np.var(naivemax)? Is this not the variance? Seems like it should be...
    #maxvars = np.var(naivemax) #F40
    maxvars = np.std(naivemax)/np.mean(naivemax) #F40
    # I don't understand what this is.
    maxvarsr = np.sum(abs(naivemax-np.mean(naivemax)))/np.mean(naivemax) #F41

    emin = naivemins[::2] # even indice minimums
    omin = naivemins[1::2] # odd indice minimums
    meanemin = np.mean(emin)
    meanomin = np.mean(omin)
    oeratio = meanomin/meanemin #F42

    peaktopeak_array = [len_nmax, len_nmin, mautocorrcoef, ptpslopes, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio]

    return peaktopeak_array, naivemax, naivemins



def lc_examine(filelist, style='-'):
    """Takes a list of FITS files and plots them one by one sequentially"""
    files = [line.strip() for line in open(filelist)]

    for i in range(len(files)):
        lc = pyfits.getdata(files[i])
        t = lc.field('TIME')
        f = lc.field('PDCSAP_FLUX')
        nf = f / np.median(f)
 
        title = 'Light curve for {0}'.format(files[i])
        plt.plot(t, nf, style)
        plt.title(title)
        plt.xlabel(r'$t (days)$')
        plt.ylabel(r'$\Delta F$')
        plt.show()

    return

def feature_calc(filelist):

    files = [line.strip() for line in open(filelist)]

    # Create the appropriate arrays for the features. Length of array determined by number of files.

    numlcs = len(files)
    longtermtrend=np.zeros(numlcs)
    meanmedrat=np.zeros(numlcs)
    skews=np.zeros(numlcs)
    varss=np.zeros(numlcs)
    coeffvar =np.zeros(numlcs)
    stds =np.zeros(numlcs)
    numoutliers =np.zeros(numlcs)
    numnegoutliers =np.zeros(numlcs)
    numposoutliers =np.zeros(numlcs)
    numout1s =np.zeros(numlcs)
    kurt =np.zeros(numlcs)
    mad =np.zeros(numlcs)
    maxslope =np.zeros(numlcs)
    minslope =np.zeros(numlcs)
    meanpslope =np.zeros(numlcs)
    meannslope =np.zeros(numlcs)
    g_asymm=np.zeros(numlcs)
    rough_g_asymm =np.zeros(numlcs)
    diff_asymm =np.zeros(numlcs)
    skewslope =np.zeros(numlcs)
    varabsslope =np.zeros(numlcs)
    varslope =np.zeros(numlcs)
    meanabsslope =np.zeros(numlcs)
    absmeansecder =np.zeros(numlcs)
    num_pspikes=np.zeros(numlcs)
    num_nspikes =np.zeros(numlcs)
    num_psdspikes =np.zeros(numlcs)
    num_nsdspikes=np.zeros(numlcs)
    stdratio =np.zeros(numlcs)
    pstrend =np.zeros(numlcs)
    num_zcross =np.zeros(numlcs)
    num_pm =np.zeros(numlcs)
    len_nmax =np.zeros(numlcs)
    len_nmin =np.zeros(numlcs)
    mautocorrcoef =np.zeros(numlcs)
    ptpslopes =np.zeros(numlcs)
    periodicity =np.zeros(numlcs)
    periodicityr =np.zeros(numlcs)
    naiveperiod =np.zeros(numlcs)
    maxvars =np.zeros(numlcs)
    maxvarsr =np.zeros(numlcs)
    oeratio =np.zeros(numlcs)
    amp = np.zeros(numlcs)
    normamp=np.zeros(numlcs)
    mbp =np.zeros(numlcs)
    mid20 =np.zeros(numlcs)
    mid35 =np.zeros(numlcs)
    mid50 =np.zeros(numlcs)
    mid65 =np.zeros(numlcs)
    mid80 =np.zeros(numlcs)
    percentamp =np.zeros(numlcs)
    magratio=np.zeros(numlcs)
    sautocorrcoef =np.zeros(numlcs)
    autocorrcoef =np.zeros(numlcs)
    flatmean =np.zeros(numlcs)
    tflatmean =np.zeros(numlcs)
    roundmean =np.zeros(numlcs)
    troundmean =np.zeros(numlcs)
    roundrat =np.zeros(numlcs)
    flatrat=np.zeros(numlcs)


    for i in range(len(files)):
        # Keeping track of progress, noting every thousand files completed.
        if (i % 1000. == 0):
            print i
            # below command prints out a bit more info about the progress,
            # personal preference thing only really.
            print("%s/%s %s percent complete" % (i,len(files),100*i/len(files)))

        t,nf,err = read_kepler_curve(files[i])

        # t = time
        # err = error
        # nf = normalized flux. Same as mf but offset by 1 to center at 0?

        longtermtrend[i] = np.polyfit(t, nf, 1)[0] # Feature 1 (Abbr. F1) overall slope
        yoff = np.polyfit(t, nf, 1)[1] # Not a feature? y-intercept of linear fit
        meanmedrat[i] = np.mean(nf) / np.median(nf) # F2
        skews[i] = scipy.stats.skew(nf) # F3
        varss[i] = np.var(nf) # F4
        coeffvar[i] = np.std(nf)/np.mean(nf) #F5
        stds[i] = np.std(nf) #F6

        corrnf = nf - longtermtrend[i]*t - yoff #this removes any linear slope to lc so you can look at just troughs - is this a sign err tho?
        # D: I don't think there's a sign error
        
        # Features 7 to 10
        numoutliers[i], numposoutliers[i], numnegoutliers[i], numout1s[i] = calc_outliers_pts(t, nf)

        kurt[i] = scipy.stats.kurtosis(nf)

        mad[i] = np.median([abs(nf[j]-np.median(nf)) for j in range(len(nf))])

        # slopes array contains features 13-30
        slopes, corrslopes, secder, slopes_array = calc_slopes(t, nf, corrnf) 

        maxslope[i] = slopes_array[0]
        minslope[i] = slopes_array[1]
        meanpslope[i]  = slopes_array[2]
        meannslope[i]  = slopes_array[3]
        g_asymm[i] = slopes_array[4]
        rough_g_asymm[i]  = slopes_array[5]
        diff_asymm[i]  = slopes_array[6]
        skewslope[i]  = slopes_array[7]
        varabsslope[i]  = slopes_array[8]
        varslope[i]  = slopes_array[9]
        meanabsslope[i]  = slopes_array[10]
        absmeansecder[i] = slopes_array[11]
        num_pspikes[i] = slopes_array[12]
        num_nspikes[i]  = slopes_array[13]
        num_psdspikes[i] = slopes_array[14]
        num_nsdspikes[i] = slopes_array[15]
        stdratio[i]  = slopes_array[16]
        pstrend[i]  = slopes_array[17]
        
        # Checks if the flux crosses the zero line.
        zcrossind= [j for j in range(len(nf)-1) if corrnf[j]*corrnf[j+1]<0]
        num_zcross[i] = len(zcrossind) #F31

        plusminus=[j for j in range(1,len(slopes)) if (slopes[j]<0)&(slopes[j-1]>0)]
        num_pm[i] = len(plusminus)

        # peak to peak array contains features 33 - 42
        peaktopeak_array, naivemax, naivemins = calc_maxmin_periodics(t, nf, err)
        """D: Bookmark"""
        # amp here is actually amp_2 in revantese
        # 2x the amplitude (peak-to-peak really)
        amp[i] = np.percentile(nf,99)-np.percentile(nf,1)
        normamp[i] = amp[i] / np.mean(nf) #this should prob go, since flux is norm'd

        mbp[i] = len([nf[j] for j in range(len(nf)) if (nf[j] < (np.median(nf) + 0.1*amp[i])) & (nf[j] > (np.median(nf)-0.1*amp[i]))]) / len(nf)

        f595 = np.percentile(nf,95)-np.percentile(nf,5)
        f1090 =np.percentile(nf,90)-np.percentile(nf,10)
        f1782 =np.percentile(nf, 82)-np.percentile(nf, 17)
        f2575 =np.percentile(nf, 75)-np.percentile(nf, 25)
        f3267 =np.percentile(nf, 67)-np.percentile(nf, 32)
        f4060 =np.percentile(nf, 60)-np.percentile(nf, 40)
        mid20[i] =f4060/f595
        mid35[i] =f3267/f595
        mid50[i] =f2575/f595
        mid65[i] =f1782/f595
        mid80[i] =f1090/f595

        percentamp[i] = max([(nf[j]-np.median(nf)) / np.median(nf) for j in range(len(nf))])
        magratio[i] = (max(nf)-np.median(nf)) / amp[i]

        autopdc=[nf[j+1] for j in range(len(nf)-1)]
        autocorrcoef[i] = np.corrcoef(nf[:-1:], autopdc)[0][1]
        autocovs = np.cov(nf[:-1:], autopdc)[0][1]
 
        sautopdc=[slopes[j+1] for j in range(len(slopes)-1)]

        sautocorrcoef[i] = np.corrcoef(slopes[:-1:], sautopdc)[0][1]
        sautocovs = np.cov(slopes[:-1:],sautopdc)[0][1]
        
        flatness = [np.mean(slopes[max(0,j-6):min(j-1, len(slopes)-1):1])- np.mean(slopes[max(0,j):min(j+4, len(slopes)-1):1]) for j in range(len(slopes)) if nf[j] in naivemax]

        flatmean[i] = np.nansum(flatness)/len(flatness)

        # trying flatness w slopes and nf rather than "corr" vals, despite orig def in RN's program
        tflatness = [-np.mean(slopes[max(0,j-6):min(j-1, len(slopes)-1):1])+ np.mean(slopes[max(0,j):min(j+4, len(slopes)-1):1]) for j in range(len(slopes)) if nf[j] in naivemins]

        tflatmean[i] = np.nansum(tflatness) / len(tflatness) 

        roundness=[np.mean(secder[max(0,j-6):min(j-1, len(secder)-1):1]) + np.mean(secder[max(0,j+1):min(j+6, len(secder)-1):1]) for j in range(len(secder)) if nf[j+1] in naivemax]

        roundmean[i] = np.nansum(roundness) / len(roundness)

        troundness = [np.mean(secder[max(0,j-6):min(j-1, len(secder)-1):1]) + np.mean(secder[max(0,j+1):min(j+6, len(secder)-1):1]) for j in range(len(secder)) if nf[j+1] in naivemins]
	
        troundmean[i] = np.nansum(troundness)/len(troundness)
        roundrat[i] = roundmean[i] / troundmean[i]

        flatrat[i] = flatmean[i] / tflatmean[i]

    final_features = np.vstack((longtermtrend, meanmedrat, skews, varss, coeffvar, stds, numoutliers, numnegoutliers, numposoutliers, numout1s, kurt, mad, maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes,stdratio, pstrend, num_zcross, num_pm, len_nmax, len_nmin, mautocorrcoef, ptpslopes, periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio, amp, normamp,mbp, mid20, mid35, mid50, mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef, flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat))
    

    return final_features

#final list of features - look up vstack and/or append to consolidate these 

#things that are apparently broken:
#numoutliers   Dan: Seems like an easy fix, see comments above

f = 'speedtest'
print(feature_calc(f))