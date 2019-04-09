import numpy as np 
import pandas as pd
np.set_printoptions(threshold='nan')
from scipy import stats
from multiprocessing import Pool,cpu_count
import os
import sys
from datetime import datetime
import pyfits
from numba import njit

def fl_files(fl):
    """
    Returns an array with the files in the given filelist
    """
    return [line.strip() for line in open(fl)]

def fl_files_w_path(fl,fitsDir):
    """
    Returns an array with the files to be processed
    in the given filelist with the given path.
    """
    files = fl_files(fl)
    fl_c = fl.replace('.txt',"")+"_completed.txt"
    if os.path.isfile(fl_c):
        # Checks for a completed filelist (fl_complete.txt) so as not to reprocess any files.
        completed = fl_files(fl_c)  
        
        dff = pd.DataFrame(index = files)
        dfc = pd.DataFrame(index = completed)
        df = dff.append(dfc)
        df = df[~df.index.duplicated(keep=False)]
        files = np.chararray(len(df.index),itemsize=len(df.index[0]))
        files[:] = df.index
        
        """
        While it seems roundabout to convert to empty pandas dataframes, it's about 20% faster
        than using numpy arrays as follows.
        
        unique,counts = np.unique(np.concatenate([files,completed]),return_counts=True)
        files = unique[counts==1]
        """

        
    else:
        files_copy = files
        files = np.chararray(len(files_copy),itemsize=len(files_copy[0]))
        files[:] = files_copy
        
        # Creates a completed filelist
        fcreate = open(fl_c,'a')
        fcreate.close()
        
    return fitsDir+files

def read_kepler_curve(file):
    """
    Given the path of a fits file, this will extract the light curve and normalize it.
    """
    lc = pyfits.getdata(file)
    t = lc.field('TIME')
    f = lc.field('PDCSAP_FLUX')
    err = lc.field('PDCSAP_FLUX_ERR')
    
    err = err[np.isfinite(t)]
    f = f[np.isfinite(t)]
    t = t[np.isfinite(t)]
    err = err[np.isfinite(f)]
    t = t[np.isfinite(f)]
    f = f[np.isfinite(f)]
    err = err/np.median(f)
    nf = f / np.median(f)

    return t, nf, err

def clean_up(fl,in_file='tmp_data.csv'):
    """
    Removes files already processed from files array and
    removes the original filelist if all files have been processed.
    """
    
    files = fl_files(fl)
    df = pd.read_csv(in_file,index_col=0)
    """
    Dropping files that have already been processed from the files 
    """
    # Create a copy of df to manipulate
    dfc = df
    # create an empty dataframe with all the file names as indices
    dff = pd.DataFrame(index = files)
    dfc = dfc.append(dff)
    files = np.array(dfc[~dfc.index.duplicated(keep=False)].index)
    
    for lc in df.index:
        with open(fl.replace('.txt',"")+"_completed.txt",'a') as completed:
            completed.write(lc)
                
    if files==[]:
        print("All files from original filelist processed, deleting original filelist.")
        os.remove(fl)

    return files

def save_output(out_file,in_file='tmp_data.csv'):
    """
    Reads in the finished data file (tmp_data.csv by default), sorts it, and saves it to the
    specified output csv. Effectively just sorting a csv and renaming it.
    """
    
    df = pd.read_csv(in_file,index_col=0)
    df=df.sort_index()
    with open(out_file,'a') as of:
        df.to_csv(of)
    os.remove(in_file)
    
    return

@njit
def flat_func(slopes,inds):
    flat1 = np.zeros(len(inds))
    for i,j in enumerate(inds):
        flat1[i] = np.mean(slopes[j-6:j-1])-np.mean(slopes[j:j+5])
    return flat1

@njit
def tflat_func(slopes,inds):
    tflat1 = np.zeros(len(inds))
    for i,j in enumerate(inds):
        tflat1[i] = -np.mean(slopes[j-6:j-1])+np.mean(slopes[j:j+5])
    return tflat1

@njit
def roundness_func(secder,inds):
    roundness1 = np.zeros(len(inds))
    for i,j in enumerate(inds):
        roundness1[i] = np.mean(secder[j-6:j+6])*2
    return roundness1

def featureCalculation(nfile,t,nf,err):
    """
    This is the primary function of this code, it takes in light curve data and returns 60 derived features.
    """
    
    assert len(t)==len(nf) and len(t)==len(err), "t, nf, err arrays must be equal sizes."
        
    try:

        longtermtrend = np.polyfit(t, nf, 1)[0] # Feature 1 (Abbr. F1) overall slope
        yoff = np.polyfit(t, nf, 1)[1] # Not a feature, y-intercept of linear fit
        nf_mean = np.mean(nf)
        nf_med = np.median(nf)
        stds = np.std(nf) #f6
        meanmedrat = nf_mean / nf_med # F2
        skews = stats.skew(nf) # F3
        varss = np.var(nf) # F4
        coeffvar = stds/nf_mean #F5


        corrnf = nf - longtermtrend*t - yoff #this removes any linear trend of lc so you can look at just troughs
        
        """
        Outlying flux counts
        """

        posthreshold = nf_mean+4*stds
        negthreshold = nf_mean-4*stds

        numout1s = len(nf[np.abs(nf-nf_mean)>stds])
        numposoutliers = len(nf[nf>posthreshold])
        numnegoutliers = len(nf[nf<negthreshold])

        numoutliers=numposoutliers+numnegoutliers #F10
        
        kurt = stats.kurtosis(nf)

        mad = np.median(np.abs(nf-nf_med))
        
        # delta nf/delta t
        slopes = (nf[1:]-nf[:-1])/(t[1:]-t[:-1])

        #corrslopes (corrected slopes) removes the longterm linear trend (if any) and then looks at the slope
        corrslopes = (corrnf[1:]-corrnf[:-1])/(t[1:]-t[:-1])
        meanslope = np.mean(slopes) #F12

        # by looking at where the 99th percentile is instead of just the largest number,
        # I think it avoids the extremes which might not be relevant (might be unreliable data)
        # Is the miniumum slope the most negative one, or the flattest one? Answer: Most negative
        maxslope=np.percentile(slopes,99) #F13
        minslope=np.percentile(slopes,1) #F14

        # Separating positive slopes and negative slopes
        # Should both include the 0 slope? It doesn't matter for calculating the means later on...
        pslope = slopes[slopes>=0]
        nslope = slopes[slopes<=0]
        # Looking at the average (mean) positive and negative slopes
        if len(pslope)==0:meanpslope=0
        else:meanpslope=np.mean(pslope) #F15

        if len(nslope)==0:meannslope=0
        else:meannslope=np.mean(nslope) #F16

        # Quantifying the difference in shape.
        # if meannslope==0 (i.e., if there are no negative slopes), g_asymm is assigned a value of 10
        # This value is chosen such that 
        # a) it is positive (where g_asymm is inherently negative), 
        # b) it is a factor larger than a random signal would produce (roughly equal average of positive and negative slopes -> g_asymm=-1)
        # c) it is not orders of magnitude larger than other data, which would affect outlier analysis
        if meannslope==0:g_asymm = 10
        else:g_asymm=meanpslope / meannslope #F17

        # Won't this be skewed by the fact that both pslope and nslope have all the 0's? Eh
        if len(nslope)==0:rough_g_asymm=10
        else:rough_g_asymm=len(pslope) / len(nslope) #F18

        # meannslope is inherently negative, so this is the difference btw the 2
        diff_asymm=meanpslope + meannslope #F19
        skewslope = stats.skew(slopes) #F20
        absslopes = np.abs(slopes)
        meanabsslope=np.mean(absslopes) #F21
        varabsslope=np.var(absslopes) #F22
        varslope=np.var(slopes) #F23

        # secder = Second Derivative
        # Reminder for self: the slope is "located" halfway between the flux and time points, 
        # so the delta t in the denominator is accounting for that.
        # secder = delta slopes/delta t, delta t = ((t_j-t_(j-1))+(t_(j+1)-t_j))/2
        # secder=[(slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2) for j in range(1, len(slopes)-1)]
        # after algebraic simplification:
        secder = 2*(slopes[1:]-slopes[:-1])/(t[1:-1]-t[:-2])
        
        # abssecder=[abs((slopes[j]-slopes[j-1])/((t[j+1]-t[j])/2+(t[j]-t[j-1])/2)) for j in range (1, len(slopes)-1)]
        # simplification:

        abssecder=np.abs(secder)
        absmeansecder=np.mean(abssecder) #F24
        
        if len(pslope)==0:pslopestds=0
        else:pslopestds=np.std(pslope)
            
        if len(nslope)==0:
            nslopesstds=0
            stdratio=10
        else:
            nslopestds=np.std(nslope)
            stdratio=pslopestds/nslopestds

        sdstds=np.std(secder)
        meanstds=np.mean(secder)

        num_pspikes=len(slopes[slopes>=meanpslope+3*pslopestds]) #F25
        num_nspikes=len(slopes[slopes<=meannslope-3*nslopestds]) #F26
        
        # 5/30/18, discovered a typo here. meanslope was missing an 'n', i.e. all data
        # processed prior to this date has num_nspikes defined as meanslope-3*nslopestds
        # which will overestimate the number of negative spikes since meanslope is inherently
        # greater than meannslope.
        
        num_psdspikes = len(secder[secder>=4*sdstds]) #F27
        num_nsdspikes = len(secder[secder<=4*sdstds]) #F28
        if nslopestds==0:
            stdratio=10
        else:
            stdratio = pslopestds / nslopestds #F29

        # The ratio of postive slopes with a following postive slope to the total number of points.
        pairs = np.where((slopes[1:]>0)&(slopes[:-1]>0))[0] # where positive slopes are followed by another positive slope
        pstrend=len(pairs)/len(slopes) #F30

        # Checks if the flux crosses the 'zero' line.
        zcrossind = np.where(corrnf[:-1]*corrnf[1:]<0)
        num_zcross = len(zcrossind) #F31

        plusminus = np.where((slopes[1:]<0)&(slopes[:-1]>0))[0]
        num_pm = len(plusminus)

        # This looks up the local maximums. Adds a peak if it's the largest within 10 points on either side.
        # Q: Is there a way to do this and take into account drastically different periodicity scales?

        naivemax,nmax_times = [],[]
        naivemins,nmin_times = [],[]
        for j in range(len(nf)):
            if nf[j] == max(nf[max(j-10,0):min(j+10,len(nf)-1)]):
                naivemax.append(nf[j])
                nmax_times.append(t[j])
            elif nf[j] == min(nf[max(j-10,0):min(j+10,len(nf)-1)]):
                naivemins.append(nf[j])
                nmin_times.append(t[j])
        naivemax = np.array(naivemax)
        nmax_times = np.array(nmax_times)
        nmax_inds = np.argsort(t)[np.searchsorted(t,nmax_times,sorter=np.argsort(t))]
        naivemins = np.array(naivemins)
        nmin_times = np.array(nmin_times)
        nmin_inds = np.argsort(t)[np.searchsorted(t,nmin_times,sorter=np.argsort(t))]

        
        len_nmax=len(naivemax) #F33
        len_nmin=len(naivemins) #F34
        
        
        
        if len(naivemax)>2:
            mautocorrcoef = np.corrcoef(naivemax[:-1], naivemax[1:])[0][1] #F35
        else:
            mautocorrcoef = 0
            
        """peak to peak slopes"""
        ppslopes = (naivemax[1:]-naivemax[:-1])/(nmax_times[1:]-nmax_times[:-1])

        if len(ppslopes)==0:
            ptpslopes = 0
        else:
            ptpslopes=np.mean(ppslopes) #F36

        maxdiff = nmax_times[1:]-nmax_times[:-1]

        if len(maxdiff)==0:
            periodicity=0
            periodicityr=0
            naiveperiod=0
        else:
            periodicity=np.std(maxdiff)/np.mean(maxdiff) #F37
            periodicityr=np.sum(abs(maxdiff-np.mean(maxdiff)))/np.mean(maxdiff) #F38
            naiveperiod=np.mean(maxdiff) #F39
        if len(naivemax)==0:
            maxvars=0
            maxvarsr=0
        else:
            maxvars = np.std(naivemax)/np.mean(naivemax) #F40
            maxvarsr = np.sum(abs(naivemax-np.mean(naivemax)))/np.mean(naivemax) #F41

        emin = naivemins[::2] # even indice minimums
        omin = naivemins[1::2] # odd indice minimums
        meanemin = np.mean(emin)
        if len(omin)==0:
            meanomin=0
        else:
            meanomin = np.mean(omin)
        oeratio = meanomin/meanemin #F42

        # amp here is actually amp_2 in revantese
        # 2x the amplitude (peak-to-peak really)
        amp = np.percentile(nf,99)-np.percentile(nf,1) #F43
        normamp = amp / nf_mean #this should prob go, since flux is norm'd #F44

        # ratio of points within one fifth of the amplitude to the median to total number of points 
        mbp = len(nf[(nf<=(nf_med+0.1*amp))&(nf>=(nf_med-0.1*amp))]) / len(nf) #F45

        f595 = np.percentile(nf,95)-np.percentile(nf,5)
        f1090 =np.percentile(nf,90)-np.percentile(nf,10)
        f1782 =np.percentile(nf, 82)-np.percentile(nf, 17)
        f2575 =np.percentile(nf, 75)-np.percentile(nf, 25)
        f3267 =np.percentile(nf, 67)-np.percentile(nf, 32)
        f4060 =np.percentile(nf, 60)-np.percentile(nf, 40)
        mid20 =f4060/f595 #F46
        mid35 =f3267/f595 #F47
        mid50 =f2575/f595 #F48
        mid65 =f1782/f595 #F49
        mid80 =f1090/f595 #F50 

        percentamp = max(np.abs(nf-nf_med)/nf_med) #F51

        magratio = (max(nf)-nf_med) / amp #F52

        autocorrcoef = np.corrcoef(nf[:-1], nf[1:])[0][1] #F54

        sautocorrcoef = np.corrcoef(slopes[:-1], slopes[1:])[0][1] #F55

        #measures the slope before and after the maximums
        # reminder: 1 less slope than flux, slopes start after first flux
        # slope[0] is between flux[0] and flux[1]
            # mean of slopes before max will be positive
            # mean of slopes after max will be negative
        
        nmax_inds_subset = nmax_inds[(nmax_inds>5)&(nmax_inds<len(slopes)-5)]
        flatness = flat_func(slopes,nmax_inds_subset)
        
        if len(flatness)==0: flatmean=0
        else: flatmean = np.mean(flatness) #F55

        # measures the slope before and after the minimums
        # trying flatness w slopes and nf rather than "corr" vals, despite orig def in RN's program
          # mean of slopes before min will be negative
          # mean of slopes after min will be positive


        nmin_inds_subset = nmin_inds[(nmin_inds>5)&(nmin_inds<len(slopes)-5)]
        tflatness = tflat_func(slopes,nmin_inds_subset)
        
        # tflatness for mins, flatness for maxes
        if len(tflatness)==0: tflatmean=0
        else: tflatmean = np.mean(tflatness) #F56
            
        # reminder: 1 less second derivative than slope (2 less than flux). secder starts after first slope.
        # secder[0] is between slope[0] and slope[1], centered at flux[1]
        
        nmax_inds_subset = nmax_inds[(nmax_inds>5)&(nmax_inds<len(secder-5))]
        roundness = roundness_func(secder,nmax_inds_subset)
        
        if len(roundness)==0: roundmean=0
        else: roundmean = np.mean(roundness) #F57

        nmin_inds_subset = nmin_inds[(nmin_inds>5)&(nmin_inds<len(secder-5))]
        troundness = roundness_func(secder,nmin_inds_subset)

        if len(troundness)==0: troundmean=0
        else: troundmean = np.mean(troundness) #F58

        if troundmean==0 and roundmean==0: roundrat=1
        elif troundmean==0: roundrat=10
        else: roundrat = roundmean / troundmean #F59

        if flatmean==0 and tflatmean==0: flatrat=1
        elif tflatmean==0: flatrat=10
        else: flatrat = flatmean / tflatmean #F60

        ndata = np.array([longtermtrend, meanmedrat, skews, varss, coeffvar, stds, \
                 numoutliers, numnegoutliers, numposoutliers, numout1s, kurt, mad, \
                 maxslope, minslope, meanpslope, meannslope, g_asymm, rough_g_asymm, \
                 diff_asymm, skewslope, varabsslope, varslope, meanabsslope, absmeansecder, \
                 num_pspikes, num_nspikes, num_psdspikes, num_nsdspikes,stdratio, pstrend, \
                 num_zcross, num_pm, len_nmax, len_nmin, mautocorrcoef, ptpslopes, \
                 periodicity, periodicityr, naiveperiod, maxvars, maxvarsr, oeratio, \
                 amp, normamp, mbp, mid20, mid35, mid50, \
                 mid65, mid80, percentamp, magratio, sautocorrcoef, autocorrcoef, \
                 flatmean, tflatmean, roundmean, troundmean, roundrat, flatrat])
        
        fts = ["longtermtrend", "meanmedrat", "skews", "varss", "coeffvar", "stds", \
               "numoutliers", "numnegoutliers", "numposoutliers", "numout1s", "kurt", "mad", \
               "maxslope", "minslope", "meanpslope", "meannslope", "g_asymm", "rough_g_asymm", \
               "diff_asymm", "skewslope", "varabsslope", "varslope", "meanabsslope", "absmeansecder", \
               "num_pspikes", "num_nspikes", "num_psdspikes", "num_nsdspikes","stdratio", "pstrend", \
               "num_zcross", "num_pm", "len_nmax", "len_nmin", "mautocorrcoef", "ptpslopes", \
               "periodicity", "periodicityr", "naiveperiod", "maxvars", "maxvarsr", "oeratio", \
               "amp", "normamp", "mbp", "mid20", "mid35", "mid50", \
               "mid65", "mid80", "percentamp", "magratio", "sautocorrcoef", "autocorrcoef", \
               "flatmean", "tflatmean", "roundmean", "troundmean", "roundrat", "flatrat"]

        df = pd.DataFrame([ndata],index=[nfile[nfile.find('kplr'):]],columns=fts)

        with open('tmp_data.csv','a') as f:
            df.to_csv(f,header=False)
        if __name__=="__main__":
            return
        else:
            return nfile,ndata
        
    except TypeError:
        kml_log = 'kml_log'
        os.system('echo %s ... TYPE ERROR >> %s'%(nfile.replace(fitsDir,""),kml_log))
        return 

def features_from_fits(nfile):
    try:
        t,nf,err = read_kepler_curve(nfile)
        # t = time
        # err = error
        # nf = normalized flux.P
        
    except TypeError as err:
        # Files can be truncated by the zipping process.
        print("%s. Try downloading %s again."%(err,nfile))
        return
    
    features = featureCalculation(nfile,t,nf,err)
    if __name__=="__main__":
        return
    else:
        return features
    
def features_from_filelist(fl,fitDir,of,numCpus = cpu_count(), verbose=False):
    """
    This method calculates the features of the given filelist from the fits files located in fitsDir.
    All output is saved to a temporary csv file called tmp_data.csv.
    Run save_output(output file) and clean_up(filelist, fits file directory) to save to the desired location,
    and to clean up the filelist and tmp_data.csv. 
    
    Returns pandas dataframe of output.
    """
    global fitsDir
    fitsDir = fitDir
    
    if not os.path.isfile('tmp_data.csv'):
        df = pd.DataFrame({
        "longtermtrend":[], "meanmedrat":[], "skews":[], "varss":[], "coeffvar":[], "stds":[], \
        "numoutliers":[], "numnegoutliers":[], "numposoutliers":[], "numout1s":[], "kurt":[], "mad":[], \
        "maxslope":[], "minslope":[], "meanpslope":[], "meannslope":[], "g_asymm":[], "rough_g_asymm":[], \
        "diff_asymm":[], "skewslope":[], "varabsslope":[], "varslope":[], "meanabsslope":[], "absmeansecder":[], \
        "num_pspikes":[], "num_nspikes":[], "num_psdspikes":[], "num_nsdspikes":[],"stdratio":[], "pstrend":[], \
        "num_zcross":[], "num_pm":[], "len_nmax":[], "len_nmin":[], "mautocorrcoef":[], "ptpslopes":[], \
        "periodicity":[], "periodicityr":[], "naiveperiod":[], "maxvars":[], "maxvarsr":[], "oeratio":[], \
        "amp":[], "normamp":[], "mbp":[], "mid20":[], "mid35":[], "mid50":[], \
        "mid65":[], "mid80":[], "percentamp":[], "magratio":[], "sautocorrcoef":[], "autocorrcoef":[], \
        "flatmean":[], "tflatmean":[], "roundmean":[], "troundmean":[], "roundrat":[], "flatrat":[]})
        
        with open('tmp_data.csv','w') as f:df.to_csv(f)
    
    # files with path.
    print("Reading %s..."%fl)
    files = fl_files_w_path(fl,fitsDir)
    useCpus = min([len(files),numCpus-1])
    if verbose:
        print("Processing %s files..."%len(files))
        print("Using %s cpus to calculate features..."%useCpus)
                                         
    p = Pool(numCpus)
    # Method saves to tmp_data.csv file to save on system memory
    p.map(features_from_fits,files)
    p.close()
    p.join()
    
    if verbose:
        print("Features have been calculated")
        print("Cleaning up...")
    clean_up(fl)
    
    if verbose:print("Saving output to %s"%of)
    save_output(of)

    if verbose:print("Done.")
    
    if __name__=="__main__":
        return
    else:
        return pd.read_csv(of)
    
if __name__=="__main__":
    """
    If this is run as a script, the following will parse the arguments it is fed, 
    or prompt the user for input.
    
    python keplerml.py path/to/filelist path/to/fits_file_directory path/to/output_file
    """
    # fl - filelist, a txt file with file names, 1 per line
    if sys.argv[1]:
        fl = sys.argv[1]
    else:
        fl = raw_input("Input path: ")

    if sys.argv[2]:
        fitsDir = sys.argv[2]
    else:
        fitsDir = raw_input("Fits files directory path: ")
    # of - output file
    if sys.argv[3]:
        of = sys.argv[3]
    else:
        of = raw_input('Output path: ')
        if of  == "":
            print("No output path specified, saving to output.csv in local folder.")
            of = 'output.csv'
    from datetime import datetime
    start = datetime.now()
    features_from_filelist(fl,fitsDir,of,verbose=True)
    print(datetime.now()-start)
    