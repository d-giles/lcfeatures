import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx
import matplotlib.gridspec as gridspec

from IPython.display import display

import numpy as np
np.set_printoptions(threshold=np.nan)
import pandas as pd

import seaborn as sns
import pyfits as fits

import sys
sys.path.append('python')
from clusterOutliers import clusterOutliers

def make_sampler(inds=['8462852']): 
    """
    Args:
        inds (Array of strings) - array of indices, as identifying strings, to be pulled from a data frame, 
                                  can be with or without kplr prefix
    Returns:
        Function that will pull the data, indicated by inds, from a dataframe df 
        
    Useful to generate samples across quarters with common sources where data is contained as a 
    Pandas dataframe, with indices set to be identifying labels (i.e. kplr008462852)
    
    To use:
        Define array containing IDs of sources of interest as strings
        Define a sample generator by calling make_sampler(inds=Array of string IDs)
        Generate dataframe by calling new function.

    Example:
    tabby_sample = make_sampler(inds=['8462852'])
    Q4_sample = tabby_sample(Q4.data)
    Q8_sample = tabby_sample(Q8.data)
    etc.
    """
    return lambda df: df[df.index.str.contains('|'.join(inds))]

def import_generator(suffix='_FullSample.csv'):
    """
    Creates a function to import quarters with common suffixes (like "_output.csv" or "_PaperSample.csv")
    """
    return lambda QN: clusterOutliers("/home/dgiles/Documents/KeplerLCs/output/"+QN+suffix,"/home/dgiles/Documents/KeplerLCs/fitsFiles/"+QN+"fitsfiles")

def colors_for_plot(inds,cmap='nipy_spectral'):
    """
    Args:
        inds (array of ints size n) - array to be converted into color values
        cmap (str) - colormap of desired output
        
    Returns:
        colorVal (numpy array, size (n,4)) - numpy array containing colors as rgba array or hex color values
        
    colormap 'color_blind' consists of 3 distinct colors specifically chosen that show up well regardless of color-sight
    """
    if cmap=='color_blind':
        # Custom set of colors to use that are color blind friendly
        color_blind = {0:"#009999", 1:"#FF6633", -1:"#333366"}
        colorVal = np.array([color_blind[i] for i in inds])
    else:
        cNorm  = colors.Normalize(min(inds), max(inds))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)
        colorVal = scalarMap.to_rgba(inds)
    return colorVal

def four_panel(data, title='Data',
               col_clus='Greens_d',shade_clus=False,
               col_out='Purples_d',shade_out=False,
               col_edge='Oranges_d',shade_edge=False,
               types='all_kde',alpha=1):
    """
    Four paneled figure separating out different designations in given data.
    Args:
        data (pandas dataframe) - dataframe to be plotted, must have columns 'db_out', 'tsne_x', and 'tsne_y'
        title (str) - Desired title for whole plot
        col_clus (str) - gradient for clustered kde
        shade_clus (boolean) - for kde plots, if False, only plots lines
        col_out (str) - see col_clus
        shade_out (boolean) - see shade_clus
        col_edge (str) - see col_clus
        shade_edge (boolean) - see shade_clus
        types (str) - What each panel should be, either KDE or scatter plot
        alpha (float, 0 to 1) - how opaque layers should be
    Returns:
        None - plots desired data
    """
    
    titlesize = 36
    ticksize = 30
    
    if types=='all_kde':
        types='kkkk'
    elif types =='all_scatter':
        types='ssss'
    comb_plot=types[3:] # the combined plot type (will be k (kde) or s (scatter))
    
    # If shading is enabled, reverse the default color gradient
    if shade_clus and col_clus=='Greens_d':
        col_clus='Greens'
    if shade_out and col_out=='Purples_d':
        col_out='Purples'
    if shade_edge and col_edge=='Oranges_d':
        col_edge='Oranges'
        
    sns.set_style('white')
    labels = data.db_out
    
    outliers = data[labels==-1]
    core = data[labels==0]
    edge = data[labels==1]
    
    # for scatter plots, creates common color map to be used in different plots w/ different data
    colorVal=colors_for_plot(labels,cmap='viridis')

    fig = plt.figure(figsize=(15,15))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    plt.suptitle(title,fontsize=titlesize)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    panels = [ax1,ax2,ax3,ax4]
    
    # Panel 1
    panel(core[core.tsne_x<40],cmap=col_clus,shade=shade_clus,
          c=colorVal[data.db_out==0][core.tsne_x<40],t=types[0],ax=ax1,alpha=alpha)
    ax1.set_title('(a)',fontsize=titlesize,verticalalignment='bottom')
    ax1.tick_params(labelsize=ticksize,labelbottom='off')
    ax1.set_xlabel('')
    ax1.set_ylabel('t-SNE y',fontsize=titlesize)
    
    # Panel 2
    panel(outliers[outliers.tsne_x<40],cmap=col_out,shade=shade_out,
          c=colorVal[data.db_out==-1][outliers.tsne_x<40],t=types[1],ax=ax2,alpha=alpha)
    ax2.set_title('(b)',fontsize=titlesize,verticalalignment='bottom')
    ax2.tick_params(labelleft='off',labelbottom='off')
    ax2.set_xlabel('')
    ax2.set_ylabel('')
    
    # Panel 3
    panel(edge[edge.tsne_x<40],cmap=col_edge,shade=shade_edge,
          c=colorVal[data.db_out==1][edge.tsne_x<40],t=types[2],ax=ax3,alpha=alpha)
    ax3.set_title('(c)',fontsize=titlesize,verticalalignment='bottom')
    ax3.tick_params(labelsize=ticksize)
    ax3.set_xlabel('t-SNE x',fontsize=titlesize)
    ax3.set_ylabel('t-SNE y',fontsize=titlesize)
    
    # Panel 4
    if comb_plot=='k':
        # All kde
        panel(core[core.tsne_x<40],cmap=col_clus,shade=shade_clus,t='k',ax=ax4,alpha=alpha)
        panel(outliers[outliers.tsne_x<40],cmap=col_out,shade=shade_out,t='k',ax=ax4,alpha=alpha)
        panel(edge[edge.tsne_x<40],cmap=col_edge,shade=shade_edge,t='k',ax=ax4,alpha=alpha)
    elif comb_plot=='s':
        # All scatter
        panel(core[core.tsne_x<40],c=colorVal[data.db_out==0][core.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(outliers[outliers.tsne_x<40],c=colorVal[data.db_out==-1][outliers.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(edge[edge.tsne_x<40],c=colorVal[data.db_out==1][edge.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        
    elif comb_plot=='ks':
        # Cluster kde, outliers scatter
        panel(core[core.tsne_x<40],cmap=col_clus,shade=shade_clus,t='k',ax=ax4,alpha=alpha)
        panel(outliers[outliers.tsne_x<40],c=colorVal[data.db_out==-1][outliers.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(edge[edge.tsne_x<40],cmap=col_edge,shade=shade_edge,t='k',ax=ax4,alpha=alpha)
    
    elif comb_plot=='kands':
        panel(core[core.tsne_x<40],c=colorVal[data.db_out==0][core.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(outliers[outliers.tsne_x<40],c=colorVal[data.db_out==-1][outliers.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        panel(edge[edge.tsne_x<40],c=colorVal[data.db_out==1][edge.tsne_x<40],t='s',ax=ax4,alpha=alpha)
        
        panel(core[core.tsne_x<40],cmap=col_clus,shade=shade_clus,t='k',ax=ax4,k_alpha=.5)
        panel(outliers[outliers.tsne_x<40],cmap=col_out,shade=shade_out,t='k',ax=ax4,k_alpha=.5)
        panel(edge[edge.tsne_x<40],cmap=col_edge,shade=shade_edge,t='k',ax=ax4,k_alpha=.5)
    #"""


    ax4.set_title('(d)',fontsize=titlesize,verticalalignment='bottom')
    ax4.tick_params(labelsize=ticksize,labelleft='off')
    ax4.set_xlabel('t-SNE x',fontsize=titlesize)
    ax4.set_ylabel('')
    
    ax1.set_xlim(ax4.get_xlim())
    ax1.set_ylim(ax4.get_ylim())
    ax2.set_xlim(ax4.get_xlim())
    ax2.set_ylim(ax4.get_ylim())
    ax3.set_xlim(ax4.get_xlim())
    ax3.set_ylim(ax4.get_ylim())
    
    return


def read_kepler_curve(file):
    """
    Given the path of a fits file, this will extract the light curve and normalize it.
    """
    lc = fits.open(file)[1].data
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

def plot_lc(file,filepath,c='blue',ax=False):
    """
    kid should be full id including time information
    Args:
        file (str) - filename starting with kplr ending in .fits
        filepath (str) - path to fits file
    Returns:
        None
    """
    if not ax:
        fig = plt.figure(figsize=(16,4))
        ax = fig.add_subplot(111)
    f = filepath+file
    
    t,nf,err=read_kepler_curve(f)

    x=t
    y=nf

    axrange=0.55*(max(y)-min(y))
    mid=(max(y)+min(y))/2
    yaxmin = mid-axrange
    yaxmax = mid+axrange
    if yaxmin < .95:
        if yaxmax > 1.05:
            ax.set_ylim(yaxmin,yaxmax)
        else:
            ax.set_ylim(yaxmin,1.05)
    elif yaxmax > 1.05:
        ax.set_ylim(.95,yaxmax)
    else:
        ax.set_ylim(.95,1.05)
        
    ax.set_xlim(min(x),max(x))
    color = c
    ax.plot(x, y, 'o',markeredgecolor='none', c=color, alpha=0.2)
    ax.plot(x, y, '-',markeredgecolor='none', c=color, alpha=0.7)
    #ax2.set_title(files[index][:13],fontsize = 20)
    ax.set_ylabel(r'$\frac{\Delta F}{F}$',fontsize=25)
    ax.tick_params(labelsize=20)
    #ax.set_title('KID'+str(int(file[4:13])),fontsize=25)
    
def four_Q_lc(kid,Qa,Qb,Qc,Qd):
    fig = plt.figure(figsize=(12,12))
    
    
    title_text = 'KID '+str(int(kid[4:13]))
    
    ax1 = fig.add_subplot(411)
    plt.title(title_text,fontsize=25)

    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)
    sampler =  make_sampler([kid])
    QaID = sampler(Qa.data)
    QbID = sampler(Qb.data)
    QcID = sampler(Qc.data)
    QdID = sampler(Qd.data)
    cs = colors_for_plot([QaID.db_out[0],QbID.db_out[0],QcID.db_out[0],QdID.db_out[0]],cmap='color_blind')
    
        
    plot_lc(QaID.index[0],filepath=Qa.fitsDir,c=cs[0],ax=ax1)
    plot_lc(QbID.index[0],filepath=Qb.fitsDir,c=cs[1],ax=ax2)
    plot_lc(QcID.index[0],filepath=Qc.fitsDir,c=cs[2],ax=ax3)
    plot_lc(QdID.index[0],filepath=Qd.fitsDir,c=cs[3],ax=ax4)
    ax4.set_xlabel('Time (Days)',fontsize=22)
    
    legend_cs = colors_for_plot([0,1,-1],cmap='color_blind')
    ax1.scatter([],[],c=legend_cs[-1],s=50,label='Outlier')
    ax1.scatter([],[],c=legend_cs[0],s=50,label='Core Cluster')
    ax1.scatter([],[],c=legend_cs[1],s=50,label='Edge Cluster')
    ax1.legend(loc='upper center',bbox_to_anchor=(0.5,1.05),ncol=3, fontsize=18)
    
    fig.tight_layout()

    
catalogs = {'koi_full':['list_koi_full.txt',',',2],
            'koi_confirmed':['list_koi_confirmed.txt',',',2],
            'koi_candidate':['list_koi_candidate.txt',',',2],
            'koi_fp':['list_koi_fp.txt',',',2],
            'EB':['list_EBs.csv',',',0],
            'HB':['list_kepler_heartbeats.txt',None,37],
            'flares':['kepler_solar_flares.txt',None,22],
            'no_signal':['list_kepler_nosig.txt',None,28],
            'periodic':['list_kepler_MSperiods.txt',None,32]}

class weirdnessProfile(object):
    def __init__(self,
                 KIC='8462852',
                 Qs=['Q4','Q8','Q11','Q16'],
                 analysis_path='/home/dgiles/Documents/KeplerLCs/output/Analysis/',
                 fitsDirPath='/home/dgiles/Documents/KeplerLCs/fitsFiles/',
                 verbose=True):
        """
        This object contains summary information about anomaly detection run on KIC lightcurves.
        Provide the KIC and the quarters for which analysis has been run, and this will create 
        a weirdness dossier, or profile for that KIC.
        Quarters must have been analyzed with an _analysis.csv file produced containing scores
        and 2 pca components for each KIC. 
        At the time of writing (3/8/19), only quarters 4, 8, 11, and 16 have been analyzed.
        
        This profiler cross references a number of catalogs whose members are stored in various
        formats. The 'catalogs' dictionary contains information on how to extract these filelists
        As a matter of convenience, copies of these analysis files have been created for each 
        catalog for each quarter rather than resample for each catalog.
        This will likely become unweildy as more catalogs and quarters are added (it already is),
        and methods will be updated, potentially impacting peformance.
        """
        if KIC[:3]=='KIC':
            self.KIC = str(int(KIC[3:]))
        elif KIC[:4]=='kplr':
            self.KIC = str(int(KIC[4:]))
        else:
            try:
                self.KIC = str(int(KIC))
            except:
                print("ID not recognized. Try without a prefix?")
        self.Qs = Qs
        self.analysis_path = analysis_path
        self.fitsDirPath = fitsDirPath
        self.sampler = lambda df: df[df.index.str.contains(self.KIC)]
        self.full_analysis_data = {Q:pd.read_csv(self.analysis_path+Q+'_analysis.csv',index_col=0) for Q in self.Qs}
        self.analysis_data = {Q:self.sampler(self.full_analysis_data[Q]) for Q in self.Qs}
        self.scores = self.scoreSummary()
        self.ranks = self.rankSummary()
        self.catalogs,self.ctlg_sampler = self.catalogCheck()
        self.ctlg_summary = self.catalogCompare()
        if verbose:self.printSummary()
        
    def scoreSummary(self):
        scores = pd.DataFrame()
        for Q in self.Qs:
            dftmp = self.analysis_data[Q].loc[:,['dist_score','PCA90_score','PCA95_score','PCA99_score']]
            dftmp.index = [Q]
            scores = scores.append(dftmp)
        return scores
    
    def rankSummary(self):
        # Returns the percentile of the object in relation to the rest of the quarter's data, generally more illuminating than the score
        full_ranks={}
        for Q in self.Qs:
            full_ranks[Q] = self.full_analysis_data[Q].loc[:,[]]
            full_ranks[Q]['Full'] = self.full_analysis_data[Q].dist_score.rank(ascending=False)
            full_ranks[Q]['Full_pctl'] = self.full_analysis_data[Q].dist_score.rank(ascending=True)/len(self.full_analysis_data[Q])*100
            full_ranks[Q]['PCA90'] = self.full_analysis_data[Q].PCA90_score.rank(ascending=False)
            full_ranks[Q]['PCA90_pctl'] = self.full_analysis_data[Q].PCA90_score.rank(ascending=True)/len(self.full_analysis_data[Q])*100
            full_ranks[Q]['PCA95'] = self.full_analysis_data[Q].PCA95_score.rank(ascending=False)
            full_ranks[Q]['PCA95_pctl'] = self.full_analysis_data[Q].PCA95_score.rank(ascending=True)/len(self.full_analysis_data[Q])*100
            full_ranks[Q]['PCA99'] = self.full_analysis_data[Q].PCA99_score.rank(ascending=False)
            full_ranks[Q]['PCA99_pctl'] = self.full_analysis_data[Q].PCA99_score.rank(ascending=True)/len(self.full_analysis_data[Q])*100
        self.full_ranks = full_ranks
        
        ranks = pd.DataFrame()
        for Q in self.Qs:
            dftmp = self.sampler(full_ranks[Q])
            dftmp.index = [Q]
            ranks = ranks.append(dftmp)
            
        return ranks
    
    def plotlcs(self):
        for Q in self.Qs:
            plot_lc(self.analysis_data[Q].index[0],self.fitsDirPath+Q+'fitsfiles/')
        return
    
    def pcaPlots(self):
        # Plots each quarter's data into the top 2 principle components from a PCA reduction
        for Q in self.Qs:
            x = self.full_analysis_data[Q].pca_x
            y = self.full_analysis_data[Q].pca_y
            fig = plt.figure(figsize=(4,4))
            ax = fig.add_subplot(111)
            ax.scatter(x,y)
            ax.scatter(self.analysis_data[Q].pca_x,self.analysis_data[Q].pca_y)
            
    def catalogCheck(self):
        # Checks the available catalogs to see if the object is contained in them
        in_catalogs = []
        ctlg_sampler = {}
        # Catalogs of individual classes are looped through here
        for ctlg in catalogs:
            import_ref = catalogs[ctlg]
            ctlg_list = np.genfromtxt(import_ref[0],delimiter=import_ref[1],skip_header=import_ref[2],usecols=(0),dtype=str)
            if self.KIC in ctlg_list:
                in_catalogs.append(ctlg)
                ctlg_sampler[ctlg] = make_sampler(ctlg_list)
                
        # Debosscher and SIMBAD have a bunch of different classifications, each need parsed out
        lines = np.genfromtxt('kepler_variable_classes.txt',dtype=str,skip_header=50)
        Debosscher_full_df = pd.DataFrame(data=lines[:,4],columns=["Class"],index=lines[:,0])
        if Debosscher_full_df.index.str.contains(self.KIC).any():
            deb_class = Debosscher_full_df.loc[self.KIC,'Class']
            ctlg_sampler['Deb_Class_%s'%deb_class]=make_sampler(Debosscher_full_df[Debosscher_full_df.Class==deb_class].index)
            in_catalogs.append('Deb_Class_%s'%deb_class)
            
        lines = np.genfromtxt('simbad.csv',delimiter=';',skip_header=7,dtype=str)[:,1:]
        simbad_full_df = pd.DataFrame(data=lines[:,1],columns=["Class"],index=lines[:,0])
        if simbad_full_df.index.str.contains(self.KIC).any():
            sim_class = simbad_full_df.loc[simbad_full_df.index.str.contains(self.KIC),'Class'][0]
            ctlg_sampler['SIMBAD_Class_%s'%sim_class]=make_sampler(simbad_full_df[simbad_full_df.Class == sim_class].index)
            in_catalogs.append('SIMBAD_Class_%s'%sim_class)
            
        return in_catalogs,ctlg_sampler
    
    def catalogCompare(self):
        # Produces summary dataframes for each catalog the object belongs to, 
        # the summary includes the mean score and the standard deviation of the scores of the catalog
        catalog_data_summary = {}
        for ctlg in self.catalogs:
            catalog_data_summary[ctlg] = pd.DataFrame()
            for Q in self.Qs:
                path = self.analysis_path+Q+'_'+ctlg+'_analysis.csv'
                ctlg_data = pd.read_csv(path,index_col=0)
                tmp_df = pd.DataFrame({'dist_score':ctlg_data.dist_score.mean(),
                                       'dist_std':ctlg_data.dist_score.std(),
                                       'PCA90_score':ctlg_data.PCA90_score.mean(),
                                       'PCA90_std':ctlg_data.PCA90_score.std(),
                                       'PCA95_score':ctlg_data.PCA95_score.mean(),
                                       'PCA95_std':ctlg_data.PCA95_score.std(),
                                       'PCA99_score':ctlg_data.PCA99_score.mean(),
                                       'PCA99_std':ctlg_data.PCA99_score.std()},
                                      index=[Q])
                catalog_data_summary[ctlg] = catalog_data_summary[ctlg].append(tmp_df)
        return catalog_data_summary
    def ctlgCmpViz(self,ctlg,scoreType='dist'):
        # Plots the score of the object against the score of the catalog with errorbars
        qs = [int(i[1:]) for i in self.scores.index]
        plt.scatter(qs,self.scores.loc[:,scoreType+'_score'])
        #plt.scatter(qs,self.ctlg_summary[ctlg].loc[:,scoreType+'_score'])
        plt.errorbar(qs,
                     self.ctlg_summary[ctlg].loc[:,scoreType+'_score'],
                     self.ctlg_summary[ctlg].loc[:,scoreType+'_std'])
        
    def printSummary(self):
        # Prints out the scores for the object in each given quarter, the catalogs it's contained in, 
        # and any significant score deviations from its identified catalogs
        print("Scores of KIC {}".format(self.KIC))
        display(self.scores)
        display(self.ranks)
        print("KIC{} contained in these catalogs: {}".format(self.KIC,self.ctlg_summary.keys()))
        for k in self.ctlg_summary.keys():
            for Q in self.Qs:
                if abs(self.ctlg_summary[k].loc[Q,'dist_score']-self.scores.loc[Q,'dist_score'])>self.ctlg_summary[k].loc[Q,'dist_std']:
                    print("Significant deviation (>1 sigma) from {} catalog in Quarter {}".format(k,Q))