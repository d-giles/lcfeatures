import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx
import matplotlib.gridspec as gridspec

import numpy as np
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
