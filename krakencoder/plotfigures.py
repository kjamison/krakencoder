"""
Functions for plotting prediction heatmaps and training curves
"""

from matplotlib import pyplot as plt
from matplotlib import use as matplotlib_use_backend

#from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cycler import cycler

import numpy as np
from scipy.io import loadmat
from scipy.signal import convolve
from scipy.interpolate import interp1d
import re
import warnings
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from .data import canonical_data_flavor
from .utils import *
from ._resources import resource_path

def extra_colormap_list():
    return ['magma2','magma2_r','flare','flare_r','rocket','rocket_r']

def load_custom_colormap(colormap_name):
    #these additional colormaps were generated for https://github.com/kjamison/colormap_matplotlib_matlab
    cmap_matfile=resource_path('matplotlib_colormaps.mat')
    Ccmap=loadmat(cmap_matfile,simplify_cells=True)
    
    #add custom "magma2" colormap that starts at 10% of the way through the original "magma" colormap
    #(so that the darkest colors are not used, which are hard to see differences in)
    #Ccmap['magma2']=scipy.interpolate.interp1d(np.linspace(0,1,256),Ccmap['magma'],axis=0)(np.linspace(.10,1,256))
    
    #no... use even more customized "magma2" using colorspacious
    
    Ccmap['flare']=np.array(sns.color_palette("flare",n_colors=256))
    Ccmap['rocket']=np.array(sns.color_palette("rocket",n_colors=256))
    #Ccmap['magma2_r']=Ccmap['magma2'][::-1]
    Ccmap['flare_r']=Ccmap['flare'][::-1]
    Ccmap['rocket_r']=Ccmap['rocket'][::-1]
    
    cmap=LinearSegmentedColormap.from_list(name=colormap_name,colors=Ccmap[colormap_name])
    return cmap

#used during training to plos loss traces
def plotloss(x,plotstyle='-',lossviewskip=0,showmax=False,smoothN=0,colors=['r','b','g','m','k','c','lime','orange']):
    x=np.atleast_2d(x).T
    if colors is not None:
        if len(colors) > x.shape[1]:
            colors=colors[:x.shape[1]]
        plt.gca().set_prop_cycle(cycler('color',colors))
    
    if smoothN > 0:
        linewidth=.25
    else:
        linewidth=1
    plt.plot(range(x.shape[0])[lossviewskip:],x[lossviewskip:,:],plotstyle,linewidth=linewidth)
    if smoothN>0 and x.shape[0]>smoothN:
        linewidth_smooth=1
        f=np.ones(smoothN)/smoothN
        xsmooth=np.vstack([np.convolve(x[:,i],f,'same') for i in range(x.shape[1])]).T
        xsmooth[-int(smoothN/2):,:]=np.nan
        xsmooth[:int(smoothN/2),:]=np.nan
        plt.plot(range(x.shape[0])[lossviewskip:],xsmooth[lossviewskip:,:],plotstyle,linewidth=linewidth_smooth)
    
    try:
        if showmax:
            midx=np.nanargmax(x,axis=0)
        else:
            midx=np.nanargmin(x,axis=0)
        midx2=np.ravel_multi_index((midx,range(x.shape[1])),x.shape)
    except:
        midx=np.zeros(x.shape[1])
        midx2=None
        
    if midx2 is None:
        return
    
    #plt.plot(midx,x.flatten()[midx2],plotstyle.replace(":","")+'o',linestyle='')
    if colors is not None:
        for i in range(len(midx)):
            if plotstyle==':':
                #for dotted lines (training curves), use hollow circles
                plt.scatter(midx[i],x.flatten()[midx2[i]],facecolors='none',edgecolor=colors[i%len(colors)],label='_nolegend_')
            else:
                plt.scatter(midx[i],x.flatten()[midx2[i]],c=colors[i%len(colors)],label='_nolegend_')
        #plt.scatter(midx,x.flatten()[midx2],c=colors)
        
    else:
        plt.scatter(midx,x.flatten()[midx2],label='_nolegend_')

#used during training to plos loss traces
def update_training_figure(loss_train, loss_val, corrloss_train, corrloss_val, corrlossN_train, corrlossN_val, corrlossRank_train, corrlossRank_val, 
    avgrank_train, avgrank_val, trainpath_names_short, data_string, network_string,train_string, losstype, epoch, trainfig=None, gridalpha=.25, topN=2):

    #set agg backend since default/TkAgg/QtAgg causes error messages with VNC/remote sometimes
    matplotlib_use_backend('agg')

    if trainfig is None:
        trainfig=plt.figure(figsize=(15,10))

    trainfig.clear()
    plt.subplot(2,3,1)
    plotloss(loss_train,':')
    plotloss(loss_val,'-')
    plt.grid(True,alpha=gridalpha)
    #plt.legend(trainpath_names_short)
    plt.legend([s+".tr" for s in trainpath_names_short]+[s+".val" for s in trainpath_names_short])
    plt.title("%s loss" % (losstype))

    plt.subplot(2,3,2)
    plotloss(loss_train,':',lossviewskip=5,smoothN=5)
    plotloss(loss_val,'-',lossviewskip=5,smoothN=5)
    plt.grid(True,alpha=gridalpha)
    plt.title("%s loss (epoch>%d)" % (losstype,5))

    plt.subplot(2,3,4)
    plotloss(corrloss_train,':',showmax=True,smoothN=5)
    plotloss(corrloss_val,'-',showmax=True,smoothN=5)
    plt.grid(True,alpha=gridalpha)
    plt.title("corr top1acc")

    plt.subplot(2,3,5)
    plotloss(corrlossN_train,':',showmax=True,smoothN=5)
    plotloss(corrlossN_val,'-',showmax=True,smoothN=5)
    plt.grid(True,alpha=gridalpha)
    plt.title("corr top%dacc" % (topN))

    plt.subplot(2,3,6)
    plotloss(corrlossRank_train,':',showmax=True)
    plotloss(corrlossRank_val,'-',showmax=True)
    plt.grid(True,alpha=gridalpha)
    plt.title("corr avgrank %ile")

    plt.subplot(2,3,3)
    plotloss(avgrank_train,':',showmax=False,lossviewskip=0,smoothN=5)
    plotloss(avgrank_val,'-',showmax=False,lossviewskip=0,smoothN=5)

    if epoch > 50:
        ymax=np.nanmax(np.vstack((avgrank_train,avgrank_val))[:,50:])
        plt.ylim([.9, ymax])

    plt.grid(True,alpha=gridalpha)
    plt.title("corr avgrank")
    plt.suptitle('%s: %s, %s' % (data_string,network_string,train_string))
    
    return trainfig

######################
def shorten_names(namelist,extrashort=0,remove_common=True,remove_strings=[]):
    if len(namelist)>1 and all([x==namelist[0] for x in namelist]):
        return namelist
    namelist=[re.sub("_FC$","",x) for x in namelist]
    namelist=[re.sub("(FCcov|FCcorr|FCpcorr)_(fs86|shen268|coco439|[^_]+)_(.+)",r"\2_\1_\3",x) for x in namelist]
    namelist=[re.sub("(fs86|shen268|coco439|[^_]+)_(sdstream|ifod2act)(_volnorm|_volnormicv)?",r"\1_SC\2\3",x) for x in namelist]
    namelist=[re.sub("SC(sdstream|ifod2act|[^_]+)_(fs86|shen268|coco439|[^_]+)(_volnorm|_volnormicv)?",r"\2_SC\1\3",x) for x in namelist]
    
    if extrashort > 0:
        namelist=[re.sub("_(FCcov|FCcorr)_hpf$","_FC#hpf",x) for x in namelist]
        namelist=[re.sub("_(FCcov|FCcorr)_hpfgsr$","_FCgsr#hpf",x) for x in namelist]
        namelist=[re.sub("_FCpcorr_hpf$","_FCpc#hpf",x) for x in namelist]
        #namelist=[re.sub("_FCcov_hpf$","_FC",x) for x in namelist]
        #namelist=[re.sub("_FCcov_hpfgsr$","_FCgsr",x) for x in namelist]
        #namelist=[re.sub("_FCpcorr_hpf$","_FCpc",x) for x in namelist]
        
        namelist=[re.sub("_(FCcov|FCcorr)_bpf$","_FC#bpf",x) for x in namelist]
        namelist=[re.sub("_(FCcov|FCcorr)_bpfgsr$","_FCgsr#bpf",x) for x in namelist]
        namelist=[re.sub("_FCpcorr_bpf$","_FCpc#bpf",x) for x in namelist]
        
        #namelist=[re.sub("_(FCcov|FCcorr)_nofilt$","_FC#nf",x) for x in namelist]
        #namelist=[re.sub("_(FCcov|FCcorr)_nofiltgsr$","_FCgsr#nf",x) for x in namelist]
        #namelist=[re.sub("_FCpcorr_nofilt$","_FCpc#nf",x) for x in namelist]
        namelist=[re.sub("_(FCcov|FCcorr)_nofilt$","_FC#mpp",x) for x in namelist]
        namelist=[re.sub("_(FCcov|FCcorr)_nofiltgsr$","_FCgsr#mpp",x) for x in namelist]
        namelist=[re.sub("_FCpcorr_nofilt$","_FCpc#mpp",x) for x in namelist]
        
        namelist=[re.sub("_SCsdstream","_SCsd",x) for x in namelist]
        namelist=[re.sub("_SCifod2act","_SCifod",x) for x in namelist]
        
        #namelist=[re.sub("_count",".cnt",x) for x in namelist]
        #namelist=[re.sub("_sift2$",".sift2",x) for x in namelist]
        #namelist=[re.sub("_sift2volnorm",".sift2vn",x) for x in namelist]
        
        namelist=[re.sub("^(.+)_volnorm_(SC[^_]+)$",r"\1_\2.vn",x) for x in namelist]
        namelist=[re.sub("^(.+)_volnormicv_(SC[^_]+)$",r"\1_\2.vnicv",x) for x in namelist]
        namelist=[re.sub("^(.+)_count_(SC[^_]+)$",r"\1_\2.cnt",x) for x in namelist]
        namelist=[re.sub("^(.+)_sift2_(SC[^_]+)$",r"\1_\2.sift2",x) for x in namelist]
        namelist=[re.sub("^(.+)_sift2volnorm_(SC[^_]+)$",r"\1_\2.sift2vn",x) for x in namelist]
        
        namelist=[re.sub("^(.+)_(SC[^_]+)_volnorm$",r"\1_\2.vn",x) for x in namelist]
        namelist=[re.sub("^(.+)_(SC[^_]+)_volnormicv$",r"\1_\2.vnicv",x) for x in namelist]
        namelist=[re.sub("^(.+)_(SC[^_]+)_count$",r"\1_\2.cnt",x) for x in namelist]
        namelist=[re.sub("^(.+)_(SC[^_]+)_sift2$",r"\1_\2.sift2",x) for x in namelist]
        namelist=[re.sub("^(.+)_(SC[^_]+)_sift2volnorm$",r"\1_\2.sift2vn",x) for x in namelist]
        
        #if all FC are hpf, remove hpf string from name
        if all(["#hpf" in x for x in namelist if "FC" in x and not ("fusion" in x or "burst" in x)]):
            namelist=[x.replace("#hpf","") for x in namelist]
        
        #if all SC are volnorm, remove volnorm string from name
        if all([".vn" in x for x in namelist if "SC" in x and not ("fusion" in x or "burst" in x)]):
            namelist=[x if "vnicv" in x else x.replace(".vn","") for x in namelist]
            
        namelist=[re.sub("burst","fusion",x) for x in namelist]
        
    if extrashort == 2:
        namelist=[re.sub("shen268","sh268",x) for x in namelist]
        namelist=[re.sub("coco439","co439",x) for x in namelist]
        namelist=[re.sub("cocoyeo","yeo",x) for x in namelist]
        namelist=[re.sub("cocolaus","laus",x) for x in namelist]
        namelist=[re.sub("_SCsd","_SCdt",x) for x in namelist]
        namelist=[re.sub("_SCifod","_SCpr",x) for x in namelist]
        namelist=[re.sub(".noself","-1",x) for x in namelist]
        namelist=[re.sub(".noatlas","-parc",x) for x in namelist]
        namelist=[re.sub("burst","fusion",x) for x in namelist]
        namelist=["_".join(x.split("_")[::-1]) for x in namelist]
    
    namelist=[x.replace("#","_") for x in namelist]
    
    for r in remove_strings:
        namelist=[x.replace(r,"") for x in namelist]
        
    pref=common_prefix(namelist)
    suff=common_suffix(namelist)
    
    if len(namelist)>1:
        if remove_common and len(pref)>0:
            namelist=[x[len(pref):] for x in namelist]
        if remove_common and len(suff)>0:
            namelist=[x[:-len(suff)] for x in namelist]
    
    return namelist

def flavor2color(conntype):
    conntype=conntype.lower()
    color=[0,0,0]
    if "fs86" in conntype:
        if "fccov" in conntype or "fccorr" in conntype:
            color=[1,0,0]
        elif "pcorr" in conntype:
            color=[.5,0,0]
        elif "sdstream" in conntype:
            color=[0,.5,0]
        elif "ifod2act" in conntype:
            color=[0,.75,0]

    elif "shen268" in conntype:
        if "fccov" in conntype or "fccorr" in conntype:
            color=[0,0,1]
        elif "pcorr" in conntype:
            color=[0,0,.5]
        elif "sdstream" in conntype:
            color=[.5,0,.5]
        elif "ifod2act" in conntype:
            color=[.75,0,.75]

    elif "coco439" in conntype:
        if "fccov" in conntype or "fccorr" in conntype:
            color=[0,.5,.5]
        elif "pcorr" in conntype:
            color=[0,.25,.25]
        elif "sdstream" in conntype:
            color=[.5,.25,0]
        elif "ifod2act" in conntype:
            color=[1,.5,0]
    return color

#reorder by atlas and modality
def flavor_reorder(conntypes, pcorr_at_end=True, sort_atlas_last=False):
    atlas_order=["fs86","shen268","coco439"]
    atlas_order=["fs86","cocoyeo143","cocolaus157","cocoyeo243","cocolaus262","shen268","cocoyeo443","coco439","cocolaus491"]
    conntype_order_groups=[]
    if sort_atlas_last:
        conntype_order_groups.append([y[0] if len(y)>0 else len(atlas_order) for y in [np.where([a in x for a in atlas_order])[0] for x in conntypes]])
    conntype_order_groups.append([re.match("^(fusion|burst).*",x) is not None for x in conntypes])
    conntype_order_groups.append([re.match("^(fusion|burst).*noatlas",x) is not None for x in conntypes])
    conntype_order_groups.append([re.match("^(fusion|burst).*noself",x) is not None for x in conntypes])
    conntype_order_groups.append([x=='gap' for x in conntypes])
    conntype_order_groups.append([x=='mean' for x in conntypes])
    conntype_order_groups.append([re.match(".*(sdstream|ifod2|SC).*",x) is not None for x in conntypes])
    if pcorr_at_end:
        conntype_order_groups.append([re.match(".*pcorr.*",x) is not None for x in conntypes])
    conntype_order_groups.append([y[0] if len(y)>0 else len(atlas_order) for y in [np.where([a in x for a in atlas_order])[0] for x in conntypes]])
    conntype_order_groups.append([re.match(".*(ifod2).*",x) is not None for x in conntypes])
    conntype_order_groups.append([re.match(".*(volnorm).*",x) is not None for x in conntypes])
    conntype_order_groups.append([re.match(".*(sift2).*",x) is not None for x in conntypes])
    
    conntype_order_groups=np.stack(conntype_order_groups,axis=-1)
    
    newidx=np.lexsort(conntype_order_groups[:,::-1].T) #sort by columns
    try:
        conntypes_new=conntypes[newidx]
    except:
        conntypes_new=[conntypes[i] for i in newidx]
        
    return conntypes_new, newidx

def flavor2group(conntype):
    flavor_group=[]
    if conntype=='':
        g=''
    elif conntype=='mean':
        g='mean'
    elif 'FC' in conntype:
        g='FC'
    else:
        g='SC'
    
    if 'fusion' in conntype or 'burst' in conntype:
        g='fusion'
    if '.noself' in conntype:
        g+='.noself'
    elif '.noatlas' in conntype:
        g+='.noatlas'
    return g

def display_kraken_heatmap(trainrecord, 
                           metrictype="top1acc", 
                           single_epoch=True, epoch_spacing=None, explicit_epoch=None, epoch_filter_size=1, best_epoch_fraction=False,
                           origscale=True, training=False, addmean=True, show_epoch=False, 
                           extra_short_names=0, pcorr_at_end=True, exclude_flavors=None, 
                           show_heatmap_value_text=True, heatmap_value_text_size=10,
                           ticklabel_text_size=10,
                           colorbar=True, clim=None, colormap=None, invert_text_color=False,
                           ax=None, figsize=[18,12],
                           add_suptitle=True, add_epochtitle=True, bottomtext='<filename>',
                           outputimagefile=None, outputcsvfile=None, outputcsvfile_append=False):
    """
    Display a heatmap of the Kraken training record.

    Parameters:
    - trainrecord (dict or str): The training record data or the path to the training record file.
    - metrictype (str or list): The type of metric(s) to display. Default is "top1acc".
    - single_epoch (bool): Whether to display a single epoch or all epochs. Default is True.
    - epoch_spacing (int): The spacing between displayed epochs. Default is None.
    - explicit_epoch (int): The specific epoch to display. Default is None.
    - epoch_filter_size (int): The size of the epoch filter. Default is 1.
    - best_epoch_fraction (bool): Whether to display the best epoch as a fraction. Default is False.
    - origscale (bool): Whether to use the original scale. Default is True.
    - training (bool): Whether to display training data. Default is False.
    - addmean (bool): Whether to add the mean value. Default is True.
    - show_epoch (bool): Whether to display the epoch number. Default is False.
    - extra_short_names (int): The number of extra short names. Default is 0.
    - colorbar (bool): Whether to display the colorbar. Default is True.
    - ax (matplotlib.axes.Axes): The axes to plot on. Default is None.
    - figsize (list): The figure size. Default is [18, 12].
    - add_suptitle (bool): Whether to add a suptitle. Default is True.
    - add_epochtitle (bool): Whether to add an epoch title. Default is True.
    - bottomtext (str): The text to display at the bottom. Default is "<filename>".
    - outputimagefile (str or dict): The path to save the output image or a dictionary with additional save parameters. Default is None.

    Returns:
    - matplotlib.axes.Axes: The axes used for plotting (if only a single metric is displayed)
    - None (for a multi-metric plot)
    """

    #ignore this during nanmean() for figure generation
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        
    if isinstance(trainrecord,str):
        recordfile=trainrecord
        trainrecord=loadmat(trainrecord,simplify_cells=True)
        trainrecord['recordfile']=recordfile

    #set agg backend since default/TkAgg/QtAgg causes error messages with VNC/remote sometimes
    matplotlib_use_backend('agg')

    #from trainrecord, need:
    # recordfile, current_epoch, nbepochs, [starting_point_epoch], [starting_point_file]
    # trainpath_names, input_name_list, topN
    # (corrloss|corrlossN|corrlossRank|avgcorr|explainedvar)[_OrigScale]_(train|val)
        

    #dpi=300 for publication?
    figure_background_color='w'
    saveparams={'facecolor':figure_background_color,'dpi':100,'bbox_inches':0}
    
    suptitle_fontsize=20
    righttitle_fontsize=16
    bottomtext_fontsize=12 #might get reduced later to make text fit

    if outputimagefile is not None and not isinstance(outputimagefile,str):
        if 'file' in outputimagefile:
            tmp_saveparams=outputimagefile
            outputimagefile=tmp_saveparams['file']
            saveparams={k:tmp_saveparams[k] if k in tmp_saveparams else v for k,v in saveparams.items()}
        else:
            raise Exception('Unknown input for outputimagefile. Must be either filename or dict with ["file"] and other fields')

    if 'starting_point_file' in trainrecord and 'starting_point_epoch' in trainrecord and trainrecord['current_epoch']==0:
        bottomtext='<startingpoint>'
        trainrecord['current_epoch']=trainrecord['starting_point_epoch']
        explicit_epoch=trainrecord['starting_point_epoch']
    
    if not isinstance(metrictype,str):
        try:
            metrictype_is_iterable=len(metrictype)
        except:
            metrictype_is_iterable=False

        if metrictype_is_iterable:

            warnings.filterwarnings("ignore", category=UserWarning, message="This figure includes Axes that are not compatible with tight_layout")
            
            #seems to be necessary to avoid overlapping axes
            subplot_params={'wspace':.5,'hspace':.5}

            axrows=np.floor(np.sqrt(len(metrictype)))
            axcols=np.ceil(len(metrictype)/axrows)
            fig=plt.figure(figsize=figsize,facecolor=saveparams['facecolor'])
            ax=fig.subplots(int(axrows),int(axcols)).flatten()
            ax_return=[]
            if outputcsvfile is not None and not outputcsvfile_append:
                #if writing csv file and NOT in append mode, clear the file here
                #then each metric will be appended to this clean file
                with open(outputcsvfile,'w') as f:
                    pass
            for i,m in enumerate(metrictype):
                ax_ret=display_kraken_heatmap(trainrecord, 
                                       metrictype=m, 
                                       single_epoch=single_epoch, epoch_spacing=epoch_spacing, explicit_epoch=explicit_epoch, epoch_filter_size=epoch_filter_size, best_epoch_fraction=best_epoch_fraction,
                                       origscale=origscale, training=training, addmean=addmean, show_epoch=show_epoch, extra_short_names=extra_short_names, 
                                       pcorr_at_end=pcorr_at_end,exclude_flavors=exclude_flavors,show_heatmap_value_text=show_heatmap_value_text,heatmap_value_text_size=heatmap_value_text_size,
                                       colorbar=colorbar,clim=clim,colormap=colormap,invert_text_color=invert_text_color,ax=ax[i], ticklabel_text_size=ticklabel_text_size,
                                       outputcsvfile=outputcsvfile,outputcsvfile_append=True)
                ax_return.append(ax_ret)
            if outputcsvfile is not None:
                print("Saved %s" % (outputcsvfile))
            
            if all([x is None for x in ax_return]):
                print("WARNING: no data to display for any subplots")
                return None
            
            suptitle_str=None
            righttitle_str=None
            bottom_str=None

            if training:
                nsubj=len(trainrecord['subjidx_train'])
            else:
                nsubj=len(trainrecord['subjidx_val'])
                
            if add_suptitle:
                suptitle_list=[]
                if origscale:
                    suptitle_list+=['OrigScale']
                else:
                    suptitle_list+=['OptimScale']
                
                if training:
                    suptitle_list+=['Training(n=%d)' % (nsubj)]
                else:
                    suptitle_list+=['Validation(n=%d)' % (nsubj)]
                
                if explicit_epoch is not None:
                    suptitle_list+=['Epoch %d' % (explicit_epoch)]
                elif single_epoch:
                    suptitle_list+=['Best All-Path Epoch']
                else:
                    suptitle_list+=['Best Per-Path Epoch']

                if show_epoch:
                    suptitle_list+=['(Epoch display)']
                
                if best_epoch_fraction:
                    suptitle_list+=['(% Per-Path Epoch)']

                suptitle_str=", ".join(suptitle_list)

            if add_epochtitle:
                lastepoch=trainrecord['current_epoch']
                if lastepoch==trainrecord['nbepochs']-1:
                    lastepoch=trainrecord['nbepochs']
                righttitle_str='Final epoch: %d' % (lastepoch)

            if bottomtext=='<filename>':
                if 'recordfile' in trainrecord:
                    bottom_str=justfilename(trainrecord['recordfile'])
            elif bottomtext=='<startingpoint>':
                if 'starting_point_file' in trainrecord:
                    bottom_str=justfilename(trainrecord['starting_point_file'])
            elif bottomtext is not None:
                bottom_str=bottomtext

            if suptitle_str is not None:
                #have to hack a bit to make sure it doesn't clip when saving
                hsup=plt.suptitle(suptitle_str, x=.01,y=.99,fontsize=suptitle_fontsize,horizontalalignment='left',verticalalignment='top')
                #plt.suptitle(" ",fontsize=suptitle_fontsize)
                #plt.text(0, 1, suptitle_str, fontsize=suptitle_fontsize,transform=fig.transFigure, horizontalalignment='left')

            if bottom_str is not None:
                #if we are adding bottom text, add a blank textbox at the bottom BEFORE tight_layout so there is extra space for it
                htxt=plt.text(.5, 0, " ", fontsize=2*bottomtext_fontsize,transform=fig.transFigure, horizontalalignment='center',verticalalignment='bottom')

            #do tight_laytout after suptitle but before adding anything that would make the figure wider (right-aligned text)
            fig.tight_layout()

            if bottom_str is not None:
                htxt=plt.text(1, 0, bottom_str, fontsize=bottomtext_fontsize, transform=fig.transFigure, horizontalalignment='right',verticalalignment='bottom')
                htxt_width=htxt.get_window_extent(fig.canvas.get_renderer()).width/fig.dpi
                fig_width = fig.get_size_inches()[0]

                # If the width of the text annotation is greater than the width of the figure, reduce the font size
                if htxt_width > fig_width:
                    bottom_fontsize_new = bottomtext_fontsize * fig_width / htxt_width
                    htxt.set_fontsize(bottom_fontsize_new)
                
            if righttitle_str is not None:
                #position hack to make sure it doesn't clip when saving
                plt.text(.999, hsup.get_position()[1], righttitle_str+" ", fontsize=righttitle_fontsize,transform=fig.transFigure, horizontalalignment='right',verticalalignment='top')
                
            fig.subplots_adjust(**subplot_params)

            fig.set_size_inches(fig.get_size_inches())
            
            if outputimagefile is not None:
                fig.savefig(outputimagefile,**saveparams)
                if isinstance(outputimagefile,str):
                    print("Saved %s" % (outputimagefile))
            
            plt.close(fig)
            return None
    
    do_colorbar=colorbar
    do_addmean=addmean
    do_squareimage=False

    M=trainrecord

    M['trainpath_names']=[s.strip() for s in M['trainpath_names']]
    M['input_name_list']=[s.strip() for s in M['input_name_list']]

    if 'topN' in M:
        topN=M['topN']
    else:
        topN=2 #why isnt this saved anywhere?
 
    #print(M['current_epoch'])
    #[print(k) for k in M.keys()];

    if metrictype.lower().startswith('r2') and not 'explainedvar_OrigScale_val' in M:
        #not available in all training records
        metrictype='avgcorr'

    #compute bidirectional identifiability metrics (mean of true2pred and pred2true)
    if 'corrloss_OrigScale_pred2true_val' in M:
        for f in ['corrloss','corrlossN','corrlossRank']:
            M[f+'_bidir'+'_OrigScale_train']=(M[f+'_OrigScale_pred2true_train']+M[f+'_OrigScale_train'])/2
            M[f+'_bidir'+'_OrigScale_val']=(M[f+'_OrigScale_pred2true_val']+M[f+'_OrigScale_val'])/2

    #perfectval=1
    perfectval=np.inf #dont color perfect values red
    
    metrictitle=metrictype
    if clim is None:
        clim=[0,1]
    if colormap is None:
        colormap='magma'
        
    if colormap.lower() in extra_colormap_list():
        colormap=load_custom_colormap(colormap)
        
    if metrictype.lower() == 'top1acc':
        metricfield='corrloss'
        metrictitle='corr top1acc'

    elif metrictype.lower() == 'topnacc':
        metricfield='corrlossN'
        metrictitle='corr top%dacc' % (topN)
        
    elif metrictype.lower() == 'avgrank':
        metricfield='corrlossRank'
        metrictitle='avgrank %ile'

    elif metrictype.lower() == 'avgcorr':
        metricfield='avgcorr'
        
    elif metrictype.lower() == 'avgcorr_resid':
        metricfield='avgcorr_resid'
        metrictitle='avgcorr (resid)'

    elif metrictype.lower().startswith('r2'):
        metricfield='explainedvar'
        metrictitle='R2'
        colormap=load_custom_colormap("magma_compdiv")
        clim=[-1,1]
        if '=' in metrictype:
            #special mode to shrink r2 scale by passing "r2=.5"
            r2_clim_scale=float(metrictype.split("=")[-1])
            clim=[-r2_clim_scale,r2_clim_scale]

    elif metrictype.lower() == 'top1accx':
        metricfield='corrloss_bidir'
        metrictitle='corr top1acc (bi-dir)'

    elif metrictype.lower() == 'topnaccx':
        metricfield='corrlossN_bidir'
        metrictitle='corr top%dacc (bi-dir)' % (topN)
        
    elif metrictype.lower() == 'avgrankx':
        metricfield='corrlossRank_bidir'
        metrictitle='avgrank %ile (bi-dir)' 
    

    metricfield_origscale=""
    if origscale:
        metricfield_origscale="_OrigScale"
    
    metricfield_trainval=""
    if training:
        metricfield_trainval='_train'
    else:
        metricfield_trainval='_val'
    
    metricfield="%s%s%s" % (metricfield, metricfield_origscale ,metricfield_trainval)

    if origscale:
        metrictitle='OrigScale %s' % (metrictitle)
    else:
        metrictitle='OptimScale %s' % (metrictitle)

    #v = the [trainpath x epochs] matrix we are displaying
    v=M[metricfield].copy()
    #print("metric.shape:",v.shape)
    if v.ndim == 1:
        v=np.atleast_2d(v).T
    #print("v.shape:",v.shape)

    #conntypes=np.array(M['input_name_list'])
    conntypes=np.array(list(np.unique(flatlist([x.split("->") for x in M['trainpath_names']]))))

    do_remove_intermodal_from_noself=True
    if do_remove_intermodal_from_noself:
        for itp, tp in enumerate(M['trainpath_names']):
            tp1,tp2=tp.split("->")
            tp2g=canonical_data_flavor(tp2,accept_unknowns=True,return_groupname=True)[1]
            if tp1=='fusionFC.noself' and tp2g == 'SC':
                v[itp]=np.nan
            elif tp1=='fusionSC.noself' and tp2g == 'FC':
                v[itp]=np.nan
    
    v_exclude=np.zeros(v.shape,dtype=bool)
    
    if exclude_flavors is not None:
        
        if isinstance(exclude_flavors,str):
            exclude_flavors=[exclude_flavors]
        for itp, tp in enumerate(M['trainpath_names']):
            tp1,tp2=tp.split("->")
            for x in exclude_flavors:
                if '*' in x:
                    x=x.replace(".","\.").replace("*",".*")
                    if re.match(x,tp1) or re.match(x,tp2):
                        #v[itp]=np.nan
                        v_exclude[itp]=True
                else:
                    if x==tp1 or x==tp2:
                        #v[itp]=np.nan
                        v_exclude[itp]=True
                
    ###################
    #do best-epoch search every time
    v_search=v.copy()
    
    v_search=naninterp(v_search,axis=1,fill_value=np.nan) #origscale might not be every epoch so interp

    v_isnan=np.all(np.isnan(v),axis=0)

    if epoch_filter_size > 1:
        filt=np.ones((epoch_filter_size,1))/epoch_filter_size
        v_search=convolve(v_search,filt,mode='same')

    if epoch_spacing:
        v_search=v_search[:,::epoch_spacing]
    
    v_search_mean=np.nanmean(v_search,axis=0)
    v_search_mean[v_isnan]=np.nanmin(v_search_mean)-1
    display_epoch_singleepoch=nanargmax_safe(v_search_mean,nanval=0)

    v_search_allnanrows=np.all(np.isnan(v_search),axis=1)
    v_search[:,v_isnan]=np.nanmin(v_search)-1
    v_search[v_search_allnanrows,:]=np.nanmin(v_search)-1
    display_epoch_perpath=nanargmax_safe(v_search,nanval=0,axis=1)
    display_epoch_perpath[v_search_allnanrows]=0
    
    if epoch_spacing:
        display_epoch_singleepoch*=epoch_spacing
        display_epoch_perpath*=epoch_spacing

    if single_epoch:
        #make sure nans from orig data are set to some small value here
        #so that the nanargmax doesn't catch it in the case of a flat naninterp 
        display_epoch=display_epoch_singleepoch
    else:
        #make sure nans from orig data are set to some small value here
        #so that the nanargmax doesn't catch it in the case of a flat naninterp 
        display_epoch=display_epoch_perpath
        if all(display_epoch==display_epoch[0]):
            display_epoch=display_epoch[0]
    
    ###################

    #override epoch if explicit epoch was given
    if explicit_epoch is not None:
        display_epoch=np.array(explicit_epoch)

    if v.shape[1]<=1:
        vdisp=v[:,0]
        #print("vdisp=v[:,0]")
    elif display_epoch.size == 1:
        vdisp=v[:,display_epoch]
        #print("vdisp=v[:,display_epoch]:",display_epoch)
    else:
        vdisp=v[range(v.shape[0]),display_epoch]
        #print("vdisp=v[:,display_epoch]:",display_epoch)
    #print("vdisp.shape:",vdisp.shape)

    if best_epoch_fraction:
        if display_epoch_perpath.size == 1:
            v_perpath=v[:,display_epoch_perpath]
        else:
            v_perpath=v[range(v.shape[0]),display_epoch_perpath]
        vdisp=vdisp/v_perpath
        
    if show_epoch:
        if display_epoch.size == 1:
            vdisp=np.ones(v.shape[0])*display_epoch
        else:
            vdisp=display_epoch
        clim=[np.nanmin(vdisp),np.nanmax(vdisp)]
        colormap='magma'
    
    vmat=np.nan*np.ones((len(conntypes),len(conntypes)))
    vmat_count=np.zeros(vmat.shape)
    vmat_exclude=np.zeros(vmat.shape,dtype=bool)
    
    v_exclude=v_exclude[:,0]
    for itp,tpname in enumerate(M['trainpath_names']):
        iname=tpname.split("->")[0]
        jname=tpname.split("->")[1]
        i=np.where(conntypes==iname)[0]
        j=np.where(conntypes==jname)[0]
        vmat[i,j]=vdisp[itp]
        vmat_exclude[i,j]=v_exclude[itp]
        if vmat_exclude[i,j]:
            vmat[i,j]=np.nan
        if ~np.isnan(vmat[i,j]):
            vmat_count[i,j]+=1

    #add mean row/column
    if do_addmean:
        #first add mean across rows (for non-fusion rows only)
        conntype_idx_for_mean=[i for i,c in enumerate(conntypes) if (not "fusion" in c and not "burst" in c)]
        mean_is_last_row=len(conntype_idx_for_mean)==len(conntypes)
        
        vmat_meanrow=np.nanmean(vmat[conntype_idx_for_mean,:],axis=0,keepdims=True)
        vmat_count_meanrow=np.sum(vmat_count[conntype_idx_for_mean,:],axis=0,keepdims=True)
        vmat_exclude_meanrow=np.zeros(vmat_meanrow.shape,dtype=bool)
        
        if not mean_is_last_row:
            #add gap row
            vmat_meanrow=np.concatenate((vmat_meanrow,np.nan*vmat_meanrow),axis=0)
            vmat_count_meanrow=np.concatenate((vmat_count_meanrow,vmat_count_meanrow),axis=0)
            vmat_exclude_meanrow=np.concatenate((vmat_exclude_meanrow,vmat_exclude_meanrow),axis=0)
        
        if show_epoch:
            vmat_meanrow=np.round(vmat_meanrow)
        
        vmat=np.concatenate((vmat,vmat_meanrow),axis=0)
        vmat_count=np.concatenate((vmat_count,vmat_count_meanrow),axis=0)
        vmat_exclude=np.concatenate((vmat_exclude,vmat_exclude_meanrow),axis=0)
        
        #now add mean across columns (for non-fusion columns only)
        #but also include mean of the mean row
        vmat_meancol=np.nanmean(vmat[:,conntype_idx_for_mean],axis=1,keepdims=True)
        vmat_count_meancol=np.sum(vmat_count[:,conntype_idx_for_mean],axis=1,keepdims=True)
        vmat_exclude_meancol=np.zeros(vmat_meancol.shape,dtype=bool)
        
        if not mean_is_last_row:
            #add gap col
            vmat_meancol=np.concatenate((vmat_meancol,np.nan*vmat_meancol),axis=1)
            vmat_count_meancol=np.concatenate((vmat_count_meancol,0*vmat_count_meancol),axis=1)
            vmat_exclude_meancol=np.concatenate((vmat_exclude_meancol,vmat_exclude_meancol),axis=1)
        
        if show_epoch:
            vmat_meancol=np.round(vmat_meancol)
            
        vmat=np.concatenate((vmat,vmat_meancol),axis=1)
        vmat_count=np.concatenate((vmat_count,vmat_count_meancol),axis=1)
        vmat_exclude=np.concatenate((vmat_exclude,vmat_exclude_meancol),axis=1)
        
        conntypes=np.append(conntypes,'mean')
        if not mean_is_last_row:
            conntypes=np.append(conntypes,'gap')
    
    #print('vmat.shape:',vmat.shape)
    ######################
    _, newidx = flavor_reorder(conntypes, pcorr_at_end=pcorr_at_end)
    
    conntypes=conntypes[newidx]
    vmat=vmat[newidx,:][:,newidx]
    vmat_count=vmat_count[newidx,:][:,newidx]
    vmat_exclude=vmat_exclude[newidx,:][:,newidx]
    
    #conntypes=list(conntypes)*2
    #conntypes_short=shorten_names([canonical_data_flavor(x) for x in conntypes])
    conntypes_short=[]
    for x in conntypes:
        if x.lower() == 'encoded' or x.lower().startswith("fusion") or x.lower().startswith("burst") or x.lower()=='mean':
            conntypes_short+=[x]
        elif x.lower() in ['unrel','sib','DZ','MZ','self','retest']:
            conntypes_short+=[x]
        elif x == 'gap':
            conntypes_short+=['']
        else:
            conntypes_short+=[canonical_data_flavor(x,accept_unknowns=True)]
    conntypes_short=shorten_names(conntypes_short)

    #vmat=np.vstack((vmat,vmat))
    #vmat=np.hstack((vmat,vmat))
    
    
    titlestr=metrictitle

    if display_epoch.size > 1:
        epochstr='ep%d-%d' % (min(display_epoch),max(display_epoch))
    else:
        if display_epoch==0 and M['corrloss_train'].ndim==1 or M['corrloss_train'].shape[1]==1:
            display_epoch=M['current_epoch']
        if display_epoch==M['nbepochs']-1:
            display_epoch=M['nbepochs']
        epochstr='ep%d' % (display_epoch)

    titlestr="%s SRC(row)xTARG(col), %s" % (titlestr,epochstr)
    
    #remove rows/columns that never had any data
    #vmat_never_input=np.max(vmat_count,axis=1)==0
    #vmat_never_output=np.max(vmat_count,axis=0)==0
    
    #override excluded inputs based on explicit exclusion criteria
    #vmat_count[np.isnan(vmat)]=0
    
    #Mtmp=dict(conntypes=conntypes,vmat_exclude=vmat_exclude,vmat=vmat,vmat_count=vmat_count)
    #from scipy.io import savemat
    #savemat("test2.mat",Mtmp,format='5',do_compression=True)
    #breakpoint()
    vmat_never_output=np.max(vmat_count,axis=0)==0
    vmat_never_input_tmpcolumns=~vmat_never_output
    vmat_never_input_tmpcolumns[conntypes=='mean']=False
    vmat_never_input=np.max(vmat_count[:,vmat_never_input_tmpcolumns],axis=1)==0

    #breakpoint()
    #vmat_never_input[conntypes=='gap']=False #make sure gap INPUT is always included

    if np.any(vmat_exclude):
        #vmat_never_input=np.any(vmat_exclude,axis=1) #this was used before        
        vmat_never_input=vmat_never_input | np.all(vmat_exclude[:,vmat_never_input_tmpcolumns],axis=1)
        
        vmat_never_output_tmprows=~vmat_never_input
        vmat_never_output_tmprows[[c in ['mean','gap'] for c in conntypes]]=False
        vmat_never_output=vmat_never_output | np.all(vmat_exclude[vmat_never_output_tmprows,:],axis=0)

    vmat=vmat[~vmat_never_input,:][:,~vmat_never_output]
    xlabels_to_display=list(np.array(conntypes_short)[~vmat_never_output])
    ylabels_to_display=list(np.array(conntypes_short)[~vmat_never_input])
    
    xlabel_colors=[flavor2color(x) for x in xlabels_to_display]
    ylabel_colors=[flavor2color(x) for x in ylabels_to_display]

    #"groups" to draw lines around in heatmap and labels
    xlabel_blocks=[flavor2group(x) for x in xlabels_to_display]
    ylabel_blocks=[flavor2group(x) for x in ylabels_to_display]
    
    if extra_short_names > 0:
        #need to do extra shortening AFTER flavor2color
        #xlabels_to_display=shorten_names(xlabels_to_display,extrashort=extra_short_names)
        #ylabels_to_display=shorten_names(ylabels_to_display,extrashort=extra_short_names)
        xlabels_to_display=shorten_names(xlabels_to_display,extrashort=extra_short_names,remove_strings=['_hpf','.vn'])
        ylabels_to_display=shorten_names(ylabels_to_display,extrashort=extra_short_names,remove_strings=['_hpf','.vn'])
        
    if ax is None:
        plt.figure(figsize=figsize)
        ax=plt.gca()
    else:
        plt.sca(ax)
    
    if do_squareimage:
        image_aspect='equal'
    else:
        image_aspect='auto'

    vmat_todisplay=vmat.copy()
    vmat_todisplay[np.isnan(vmat)]=0
    vmat_alpha=1-.75*np.isnan(vmat) #make nan entries semi-transparent
    if vmat_todisplay.size == 0:
        print("WARNING: no data to display")
        return None
    
    do_print_csv=outputcsvfile is not None
    if do_print_csv:
        if outputcsvfile_append:
            csvmode='a'
        else:
            csvmode='w'
        with open(outputcsvfile,csvmode) as fcsv:
            fcsv.write(metrictitle+"\n")
            for icol in range(vmat_todisplay.shape[1]):
                if xlabels_to_display[icol] == 'mean':
                    fcsv.write(',"average"')
                else:
                    fcsv.write(',"target:%s"' % (xlabels_to_display[icol]))
            fcsv.write("\n")
            for irow in range(vmat_todisplay.shape[0]):
                if ylabels_to_display[irow]=='':
                    continue
                for icol in range(vmat_todisplay.shape[1]):
                    if icol==0:
                        if ylabels_to_display[irow]=='mean':
                            fcsv.write('"average"')
                        else:
                            fcsv.write('"source:%s"' % (ylabels_to_display[irow]))
                    fcsv.write(",%f" % (vmat_todisplay[irow,icol]))
                #print("\n",end="")
                fcsv.write("\n")
    
    him=plt.imshow(vmat_todisplay,cmap=colormap,vmin=clim[0],vmax=clim[1],aspect=image_aspect,alpha=vmat_alpha)

    #print("vmat_alpha.shape:",vmat_alpha.shape)

    color1='k'
    color2='w'
    if invert_text_color:
        color1='w'
        color2='k'
        
    if show_heatmap_value_text:
        txtargs={'horizontalalignment':'center','verticalalignment':'center','fontsize':heatmap_value_text_size}
        for i in range(vmat.shape[0]):
            for j in range(vmat.shape[1]):
                x=vmat[i,j]
                if np.isnan(x):
                    continue
                vstr="%.2f" % (x)
                vstr=re.sub(r"\.0+$","",vstr)
                vstr=re.sub(r"(-?)0+(\.)",r"\1\2",vstr)
                
                txtcolor=color1
                if clim[0]<0:
                    #diverging/posneg colormaps need abs()
                    if (abs(x)-0)/max(np.abs(clim)-0) < .7:
                        txtcolor=color2
                else:
                    if (x-clim[0])/(clim[1]-clim[0]) < .7:
                        txtcolor=color2
                if x==perfectval:
                    txtcolor='r'
                plt.text(j,i,vstr,color=txtcolor,**txtargs)
                
    rightbump=.01 #tiny extra bump because heatmap is 1-2 pixels wider than it should be
    
    xl=[-.5,vmat.shape[1]-.5+rightbump]
    yl=[vmat.shape[0]-.5,-.5]


    #add lines around "mean" row/col in heatmap
    meanlineval_color_thresh=.45
    row_lineidx_list=[]
    col_lineidx_list=[]
    for i,x in enumerate(ylabel_blocks):
        if i==0:
            continue
        if x=='' or ylabel_blocks[i-1]=='':
            continue
        if x != ylabel_blocks[i-1]:
            row_lineidx_list.append(i)
    for i,x in enumerate(xlabel_blocks):
        if i==0:
            continue
        if x=='' or xlabel_blocks[i-1]=='':
            continue
        if x != xlabel_blocks[i-1]:
            col_lineidx_list.append(i)
        
    for meanrow in row_lineidx_list:
        meanlinecolor=color1
        meanlineval=np.nanmean(vmat[meanrow-1:meanrow,:])
        #print("meanrow:",meanlineval,ylabels_to_display[meanrow-1],ylabels_to_display[meanrow],"thresh=",meanlineval_color_thresh)
        if meanlineval < meanlineval_color_thresh:
            meanlinecolor=color2
        if ylabels_to_display[meanrow]=='mean':
            plt.plot(xl,[meanrow-.53,meanrow-.53],color=meanlinecolor,linewidth=1,zorder=100)
            #plt.plot(xl,[meanrow+.5,meanrow+.5],color=meanlinecolor,linewidth=1,zorder=100)
        else:
            plt.plot(xl,[meanrow-.53,meanrow-.53],color=meanlinecolor,linewidth=.5,zorder=100)

    for meancol in col_lineidx_list:
        meanlinecolor=color1
        meanlineval=np.nanmean(vmat[:,meancol-1:meancol])
        #print("meancol:",meanlineval,xlabels_to_display[meancol-1],xlabels_to_display[meancol],"thresh=",meanlineval_color_thresh)
        if meanlineval < meanlineval_color_thresh:
            meanlinecolor=color2
        if xlabels_to_display[meancol]=='mean':
            plt.plot([meancol-.5,meancol-.5],yl,color=meanlinecolor,linewidth=1,zorder=100)
            #plt.plot([meancol+.5+rightbump,meancol+.5+rightbump],yl,color=meanlinecolor,linewidth=1,zorder=100)
        else:
            plt.plot([meancol-.5,meancol-.5],yl,color=meanlinecolor,linewidth=.5,zorder=100)

    #draw outer border for heatmap
    plt.plot([xl[0],xl[1],xl[1],xl[0],xl[0]],[yl[0],yl[0],yl[1],yl[1],yl[0]],color='k',linewidth=2,zorder=300)
    
    #fill white for 'gap' row/col in heatmap
    if '' in ylabels_to_display:
        for gaprow in [i for i,x in enumerate(ylabels_to_display) if x=='']:
            plt.fill([xl[0]-.5,xl[0]-.5,xl[1]+.5,xl[1]+.5],[gaprow-.5,gaprow+.5,gaprow+.5,gaprow-.5],figure_background_color,edgecolor='k',linewidth=1,zorder=400)
    if '' in xlabels_to_display:
        for gapcol in [i for i,x in enumerate(xlabels_to_display) if x=='']:
            plt.fill([gapcol-.5,gapcol+.5,gapcol+.5,gapcol-.5],[yl[0],yl[0],yl[1],yl[1]],figure_background_color,edgecolor='k',linewidth=1,zorder=400)
    
    plt.box(False)
    
    plt.xlim(xl)
    plt.ylim([max(yl),min(yl)]) #invert y axis

    xticks=list(np.arange(len(xlabels_to_display)))
    yticks=list(np.arange(len(ylabels_to_display)))

    plt.xticks(ticks=[])
    plt.yticks(ticks=[])
    #plt.xticks(ticks=xticks,labels=xlabels_to_display)
    #plt.yticks(ticks=yticks,labels=ylabels_to_display)

    #use text() to draw xtick and ytick labels for more control
    #
    #textbold=lambda x: {'fontweight':'bold'} if x=='mean' else {}
    textbold=lambda x: {'style':'italic'} if x=='mean' else {}
    
    textrelabel=lambda x: 'average' if x=='mean' else x
    
    tickfontsizeargs={}
    tickfontsizeargs={'fontsize':ticklabel_text_size}
    [plt.text(-.5,yticks[i],textrelabel(ylabels_to_display[i])+" ",color=ylabel_colors[i],horizontalalignment='right',verticalalignment='center',**tickfontsizeargs,**textbold(ylabels_to_display[i])) for i in range(len(yticks))]
    [plt.text(xticks[i],max(yticks)+1-.5,textrelabel(xlabels_to_display[i])+" ",color=xlabel_colors[i],rotation=45,rotation_mode='anchor',horizontalalignment='right',verticalalignment='top',**tickfontsizeargs,**textbold(xlabels_to_display[i])) for i in range(len(xticks))]

    add_block_label_lines=True
    if add_block_label_lines:
        #add lines to separate SC and FC blocks in xlabels and ylabels
        for blockend in row_lineidx_list:
            plt.text(-.5,yticks[blockend-1]*.9+yticks[blockend]*.1,'_'*12+' ',color='black',horizontalalignment='right',verticalalignment='center')
        
        for blockend in col_lineidx_list:
            plt.text((xticks[blockend-1]*2+xticks[blockend])/3,max(yticks)+1-.75,'_'*13,color='black',rotation=45,rotation_mode='anchor',horizontalalignment='right',verticalalignment='top')
            
    #add "SOURCE" and "TARGET" above tick labels
    plt.text(-.5,-.5,"SOURCE",color='k',fontweight='bold',horizontalalignment='right',verticalalignment='bottom')
    plt.text(-1,max(yticks)+1-.5,"TARGET",color='k',fontweight='bold',rotation=45,rotation_mode='anchor',horizontalalignment='right',verticalalignment='top')

    if titlestr:
        plt.title(titlestr,fontdict={'fontweight':'bold'})

    if do_colorbar:
        #make sure colorbar scales correctly
        if False:
            #this causes problems in some versions of matplotlib
            inset_axesaxins = inset_axes(
                ax,
                width=.25,  # width: 5% of parent_bbox width
                height="100%",  # height: 50%
                loc="lower left",
                bbox_to_anchor=(1.05, 0., 1, 1),
                bbox_transform=ax.transAxes,
                borderpad=0,
            )
        inset_axesaxins = ax.inset_axes(
                [1.05, 0., 0.05, 1],
                transform=ax.transAxes
            )
        
        if clim[0]==-1 and clim[1]==1:
            cticks=np.linspace(-1,1,11)
        else:
            cticks=None
        hcbar=plt.colorbar(him,cax=inset_axesaxins,ticks=cticks)
        hcbar.ax.tick_params(direction='in')
    #hcbar=plt.colorbar(him)
    #hcbar.ax.tick_params(direction='in')
    
    if outputimagefile is not None:
        ax.figure.savefig(outputimagefile,**saveparams)
        if isinstance(outputimagefile,str):
            print("Saved %s" % (outputimagefile))
        
    return ax