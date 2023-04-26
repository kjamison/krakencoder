from krakencoder import *
from loss import *

import torch.utils.data as data_utils
import torch.optim as optim

import os
import platform
import time
from datetime import datetime
import random
import re
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from scipy.spatial.distance import cdist as scipy_cdist

#!conda install -c conda-forge scikit-learn -y

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import FunctionTransformer

from sklearn.model_selection import GroupShuffleSplit

import matplotlib.pyplot as plt
#from IPython import display #only in jupyter
from cycler import cycler

#####################################
#some useful functions

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def numpyvar(x):
    if not torch.is_tensor(x):
        return x
    return x.numpy()

def torchvar(x, astype=None):
    if torch.is_tensor(x):
        return x
    
    #cast variable to torch (use cuda if available)
    try:
        _ = iter(x)
    except TypeError:
        islist=False
        input_type=type(x)
    else:
        islist=True
        input_type=type(x[0])
    
    if astype is None:
        astype=input_type
    
    if torch.cuda.is_available():
        if astype is int:
            torchfun=torch.cuda.IntTensor
        else:
            torchfun=torch.cuda.FloatTensor
    else:
        if astype is int:
            torchfun=torch.IntTensor
        else:
            torchfun=torch.FloatTensor
    
    if islist:
        return torchfun(x)
    else:
        return torchfun([x])

def torchfloat(x):
    return torchvar(x,astype=float)

def torchint(x):
    return torchvar(x,astype=int)

def random_train_test_split(numsubj=None, subjlist=None, train_frac=.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if subjlist is not None:
        numsubj=len(subjlist)
    else:
        subjlist=np.arange(numsubj)
    
    subjidx_random=np.argsort(np.random.random_sample(numsubj))
    #subjidx_random=np.arange(numsubj)

    nsubj_train=int(numsubj*train_frac)
    nsubj_test=numsubj-nsubj_train

    subjidx_train=subjidx_random[:nsubj_train]
    subjidx_test=subjidx_random[nsubj_train:]

    return subjlist[subjidx_train],subjlist[subjidx_test]

def random_train_test_split_groups(groups, numsubj=None, subjlist=None, train_frac=.5, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if subjlist is not None:
        numsubj=len(subjlist)
    else:
        subjlist=np.arange(numsubj)
    
    if numsubj != len(groups):
        return None
    
    gss=GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    subjidx_train, subjidx_test = next(gss.split(X=subjlist,groups=groups))
    
    subjidx_train=subjidx_train[np.argsort(np.random.random_sample(len(subjidx_train)))]
    subjidx_test=subjidx_test[np.argsort(np.random.random_sample(len(subjidx_test)))]
    return subjlist[subjidx_train],subjlist[subjidx_test],groups[subjidx_train],groups[subjidx_test]

#take a dictionary where each field value is a list
#and generate a new list of dictionaries with every combination of fields
def dict_combination_list(listdict, reverse_field_order=False):
    keylist=list(listdict.keys())
    if reverse_field_order:
        keylist=keylist[::-1]
    permlist=[]
    for k in keylist:
        vlist=listdict[k]
        if len(permlist)==0:
            permlist=[{k:v} for v in vlist]
            continue
        permlist_new=[]
        for v in vlist:
            for p in permlist:
                p[k]=v
                permlist_new+=[p.copy()]
        permlist=permlist_new
    return permlist

def flatlist(l):
    if l is None:
        return []
    return [x for y in l for x in y]

def common_prefix(strlist):
    strlen=[len(s) for s in strlist]
    result=""
    for i in range(1,min(strlen)):
        if all([s[:i]==strlist[0][:i] for s in strlist]):
            result=strlist[0][:i]
        else:
            break
    return result

def common_suffix(strlist):
    strlen=[len(s) for s in strlist]
    result=""
    for i in range(1,min(strlen)):
        if all([s[-i:]==strlist[0][-i:] for s in strlist]):
            result=strlist[0][-i:]
        else:
            break
    return result

def trim_string(s,left=0,right=0):
    snew=s
    if left>0:
        snew=snew[left:]
    if right>0:
        snew=snew[:-right]
    return snew

def loss_string_to_dict(loss_string, override_weight_dict={}, lossgroup_default_weight={}):
    
    loss_items_dict=OrderedDict()
    #merge info from loss_string split and default_weight_dict
    for lt_item in loss_string.split("+"):
        lt=lt_item.split(".w")
        if len(lt) > 1:
            w=float(lt[1])
        else:
            w=None
        lt=lt[0]
        lt_suffix=None
        
        if lt.endswith('B'):
            lt_suffix='B'
            lt=lt.replace(lt_suffix,'')
        
        loss_items_dict[lt]={"name":lt,"weight":w}
        
        if lt_suffix is not None:
            loss_items_dict[lt]['suffix']=lt_suffix
    
    for lt,w in override_weight_dict.items():
        if lt in loss_items_dict:
            loss_items_dict[lt]["weight"]=w
        else:
            loss_items_dict[lt]={"name":lt,"weight":w}

    new_loss_items_dict=OrderedDict()
    for lt,lt_item in loss_items_dict.items():
        w=lt_item["weight"]

        if lt in ['mse','msesum','corrtrace','correye','corrmatch','dist','neidist']:
            lt_group='output'
        elif lt in ['enceye','encdist','encneidist','encdot','encneidot']:
            lt_group='encoded'
        elif lt in ['latentnormloss','latentsimloss','latentmaxradloss','latentsimloss']:
            lt_group='encoded_meta'
        else:
            raise Exception("Unknown losstype: %s" % (lt))
        
        if w is None and lt_group in lossgroup_default_weight:
            w=lossgroup_default_weight[lt_group]
        
        lt_suffix=""
        if "suffix" in lt_item:
            lt_suffix=lt_item['suffix']
        
        if w is None or w==0:
            lt_string=lt+lt_suffix
        elif w == 1:
            lt_string=lt+lt_suffix
        else:
            lt_string="%s%s.w%g" % (lt,lt_suffix,w)

        new_loss_items_dict[lt]={"name":lt, "weight":w, "string":lt_string, "lossgroup":lt_group}
        if "suffix" in lt_item:
            new_loss_items_dict[lt]['suffix']=lt_item['suffix']
    
    return new_loss_items_dict

def loss_dict_to_string(loss_info_dict):
    loss_str="+".join([v['string'] for k,v in loss_info_dict.items() if v['weight'] is not None and v['weight']!=0])
    return loss_str

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
            plt.scatter(midx[i],x.flatten()[midx2[i]],c=colors[i%len(colors)])
        #plt.scatter(midx,x.flatten()[midx2],c=colors)
        
    else:
        plt.scatter(midx,x.flatten()[midx2])

def update_training_figure(loss_train, loss_val, corrloss_train, corrloss_val, corrlossN_train, corrlossN_val, corrlossRank_train, corrlossRank_val, 
    avgrank_train, avgrank_val, trainpath_names_short, data_string, network_string,train_string, losstype, epoch, trainfig=None, gridalpha=.25, topN=2):

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

#################################
#################################
#create data input/output transformers, datasets/dataloaders, and TRAINING PATHS

def generate_transformer(traindata, transformer_type, transformer_param_dict=None, precomputed_transformer_params=None, return_components=True):
    if transformer_param_dict:
        transformer_info=transformer_param_dict
    else:
        transformer_info={}
    
    if precomputed_transformer_params and precomputed_transformer_params["type"] != transformer_type:
        raise Exception("Precomputed transformer was %s, expected %s" % (precomputed_transformer_params["type"],transformer_type))
    
    if transformer_type == "none":
        transformer=FunctionTransformer(func=lambda x:x,
                                        inverse_func=lambda x:x)
        transformer_info["type"]="none"
        transformer_info["params"]={}
    
    elif transformer_type == "pca":
        if precomputed_transformer_params:
            if transformer_param_dict and precomputed_transformer_params["reduce_dimension"] != transformer_param_dict['reduce_dimension']:
                raise Exception("Precomputed transformer dimension was %d, expected %d" % (precomputed_transformer_params["reduce_dimension"],
                    transformer_param_dict['reduce_dimension']))
            data_normscale=precomputed_transformer_params['input_normscale']
            normscale=precomputed_transformer_params['output_normscale']
            pca_components=precomputed_transformer_params['pca_components']
            pca_input_mean=precomputed_transformer_params['pca_input_mean']
            pca_dim=precomputed_transformer_params["reduce_dimension"]
        else:
            pca_xform=PCA(n_components=transformer_param_dict['reduce_dimension'],random_state=0).fit(traindata)
            data_normscale=np.linalg.norm(pca_xform.transform(traindata))
            normscale=100
            
            pca_components=pca_xform.components_
            pca_input_mean=pca_xform.mean_
            pca_dim=transformer_param_dict['reduce_dimension']

        #just make it explicitly a lambda function instead of using the PCA.transform() so we can be certain it's reproducible during
        #training and later evaluation
        #Xpc = np.dot(X-pca_input_mean,pca_components.T)
        #Xnew = np.dot(Xpc,pca_components)+pca_input_mean
        #
        #old version inverse_func was incorrect (pre 4/20/2023). Does not affect model training but DOES affect origspace outputs:
        #BAD! transformer=FunctionTransformer(func=lambda x:normscale*(np.dot(x-pca_input_mean,pca_components.T)/data_normscale),
        #                                inverse_func=lambda x:((np.dot(x,pca_components)+pca_input_mean)/normscale)*data_normscale)
        
        transformer=FunctionTransformer(func=lambda x:normscale*(np.dot(x-pca_input_mean,pca_components.T)/data_normscale),
                                        inverse_func=lambda x:np.dot((x/normscale)*data_normscale,pca_components)+pca_input_mean)
        
        transformer_info["type"]="pca"
        transformer_info["params"]={"reduce_dimension":pca_dim,
                                    "input_normscale":data_normscale, 
                                    "output_normscale":normscale}
                                    
        if return_components:
            transformer_info["params"]["pca_components"]=pca_components
            transformer_info["params"]["pca_input_mean"]=pca_input_mean
            
    elif transformer_type == "tsvd":
        if precomputed_transformer_params:
            if transformer_param_dict and precomputed_transformer_params["reduce_dimension"] != transformer_param_dict['reduce_dimension']:
                raise Exception("Precomputed transformer dimension was %d, expected %d" % (precomputed_transformer_params["reduce_dimension"],
                    transformer_param_dict['reduce_dimension']))
            data_normscale=precomputed_transformer_params['input_normscale']
            normscale=precomputed_transformer_params['output_normscale']
            tsvd_components=precomputed_transformer_params['tsvd_components']
            tsvd_dim=precomputed_transformer_params["reduce_dimension"]
        else:
            pca_xform=TruncatedSVD(n_components=transformer_param_dict['reduce_dimension'],random_state=0).fit(traindata)
            data_normscale=np.linalg.norm(pca_xform.transform(traindata))
            normscale=100
            
            tsvd_components=pca_xform.components_
            tsvd_dim=transformer_param_dict['reduce_dimension']
        
        #use lambda with component data instead of TSVD.transform() to be certain training and evaluation are the same
        #Xtsvd = np.dot(X,tsvd_components.T)
        #Xnew = np.dot(Xtsvd,tsvd_components)
        transformer=FunctionTransformer(func=lambda x:normscale*(np.dot(x,tsvd_components.T)/data_normscale),
                                        inverse_func=lambda x:(np.dot(x,tsvd_components)/normscale)*data_normscale)
        transformer_info["type"]="tsvd"
        transformer_info["params"]={"reduce_dimension":tsvd_dim,
                                    "input_normscale":data_normscale, 
                                    "output_normscale":normscale}
        if return_components:
            transformer_info["params"]["tsvd_components"]=tsvd_components
            
    elif transformer_type == "norm":
        if precomputed_transformer_params:
            data_normscale=precomputed_transformer_params['input_normscale']
            normscale=precomputed_transformer_params['output_normscale']
        else:
            #normalize each SC matrix by the total L2 norm of the [trainsubj x pairs] for that type
            data_normscale=np.linalg.norm(traindata)
            data_normscale/=np.sqrt(traindata.size)

            normscale=100
            
        transformer=FunctionTransformer(func=lambda x:normscale*(x/data_normscale),
                                        inverse_func=lambda x:(x/normscale)*data_normscale)
        transformer_info["type"]="norm"
        transformer_info["params"]={"input_normscale":data_normscale, 
                                    "output_normscale":normscale}
    
    elif transformer_type == "zscore":
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
            data_stdev=precomputed_transformer_params['input_stdev']
        else:
            #normalize each SC matrix by the total L2 norm of the [trainsubj x pairs] for that type
            data_mean=np.mean(traindata)
            data_stdev=np.std(traindata)

        transformer=FunctionTransformer(func=lambda x:(x-data_mean)/data_stdev,
                                        inverse_func=lambda x:(x*data_stdev)+data_mean)
        transformer_info["type"]="zscore"
        transformer_info["params"]={"input_mean":data_mean, 
                                    "input_stdev":data_stdev}
                                    
    elif transformer_type == "zfeat":
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
            data_stdev=precomputed_transformer_params['input_stdev']
        else:
            #z-score each FEATURE (column) based on training
            data_mean=np.mean(traindata,axis=0,keepdims=True)
            data_stdev=np.std(traindata,axis=0,keepdims=True)
            data_stdev[data_stdev==0]=1.0
        
        transformer=FunctionTransformer(func=lambda x:(x-data_mean)/data_stdev,
                                        inverse_func=lambda x:(x*data_stdev)+data_mean)
        transformer_info["type"]="zfeat"
        transformer_info["params"]={"input_mean":data_mean, 
                                    "input_stdev":data_stdev}
        
    elif transformer_type == "cfeat":
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
        else:
            #demean each FEATURE (column) based on training
            data_mean=np.mean(traindata,axis=0,keepdims=True)
        
        transformer=FunctionTransformer(func=lambda x:x-data_mean,
                                        inverse_func=lambda x:x+data_mean)
        transformer_info["type"]="cfeat"
        transformer_info["params"]={"input_mean":data_mean}
        
    elif transformer_type == "cfeat+norm":
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
            data_rownorm_mean=precomputed_transformer_params['input_rownorm_mean']
        else:
            #demean each FEATURE (column) based on training
            data_mean=np.mean(traindata,axis=0,keepdims=True)
            data_rownorm_mean=np.mean(np.sqrt(np.sum((traindata-data_mean)**2,axis=1)))
        
        transformer=FunctionTransformer(func=lambda x:(x-data_mean)/data_rownorm_mean,
                                        inverse_func=lambda x:x*data_rownorm_mean+data_mean)
        transformer_info["type"]="cfeat+norm"
        transformer_info["params"]={"input_mean":data_mean,"input_rownorm_mean":data_rownorm_mean}
    
    elif transformer_type == "zscore+rownorm":
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
            data_stdev=precomputed_transformer_params['input_stdev']
            data_z_rownorm_mean=precomputed_transformer_params['input_rownorm_mean']
        else:
            #normalize each SC matrix by the total L2 norm of the [trainsubj x pairs] for that type
            data_mean=np.mean(traindata)
            data_stdev=np.std(traindata)
            data_z_rownorm_mean=np.mean(np.sqrt(np.sum(((traindata-data_mean)/data_stdev)**2,axis=1)))
            
        transformer=FunctionTransformer(func=lambda x:(x-data_mean)/(data_stdev*data_z_rownorm_mean),
                                        inverse_func=lambda x:(x*data_stdev*data_z_rownorm_mean)+data_mean)
        transformer_info["type"]="zscore+rownorm"
        transformer_info["params"]={"input_mean":data_mean, 
                                    "input_stdev":data_stdev,
                                    "input_rownorm_mean":data_z_rownorm_mean}
        
    elif transformer_type == "lognorm+rownorm":
        if precomputed_transformer_params:
            min_nonzero=precomputed_transformer_params['input_minimum_nonzero']
            logmean_nonzero=precomputed_transformer_params['input_logmean_nonzero']
            logstd_nonzero=precomputed_transformer_params['input_logstdev_nonzero']
            logmean_new=precomputed_transformer_params['output_logmean_nonzero']
            logstd_new=precomputed_transformer_params['output_logstdev_nonzero']
            dasta_rownorm_mean=precomputed_transformer_params['input_rownorm_mean']
        else:
            #map SC data x->log(x), scaling mean/std (excluding zeros) to 0.5, .1 to be SOMEWHAT closer to FC
            min_nonzero=np.min(traindata[traindata>0])
        
            logmean_nonzero=np.mean(np.log10(traindata[traindata>min_nonzero]))
            logstd_nonzero=np.std(np.log10(traindata[traindata>min_nonzero]))

            logmean_new=.5
            logstd_new=.1

            lognorm_func = lambda x: (((np.log10(np.clip(x,min_nonzero,None))-logmean_nonzero)/logstd_nonzero)*logstd_new+logmean_new)*(x>=min_nonzero)

            data_rownorm_mean=np.mean(np.sqrt(np.sum(lognorm_func(traindata)**2,axis=1)))
        
        lognorm_func = lambda x: (((np.log10(np.clip(x,min_nonzero,None))-logmean_nonzero)/logstd_nonzero)*logstd_new+logmean_new)*(x>=min_nonzero)/data_rownorm_mean
        #for setting x<min_nonzero to 0... how?
        lognorm_inv_func = lambda y: np.clip(10**(((y*data_rownorm_mean-logmean_new)/logstd_new)*logstd_nonzero+logmean_nonzero),0,None)
        
        transformer=FunctionTransformer(func=lognorm_func,
                                        inverse_func=lognorm_inv_func)
        transformer_info["type"]="lognorm"
        transformer_info["params"]={"input_minimum_nonzero":min_nonzero,
                                    "input_logmean_nonzero":logmean_nonzero,
                                    "input_logstdev_nonzero":logstd_nonzero, 
                                    "output_logmean_nonzero":logmean_new, 
                                    "output_logstdev_nonzero":logstd_new, 
                                    "input_rownorm_mean":data_rownorm_mean}
        
    elif transformer_type == "torchfile":
        raise Exception("Not handling pretrained input transformer networks yet")
        ptfile=transformer_param_dict['transformer_file']
        
        encoder_net, encoder_checkpoint = Krakencoder.load_checkpoint(ptfile)
        
        pca_transformer=None
        if not encoder_checkpoint['leave_data_alone'] and encoder_checkpoint['reduce_dimension']:
            pca_transformer, pca_dict = generate_transformer(traindata, transformer_type="pca", 
                transformer_param_dict={'reduce_dimension':encoder_checkpoint['reduce_dimension']})
        
        if pca_transformer is not None:
            transformer=FunctionTransformer(func=lambda x:encoder_net.encoder_list[0](pca_transformer.transform(x)),
                                             inverse_func=lambda x:pca_transformer.inverse_transform(encoder_net.decoder_list[0](x)).cpu())
        else:
            transformer=FunctionTransformer(func=lambda x:encoder_net.encoder_list[0](x),
                                             inverse_func=lambda x:encoder_net.decoder_list[0](x))
        
        data_transformer_dict["type"]="torchfile"
        data_transformer_dict["params"]={"transformer_params":transformer_param_dict['transformer_file']}
        if pca_transformer is not None:
            data_transformer_dict["pca_dimension"]=encoder_checkpoint['reduce_dimension']
    
    #add type into "params" sub-dict for later saving
    transformer_info["params"]["type"]=transformer_info["type"]
    
    return transformer, transformer_info
    
def generate_training_paths(conndata_alltypes, conn_names, subjects, subjidx_train, subjidx_val, trainpath_pairs=[],trainpath_group_pairs=[], 
                            data_string=None, batch_size=6, skip_selfs=False, crosstrain_repeats=1, 
                            reduce_dimension=None, leave_data_alone=False, use_pretrained_encoder=False, keep_origscale_data=False, quiet=False,
                            use_lognorm_for_sc=False, use_truncated_svd=False, use_truncated_svd_for_sc=False, input_transformation_info=None,
                            precomputed_transformer_info_list={}):

    #data_string='fs86_volnorm'
    if not data_string:
        data_string=common_prefix(conn_names)+common_suffix(conn_names)

    if skip_selfs:
        data_string+='_noself'
    
    default_transformation_type="zscore+rownorm"
    
    if input_transformation_info is not None and input_transformation_info is not False:
        if re.search("^pc[0-9]+$",input_transformation_info):
            reduce_dimension=int(input_transformation_info.replace("pc",""))
            use_truncated_svd=False
        elif re.search("^tsvd[0-9]+$",input_transformation_info):
            reduce_dimension=int(input_transformation_info.replace("tsvd",""))
            use_truncated_svd=True
        elif re.search("^pc\+tsvd[0-9]+$",input_transformation_info):
            reduce_dimension=int(input_transformation_info.split("+")[-1].replace("pc","").replace("tsvd",""))
            use_truncated_svd=False
            use_truncated_svd_for_sc=True
        elif input_transformation_info.upper() == "NONE":
            reduce_dimension=0
            leave_data_alone=True
        elif input_transformation_info.upper() == "ZFEAT":
            reduce_dimension=0
            use_lognorm_for_sc=False
            default_transformation_type="zfeat"
        elif input_transformation_info.upper() == "CFEAT":
            reduce_dimension=0
            use_lognorm_for_sc=False
            default_transformation_type="cfeat"
        elif input_transformation_info.upper() == "CFEATNORM" or input_transformation_info.upper() == "CFEAT+NORM":
            reduce_dimension=0
            use_lognorm_for_sc=False
            default_transformation_type="cfeat+norm"
        elif input_transformation_info.upper() == "ZROWNORM" or input_transformation_info.upper() == "ZSCORE+ROWNORM":
            reduce_dimension=0
            use_lognorm_for_sc=False
        elif input_transformation_info.upper() == "LOGNORM+ROWNORM":
            reduce_dimension=0
            use_lognorm_for_sc=True
        elif input_transformation_info.upper() == "TORCHFILE":
            reduce_dimension=0
            use_pretrained_encoder=True
    
    xformstr=""
    if leave_data_alone:
        xformstr+="nonorm"
        input_transformation_info="none"
        
    elif reduce_dimension:
        if use_truncated_svd_for_sc:
            input_transformation_info="pc+tsvd%d" % (reduce_dimension)
            xformstr=input_transformation_info
        elif use_truncated_svd:
            input_transformation_info="tsvd%d" % (reduce_dimension)
            xformstr=input_transformation_info
        else:
            input_transformation_info="pc%d" % (reduce_dimension)
            xformstr=input_transformation_info
    else:
        #data_string+="_z"
        #data_string+="_zrownorm"
        if use_lognorm_for_sc:
            input_transformation_info="lognorm+rownorm"
            xformstr="z+sclog+rownorm"
        else:
            if not input_transformation_info:
                input_transformation_info=default_transformation_type
            xformstr=input_transformation_info
            xformstr=xformstr.replace("zscore+rownorm","zrownorm").replace("lognorm+rownorm","z+sclog+rownorm")
    
    data_string+="_"+xformstr
    
    data_string=data_string.replace('__','_')

    conn_names_short=[trim_string(s,left=len(common_prefix(conn_names)),right=len(common_suffix(conn_names))) for s in conn_names]

    if trainpath_group_pairs:
        trainpath_pairs=[]
        for g1,g2 in trainpath_group_pairs:
            c1list=[i for i,x in enumerate(conn_names) if conndata_alltypes[x]['group']==g1]
            c2list=[i for i,x in enumerate(conn_names) if conndata_alltypes[x]['group']==g2]
            for i in c1list:
                for j in c2list:
                    if skip_selfs and i==j:
                        if not quiet:
                            print("Skipping self-encoder %s->%s" % (i,j))
                        continue
                    trainpath_pairs+=[[i,j]]

    if trainpath_pairs == "self":
        trainpath_pairs=[]
        for i1,c1 in enumerate(conn_names):
            trainpath_pairs+=[[c1,c1]]
    
    #if we haven't already set a trainpath_pairs list, generate all pairs
    if not trainpath_pairs:
        for i1,c1 in enumerate(conn_names):
            for i2,c2 in enumerate(conn_names):
                if skip_selfs and c1==c2:
                    if not quiet:
                        print("Skipping self-encoder %s->%s" % (c1,c2))
                    continue
                trainpath_pairs+=[[c1,c2]]

    data_string+="_%dpaths" % (len(trainpath_pairs))
    
    #create/fit transformers for each of the input and output datasets
    #just once so we dont have to redo PCA for every single path
    data_transformer_list={}
    data_transformer_info_list={}
    traindata_list={}
    valdata_list={}
    
    traindata_marginmin_list={}
    traindata_marginmax_list={}
    traindata_marginmean_list={}
    
    traindata_origscale_list={}
    valdata_origscale_list={}
    
    unames=list(set(flatlist(trainpath_pairs)))
    #unames=[str(s) for s in np.unique(np.stack(trainpath_pairs).flatten())]
    for conn_name in unames:

        if type(conn_name) != str:
            #if index was provided
            i1=conn_name
            conn_name=conn_names[i1]
        
        if conn_name in data_transformer_list:
            continue
        
        if not quiet:
            print("Transforming input data for %s" % (conn_name), end="")
        
        traindata=conndata_alltypes[conn_name]['data'][subjidx_train]
        valdata=conndata_alltypes[conn_name]['data'][subjidx_val]
        datagroup=conndata_alltypes[conn_name]['group']
        
        if keep_origscale_data:
            #traindata_origscale_list[conn_name]=traindata.copy()
            #valdata_origscale_list[conn_name]=valdata.copy()
            traindata_origscale_list[conn_name]=conndata_alltypes[conn_name]['data'][subjidx_train]
            valdata_origscale_list[conn_name]=conndata_alltypes[conn_name]['data'][subjidx_val]
        
        #maybe need to be able to pass in a transformer
        #so we have the option of using a pretrained model
        data_transformer_dict={}
        
        precomputed_transformer_params=None
        precomputed_transformer_string=""
        if precomputed_transformer_info_list:
            if not conn_name in precomputed_transformer_info_list:
                raise Exception("'%s' not found in precomputed transformer file" % (conn_name))
            precomputed_transformer_params=precomputed_transformer_info_list[conn_name]
            precomputed_transformer_string=" (from file)"
        
        if use_pretrained_encoder and conndata_alltypes[conn_name]['transformer_file'] is not None:
            data_transformer_list[conn_name], data_transformer_dict = generate_transformer(traindata, transformer_type="torchfile", 
                transformer_param_dict={'transformer_file':conndata_alltypes[conn_name]['transformer_file']},
                precomputed_transformer_params=precomputed_transformer_params)
        
        elif leave_data_alone:
            data_transformer_list[conn_name], data_transformer_dict = generate_transformer(traindata,transformer_type="none",
                precomputed_transformer_params=precomputed_transformer_params)
        
        elif reduce_dimension:
            #reduce dimensionality using PCA on training data
            if use_truncated_svd_for_sc and datagroup == "SC":
                data_transformer_list[conn_name], data_transformer_dict = generate_transformer(traindata,transformer_type="tsvd",
                    transformer_param_dict={'reduce_dimension':reduce_dimension},
                    precomputed_transformer_params=precomputed_transformer_params)
            
                data_transformer_dict["tsvd_params"]=data_transformer_list[conn_name].get_params()
            elif use_truncated_svd:
                data_transformer_list[conn_name], data_transformer_dict = generate_transformer(traindata,transformer_type="tsvd",
                    transformer_param_dict={'reduce_dimension':reduce_dimension},
                    precomputed_transformer_params=precomputed_transformer_params)
            
                data_transformer_dict["tsvd_params"]=data_transformer_list[conn_name].get_params()
            else:
                data_transformer_list[conn_name], data_transformer_dict = generate_transformer(traindata,transformer_type="pca",
                    transformer_param_dict={'reduce_dimension':reduce_dimension},
                    precomputed_transformer_params=precomputed_transformer_params)
            
                data_transformer_dict["pca_params"]=data_transformer_list[conn_name].get_params()
        else:
            #normalize each SC matrix by the total L2 norm of the [trainsubj x pairs] for that type
            #data_transformer_list[conn_name], data_transformer_dict = generate_transformer(traindata,transformer_type="norm")
            if use_lognorm_for_sc and datagroup == "SC":
                xtype="lognorm+rownorm"
            else:
                #xtype="zscore+rownorm"
                xtype=default_transformation_type
            #xtype="zscore+rownorm"
            data_transformer_list[conn_name], data_transformer_dict = generate_transformer(traindata,transformer_type=xtype,
                    precomputed_transformer_params=precomputed_transformer_params)
    
        if not quiet:
            print(" using %s%s" % (data_transformer_dict['type'],precomputed_transformer_string))
        
        traindata_list[conn_name]=data_transformer_list[conn_name].transform(traindata)
        valdata_list[conn_name]=data_transformer_list[conn_name].transform(valdata)
        data_transformer_info_list[conn_name]=data_transformer_dict
        
        #compute min,mean,max of intersubject euclidean distances for this dataset
        #traindist=torch.cdist(traindata_list[conn_name],traindata_list[conn_name],p=2.0)
        traindist=scipy_cdist(traindata_list[conn_name],traindata_list[conn_name])
        traindist_triidx=np.triu_indices(traindist.shape[0],k=1)
        traindata_marginmin_list[conn_name]=traindist[traindist_triidx].min()
        traindata_marginmax_list[conn_name]=traindist[traindist_triidx].max()
        traindata_marginmean_list[conn_name]=traindist[traindist_triidx].mean()
        
    #now loop through the input/output pairs list
    trainpath_list=[]
    for c1,c2 in trainpath_pairs:
        if type(c1) == str:
            #if provided strings, find the index
            i1=[i for i,x in enumerate(conn_names) if x==c1][0]
            i2=[i for i,x in enumerate(conn_names) if x==c2][0]
        else:
            #if provided index, find the strings
            i1,i2=c1,c2
            c1,c2=conn_names[i1],conn_names[i2]
        
        c1_short=trim_string(c1,left=len(common_prefix(conn_names)),right=len(common_suffix(conn_names)))
        c2_short=trim_string(c2,left=len(common_prefix(conn_names)),right=len(common_suffix(conn_names)))

        input_npairs=conndata_alltypes[c1]['data'].shape[1]
        output_npairs=conndata_alltypes[c2]['data'].shape[1]
        
        input_transformer=data_transformer_list[c1]
        output_transformer=data_transformer_list[c2]
        
        train_inputs=traindata_list[c1]
        val_inputs=valdata_list[c1]
        train_outputs=traindata_list[c2]
        val_outputs=valdata_list[c2]

        train_encoded=None
        val_encoded=None
        
        if 'encoded' in conndata_alltypes[c1]:
            train_encoded=conndata_alltypes[c1]['encoded'][subjidx_train]
            val_encoded=conndata_alltypes[c1]['encoded'][subjidx_val]
        
        input_npairs=train_inputs.shape[1]
        output_npairs=train_outputs.shape[1]

        ########################
        ########################

        trainpath_tmp={}
        trainpath_tmp['input_name']=c1
        trainpath_tmp['input_name_short']=c1_short
        trainpath_tmp['input_npairs']=input_npairs
        trainpath_tmp['output_name']=c2
        trainpath_tmp['output_name_short']=c2_short
        trainpath_tmp['output_npairs']=output_npairs
        trainpath_tmp['input_transformer']=input_transformer
        trainpath_tmp['output_transformer']=output_transformer
        trainpath_tmp['encoder_index']=i1
        trainpath_tmp['decoder_index']=i2

        #train data to pytorch        
        train_inputs = torchfloat(train_inputs)
        train_outputs = torchfloat(train_outputs)
        val_inputs = torchfloat(val_inputs)
        val_outputs = torchfloat(val_outputs)
        
        if train_encoded is None:
            trainset = data_utils.TensorDataset(train_inputs, train_outputs)
        else:
            train_encoded=torchfloat(train_encoded)
            trainset = data_utils.TensorDataset(train_inputs, train_outputs, train_encoded)
        
        trainloader = data_utils.DataLoader(trainset, batch_size=batch_size,
                                                  shuffle=True, drop_last=True)
        
        trainpath_tmp['trainloader']=trainloader
        trainpath_tmp['train_inputs']=train_inputs
        trainpath_tmp['train_outputs']=train_outputs
        if train_encoded is not None:
            trainpath_tmp['train_encoded']=train_encoded

        trainpath_tmp['train_marginmin_inputs']=traindata_marginmin_list[c1]
        trainpath_tmp['train_marginmax_inputs']=traindata_marginmax_list[c1]
        trainpath_tmp['train_marginmean_inputs']=traindata_marginmean_list[c1]
        trainpath_tmp['train_marginmin_outputs']=traindata_marginmin_list[c2]
        trainpath_tmp['train_marginmax_outputs']=traindata_marginmax_list[c2]
        trainpath_tmp['train_marginmean_outputs']=traindata_marginmean_list[c2]
        
        #val data to pytorch

        #val_batchsize=batch_size
        val_batchsize=val_inputs.shape[0] #no reason to batch this and it just makes selecting batch size hard!
        
        if val_encoded is None:
            valset = data_utils.TensorDataset(val_inputs, val_outputs)
        else:
            val_encoded=torchfloat(val_encoded)
            valset = data_utils.TensorDataset(val_inputs, val_outputs, val_encoded)
        
        valloader = data_utils.DataLoader(valset, batch_size= val_batchsize,
                                                  shuffle=False, drop_last=False)

        trainpath_tmp['valloader']=valloader
        trainpath_tmp['val_inputs']=val_inputs
        trainpath_tmp['val_outputs']=val_outputs
        if val_encoded is not None:
            trainpath_tmp['val_encoded']=val_encoded
        
        #since it is easier to learn the self encoder, 
        # loop through the non-self ones X times for each epoch
        if c1 == c2:
            if skip_selfs:
                print("Skipping selfs")
                continue
            trainpath_tmp['trainloops']=1
        else:
            if skip_selfs:
                trainpath_tmp['trainloops']=1
            else:
                trainpath_tmp['trainloops']=crosstrain_repeats
                


        #these aren't specific to the given encoder but it is useful to return it here
        trainpath_tmp['data_string']=data_string
        raw_input_size_list=[conndata_alltypes[x]['numpairs'] for x in conn_names if x in conndata_alltypes]
        input_size_list=raw_input_size_list
        if reduce_dimension:
            input_size_list=[reduce_dimension for x in conn_names if x in conndata_alltypes]
        trainpath_tmp['subjects']=subjects
        trainpath_tmp['subjidx_train']=subjidx_train
        trainpath_tmp['subjidx_val']=subjidx_val
        trainpath_tmp['input_name_list']=conn_names
        trainpath_tmp['input_size_list']=input_size_list
        trainpath_tmp['raw_input_size_list']=raw_input_size_list
        trainpath_tmp['reduce_dimension']=reduce_dimension
        trainpath_tmp['input_transformation_info']=input_transformation_info
        trainpath_tmp['use_truncated_svd']=use_truncated_svd
        trainpath_tmp['use_truncated_svd_for_sc']=use_truncated_svd
        trainpath_tmp['use_lognorm_for_sc']=use_lognorm_for_sc
        trainpath_tmp['leave_data_alone']=leave_data_alone
        trainpath_tmp['skip_selfs']=skip_selfs
        if not reduce_dimension:
            trainpath_tmp['reduce_dimension']=0
        if not input_transformation_info:
            trainpath_tmp['input_transformation_info']=False #avoid saving None type
        
        trainpath_list.append(trainpath_tmp)

    data_origscale_list=None
    if keep_origscale_data:
        data_origscale_list={}
        data_origscale_list['traindata_origscale']=traindata_origscale_list
        data_origscale_list['valdata_origscale']=valdata_origscale_list
    return trainpath_list, data_origscale_list, data_transformer_info_list

##############################
##############################
# compute loss for a given path
def compute_path_loss(conn_predicted=None, conn_targets=None, conn_encoded=None, conn_encoded_targets=None, criterion=[], encoded_criterion=[], output_margin=None, encoder_margin=None, 
                latentnorm_loss_weight=0, latent_maxrad_weight=0, latent_maxrad=None):
    
    loss=0
    for crit in criterion:
        w=1
        if "weight" in crit:
            w=crit['weight']
        if "pass_margin" in crit:
            loss+=w*crit['function'](conn_predicted, conn_targets, margin=output_margin)
        else:
            #loss+=w*crit['function'](conn_predicted, conn_targets)
            #for some reason += causes problems with MSELoss because it returns a flat tensor, not an array
            #but loss=loss+ is OK with this
            loss=loss+w*crit['function'](conn_predicted, conn_targets)

    for enc_crit in encoded_criterion:
        w=1
        if "weight" in enc_crit:
            w=enc_crit['weight']
        if "pass_margin" in enc_crit:
            if conn_encoded_targets is None:
                loss_enc=w*enc_crit['function'](conn_encoded, conn_encoded, margin=encoder_margin)
            else:
                loss_enc=w*enc_crit['function'](conn_encoded, conn_encoded_targets, margin=encoder_margin)
        else:
            if conn_encoded_targets is None:
                loss_enc=w*enc_crit['function'](conn_encoded, conn_encoded)
            else:
                loss_enc=w*enc_crit['function'](conn_encoded, conn_encoded_targets)
        loss += loss_enc

    if latentnorm_loss_weight > 0:
        loss_latentnorm = torch.linalg.norm(conn_encoded)
        loss += latentnorm_loss_weight*loss_latentnorm

    if latent_maxrad_weight > 0:
        loss_latentrad = torch.mean(torch.nn.ReLU()(torch.sum(conn_encoded**2,axis=1)-latent_maxrad))
        loss += latent_maxrad_weight*loss_latentrad

    return loss

###################################
###################################
#define network training FUNCTION
def train_network(trainpath_list, training_params, net=None, data_origscale_list=None, trainfig=None, 
                  trainthreads=16, display_epochs=20, display_seconds=None, 
                  save_epochs=100, checkpoint_epochs=None, update_single_checkpoint=True, save_optimizer_params=True,
                  explicit_checkpoint_epoch_list=[], precomputed_transformer_info_list={}, save_input_transforms=True):

    trainpath_names=['%s->%s' % (tp['input_name'],tp['output_name']) for tp in trainpath_list]
    trainpath_names_short=['%s->%s' % (tp['input_name_short'],tp['output_name_short']) for tp in trainpath_list]
    trainpath_multiple=[tp['trainloops'] for tp in trainpath_list]
    data_string=trainpath_list[0]['data_string']
    coencoder_size_list=trainpath_list[0]['input_size_list']
    subjects=trainpath_list[0]['subjects']
    subjidx_train=trainpath_list[0]['subjidx_train']
    subjidx_val=trainpath_list[0]['subjidx_val']
    
    latentsize=training_params['latentsize']
    dropout=training_params['dropout']
    lr=training_params['learningrate']
    nbepochs=training_params['nbepochs']
    skip_relu=training_params['skip_relu']
    hiddenlayers=training_params['hiddenlayers']
    do_separate_optimizer=training_params['separate_optimizer']
    
    optimizer_name='adam'
    if 'optimizer_name' in training_params:
        optimizer_name=training_params['optimizer_name']
        
    do_zerograd_none=False
    if 'zerograd_none' in training_params:
        do_zerograd_none=training_params['zerograd_none']
        
    losstype='mse'
    if 'losstype' in training_params:
        losstype=training_params['losstype']
        
    do_skip_accurate_paths=False
    if 'skip_accurate_paths' in training_params:
        do_skip_accurate_paths=training_params['skip_accurate_paths']
    
    do_early_stopping_for_skipacc=True
    if 'early_stopping' in training_params:
        do_early_stopping_for_skipacc=training_params['early_stopping']
        
    latent_inner_loss_weight=1
    if 'latent_inner_loss_weight' in training_params:
        latent_inner_loss_weight=training_params['latent_inner_loss_weight']
        
    latentsim_loss_weight=0
    if 'latentsim_loss_weight' in training_params:
        latentsim_loss_weight=training_params['latentsim_loss_weight']
    
    latentnorm_loss_weight=0
    if 'latentnorm_loss_weight' in training_params:
        latentnorm_loss_weight=training_params['latentnorm_loss_weight']
        
    latent_maxrad=1
    if 'latent_maxrad' in training_params:
        latent_maxrad=training_params['latent_maxrad']
    
    latent_maxrad_weight=0
    if 'latent_maxrad_weight' in training_params:
        latent_maxrad_weight=training_params['latent_maxrad_weight']
    
    latent_normalize=False
    if 'latent_normalize' in training_params:
        latent_normalize=training_params['latent_normalize']
    
    relu_tanh_alternate=False
    if 'relu_tanh_alternate' in training_params:
        relu_tanh_alternate=training_params['relu_tanh_alternate']
        
    leakyrelu_negative_slope=False
    if 'leakyrelu_negative_slope' in training_params:
        leakyrelu_negative_slope=training_params['leakyrelu_negative_slope']
        
    mse_weight=0
    if 'mse_weight' in training_params:
        mse_weight=training_params['mse_weight']
    
    latent_activation='none'
    if 'latent_activation' in training_params:
        latent_activation=training_params['latent_activation']
    
    init_type=None
    if 'init_type' in training_params:
        init_type=training_params['init_type']
    
    adam_decay=None
    if 'adam_decay' in training_params:
        adam_decay=training_params['adam_decay']
    
    do_trainpath_shuffle=False
    if 'trainpath_shuffle' in training_params:
        do_trainpath_shuffle=training_params['trainpath_shuffle']
        
    do_roundtrip=False
    if 'roundtrip' in training_params:
        do_roundtrip=training_params['roundtrip']
    
    #how often to compute selfcc and othercc for untransformed space? (eg not PCA)
    origscalecorr_epochs=0
    if 'origscalecorr_epochs' in training_params:
        origscalecorr_epochs=training_params['origscalecorr_epochs']

    
    do_batchwise_latentsim=False
    latentsim_batchsize=0
    if 'latentsim_batchsize' in training_params:
        latentsim_batchsize=training_params['latentsim_batchsize']
        do_batchwise_latentsim=latentsim_batchsize>0
    
    do_meantarget_latentsim = False
    if 'meantarget_latentsim' in training_params:
        do_meantarget_latentsim = training_params['meantarget_latentsim']
    
    do_fixed_encoding = False
    if 'fixed_encoding' in training_params:
        do_fixed_encoding = training_params['fixed_encoding']
    
    #for 'fixed_encoding' mode, do not skip accurate paths and do not use latentsim
    if do_fixed_encoding:
        do_skip_accurate_paths=False
        latentsim_loss_weight=0
        latent_maxrad_weight=0
        latentnorm_loss_weight=0
        
        
    #if latent_normalize is true, use dot product for distance
    #do_latent_dotproduct_distance=latent_normalize
    do_latent_dotproduct_distance=False
    

    do_use_existing_net=net is not None
    do_initialize_net=False
    if net is None:
        net = Krakencoder(coencoder_size_list, latentsize=latentsize, 
                            hiddenlayers=hiddenlayers, skip_relu=skip_relu, dropout=dropout,
                            relu_tanh_alternate=relu_tanh_alternate, leakyrelu_negative_slope=leakyrelu_negative_slope,
                            latent_activation=latent_activation, latent_normalize=latent_normalize)
        do_initialize_net=True

    network_string=net.prettystring()
    network_parameter_count=sum([p.numel() for p in net.parameters()]) #count total weights in model
    network_description_string=str(net) #long multi-line pytorch-generated description
    
    lrstr="lr%g" % (lr)
    optimstr=""
    #if do_separate_optimizer:
    #    optimstr="_pathoptim"
    if not do_separate_optimizer:
        optimstr="_1op"
    
    optimname_str=""
    if optimizer_name != "adam":
        optimname_str="_%s" % (optimizer_name)
    if adam_decay is not None:
        optimname_str+=".w%g" % (adam_decay)
    
    zgstr=""
    #if do_zerograd_none:
    #    zgstr="_zgnone"
    skipaccstr=""
    if do_skip_accurate_paths:
        skipaccstr="_skipacc"
        if not do_early_stopping_for_skipacc:
            skipaccstr+="GO"
    

    timestamp=datetime.now()
    timestamp_suffix=timestamp.strftime("%Y%m%d_%H%M%S")

    print(net.parameters)

    #store this in record to help tell where training ran (cloud? cuda? local?)
    env_string="HOME=%s:CWD=%s:PLATFORM=%s:CUDA=%s" % (os.getenv('HOME'),os.getcwd(),platform.platform(),torch.cuda.is_available())
    
    if torch.cuda.is_available():
        net = net.cuda()

    maxthreads=trainthreads
    if not torch.cuda.is_available():
        torch.set_num_threads(maxthreads)
    
    initstr=""
    if init_type == "xavier" and do_initialize_net:
        #note: pytorch >1.0 uses kaiming by default now
        #so probably dont need this option
        def init_weights(m):
            if isinstance(m,nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)
        net.apply(init_weights)
        initstr="_init.xav"
    
    def makeoptim(optname="adam"):
        if optname == "adam":
            if adam_decay is not None:
                opt = optim.Adam(net.parameters(),lr=lr, weight_decay=adam_decay)
            else:
                opt = optim.Adam(net.parameters(),lr=lr)
        elif optname == "adamw":
            if adam_decay is not None:
                opt = optim.AdamW(net.parameters(),lr=lr, weight_decay=adam_decay)
            else:
                opt = optim.AdamW(net.parameters(),lr=lr)
        return opt

    ###############################

    do_pass_margins_to_loss=[]
    do_pass_margins_to_encoded_loss=[]

    criterion=[]
    encoded_criterion=[]
    latentsim_criterion=None

    skipacc_top1acc_function=corrtop1acc

    loss_override_dict=OrderedDict()
    if mse_weight > 0:
        loss_override_dict['mse']=mse_weight
    loss_override_dict['latentsimloss']=latentsim_loss_weight
    loss_override_dict['latentnormloss']=latentnorm_loss_weight
    loss_override_dict['latentmaxradloss']=latent_maxrad_weight

    lossgroup_default_weight={'output':1,'encoded':latent_inner_loss_weight,'encoded_meta':None}

    losstype_dict=loss_string_to_dict(losstype,override_weight_dict=loss_override_dict, lossgroup_default_weight=lossgroup_default_weight)

    if 'latentsimloss' in losstype_dict and 'suffix' in losstype_dict['latentsimloss'] and losstype_dict['latentsimloss']['suffix']=='B':
        do_batchwise_latentsim=True

    for lt, lt_item in losstype_dict.items():
        lt_w=lt_item['weight']
        lt_group=lt_item['lossgroup']
        
        if lt == 'mse':
            criterion += [{"function":nn.MSELoss(), "weight": torchfloat(lt_w)}]
        elif lt == 'msesum':
            criterion += [{"function":nn.MSELoss(reduction='sum'), "weight": torchfloat(lt_w)}]
        elif lt == 'corrtrace':
            criterion += [{"function":corrtrace, "weight": torchfloat(lt_w)}]
        elif lt == 'correye':
            criterion += [{"function":correye, "weight": torchfloat(lt_w)}]
        elif lt == 'corrmatch':
            criterion += [{"function":corrmatch, "weight": torchfloat(lt_w)}]
        elif lt == 'dist':
            criterion += [{"function":distance_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]
            skipacc_top1acc_function=disttop1acc
        elif lt == 'neidist':
            criterion += [{"function":distance_neighbor_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]
            skipacc_top1acc_function=disttop1acc

        elif lt == 'enceye':
            encoded_criterion += [{"function":correye, "weight": torchfloat(lt_w)}]
        elif lt == 'encdist':
            encoded_criterion += [{"function":distance_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]
        elif lt == 'encneidist':
            encoded_criterion += [{"function":distance_neighbor_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]
        elif lt == 'encdot':
            encoded_criterion += [{"function":dotproduct_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]
        elif lt == 'encneidot':
            encoded_criterion += [{"function":dotproduct_neighbor_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]

        elif lt_group=='encoded_meta':
            pass

        else:
            raise Exception("Unknown losstype: %s" % (lt_item))
    
    if do_fixed_encoding and mse_weight > 0:
        encoded_criterion += [{"function":nn.MSELoss(), "weight": torchfloat(mse_weight)}]

    latentnorm_loss_weight_torch=torchfloat(losstype_dict['latentnormloss']['weight'])
    latent_maxrad_weight_torch=torchfloat(losstype_dict['latentmaxradloss']['weight'])
    latent_maxrad_torch=torchfloat(latent_maxrad)
    latentsim_loss_weight_torch=torchfloat(losstype_dict['latentsimloss']['weight'])

    
    loss_str="_"+loss_dict_to_string(losstype_dict)
    if do_fixed_encoding:
        loss_str+="+fixlatent"

    train_string="%depoch_%s%s%s%s%s%s%s" % (nbepochs,lrstr,loss_str,optimstr,optimname_str,zgstr,skipaccstr,initstr)
    #if do_trainpath_shuffle:
    #       train_string+="_tpshuffle"
    if do_roundtrip:
        if do_use_existing_net:
            train_string+="_addroundtrip"
        else:
            train_string+="_roundtrip"
    
    optimizer_list=[]

    if do_separate_optimizer:
        optimizer_list=[makeoptim(optimizer_name) for i in range(len(trainpath_list))]
        optimizer_latentsim=makeoptim(optimizer_name) #make additional optimizer for just the latent similarity
    else:
        optimizer = makeoptim(optimizer_name)
        optimizer_latentsim = None
    ###############################

    epoch_timestamp = np.nan*np.zeros(nbepochs)
    
    loss_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    loss_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrloss_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrloss_val = np.nan*np.zeros((len(trainpath_list),nbepochs))

    corrlossN_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossN_val = np.nan*np.zeros((len(trainpath_list),nbepochs))

    corrlossRank_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossRank_val = np.nan*np.zeros((len(trainpath_list),nbepochs))

    avgcorr_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    avgcorr_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    
    avgcorr_other_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    avgcorr_other_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    
    latentsim_mse_train = np.nan*np.zeros(nbepochs)
    latentsim_corr_train = np.nan*np.zeros(nbepochs)
    latentsim_corr_other_train = np.nan*np.zeros(nbepochs)
    
    latentsim_mse_val = np.nan*np.zeros(nbepochs)
    latentsim_corr_val = np.nan*np.zeros(nbepochs)
    latentsim_corr_other_val = np.nan*np.zeros(nbepochs)
    
    latentsim_top1acc_train = np.nan*np.zeros(nbepochs)
    latentsim_topNacc_train = np.nan*np.zeros(nbepochs)
    latentsim_avgrank_train = np.nan*np.zeros(nbepochs)
        
    latentsim_top1acc_val = np.nan*np.zeros(nbepochs)
    latentsim_topNacc_val = np.nan*np.zeros(nbepochs)
    latentsim_avgrank_val = np.nan*np.zeros(nbepochs)
    
    avgcorr_OrigScale_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    avgcorr_OrigScale_other_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    avgcorr_OrigScale_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    avgcorr_OrigScale_other_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    
    corrloss_OrigScale_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossN_OrigScale_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossRank_OrigScale_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrloss_OrigScale_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossN_OrigScale_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossRank_OrigScale_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    
    topN=2
    batchsize=trainpath_list[0]['trainloader'].batch_size
    
    ################
    #make subject index dataloader for latentsimloss
    numsubjects_train=len(trainpath_list[0]['subjidx_train'])
    
    if do_batchwise_latentsim:
        tmp_latent_batchsize=latentsim_batchsize
    else:
        tmp_latent_batchsize=numsubjects_train
    
    latentsimloss_subjidx_dataloader=data_utils.DataLoader(np.arange(numsubjects_train), batch_size=tmp_latent_batchsize, shuffle=True, drop_last=True)

    recordfile="connae_trainrecord_%s_%s_%s_%s.mat" % (data_string,network_string,train_string,timestamp_suffix)
    imgfile="connae_%s_%s_%s_%s.png" % (data_string,network_string,train_string,timestamp_suffix)
    checkpoint_filebase="connae_chkpt_%s_%s_%s_%s" % (data_string,network_string,train_string,timestamp_suffix)
    
    #if saving transformer info (eg: precomputed PCA weights), need to remove the transformer OBJECT with embedded functions etc
    # from the list. Just save params
    input_transformer_file=""
    if precomputed_transformer_info_list and save_input_transforms:
         input_transformer_file="connae_ioxfm_%s_%s_%s_%s.npy" % (data_string,network_string,train_string,timestamp_suffix)
         transformer_params_to_save={}
         for k_iox in precomputed_transformer_info_list.keys():
             transformer_params_to_save[k_iox]=precomputed_transformer_info_list[k_iox]["params"]
             transformer_params_to_save[k_iox]["type"]=precomputed_transformer_info_list[k_iox]["type"]
             transformer_params_to_save[k_iox]["train_subjects"]=subjects[subjidx_train]
         np.save(input_transformer_file,transformer_params_to_save)
         del transformer_params_to_save #clear up memory
         print("Saved transforms: %s" % (input_transformer_file))

    trainrecord={}
    trainrecord['subjects'] = subjects
    trainrecord['subjidx_train']=subjidx_train
    trainrecord['subjidx_val']=subjidx_val
    trainrecord['numsubjects']=len(subjects)
    trainrecord['numsubjects_train']=len(subjidx_train)
    trainrecord['numsubjects_val']=len(subjidx_val)
    trainrecord['trainpath_names']=trainpath_names
    trainrecord['trainpath_names_short']=trainpath_names_short
    trainrecord['trainpath_multiple']=trainpath_multiple
    trainrecord['data_string']=data_string
    trainrecord['network_string']=network_string
    trainrecord['environment_string']=env_string
    trainrecord['trainpath_shuffle']=do_trainpath_shuffle
    trainrecord['roundtrip']=do_roundtrip
    trainrecord['train_string']=train_string
    trainrecord['nbepochs']=nbepochs
    trainrecord['learningrate']=lr
    trainrecord['separate_optimizer']=do_separate_optimizer
    trainrecord['batchsize']=batchsize
    trainrecord['latentsim_batchsize']=latentsim_batchsize
    trainrecord['losstype']=losstype
    trainrecord['latent_inner_loss_weight']=latent_inner_loss_weight
    trainrecord['mse_weight']=mse_weight
    trainrecord['latentsim_loss_weight']=latentsim_loss_weight
    trainrecord['latentnorm_loss_weight']=latentnorm_loss_weight
    trainrecord['latent_maxrad']=latent_maxrad
    trainrecord['latent_maxrad_weight']=latent_maxrad_weight
    trainrecord['latent_normalize']=latent_normalize
    trainrecord['optimizer_name']=optimizer_name
    trainrecord['zerograd_none']=do_zerograd_none
    trainrecord['skip_accurate_paths']=do_skip_accurate_paths
    trainrecord['early_stopping']=do_early_stopping_for_skipacc
    trainrecord['model_description']=network_description_string
    trainrecord['total_parameter_count']=network_parameter_count
    trainrecord['origscalecorr_epochs']=origscalecorr_epochs
    trainrecord['origscalecorr_inputtype']='inverse'
    trainrecord['saved_input_transformer_file']=input_transformer_file
    trainrecord['fixed_encoding']=do_fixed_encoding
    trainrecord['meantarget_latentsim']=do_meantarget_latentsim
    
    if data_origscale_list is not None:
        trainrecord['origscalecorr_inputtype']='original'
    
    if init_type:
        trainrecord['initialization']=init_type
    
    if adam_decay is not None:
        trainrecord['adam_decay']=adam_decay
    
    #save these for inclusion in checkpoint
    trainrecord_params=trainrecord.copy()
    
    #from trainpath generation:
    networkinfo={}
    
    networkinfo['reduce_dimension']=trainpath_list[0]['reduce_dimension']
    networkinfo['use_truncated_svd']=trainpath_list[0]['use_truncated_svd']
    networkinfo['input_transformation_info']=trainpath_list[0]['input_transformation_info']
    networkinfo['use_lognorm_for_sc']=trainpath_list[0]['use_lognorm_for_sc']
    networkinfo['use_truncated_svd_for_sc']=trainpath_list[0]['use_truncated_svd_for_sc']
    networkinfo['input_size_list']=trainpath_list[0]['input_size_list']
    networkinfo['orig_input_size_list']=trainpath_list[0]['raw_input_size_list']
    networkinfo['leave_data_alone']=trainpath_list[0]['leave_data_alone']
    networkinfo['skip_selfs']=trainpath_list[0]['skip_selfs']
    networkinfo['input_name_list']=trainpath_list[0]['input_name_list']
    networkinfo['trainpath_encoder_index_list']=[tp['encoder_index'] for tp in trainpath_list]
    networkinfo['trainpath_decoder_index_list']=[tp['decoder_index'] for tp in trainpath_list]
    
    networkinfo['latentsize']=latentsize
    networkinfo['hiddenlayers']=hiddenlayers
    networkinfo['skip_relu']=skip_relu
    networkinfo['leakyrelu_negative_slope']=leakyrelu_negative_slope
    networkinfo['relu_tanh_alternate']=relu_tanh_alternate
    networkinfo['dropout']=dropout
    networkinfo['latent_activation']=latent_activation
    networkinfo['latent_normalize']=latent_normalize

    
    for k in networkinfo.keys():
        trainrecord[k]=networkinfo[k]
    
    neg1_index=torchint(-1)

    #encoder margin=1.0
    encoder_margin_torch=torchfloat(1.0)
    
    #make the cpu-only copy of the data OUTSIDE the loop
    train_outputs_list_np={}
    val_outputs_list_np={}

    for itp,trainpath in enumerate(trainpath_list):
        c2=trainpath['output_name']
        if c2 in train_outputs_list_np:
            continue

        #make cpu copy of train and val OUTPUTS for top1acc testing in loop
        train_outputs_list_np[c2]=trainpath['train_outputs'].cpu().detach().numpy()
        val_outputs_list_np[c2]=trainpath['val_outputs'].cpu().detach().numpy()
            
    starttime=time.time()
    timestamp_last_display=starttime    
    
    
    do_eval_only = nbepochs==1
    if do_eval_only:
        print("Only evaluating network")
    
    skipped_epoch_counter=0
    do_compute_performance_AFTER_latentsim=True #this way recorded performance matches checkpoint

    #######################
    for epoch in range(nbepochs):
        
        all_train_acc=[False]*len(trainpath_list)
        all_val_acc=[False]*len(trainpath_list)
        
        allpath_train_enc=[None]*len(trainpath_list)
        allpath_val_enc=[None]*len(trainpath_list)
        
        if do_trainpath_shuffle:
            trainpath_order=np.argsort(np.random.random_sample(len(trainpath_list)))
        else:
            trainpath_order=np.arange(len(trainpath_list))
        
        #for itp,trainpath in enumerate(trainpath_list):
        for itp in trainpath_order:
            trainpath=trainpath_list[itp]
            
            if optimizer_list:
                optimizer=optimizer_list[itp]
            
            trainloader=trainpath['trainloader']
            valloader=trainpath['valloader']
            encoder_index=trainpath['encoder_index']
            decoder_index=trainpath['decoder_index']

            trainloops=trainpath['trainloops']

            train_inputs=trainpath['train_inputs']
            train_outputs=trainpath['train_outputs']
            val_inputs=trainpath['val_inputs']
            val_outputs=trainpath['val_outputs']
            train_encoded=None
            val_encoded=None
            if 'train_encoded' in trainpath:
                train_encoded=trainpath['train_encoded']
            if 'val_encoded' in trainpath:
                val_encoded=trainpath['val_encoded']
            
            output_transformer=trainpath['output_transformer']

            #make cpu copy of train and val OUTPUTS for top1acc testing in loop
            train_outputs_np=train_outputs_list_np[trainpath['output_name']]
            val_outputs_np=val_outputs_list_np[trainpath['output_name']]
            #train_outputs_np=train_outputs.cpu().detach().numpy()
            #val_outputs_np=val_outputs.cpu().detach().numpy()
            
            #should already have done this
            #if torch.cuda.is_available():
            #    train_inputs, train_outputs = train_inputs.cuda(), train_outputs.cuda()
            #    val_inputs, val_outputs = val_inputs.cuda(), val_outputs.cuda()

            encoder_index_torch=torchint(encoder_index)
            decoder_index_torch=torchint(decoder_index)
            output_margin_torch=torchfloat(trainpath['train_marginmean_outputs'])

            transcoder_list=None
            if do_roundtrip:
                transcoder_list=[decoder_index_torch]
                decoder_index_torch=encoder_index_torch
                
            #for some of the paths, we might find perfect fit early, 
            # so skip those (until we notice it starts to decline)
            #but for training this is maybe true TOO often?
            do_check_trainacc=False
            if do_skip_accurate_paths and do_check_trainacc:
                net.eval()
                with torch.no_grad():
                    conn_encoded, conn_predicted = net(train_inputs, encoder_index_torch, decoder_index_torch, transcoder_index_list=transcoder_list)
                acc=skipacc_top1acc_function(train_outputs, conn_predicted)
                if acc == 1:
                    #this is fitting all training data, so don't train on it for this epoch
                    #print("Skipping %s due to train acc" % (trainpath_names_short[itp]))
                    all_train_acc[itp]=True
                    trainloops=0
                    
            #for some of the paths, we might find perfect fit early, 
            # so skip those (until we notice it starts to decline)
            do_check_valacc=True
            if do_skip_accurate_paths and do_check_valacc:
                net.eval()
                with torch.no_grad():
                    conn_encoded, conn_predicted = net(val_inputs, encoder_index_torch, decoder_index_torch, transcoder_index_list=transcoder_list)
                acc=skipacc_top1acc_function(val_outputs,conn_predicted)
                if acc == 1:
                    #this is fitting all validation data, so don't train on it for this epoch
                    #print("Skipping %s due to val acc" % (trainpath_names_short[itp]))
                    all_val_acc[itp]=True
                    trainloops=0
            
            if do_eval_only:
                trainloops=0
            

            criterion_latent_mse=nn.MSELoss()
            
            net.train()

            for iloop in range(trainloops):
                train_running_loss = 0

                for batch_idx, train_data in enumerate(trainloader):
                    
                    if do_fixed_encoding:
                        #pulls out <batchsize> at a time
                        conn_inputs, conn_targets, conn_encoded_targets = train_data
                        
                        #first compute current encoding and backprop encoder loss
                        optimizer.zero_grad(set_to_none=do_zerograd_none)
                                            
                        conn_encoded = net(conn_inputs, encoder_index_torch, neg1_index)
                        
                        #loss = criterion_latent_mse(conn_encoded,conn_encoded_targets) #where can we use this?
                        
                        loss = compute_path_loss(conn_encoded=conn_encoded, conn_encoded_targets=conn_encoded_targets, encoded_criterion=encoded_criterion, encoder_margin=encoder_margin_torch, 
                                        latentnorm_loss_weight=latentnorm_loss_weight_torch, latent_maxrad_weight=latent_maxrad_weight_torch, latent_maxrad=latent_maxrad_torch)
                        
                        loss.backward()
                        optimizer.step()
                        
                        #Then compute predicted output and backprop decoder loss
                        optimizer.zero_grad(set_to_none=do_zerograd_none)
                        _ , conn_predicted = net(conn_encoded_targets, neg1_index, decoder_index_torch)
                    
                        loss = compute_path_loss(conn_predicted=conn_predicted, conn_targets=conn_targets, criterion=criterion, output_margin=output_margin_torch)
                        loss.backward()
                        optimizer.step()
                    else:
                        optimizer.zero_grad(set_to_none=do_zerograd_none)
                        
                        #pulls out <batchsize> at a time
                        conn_inputs, conn_targets = train_data
                        conn_encoded, conn_predicted = net(conn_inputs,encoder_index_torch,decoder_index_torch, transcoder_index_list=transcoder_list)
                    
                        #loss.backward() ACCUMULATES gradients in net parameters
                        #optimizer.step() propagates according to those gradients
                        #optimizer.zero_grad() ZEROS gradients so backprop is only based on current step

                        ######################
                        # loss terms (training)
                        loss = compute_path_loss(conn_predicted=conn_predicted, conn_targets=conn_targets, conn_encoded=conn_encoded, criterion=criterion, encoded_criterion=encoded_criterion, 
                            output_margin=output_margin_torch, encoder_margin=encoder_margin_torch, latentnorm_loss_weight=latentnorm_loss_weight_torch, 
                            latent_maxrad_weight=latent_maxrad_weight_torch, latent_maxrad=latent_maxrad_torch)

                        loss.backward()
                        optimizer.step()

                        train_running_loss += loss.item() 

                loss_train[itp,epoch] = train_running_loss/(batch_idx+1)    

            # validation
            net.eval()
            val_running_loss = 0

            for batch_idx, val_data in enumerate(valloader):
                    
                if do_fixed_encoding:
                    #pulls out <batchsize> at a time
                    conn_inputs, conn_targets, conn_encoded_targets = val_data
                    
                    with torch.no_grad():
                        conn_encoded = net(conn_inputs, encoder_index_torch, neg1_index)
                        _ , conn_predicted = net(conn_encoded_targets, neg1_index, decoder_index_torch)
                        
                    #loss = criterion_latent_mse(conn_encoded,conn_encoded_targets) #where can we use this?
                    loss = compute_path_loss(conn_encoded=conn_encoded, conn_encoded_targets=conn_encoded_targets, encoded_criterion=encoded_criterion, encoder_margin=encoder_margin_torch, 
                                    latentnorm_loss_weight=latentnorm_loss_weight_torch, latent_maxrad_weight=latent_maxrad_weight_torch, latent_maxrad=latent_maxrad_torch)
                    
                    loss = compute_path_loss(conn_predicted=conn_predicted, conn_targets=conn_targets, criterion=criterion, output_margin=output_margin_torch)
                else:
                    #with new val_batchsize, this should be all the val data at once
                    conn_inputs, conn_targets = val_data
                
                    with torch.no_grad():
                        #conn_encoded, conn_predicted = net(conn_inputs)
                        conn_encoded, conn_predicted = net(conn_inputs,encoder_index_torch,decoder_index_torch, transcoder_index_list=transcoder_list)

                    ######################
                    # loss terms (validation)
                    loss = compute_path_loss(conn_predicted=conn_predicted, conn_targets=conn_targets, conn_encoded=conn_encoded, criterion=criterion, encoded_criterion=encoded_criterion, 
                        output_margin=output_margin_torch, encoder_margin=encoder_margin_torch, latentnorm_loss_weight=latentnorm_loss_weight_torch, 
                        latent_maxrad_weight=latent_maxrad_weight_torch, latent_maxrad=latent_maxrad_torch)
                
                #######################
                val_running_loss += loss.item() 
            
            loss_val[itp,epoch] = val_running_loss/(batch_idx+1)

        ####################################
        ####################################
        # compute between-path latent space similiarity loss and backpropagate
        
        if latentsim_loss_weight>0 and len(trainpath_list)>1 and not do_eval_only:
            criterion_latentsim=nn.MSELoss()
            
            do_inputwise_latentsim = True
            
            trainrecord['inputwise_latentsim_loss']=do_inputwise_latentsim
            trainrecord['meantarget_loss']=do_meantarget_latentsim
            #note: for contrastive, do we want batchwise? (need batchsize larger to make sure we have each encoder to train enough times)
            #like:
            #allinput_train_encoded stacked into (numencoders*numtrainsubj) x latentsize = 1470 x 128, and a [1470 x 1] subject label
            #except we need the input data during net.train(), so .... allinput_train = [numencoders*numtrainsubj] x inputsize = 5*294 x 256 = 1470x256 and 1470x1 subject label
            #for batchidx, trainset in dataloader(batchsize=42*numencoders):
            #   traininputs, subjlabel, encoderlabel = trainset
            #   net.train()
            #   for ienc in unique(encoderlabel):
            #       train_encoded[encoderlabel==ienc] = net(traininputs[encoderlabel==ienc,encoder_index=ienc,decoder_index=-1)
            #   for subj in np.unique(subjlabel):
            #       subjdata=train_encoded[subjlabel==subj]
            #       posdist=similarity(subjdata,normalize(subjdata.mean(axis=0,keepdim=True),axis=1)) #note: need to normalize after averaging
            #       negdist=similarity(subjdata,train_encoded[subjlabel!=subj])
            #       #note similarity(x,y) = x@y.T which gives (numencoder x 1) for proto and (numencoder x numothersubj*numencoder) for negative
            #       #so we can then sum those or something
            #       loss+=posdist-negdist 
            #       #using loss with logs and exponents and all that fun stuff from Konkle paper
            #   loss.backwards()
            #   optim.step()
            
            do_contrastive_latentsim=False
            
            if do_inputwise_latentsim:
                encidx , tpidx=np.unique([tp['encoder_index'] for tp in trainpath_list],return_index=True)
                
                
                #if epoch % 10 == 0:
                #    savemat("testCC%05d.mat" % (epoch),{"allinput_train_enc":[x.numpy().copy() for x in allinput_train_encoded]},format='5',do_compression=True)
                
                
                loss=0
                if optimizer_latentsim is not None:
                    optimlat=optimizer_latentsim
                else:
                    optimlat=optimizer
                
                #compute the total inter-input encoding similarity loss
                for batch_idx, batchsubjidx in enumerate(latentsimloss_subjidx_dataloader):
                    
                    batchinput_train_encoded=[]
                    batchinput_train_encoded_mean=0
                    batchinput_train_encoded_count=0
                    net.eval()
                
                    #compute the training data encoding for each input
                    for ienc, itp in enumerate(tpidx):
                        train_inputs=trainpath_list[itp]['train_inputs'][batchsubjidx]
                        encoder_index=trainpath_list[itp]['encoder_index']
                        encoder_index_torch=torchint(encoder_index)
                        
                        with torch.no_grad():
                            conn_encoded = net(train_inputs,encoder_index_torch,neg1_index)
                        
                        if do_meantarget_latentsim:
                            batchinput_train_encoded_mean+=conn_encoded
                            batchinput_train_encoded_count+=1
                        else:
                            batchinput_train_encoded+=[conn_encoded]
                    if do_meantarget_latentsim:
                        batchinput_train_encoded_mean=batchinput_train_encoded_mean/batchinput_train_encoded_count
                        if latent_normalize:
                            #renormalize mean to hypersphere shell
                            batchinput_train_encoded_mean=nn.functional.normalize(batchinput_train_encoded_mean,p=2,dim=1)
                        
                    net.train()
                    optimizer.zero_grad(set_to_none=do_zerograd_none)
                    
                    loss=0
                    for ienc, itp in enumerate(tpidx):
                        train_inputs=trainpath_list[itp]['train_inputs'][batchsubjidx]
                        encoder_index=trainpath_list[itp]['encoder_index']
                        encoder_index_torch=torchint(encoder_index)
                        
                        conn_encoded = net(train_inputs,encoder_index_torch,neg1_index)
                        
                        #conn_encoded = numtrainsubj x latentsize
                        #allinput_train_encoded = [num encoders] list of conn_encodeds
                        
                        if do_meantarget_latentsim:
                            loss+=criterion_latentsim(conn_encoded,batchinput_train_encoded_mean)
                        else:
                            loss+=sum([criterion_latentsim(conn_encoded,x) for jenc,x in enumerate(batchinput_train_encoded) if ienc != jenc])
                        #maybe also add some version of this that isn't sensitive to overall scale?
                
                    loss=loss*latentsim_loss_weight_torch
                    loss.backward()
                    optimlat.step()
            else:
                #if epoch % 10 == 0:
                #    savemat("testB%05d.mat" % (epoch),{"allpath_train_enc":[x.numpy().copy() for x in allpath_train_enc]},format='5',do_compression=True)
                
                net.train()
                
                if optimizer_latentsim is not None:
                    optimlat=optimizer_latentsim
                else:
                    optimlat=optimizer
                optimlat.zero_grad(set_to_none=do_zerograd_none)
                
                loss=0
                #for itp,trainpath in enumerate(trainpath_list):
                for itp in trainpath_order:
                    trainpath = trainpath_list[itp]
                    encoder_index=trainpath['encoder_index']
                    decoder_index=trainpath['decoder_index']

                    trainloops=trainpath['trainloops']

                    train_inputs=trainpath['train_inputs']
                    train_outputs=trainpath['train_outputs']
                
                    #if optimizer_list:
                    #    optimizer=optimizer_list[itp]
                    encoder_index_torch=torchint(encoder_index)
                    decoder_index_torch=torchint(decoder_index)
                    
                    #assume this is done during trainpath generation
                    #if torch.cuda.is_available():
                    #    train_inputs = train_inputs.cuda()
                
                    #this was being called AFTER net(...) until the first mse_weight tests 3/27!   
                    #I dont think there was a reason for this, and it shouldn't matter (confirmed no change)
                    #but moving it before net(...) is what we do elsewhere
                    #if optimizer_list:
                    #    optimizer.zero_grad(set_to_none=do_zerograd_none)
                
                    #use encoder-only mode!
                    conn_encoded = net(train_inputs,encoder_index_torch,neg1_index)
             
                
                    tmploss=latentsim_loss_weight_torch*sum([criterion_latentsim(conn_encoded,x) for jtp,x in enumerate(allpath_train_enc) if itp != jtp])
                    #if optimizer_list:
                    #    tmploss.backward()
                    #    optimizer.step()
                
                    loss+=tmploss
                    
                #if not optimizer_list:
                #    loss.backward()
                #    optimizer.step()
                loss.backward()
                optimlat.step()
            
            net.eval()
        ############################
        #computing performance AFTER latentsim
        trainrecord['compute_performance_after_latentsim']=do_compute_performance_AFTER_latentsim
        if do_compute_performance_AFTER_latentsim:
            
            net.eval()
            for itp in trainpath_order:
                trainpath=trainpath_list[itp]
            
                if optimizer_list:
                    optimizer=optimizer_list[itp]
            
                trainloader=trainpath['trainloader']
                valloader=trainpath['valloader']
                encoder_index=trainpath['encoder_index']
                decoder_index=trainpath['decoder_index']

                trainloops=trainpath['trainloops']

                train_inputs=trainpath['train_inputs']
                train_outputs=trainpath['train_outputs']
                val_inputs=trainpath['val_inputs']
                val_outputs=trainpath['val_outputs']
                train_encoded=None
                val_encoded=None
                if 'train_encoded' in trainpath:
                    train_encoded=trainpath['train_encoded']
                if 'val_encoded' in trainpath:
                    val_encoded=trainpath['val_encoded']
                    
                output_transformer=trainpath['output_transformer']

                #make cpu copy of train and val OUTPUTS for top1acc testing in loop
                train_outputs_np=train_outputs_list_np[trainpath['output_name']]
                val_outputs_np=val_outputs_list_np[trainpath['output_name']]
                #train_outputs_np=train_outputs.cpu().detach().numpy()
                #val_outputs_np=val_outputs.cpu().detach().numpy()
            
                #should already have done this
                #if torch.cuda.is_available():
                #    train_inputs, train_outputs = train_inputs.cuda(), train_outputs.cuda()
                #    val_inputs, val_outputs = val_inputs.cuda(), val_outputs.cuda()

                encoder_index_torch=torchint(encoder_index)
                decoder_index_torch=torchint(decoder_index)
                output_margin_torch=torchfloat(trainpath['train_marginmean_outputs'])

                transcoder_list=None
                if do_roundtrip:
                    transcoder_list=[decoder_index_torch]
                    decoder_index_torch=encoder_index_torch
        
                #compute full train set identifiability
                with torch.no_grad():
                    if do_fixed_encoding:
                        conn_encoded = net(train_inputs, encoder_index_torch, neg1_index)
                        _,conn_predicted = net(train_encoded, neg1_index, decoder_index_torch)
                    else:
                        #conn_encoded, conn_predicted = net(train_inputs)
                        conn_encoded, conn_predicted = net(train_inputs, encoder_index_torch, decoder_index_torch, transcoder_index_list=transcoder_list)

                #fc_preds = conn_predicted.cpu().detach().numpy()
                train_predicted=conn_predicted.cpu()
        
                allpath_train_enc[itp]=conn_encoded #store these encoded outputs to check across paths
                train_cc=xycorr(train_outputs,conn_predicted)
        
                corrloss_train[itp,epoch]=corrtop1acc(cc=train_cc)
                corrlossN_train[itp,epoch]=corrtopNacc(cc=train_cc,topn=topN)
                corrlossRank_train[itp,epoch]=corravgrank(cc=train_cc)
                avgcorr_train[itp,epoch],avgcorr_other_train[itp,epoch]=corr_ident_parts(cc=train_cc)
        
                #compute full val set identifiability
                with torch.no_grad():
                    if do_fixed_encoding:
                        conn_encoded = net(val_inputs, encoder_index_torch, neg1_index)
                        _, conn_predicted = net(val_encoded, neg1_index, decoder_index_torch)
                    else:
                        conn_encoded, conn_predicted = net(val_inputs, encoder_index_torch, decoder_index_torch, transcoder_index_list=transcoder_list)
                #fc_preds = conn_predicted.cpu().detach().numpy()

                allpath_val_enc[itp]=conn_encoded #store these encoded outputs to check across paths
                val_cc=xycorr(val_outputs,conn_predicted)
                corrloss_val[itp,epoch]=corrtop1acc(cc=val_cc)
                corrlossN_val[itp,epoch]=corrtopNacc(cc=val_cc,topn=topN)
                corrlossRank_val[itp,epoch]=corravgrank(cc=val_cc)
                avgcorr_val[itp,epoch],avgcorr_other_val[itp,epoch]=corr_ident_parts(cc=val_cc)
                
                #this was negligible for non-PCA transform (norm only)
                #added about 33% more time to each epoch for PCA
                origscale_this_epoch=epoch==nbepochs-1 or origscalecorr_epochs>0 and (epoch % origscalecorr_epochs == 0)
                
                if origscale_this_epoch and output_transformer is not None:
                    if data_origscale_list is not None:
                        valOrig_outputs=data_origscale_list['valdata_origscale'][trainpath['output_name']]
                    else:
                        valOrig_outputs=output_transformer.inverse_transform(val_outputs.cpu())
                    valOrig_predicted=output_transformer.inverse_transform(conn_predicted.cpu())
            
                    #if requested, look at the difference between mean(traindata) and mean(restore(train predicted))
                    #and shift the origscale restore(val predicted) by that amount 
                    do_adjust_train_restore_mean=False
                    if do_adjust_train_restore_mean:
                        if data_origscale_list is not None:
                            trainOrig_outputs=data_origscale_list['traindata_origscale'][trainpath['output_name']]
                        else:
                            trainOrig_outputs=output_transformer.inverse_transform(train_outputs.cpu())
                        trainOrig_predicted=output_transformer.inverse_transform(train_predicted)
                
                        trainOrig_output_mean=trainOrig_outputs.mean(axis=0,keepdims=True)
                        trainOrig_predicted_mean=trainOrig_predicted.mean(axis=0,keepdims=True)
                        trainOrig_meandiff=trainOrig_predicted_mean-trainOrig_output_mean
                
                        valOrig_predicted-=trainOrig_meandiff
                
                    #################
                    if True:
                        valOrig_outputs=torchfloat(valOrig_outputs).cpu()
                        valOrig_predicted=torchfloat(valOrig_predicted).cpu()
                    else:
                        if not torch.is_tensor(valOrig_outputs):
                            #for some reason we need to explicitly convert this sometimes
                            valOrig_outputs=torch.from_numpy(valOrig_outputs).float()
            
                        if not torch.is_tensor(valOrig_predicted):
                            #for some reason we need to explicitly convert this sometimes
                            valOrig_predicted=torch.from_numpy(valOrig_predicted).float()
            
                    #for even less clear reason, these are sometimes double() instead of float
                    valOrig_cc=xycorr(valOrig_outputs.float(),valOrig_predicted.float())
            
                    corrloss_OrigScale_val[itp,epoch]=corrtop1acc(cc=valOrig_cc)
                    corrlossN_OrigScale_val[itp,epoch]=corrtopNacc(cc=valOrig_cc,topn=topN)
                    corrlossRank_OrigScale_val[itp,epoch]=corravgrank(cc=valOrig_cc)
                    avgcorr_OrigScale_val[itp,epoch],avgcorr_OrigScale_other_val[itp,epoch]=corr_ident_parts(cc=valOrig_cc)
                
                
        ####################################
        #compute current encoding correlation and mse for training curve info
        #needs to happen after "performance BEFORE latentsim" or "performance AFTER latentsim" sections, otherwise we get zeros
        tmp_encmse_val=0
        tmp_encmse_train=0
        tmp_encmse_count=0
        
        tmp_enccorr_val=0
        tmp_enccorr_other_val=0
        tmp_enccorr_train=0
        tmp_enccorr_other_train=0
        
        tmp_enccorr_top1acc_val=0
        tmp_enccorr_top1acc_train=0
        
        tmp_enccorr_topNacc_val=0
        tmp_enccorr_topNacc_train=0
        
        tmp_enccorr_avgrank_val=0
        tmp_enccorr_avgrank_train=0
        
        for itp in range(len(allpath_val_enc)):
            #break
            if allpath_val_enc[itp] is None or allpath_train_enc[itp] is None:
                continue
            for jtp in range(itp+1,len(allpath_val_enc)):
                if allpath_val_enc[jtp] is None or allpath_train_enc[jtp] is None:
                    continue
                
                tmp_encmse_count+=1
                #epoch time is 2.1sec for just mse
                #epoch time is 7.8sec for mse+ val and train corrself/other
                #epoch time is 2.2sec for NONE
                #epoch time is 2.2sec for mse+ VAL corrself/other
                #epoch time is 2.5sec for mse+ VAL full corr set
                tmp_encmse_val+=torch.mean((allpath_val_enc[itp]-allpath_val_enc[jtp])**2)
                tmp_encmse_train+=torch.mean((allpath_train_enc[itp]-allpath_train_enc[jtp])**2)
                
                
                #train cc
                #this makes it take a long time
                #train_cc=xycorr(allpath_train_enc[itp],allpath_train_enc[jtp]) #subj x subj
                #tmp_self,tmp_other=corr_ident_parts(cc=train_cc)
                #tmp_enccorr_train+=tmp_self
                #tmp_enccorr_other_train+=tmp_other
                #tmp_enccorr_top1acc_train+=corrtop1acc(cc=train_cc)
                #tmp_enccorr_topNacc_train+=corrtopNacc(cc=train_cc,topn=topN)
                #tmp_enccorr_avgrank_train+=corravgrank(cc=train_cc)
            
                #val cc
                val_cc=xycorr(allpath_val_enc[itp],allpath_val_enc[jtp]) #subj x subj
                tmp_self,tmp_other=corr_ident_parts(cc=val_cc)
                tmp_enccorr_val+=tmp_self
                tmp_enccorr_other_val+=tmp_other
                #tmp_enccorr_top1acc_val+=corrtop1acc(cc=val_cc)
                #tmp_enccorr_topNacc_val+=corrtopNacc(cc=val_cc,topn=topN)
                #tmp_enccorr_avgrank_val+=corravgrank(cc=val_cc)
                
        if tmp_encmse_count == 0:
            tmp_encmse_count = 1
        latentsim_mse_val[epoch]=tmp_encmse_val/tmp_encmse_count
        latentsim_mse_train[epoch]=tmp_encmse_train/tmp_encmse_count
        
        latentsim_corr_val[epoch]=tmp_enccorr_val/tmp_encmse_count
        latentsim_corr_other_val[epoch]=tmp_enccorr_other_val/tmp_encmse_count
        
        #latentsim_corr_train[epoch]=tmp_enccorr_train/tmp_encmse_count
        #latentsim_corr_other_train[epoch]=tmp_enccorr_other_train/tmp_encmse_count
        
        #latentsim_top1acc_train[epoch]=tmp_enccorr_top1acc_train/tmp_encmse_count
        #latentsim_topNacc_train[epoch]=tmp_enccorr_topNacc_train/tmp_encmse_count
        #latentsim_avgrank_train[epoch]=tmp_enccorr_avgrank_train/tmp_encmse_count
        
        #latentsim_top1acc_val[epoch]=tmp_enccorr_top1acc_val/tmp_encmse_count
        #latentsim_topNacc_val[epoch]=tmp_enccorr_topNacc_val/tmp_encmse_count
        #latentsim_avgrank_val[epoch]=tmp_enccorr_avgrank_val/tmp_encmse_count
        
        
        ############################
        epoch_timestamp[epoch]=time.time()-starttime
        
        display_on_this_loop=epoch % display_epochs == 0 or epoch==nbepochs-1
        save_on_this_loop=epoch>0 and epoch % save_epochs == 0
        exit_on_this_loop=False
        checkpoint_on_this_loop=checkpoint_epochs and epoch > 0 and (epoch % checkpoint_epochs == 0 or epoch==nbepochs-1)

        if epoch in explicit_checkpoint_epoch_list:
            checkpoint_on_this_loop=True
        
        if do_skip_accurate_paths:
            #if we skipped all paths because they were accurate (via correlation top1acc)
            #then we can exit training early
            #ALTHOUGH: do we want to keep training for latent MSE similarity?
            if all(all_train_acc) or all(all_val_acc):
                skipped_epoch_counter+=1
                #print("epoch %s: All paths were skipped!" % (epoch))
                if do_early_stopping_for_skipacc:
                    print("Exiting now because early stopping was selected")
                    exit_on_this_loop=True
                    save_on_this_loop=True
                    display_on_this_loop=True
                    checkpoint_on_this_loop=checkpoint_epochs is not None and checkpoint_epochs>0
                
                
        if display_on_this_loop:
            tmp_time=time.time()
            if not display_seconds or (tmp_time-timestamp_last_display)>=display_seconds:
                timestamp_last_display=tmp_time
                if(False and len(trainpath_list)==1):
                    print('epoch: {}, train loss: {:.6f}, val loss:{:.6f}, train top1cc:{:.0f}/{:d}={:.6f}, val top1acc:{:.0f}/{:d}={:.6f}'.format(
                        epoch, loss_train[0,epoch], loss_val[0,epoch], 
                        corrloss_train[0,epoch]*train_outputs_np.shape[0], train_outputs_np.shape[0], corrloss_train[0,epoch], 
                        corrloss_val[0,epoch]*val_outputs_np.shape[0],val_outputs_np.shape[0], corrloss_val[0,epoch]
                    ))
                else:
                    curtime=time.time()-starttime
                    print("trainrecord: %s" % (recordfile))
                    print('epoch: {}, {:.2f} seconds ({:.2f} sec/epoch)'.format(epoch,curtime,curtime/(epoch+1)))
                    print('  path, train loss, val loss, train top1acc, val top1acc, train topNacc, val topNacc, train rank, val rank')
                    for itp in range(len(trainpath_list)):

                        print('  path{}:{}->{}, {:12.6f}, {:12.6f}, {:.0f}/{:d}={:.6f}, {:.0f}/{:d}={:.6f}, {:.6f}, {:.6f}, {:.3f}, {:.3f}'.format(
                            itp, trainpath_list[itp]['encoder_index'],trainpath_list[itp]['decoder_index'],
                            loss_train[itp,epoch], loss_val[itp,epoch], 
                            corrloss_train[itp,epoch]*train_outputs_np.shape[0], train_outputs_np.shape[0], corrloss_train[itp,epoch], 
                            corrloss_val[itp,epoch]*val_outputs_np.shape[0],val_outputs_np.shape[0], corrloss_val[itp,epoch],
                            corrlossN_train[itp,epoch],
                            corrlossN_val[itp,epoch],
                            corrlossRank_train[itp,epoch],
                            corrlossRank_val[itp,epoch]
                        ))
                    if skipped_epoch_counter > 0:
                        print('  skipped all paths for {}/{} epochs'.format(skipped_epoch_counter,display_epochs))
                    print('epoch: {}, {:.2f} seconds ({:.2f} sec/epoch)'.format(epoch,curtime,curtime/(epoch+1)))

                skipped_epoch_counter=0
                
                avgrank_train=1+(1-corrlossRank_train)*len(subjidx_train)
                avgrank_val=1+(1-corrlossRank_val)*len(subjidx_val)

                trainfig=update_training_figure(loss_train, loss_val, corrloss_train, corrloss_val, corrlossN_train, corrlossN_val, corrlossRank_train,
                    corrlossRank_val, avgrank_train, avgrank_val, trainpath_names_short, data_string, network_string, train_string, losstype, epoch, 
                    trainfig=trainfig, gridalpha=.25, topN=topN)

                #only in jupyter
                #display.display(trainfig)
                #display.clear_output(wait=True)

        if save_on_this_loop:

            trainduration=time.time()-starttime
            
            avgrank_train=1+(1-corrlossRank_train)*len(subjidx_train)
            avgrank_val=1+(1-corrlossRank_val)*len(subjidx_val)
                
            trainfig.savefig(imgfile,facecolor='w',dpi=100)
            print("Ep %d) Saved %s" % (epoch, imgfile))
            
            trainrecord['loss_train']=loss_train
            trainrecord['loss_val']=loss_val
            trainrecord['corrloss_train']=corrloss_train
            trainrecord['corrloss_val']=corrloss_val
            trainrecord['corrlossN_train']=corrlossN_train
            trainrecord['corrlossN_val']=corrlossN_val
            trainrecord['corrlossRank_train']=corrlossRank_train
            trainrecord['corrlossRank_val']=corrlossRank_val
            trainrecord['avgrank_train']=avgrank_train
            trainrecord['avgrank_val']=avgrank_val
            trainrecord['avgcorr_train']=avgcorr_train
            trainrecord['avgcorr_other_train']=avgcorr_other_train
            trainrecord['avgcorr_val']=avgcorr_val
            trainrecord['avgcorr_other_val']=avgcorr_other_val
            trainrecord['avgcorr_OrigScale_train']=avgcorr_OrigScale_train
            trainrecord['avgcorr_OrigScale_other_train']=avgcorr_OrigScale_other_train
            trainrecord['avgcorr_OrigScale_val']=avgcorr_OrigScale_val
            trainrecord['avgcorr_OrigScale_other_val']=avgcorr_OrigScale_other_val
            trainrecord['corrloss_OrigScale_train']=corrloss_OrigScale_train
            trainrecord['corrloss_OrigScale_val']=corrloss_OrigScale_val
            trainrecord['corrlossN_OrigScale_train']=corrlossN_OrigScale_train
            trainrecord['corrlossN_OrigScale_val']=corrlossN_OrigScale_val
            trainrecord['corrlossRank_OrigScale_train']=corrlossRank_OrigScale_train
            trainrecord['corrlossRank_OrigScale_val']=corrlossRank_OrigScale_val
            trainrecord['latentsim_mse_train']=latentsim_mse_train
            trainrecord['latentsim_corr_train']=latentsim_corr_train
            trainrecord['latentsim_corr_other_train']=latentsim_corr_other_train
            trainrecord['latentsim_mse_val']=latentsim_mse_val
            trainrecord['latentsim_corr_val']=latentsim_corr_val
            trainrecord['latentsim_corr_other_val']=latentsim_corr_other_val
            trainrecord['latentsim_top1acc_train']=latentsim_top1acc_train
            trainrecord['latentsim_topNacc_train']=latentsim_topNacc_train
            trainrecord['latentsim_avgrank_train']=latentsim_avgrank_train
            trainrecord['latentsim_top1acc_val']=latentsim_top1acc_val
            trainrecord['latentsim_topNacc_val']=latentsim_topNacc_val
            trainrecord['latentsim_avgrank_val']=latentsim_avgrank_val
            trainrecord['maxthreads']=maxthreads
            trainrecord['timestamp']=timestamp_suffix
            trainrecord['epoch_timestamp']=epoch_timestamp
            trainrecord['trainduration']=trainduration
            trainrecord['current_epoch']=epoch
            trainrecord['seconds_per_epoch']=trainduration/(epoch+1)

            savemat(recordfile,trainrecord,format='5',do_compression=True)
            print("Ep %d) Saved %s" % (epoch, recordfile))
            
        if checkpoint_on_this_loop:
            if update_single_checkpoint:
                statefile=checkpoint_filebase+".pt"
            else:
                if epoch == nbepochs-1:
                    statefile=checkpoint_filebase+"_ep%06d.pt" % (nbepochs)
                else:
                    statefile=checkpoint_filebase+"_ep%06d.pt" % (epoch)
            #checkpoint={"state_dict": net.state_dict(),"epoch": epoch}
            checkpoint={"epoch": epoch}
            
            #copy network description fields into checkpoint
            for k in networkinfo.keys():
                checkpoint[k]=networkinfo[k]
                
            checkpoint['training_params']=trainrecord_params
            
            if save_optimizer_params and (epoch == nbepochs-1 or exit_on_this_loop):
                #include optimizer in final checkpoint (so we could resume training)
                if optimizer_list:
                    checkpoint['optimizer']=[opt.state_dict() for opt in optimizer_list]
                else:
                    checkpoint['optimizer']=optimizer.state_dict()
            #torch.save(net.state_dict(), statefile)
            #torch.save(checkpoint, statefile)
            net.save_checkpoint(statefile, checkpoint)
            print("Ep %d) Saved %s" % (epoch, statefile))
            
        if exit_on_this_loop:
            break


    trainduration=time.time()-starttime
    print("Training took %.3f seconds" % (trainduration))

    trainfig.savefig(imgfile,facecolor='w',dpi=100)
    print("Saved %s" % (imgfile))

    avgrank_train=1+(1-corrlossRank_train)*len(subjidx_train)
    avgrank_val=1+(1-corrlossRank_val)*len(subjidx_val)
            

    trainrecord['loss_train']=loss_train
    trainrecord['loss_val']=loss_val
    trainrecord['corrloss_train']=corrloss_train
    trainrecord['corrloss_val']=corrloss_val
    trainrecord['corrlossN_train']=corrlossN_train
    trainrecord['corrlossN_val']=corrlossN_val
    trainrecord['corrlossRank_train']=corrlossRank_train
    trainrecord['corrlossRank_val']=corrlossRank_val
    trainrecord['avgrank_train']=avgrank_train
    trainrecord['avgrank_val']=avgrank_val
    trainrecord['avgcorr_train']=avgcorr_train
    trainrecord['avgcorr_other_train']=avgcorr_other_train
    trainrecord['avgcorr_val']=avgcorr_val
    trainrecord['avgcorr_other_val']=avgcorr_other_val
    trainrecord['avgcorr_OrigScale_train']=avgcorr_OrigScale_train
    trainrecord['avgcorr_OrigScale_other_train']=avgcorr_OrigScale_other_train
    trainrecord['avgcorr_OrigScale_val']=avgcorr_OrigScale_val
    trainrecord['avgcorr_OrigScale_other_val']=avgcorr_OrigScale_other_val
    trainrecord['corrloss_OrigScale_train']=corrloss_OrigScale_train
    trainrecord['corrloss_OrigScale_val']=corrloss_OrigScale_val
    trainrecord['corrlossN_OrigScale_train']=corrlossN_OrigScale_train
    trainrecord['corrlossN_OrigScale_val']=corrlossN_OrigScale_val
    trainrecord['corrlossRank_OrigScale_train']=corrlossRank_OrigScale_train
    trainrecord['corrlossRank_OrigScale_val']=corrlossRank_OrigScale_val
    trainrecord['latentsim_mse_train']=latentsim_mse_train
    trainrecord['latentsim_corr_train']=latentsim_corr_train
    trainrecord['latentsim_corr_other_train']=latentsim_corr_other_train
    trainrecord['latentsim_mse_val']=latentsim_mse_val
    trainrecord['latentsim_corr_val']=latentsim_corr_val
    trainrecord['latentsim_corr_other_val']=latentsim_corr_other_val
    trainrecord['latentsim_top1acc_train']=latentsim_top1acc_train
    trainrecord['latentsim_topNacc_train']=latentsim_topNacc_train
    trainrecord['latentsim_avgrank_train']=latentsim_avgrank_train
    trainrecord['latentsim_top1acc_val']=latentsim_top1acc_val
    trainrecord['latentsim_topNacc_val']=latentsim_topNacc_val
    trainrecord['latentsim_avgrank_val']=latentsim_avgrank_val
    trainrecord['maxthreads']=maxthreads
    trainrecord['timestamp']=timestamp_suffix
    trainrecord['epoch_timestamp']=epoch_timestamp
    trainrecord['trainduration']=trainduration
    trainrecord['current_epoch']=epoch
    trainrecord['seconds_per_epoch']=trainduration/(epoch+1)
    
    savemat(recordfile,trainrecord,format='5',do_compression=True)
    print("Saved %s" % (recordfile))
        
    #torch.save(net, statefile) #saves whole class
    #torch.save(net.state_dict(), statefile) #saves the weights (less sensitive to class changes)
    return net, trainrecord
    #interesting re single-path: 
    # with latentsize=16, I get to val (n=42) ident top1 acc=1.0 after 150 or so epochs
    # with latentsize=4, I get val id=0.80 after 200 epochs and then val decreases

def run_network(net, trainpath_list, maxthreads=1, burstmode=False, pathfinder_list=[], burstmode_search_list=[], burstmode_do_not_normalize=False):
    trainpath_names=['%s->%s' % (tp['input_name'],tp['output_name']) for tp in trainpath_list]
    trainpath_names_short=['%s->%s' % (tp['input_name_short'],tp['output_name_short']) for tp in trainpath_list]
    data_string=trainpath_list[0]['data_string']
    coencoder_size_list=trainpath_list[0]['input_size_list']
    subjects=trainpath_list[0]['subjects']
    subjidx_train=trainpath_list[0]['subjidx_train']
    subjidx_val=trainpath_list[0]['subjidx_val']
    
    pathfinder_transcoder_list=[]
    if pathfinder_list:
        pathfinder_trainpath_list=[tp for tp in trainpath_list if tp['input_name']==pathfinder_list[0] and tp['output_name']==pathfinder_list[-1]]
        for p in pathfinder_list[1:-1]:
            itmp=[tp['encoder_index'] for tp in trainpath_list if tp['input_name']==p]
            if len(itmp) == 0:
                print("Available names:",pathfinder_trainpath_list[0]['input_name_list'])
                raise Exception("Unknown conndata name for pathfinder evaluation: %s" % (p))
            pathfinder_transcoder_list+=[itmp[0]]
        pathfinder_trainpath_list[0]['transcoder_list']=pathfinder_transcoder_list
    
    if not torch.cuda.is_available():
        torch.set_num_threads(maxthreads)
        
    if torch.cuda.is_available():
        net = net.cuda()
    
    net.eval()
    if pathfinder_transcoder_list:
        pathfinder_pathnames_str=[pathfinder_trainpath_list[0]['input_name']]
        for i in pathfinder_trainpath_list[0]['transcoder_list']:
            pathfinder_pathnames_str+=[pathfinder_trainpath_list[0]['input_name_list'][i]]
        pathfinder_pathnames_str+=[pathfinder_trainpath_list[0]['output_name']]
        pathfinder_pathnames_str="->".join(pathfinder_pathnames_str)
        print("Pathfinder mode: %s" % (pathfinder_pathnames_str))
        trainpath_list=pathfinder_trainpath_list

    for itp,trainpath in enumerate(trainpath_list):
        encoder_index=trainpath['encoder_index']
        decoder_index=trainpath['decoder_index']
        
        transcoder_list=None
        if 'transcoder_list' in trainpath:
            transcoder_list=trainpath['transcoder_list']

        train_inputs=trainpath['train_inputs']
        train_outputs=trainpath['train_outputs']
        val_inputs=trainpath['val_inputs']
        val_outputs=trainpath['val_outputs']
        output_transformer=trainpath['output_transformer']

        #make cpu copy of train and val OUTPUTS for top1acc testing in loop
        #train_outputs_np=train_outputs_list_np[trainpath['output_name']]
        #val_outputs_np=val_outputs_list_np[trainpath['output_name']]
    
        #train_outputs_np=train_outputs.cpu().detach().numpy()
        #val_outputs_np=val_outputs.cpu().detach().numpy()

        #should already have done this
        #if torch.cuda.is_available():
        #    train_inputs, train_outputs = train_inputs.cuda(), train_outputs.cuda()
        #    val_inputs, val_outputs = val_inputs.cuda(), val_outputs.cuda()

        encoder_index_torch=torchint(encoder_index)
        decoder_index_torch=torchint(decoder_index)
        
        if transcoder_list is None:
            transcoder_list_torch=None
        else:
            transcoder_list_torch=torchint(transcoder_list)
        
        if train_inputs is not None:
            with torch.no_grad():
                conn_encoded, conn_predicted = net(train_inputs, encoder_index_torch, decoder_index_torch, transcoder_list_torch)
    
            trainpath_list[itp]['train_outputs_predicted']=torch.clone(conn_predicted)
            trainpath_list[itp]['train_inputs_encoded']=torch.clone(conn_encoded)
    
        with torch.no_grad():
            conn_encoded, conn_predicted = net(val_inputs, encoder_index_torch, decoder_index_torch, transcoder_list_torch)
    
        #val_cc=xycorr(val_outputs,conn_predicted)
        #val_top1acc=corrtop1acc(cc=val_cc)
        #print("path%d: %f" % (itp,val_top1acc))
        trainpath_list[itp]['val_outputs_predicted']=torch.clone(conn_predicted)
        trainpath_list[itp]['val_inputs_encoded']=torch.clone(conn_encoded)
    
    if burstmode:
        burstmode_trainpath_mask=np.ones(len(trainpath_list))>0
        if len(burstmode_search_list)>0:
            burstmode_trainpath_mask=[any([x in tp["input_name"] for x in burstmode_search_list]) for tp in trainpath_list]
            for itp, trainpath in enumerate(trainpath_list):
                print(burstmode_trainpath_mask[itp],trainpath["input_name"],trainpath["output_name"])
        
        if train_inputs is not None:
            traindata_encoded_mean=torch.mean(torch.stack([tp['train_inputs_encoded'] for itp,tp in enumerate(trainpath_list) if burstmode_trainpath_mask[itp]]),axis=0)
        valdata_encoded_mean=torch.mean(torch.stack([tp['val_inputs_encoded'] for itp,tp in enumerate(trainpath_list) if burstmode_trainpath_mask[itp]]),axis=0)
        
        if net.latent_normalize and not burstmode_do_not_normalize:
            if train_inputs is not None:
                traindata_encoded_mean=nn.functional.normalize(traindata_encoded_mean,p=2,dim=1)
            valdata_encoded_mean=nn.functional.normalize(valdata_encoded_mean,p=2,dim=1)
        
        neg1_index=torchint(-1)
        
        for itp,trainpath in enumerate(trainpath_list):
            encoder_index=trainpath['encoder_index']
            decoder_index=trainpath['decoder_index']
            
            train_outputs=trainpath['train_outputs']
            val_outputs=trainpath['val_outputs']
            output_transformer=trainpath['output_transformer']
            
            #make cpu copy of train and val OUTPUTS for top1acc testing in loop
            #train_outputs_np=train_outputs_list_np[trainpath['output_name']]
            #val_outputs_np=val_outputs_list_np[trainpath['output_name']]
    
            #train_outputs_np=train_outputs.cpu().detach().numpy()
            #val_outputs_np=val_outputs.cpu().detach().numpy()

            #should already have done this
            #if torch.cuda.is_available():
            #    train_inputs, train_outputs = train_inputs.cuda(), train_outputs.cuda()
            #    val_inputs, val_outputs = val_inputs.cuda(), val_outputs.cuda()

            decoder_index_torch=torchint(decoder_index)

            if train_inputs is not None:
                with torch.no_grad():
                    conn_encoded, conn_predicted = net(traindata_encoded_mean, neg1_index, decoder_index_torch)
    
                trainpath_list[itp]['train_outputs_predicted']=torch.clone(conn_predicted)
                trainpath_list[itp]['train_inputs_encoded']=torch.clone(conn_encoded)
    
            with torch.no_grad():
                conn_encoded, conn_predicted = net(valdata_encoded_mean, neg1_index, decoder_index_torch)
    
            trainpath_list[itp]['val_outputs_predicted']=torch.clone(conn_predicted)
            trainpath_list[itp]['val_inputs_encoded']=torch.clone(conn_encoded)
        
    return net, trainpath_list
