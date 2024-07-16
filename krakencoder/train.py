"""
Functions for training the network
"""

from .model import *
from .adaptermodel import *
from .loss import *
from .utils import *
from .data import generate_transformer
from .plotfigures import *

import torch.utils.data as data_utils
import torch.optim as optim

import os
import platform
import time
from datetime import datetime
import re
from collections import OrderedDict

import numpy as np
from scipy.io import loadmat, savemat

from scipy.spatial.distance import cdist as scipy_cdist

from sklearn.model_selection import GroupShuffleSplit

#####################################
#some useful functions

def random_train_test_split(numsubj=None, subjlist=None, train_frac=.5, seed=None):
    """
    Randomly split subjects into training and test sets, according to train_frac
    
    Parameters: 
    (must include either numsubj or subjlist)
    numsubj: int (optional), number of subjects. If not provided, will use len(subjlist)
    subjlist: list (optional), list of subject IDs or indices. If not provided, will use np.arange(numsubj)
    train_frac: float (default 0.5), fraction of subjects to use for training (0-1.0)
    seed: int (optional), random seed for reproducibility
    
    Returns:
    (trainsubj, testsubj): tuple of lists of subject IDs or indices
    """
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
    """
    Randomly split subject into training and test sets, according to train_frac, while keeping groups together
    (e.g., use groups=familyID to keep families/twins in same split to avoid data leakage)
    
    Parameters:
    groups: list of group IDs for each subject (must be same length as subjlist or numsubj)
    numsubj: int (optional), number of subjects. If not provided, will use len(subjlist)
    subjlist: list (optional), list of subject IDs or indices. If not provided, will use np.arange(numsubj)
    train_frac: float (default 0.5), fraction of subjects to use for training (0-1.0)
    seed: int (optional), random seed for reproducibility
    
    Returns:
    (trainsubj, testsubj, group_train, group_test): tuple of lists of subject IDs or indices and group IDs
    """
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

def loss_string_to_dict(loss_string, override_weight_dict={}, default_weight_dict={}, lossgroup_default_weight={}):
    """
    Parse loss string into a dictionary with keys for each loss type
    
    loss_string: string with loss types separated by '+', with each loss type optionally followed by '.w' and a weight
        (e.g. 'correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000')
    
    Defaults and overrides for specific weights:
    override_weight_dict: dictionary with loss type as key and weight as value, to override default weights
    default_weight_dict: dictionary with loss type as key and default weight as value (eg. {'mse':100,'correye':10})
    loss_group_default_weight: dictionary with loss group as key and default weight as value (eg. {'output':1,'encoded':10})
    """
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
    
    for lt,w in default_weight_dict.items():
        if not lt in loss_items_dict:
            loss_items_dict[lt]={"name":lt,"weight":w}
    
    for lt,w in override_weight_dict.items():
        if lt in loss_items_dict:
            loss_items_dict[lt]["weight"]=w
        else:
            loss_items_dict[lt]={"name":lt,"weight":w}

    new_loss_items_dict=OrderedDict()
    for lt,lt_item in loss_items_dict.items():
        w=lt_item["weight"]

        if lt in ['mse','msesum','corrtrace','correye','corrmatch','dist','neidist','var']:
            lt_group='output'
        elif lt in ['enceye','encdist','encneidist','encdot','encneidot']:
            lt_group='encoded'
        elif lt in ['latentnormloss','latentsimloss','latentmaxradloss']:
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
    """
    Convert loss_info_dict to a string (e.g. for including in filename)
    """
    loss_str="+".join([v['string'] for k,v in loss_info_dict.items() if v['weight'] is not None and v['weight']!=0])
    return loss_str

def generate_training_paths(conndata_alltypes, conn_names, subjects, subjidx_train, subjidx_val, trainpath_pairs=[],trainpath_group_pairs=[], 
                            data_string=None, batch_size=40, skip_selfs=False, crosstrain_repeats=1, 
                            reduce_dimension=None, leave_data_alone=False, use_pretrained_encoder=False, keep_origscale_data=False, quiet=False,
                            use_lognorm_for_sc=False, use_truncated_svd=False, use_truncated_svd_for_sc=False, input_transformation_info=None,
                            precomputed_transformer_info_list={}, create_data_loader=True):
    """
    Generate data structures for all training paths, including data transformers, data loaders
    
    === Input data parameters ===
    conndata_alltypes: dictionary with keys for each connectivity type, as returned by load_hcp_data()
        conndata[conntype]=a dictionary with keys:
            'data' (Nsubj x Npairs) torch tensor or numpy ndarray
            'group' (string) group name for this connectivity type (e.g., 'SC','FC')
                used when training on specific groups (eg only SC->FC), or when using group-specific transformations like TSVD for 'SC'
                or for inter-group loss functions
            'numpairs' (int) Npairs for this flavor
    conn_names: list of connectivity type names
    subjects: list of subject IDs
    subjidx_train: list of indices for training subjects
    subjidx_val: list of indices for validation subjects
    
    === Training path generation ===
    trainpath_pairs: list of pairs of connectivity types to train on (e.g., [['SCsdstream_fs86_volnorm','SCifod2act_shen268_volnorm']]). default=all pairs
    trainpath_group_pairs: list of pairs of connectivity type groups to train on (e.g., [['SC','SC']]). default=all group pairs
    skip_selfs: bool (default False), skip auto-encoders (e.g., SCsdstream_fs86_volnorm->SCsdstream_fs86_volnorm)
    crosstrain_repeats: int (default 1), number of times to train transcoders for each epoch (to upweight versus auto-encoders)
    
    === Data transformation parameters ===
    input_transformation_info: string (default None), specify a transformation type 
        (e.g., 'pc256', 'tsvd256', 'pc+tsvd256', 'none', 'varnorm', 'zfeat', 'cfeat', 'cfeat+norm', 'cfeat+varnorm', 
            'zscore+rownorm', 'lognorm+rownorm', 'torchfile')
        - overrides other transformation arguments
    precomputed_transformer_info_list: dictionary (default {}), precomputed transformer information for each connectivity type
        - keys are connectivity type names
        - values are dictionaries with parameters or components that define the transformation (eg PCA weights)
        - overrides other transformation arguments
    
    reduce_dimension: int (default None), number of dimensions to reduce to using PCA or truncated SVD
    leave_data_alone: bool (default False), do not normalize or reduce dimensionality
    use_pretrained_encoder: bool (default False), use a pretrained encoder (e.g., from a torch file)
    keep_origscale_data: bool (default False), keep original scale data in data_origscale_list
        - useful for tracking prediction versus original data, but takes up more memory
        - if False, only the reduced input data is kept and OrigScale metrics are computed by inverse transforming the reduced data
    use_lognorm_for_sc: bool (default False), use lognorm+rownorm for SC data (to try to transform raw SC data to a more FC-like distribution)
    use_truncated_svd: bool (default False), use truncated SVD for dimensionality reduction instead of PCA
    use_truncated_svd_for_sc: bool (default False), use truncated SVD for SC data only (but PCA for FC)
    
    === Misc parameters ===
    batch_size: int (default 40), batch size for training
    create_data_loader: bool (default True), create DataLoader objects for training and validation data
    data_string: string to identify this dataset (e.g., 'SCFC_fs86+shen268+coco439_993subj')
    quiet: bool (default False), suppress output messages about data transformations, variance explained, etc
    
    === Returns ===
    trainpath_list: list of dictionaries with training path information for each path (e.g., FCcorr_fs86_hpf -> SCifod2act_shen268_volnorm)
        trainpath_list[i]['input_name'] ['output_name']: name of input/output connectivity type for this path
        trainpath_list[i]['input_npairs'] ['output_npairs']: number of pairs in input/output data for this path
        trainpath_list[i]['input_transformer'] ['output_transformer']: transformer object for input/output type
        trainpath_list[i]['trainloader'] ['valloader']: DataLoader object for training/validation data
        trainpath_list[i]['train_inputs'] ['train_outputs']: torch tensor with training input/output data (Ntrainsubj x Nfeat)
        trainpath_list[i]['train_marginmin_outputs']: minimum intersubject distance for training output data
        trainpath_list[i]['train_marginmax_outputs']: maximum intersubject distance for training output data
        trainpath_list[i]['train_marginmean_outputs']: mean intersubject distance for training output data
        trainpath_list[i]['trainloops']: number of times to train this path for each epoch
    data_optimscale_list: dictionary with transformed data for training and validation splits (e.g., reduced dimensionality, normalized, etc)
        data_optimscale_list['traindata'][conn_name]: torch tensor with transformed training data for each connectivity type (Ntrainsubj x Nfeat)
        data_optimscale_list['valdata'][conn_name]: torch tensor with transformed validation data for each connectivity type (Nvalsubj x Nfeat)
    data_origscale_list: dictionary with original, untransformed data for training and validation data (if keep_origscale_data=True, otherwise None)
        data_origscale_list['traindata_origscale'][conn_name]: torch tensor with original training data for each connectivity type (Ntrainsubj x Npairs)
        data_origscale_list['valdata_origscale'][conn_name]: torch tensor with original validation data for each connectivity type (Nvalsubj x Npairs)
        data_origscale_list['traindata_origscale_mean'][conn_name]: torch tensor with pop. mean of original training data for each connectivity type (1 x Npairs)
    data_transformer_info_list: dictionary with information about the transformers used for each connectivity type
    """
    
    if not data_string:
        data_string=common_prefix(conn_names)+common_suffix(conn_names)

    if skip_selfs:
        data_string+='_noself'
    
    default_transformation_type="zscore+rownorm"
    
    if len(precomputed_transformer_info_list) > 0:
        #when calling generate_training_paths with precomputed transformers, pull "type" from first one
        default_transformation_type=[v['type'] for k,v in precomputed_transformer_info_list.items()][0]
    elif input_transformation_info is not None and input_transformation_info is not False:
        if re.search("^pc[0-9]+$",input_transformation_info):
            reduce_dimension=int(input_transformation_info.replace("pc",""))
            use_truncated_svd=False
        elif re.search("^tsvd[0-9]+$",input_transformation_info):
            reduce_dimension=int(input_transformation_info.replace("tsvd",""))
            use_truncated_svd=True
        elif re.search(r"^pc\+tsvd[0-9]+$",input_transformation_info):
            reduce_dimension=int(input_transformation_info.split("+")[-1].replace("pc","").replace("tsvd",""))
            use_truncated_svd=False
            use_truncated_svd_for_sc=True
        elif input_transformation_info.upper() == "NONE":
            reduce_dimension=0
            leave_data_alone=True
        elif input_transformation_info.upper() == "VARNORM":
            reduce_dimension=0
            use_lognorm_for_sc=False
            default_transformation_type="varnorm"
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
        elif input_transformation_info.upper() == "CFEATVARNORM" or input_transformation_info.upper() == "CFEAT+VARNORM":
            reduce_dimension=0
            use_lognorm_for_sc=False
            default_transformation_type="cfeat+varnorm"
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
    for iconn,conn_name in enumerate(unames):

        if type(conn_name) != str:
            #if index was provided
            i1=conn_name
            conn_name=conn_names[i1]
        
        if conn_name in data_transformer_list:
            continue
        
        if not quiet:
            print("Transforming input data %d/%d: %s" % (iconn+1,len(unames),conn_name), end="")
        
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
            print(" using %s%s" % (data_transformer_dict['type'],precomputed_transformer_string), end="")
        
        traindata_list[conn_name]=data_transformer_list[conn_name].transform(traindata)
        valdata_list[conn_name]=data_transformer_list[conn_name].transform(valdata)
        data_transformer_info_list[conn_name]=data_transformer_dict
        
        if not quiet:
            var_explained_ratio=explained_variance_ratio(torchfloat(traindata), data_transformer_list[conn_name].inverse_transform(traindata_list[conn_name]))
            print(" (training variance maintained: %.2f%%)" % (var_explained_ratio*100),end="")

        if not quiet:
            print("")

        #compute min,mean,max of intersubject euclidean distances for this dataset
        traindist=scipy_cdist(traindata_list[conn_name].cpu(),traindata_list[conn_name].cpu())
        traindist[np.eye(traindist.shape[0])>0]=np.nan
        #min/max are the MEAN of the nearest and farthest distances from each subject
        traindata_marginmin_list[conn_name]=np.nanmin(traindist,axis=1).mean()
        traindata_marginmax_list[conn_name]=np.nanmax(traindist,axis=1).mean()
        traindata_marginmean_list[conn_name]=np.nanmean(traindist)
        
    data_optimscale_list={}
    data_optimscale_list['traindata']={k:torchfloat(v) for k,v in traindata_list.items()}
    data_optimscale_list['valdata']={k:torchfloat(v) for k,v in valdata_list.items()}

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

        #make new copies of data for each training path
        #to make sure gradients do not interact with each other across paths?
        #train_inputs = torchfloat(train_inputs).detach()
        #train_outputs = torchfloat(train_outputs).detach()
        #val_inputs = torchfloat(val_inputs).detach()
        #val_outputs = torchfloat(val_outputs).detach()

        #alternatively, use shared tensors, which saves memory
        #somehow, the training results seem numerically identical
        #with shared or copied data, so use the shared
        train_inputs=data_optimscale_list['traindata'][c1]
        train_outputs=data_optimscale_list['traindata'][c2]
        val_inputs=data_optimscale_list['valdata'][c1]
        val_outputs=data_optimscale_list['valdata'][c2]

        trainloader=None
        if create_data_loader:
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
        
        valloader=None
        if create_data_loader:
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
        trainpath_tmp['input_group_list']=[conndata_alltypes[x]['group'] for x in conn_names]
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
        data_origscale_list['traindata_origscale_mean']={x: traindata_origscale_list[x].mean(axis=0,keepdims=True) for x in conn_names}
        data_origscale_list['traindata_origscale']=traindata_origscale_list
        data_origscale_list['valdata_origscale']=valdata_origscale_list
    return trainpath_list, data_optimscale_list, data_origscale_list, data_transformer_info_list

##############################
##############################
# compute loss for a given path
def compute_path_loss(conn_predicted=None, conn_targets=None, conn_encoded=None, conn_encoded_targets=None, 
                      criterion=[], encoded_criterion=[], output_margin=None, encoder_margin=None, 
                      latentnorm_loss_weight=0, latent_maxrad_weight=0, latent_maxrad=None, return_list=False):
    """
    Compute batch loss for a given path (e.g., FCcorr_fs86 -> SCifod2act_shen268)
    Depending on inputs, it may compute reconstruction loss and/or latent space loss
    Loops through each reconstruction loss function in OrderedDict 'criterion' and latent space loss function in OrderedDict 'encoded_criterion'
    
    Parameters:
    Reconstruction loss inputs:
        conn_predicted: torch tensor, predicted connectivity matrix (Nsubj x Nfeat)
        conn_targets: torch tensor, target/measured connectivity matrix (Nsubj x Nfeat)
        criterion: list of dictionaries, each with keys 'function' (loss function) and optional 'weight' (loss weight)
            e.g., [ dict(name='correye', function=correye,weight=1), 
                    dict(name='neidist', function=distance_neighbor_loss, pass_margins=True, weight=1) ]
        output_margin: float (optional), margin for reconstruction loss functions that use it (e.g., contrastive loss)
        
    Latent space loss inputs:
        conn_encoded: torch tensor, encoded representation of the input data (Nsubj x Nlatent)
        conn_encoded_targets: torch tensor (optional), target encoded representation of the input data (Nsubj x Nlatent)
            (Could be used to train new encoders to an existing latent representation)
        encoded_criterion: list of dictionaries per loss type, each with keys 'function' (loss function) and optional 'weight' (loss weight)
             e.g., [ dict(name='enceye', function=correye,weight=1), dict(name='encdist',function=distance_loss,weight=1) ]
        encoder_margin: float (optional), margin for latent space loss functions that use it (e.g., contrastive loss)
        latentnorm_loss_weight: float (default 0), weight to apply to latent space norm (to keep it from getting large)
        latent_maxrad_weight: float (default 0), weight to apply to latent space max radius (to keep it from getting > latent_maxrad)
        latent_maxrad: float (default 1), maximum radius for latent space (After which latent_maxrad_weight is applied)
    
    return_list: bool (default False), return a list of loss components (name, loss value, weight, margin) as well as total loss
    
    Returns:
    loss: torch tensor, total loss for the given path
    
    or (loss, loss_list) if return_list=True
    """
    loss=torchfloat([[0]])
    loss_list=[]
    for crit in criterion:
        w=1
        if "weight" in crit:
            w=crit['weight']
        if "pass_margin" in crit:
            #loss=loss+w*crit['function'](conn_predicted, conn_targets, margin=output_margin)
            thisloss=crit['function'](conn_targets, conn_predicted, margin=output_margin) # compute loss(target,pred) 4/5/2024
            loss=loss+w*thisloss
        else:
            #loss=loss+w*crit['function'](conn_predicted, conn_targets)
            thisloss=crit['function'](conn_targets, conn_predicted) # compute loss(target,pred) 4/5/2024
            loss=loss+w*thisloss
        loss_list+=[{'name':crit['name'],'loss':thisloss, 'weight':w, 'margin':output_margin}]

    for enc_crit in encoded_criterion:
        w=1
        if "weight" in enc_crit:
            w=enc_crit['weight']
        if "pass_margin" in enc_crit:
            if conn_encoded_targets is None:
                thisloss=enc_crit['function'](conn_encoded, conn_encoded, margin=encoder_margin)
                loss_enc=w*thisloss
            else:
                #loss_enc=w*enc_crit['function'](conn_encoded, conn_encoded_targets, margin=encoder_margin)
                thisloss=enc_crit['function'](conn_encoded_targets, conn_encoded, margin=encoder_margin) #compute loss(target,pred) 4/5/2024
                loss_enc=w*thisloss
        else:
            if conn_encoded_targets is None:
                thisloss=enc_crit['function'](conn_encoded, conn_encoded)
                loss_enc=w*thisloss
            else:
                #loss_enc=w*enc_crit['function'](conn_encoded, conn_encoded_targets)
                thisloss=enc_crit['function'](conn_encoded_targets, conn_encoded) #compute loss(target,pred) 4/5/2024
                loss_enc=w*thisloss
        loss_list+=[{'name':crit['name'],'loss':thisloss, 'weight':w, 'margin':output_margin}]
        loss += loss_enc

    if latentnorm_loss_weight > 0:
        loss_latentnorm = torch.linalg.norm(conn_encoded)
        loss_list+=[{'name':'latentnorm','loss':loss_latentnorm, 'weight':latentnorm_loss_weight, 'margin':None}]
        loss += latentnorm_loss_weight*loss_latentnorm

    if latent_maxrad_weight > 0:
        loss_latentrad = torch.mean(torch.nn.ReLU()(torch.sum(conn_encoded**2,axis=1)-latent_maxrad))
        loss_list+=[{'name':'latent_maxrad','loss':loss_latentrad, 'weight':latent_maxrad_weight, 'margin':None}]
        loss += latent_maxrad_weight*loss_latentrad

    if return_list:
        return loss, loss_list
    else:
        return loss

###################################
###################################
#define network training FUNCTION
def train_network(trainpath_list, training_params, net=None, data_optimscale_list=None, data_origscale_list=None, trainfig=None, 
                  trainthreads=16, display_epochs=20, display_seconds=None, 
                  save_epochs=100, checkpoint_epochs=None, update_single_checkpoint=True, save_optimizer_params=True,
                  explicit_checkpoint_epoch_list=[], precomputed_transformer_info_list={}, save_input_transforms=True,
                  output_file_prefix="kraken",logger=None, extra_trainrecord_dict={}):
    """
    Train a network on a set of training paths. 
    This function is designed to be called from a script or notebook, and will handle all the training details.
    
    Parameters:
    trainpath_list: list of dictionaries with training path information for input/output pair (from generate_training_paths)
    training_params: dictionary with training parameters. See run_training.py for usage
    
        === Architecture parameters ===
        training_params['latentsize']: int, size of latent space
        training_params['skip_relu']: bool, skip ReLU activation in the network
        training_params['hiddenlayers']: int, number of hidden layers in the network
        training_params['latent_normalize']: bool (default=False), normalize latent space vectors to unit length
        training_params['relu_tanh_alternate']: bool (default=False), alternate between ReLU and tanh activations (Sarwar 2021 style deep network)
        training_params['leakyrelu_negative_slope']: float (default=0), use leaky ReLU with negative slope (if > 0, otherwise use ReLU)
        training_params['latent_activation']: string (default='none'), activation function for latent space (e.g., 'tanh', 'relu', 'none')
        
        ==== Loss parameters ====
        training_params['losstype']: string (default='mse'), type of loss function to use (e.g., 'mse', 'correye+enceye.w10+neidist+encdist.w10+mse.w1000')
            - overrides other loss parameters
        training_params['latent_inner_loss_weight']: float (default=1.0), default weight for latent space loss terms
        training_params['latentsim_loss_weight']: float (default=0), weight for latent space intra-subject, inter-flavor similarity loss
        training_params['latentnorm_loss_weight']: float (default=0), weight for latent space norm loss
        training_params['latent_maxrad']: float (default=1), maximum radius for latent space (if latent_maxrad_weight > 0)
        training_params['latent_maxrad_weight']: float (default=0), weight for latent space max radius loss
        training_params['mse_weight']: float (default=0), weight for mean squared error loss
        
        === Training parameters ===
        training_params['nbepochs']: int, number of epochs to train
        training_params['dropout']: float, dropout rate (0-1.0)
        training_params['dropout_schedule_list']: list of float (default=None), list of dropout rates to interpolate through over epochs
            (eg [0.5,0] starts at 0.5 and ends at 0), if None, just use 'dropout' parameter for all epochs
        training_params['dropout_final_layer']: float (default=None), separate dropout rate for the final DECODER layer only (eg: for the final reconstruction). None=use 'dropout'
        training_params['dropout_final_layer_list']: list of float (default=None), separate dropout rate for the final DECODER layer for each input type
        training_params['origscalecorr_epochs']: int (default=0=never), number of epochs between computing performance metrics for inverse-transformed 'origscale' data
        training_params['trainpath_shuffle']: bool (default=False), shuffle training paths each epoch
        training_params['roundtrip']: bool (default=False), train encoder[i]->latent->decoder[j]->encoder[j]->latent->decoder[i], loss(meas_i, pred_i)
        training_params['meantarget_latentsim']: bool (default=False), use inter-flavor mean as target for intra-subject similarity loss
        training_params['target_encoding']: bool (default=False), a target encoding is provided for each subject, to LOOSELY guide the encoder and decoder:
            - We compute input->latent->pred_output, then loss(latent, target latent) and loss(pred_output, target output)
        training_params['fixed_target_encoding']: bool (default=False), a target encoding is provided for each subject, to STRICTLY guide the encoder and decoder:
            - We computer input->latent, but TARGET_latent->output, then loss(latent, TARGET_latent) + loss(pred_output, target_output)
        
        === Optimizer parameters ===
        training_params['optimizer_name']: string (default 'adam'), name of optimizer to use
        training_params['learningrate']: float, learning rate
        training_params['zerograd_none']: bool (default=False), zero gradients for None values in the network
        training_params['init_type']: string (default=None->'kaiming'), type of initialization to use for network weights. (Options: 'xavier', 'kaiming')
        training_params['adam_decay']: float (default=None), weight decay for Adam optimizer
        
        === Experimental: Intergroup parameters ===
        training_params['intergroup']: bool (default=False) if True, create and apply additional intergroup transformations between groups of input types (eg SC->FC, FC->SC)
        training_params['intergroup_extra_layer_count']: int (default=0), number of extra intergroup transformation layers (0=single Nlatent->Nlatent layer)
        training_params['intergroup_relu']: bool (default=True), if True, apply ReLU activation after each intergroup transformation layer
        training_params['intergroup_dropout']: float (default=None), separate dropout rate for all intergroup transformation layers. None=use 'dropout'
        
    net: (optional) Krakencoder object to train (if None, will create a new Krakencoder object)
    data_optimscale_list: dictionary with transformed data for training and validation splits (from generate_training_paths)
    data_origscale_list: dictionary with original, untransformed data for training and validation data (from generate_training_paths)
    trainfig: (optional) matplotlib figure object to display/update training progress (if None, will create a new figure)
    trainthreads: int (default=16), number of CPU threads to use for training
    display_epochs: int (default=20), number of epochs between display updates
    display_seconds: int (default=None), number of seconds between display updates (overrides display_epochs)
    save_epochs: int (default=100), number of epochs between saving training record
    checkpoint_epochs: int (default=None), number of epochs between saving checkpoints
    update_single_checkpoint: bool (default=True), only keep the most recent checkpoint file (if False, keep all)
    save_optimizer_params: bool (default=True), save optimizer parameters in checkpoint files (allows resuming training, but much larger checkpoint files)
    explicit_checkpoint_epoch_list: list of epochs to save checkpoints (overrides checkpoint_epochs)
    precomputed_transformer_info_list: dictionary with precomputed transformer information for each connectivity type
    save_input_transforms: bool (default=True), save input transformation information to an '<prefix>_ioxfm_*.npy' file
    output_file_prefix: string (default="kraken"), prefix for output files (e.g., training record, checkpoint files)
    logger: (optional) Logger object to log training progress (see log.py)
    extra_trainrecord_dict: dictionary with additional fields to save in the training record (e.g., include the command line inputs used to call this function)
    """
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
    optimizer_name=training_params['optimizer_name'] if 'optimizer_name' in training_params else 'adam'
    do_zerograd_none=training_params['zerograd_none'] if 'zerograd_none' in training_params else False
    losstype=training_params['losstype'] if 'losstype' in training_params else 'mse'
    latent_inner_loss_weight=training_params['latent_inner_loss_weight'] if 'latent_inner_loss_weight' in training_params else 1
    latentsim_loss_weight=training_params['latentsim_loss_weight'] if 'latentsim_loss_weight' in training_params else 0
    latentnorm_loss_weight=training_params['latentnorm_loss_weight'] if 'latentnorm_loss_weight' in training_params else 0
    latent_maxrad=training_params['latent_maxrad'] if 'latent_maxrad' in training_params else 1
    latent_maxrad_weight=training_params['latent_maxrad_weight'] if 'latent_maxrad_weight' in training_params else 0
    latent_normalize=training_params['latent_normalize'] if 'latent_normalize' in training_params else False
    relu_tanh_alternate=training_params['relu_tanh_alternate'] if 'relu_tanh_alternate' in training_params else False
    leakyrelu_negative_slope=training_params['leakyrelu_negative_slope'] if 'leakyrelu_negative_slope' in training_params else 0
    mse_weight=training_params['mse_weight'] if 'mse_weight' in training_params else 0
    latent_activation=training_params['latent_activation'] if 'latent_activation' in training_params else 'none'
    adam_decay=training_params['adam_decay'] if 'adam_decay' in training_params else None
    do_trainpath_shuffle=training_params['trainpath_shuffle'] if 'trainpath_shuffle' in training_params else False
    do_roundtrip=training_params['roundtrip'] if 'roundtrip' in training_params else False
    
    #how often to compute selfcc and othercc for untransformed space? (eg not PCA)
    origscalecorr_epochs=training_params['origscalecorr_epochs'] if 'origscalecorr_epochs' in training_params else 0
    
    
    do_meantarget_latentsim = training_params['meantarget_latentsim'] if 'meantarget_latentsim' in training_params else False
    do_target_encoding = training_params['target_encoding'] if 'target_encoding' in training_params else False
    do_fixed_target_encoding = training_params['fixed_target_encoding'] if 'fixed_target_encoding' in training_params else False
    do_target_encoding = do_target_encoding or do_fixed_target_encoding
    
    dropout_schedule_list=training_params['dropout_schedule_list'] if 'dropout_schedule_list' in training_params else None
    
    dropout_final_layer=training_params['dropout_final_layer'] if 'dropout_final_layer' in training_params else None
    dropout_final_layer_list=training_params['dropout_final_layer_list'] if 'dropout_final_layer_list' in training_params else None
    ############# intergroup
    intergroup=training_params['intergroup'] if 'intergroup' in training_params else False
    intergroup_hiddenlayers=training_params['intergroup_extra_layer_count'] if 'intergroup_extra_layer_count' in training_params else 0
    #intergroup_inputgroup_list=['fc' if 'FC' in conntype else 'sc' for conntype in trainpath_list[0]['input_name_list']]
    intergroup_inputgroup_list=trainpath_list[0]['input_group_list'] #these are all upper case for now
    intergroup_relu=training_params['intergroup_relu'] if 'intergroup_relu' in training_params else True
    intergroup_dropout=training_params['intergroup_dropout'] if 'intergroup_dropout' in training_params else None
    intergroup_dropout_schedule_list=[intergroup_dropout,intergroup_dropout]
    ############# end intergroup
    
    if dropout_schedule_list is None:
        dropout_schedule_list=[dropout,dropout]
    
    if do_fixed_target_encoding:
        # for FIXED target encoding, do not use latentsim
        latentsim_loss_weight=0
        
        
    #if latent_normalize is true, use dot product for distance
    #do_latent_dotproduct_distance=latent_normalize
    do_latent_dotproduct_distance=False
    

    do_use_existing_net=net is not None
    if net is None:
        if intergroup:
            ############# intergroup
            #intergroup_layers=[latentsize] # FC --(encode)--> FClatent -> [128+relu + 128+relu] --(decoder)--> SC
            #intergroup_layers=[] # FC --(encode)--> FClatent -> [128+relu] --(decoder)--> SC
            intergroup_layers=[latentsize]*intergroup_hiddenlayers
            
            net = Krakencoder(coencoder_size_list, latentsize=latentsize, 
                                hiddenlayers=hiddenlayers, skip_relu=skip_relu, dropout=dropout,
                                dropout_schedule_list=dropout_schedule_list,
                                relu_tanh_alternate=relu_tanh_alternate, leakyrelu_negative_slope=leakyrelu_negative_slope,
                                latent_activation=latent_activation, latent_normalize=latent_normalize,
                                intergroup=True, intergroup_layers=intergroup_layers, intergroup_inputgroup_list=intergroup_inputgroup_list, 
                                intergroup_relu=intergroup_relu,intergroup_dropout=intergroup_dropout,
                                dropout_final_layer=dropout_final_layer, dropout_final_layer_list=dropout_final_layer_list)
            ############# end intregroup
        else:
            net = Krakencoder(coencoder_size_list, latentsize=latentsize, 
                                hiddenlayers=hiddenlayers, skip_relu=skip_relu, dropout=dropout,
                                dropout_schedule_list=dropout_schedule_list,
                                relu_tanh_alternate=relu_tanh_alternate, leakyrelu_negative_slope=leakyrelu_negative_slope,
                                latent_activation=latent_activation, latent_normalize=latent_normalize,
                                dropout_final_layer=dropout_final_layer, dropout_final_layer_list=dropout_final_layer_list)
    
    network_string=net.prettystring()
    network_parameter_count=sum([p.numel() for p in net.parameters()]) #count total weights in model
    network_description_string=str(net) #long multi-line pytorch-generated description
    
    lrstr="lr%g" % (lr)
    optimstr=""
    optimname_str=""
    if optimizer_name != "adam":
        optimname_str="_%s" % (optimizer_name)
    if adam_decay is not None:
        optimname_str+=".w%g" % (adam_decay)
    
    zgstr=""
    
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

    do_eval_only = nbepochs<=1
    if do_eval_only:
        nbepochs=1
        training_params['nbepochs']=nbepochs
        print("Only evaluating network")
    
    ###############################

    do_pass_margins_to_loss=[]
    do_pass_margins_to_encoded_loss=[]

    criterion=[]
    encoded_criterion=[]
    latentsim_criterion=None

    loss_default_dict=OrderedDict()
    if mse_weight > 0:
        loss_default_dict['mse']=mse_weight
    loss_default_dict['latentsimloss']=latentsim_loss_weight
    loss_default_dict['latentnormloss']=latentnorm_loss_weight
    loss_default_dict['latentmaxradloss']=latent_maxrad_weight

    lossgroup_default_weight={'output':1,'encoded':latent_inner_loss_weight,'encoded_meta':None}

    losstype_dict=loss_string_to_dict(losstype,default_weight_dict=loss_default_dict, lossgroup_default_weight=lossgroup_default_weight)
    
    if 'latentsimloss' in losstype_dict and 'suffix' in losstype_dict['latentsimloss'] and losstype_dict['latentsimloss']['suffix']=='B':
        do_batchwise_latentsim=True

    for lt, lt_item in losstype_dict.items():
        lt_w=lt_item['weight']
        lt_group=lt_item['lossgroup']
        
        if lt == 'mse':
            criterion += [{"name":lt, "function":nn.MSELoss(), "weight": torchfloat(lt_w)}]
        elif lt == 'msesum':
            criterion += [{"name":lt, "function":nn.MSELoss(reduction='sum'), "weight": torchfloat(lt_w)}]
        elif lt == 'var':
            criterion += [{"name":lt, "function":var_match_loss, "weight": torchfloat(lt_w)}]            
        elif lt == 'corrtrace':
            criterion += [{"name":lt, "function":corrtrace, "weight": torchfloat(lt_w)}]
        elif lt == 'correye':
            criterion += [{"name":lt, "function":correye, "weight": torchfloat(lt_w)}]
        elif lt == 'corrmatch':
            criterion += [{"name":lt, "function":corrmatch, "weight": torchfloat(lt_w)}]
        elif lt == 'dist':
            criterion += [{"name":lt, "function":distance_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]
        elif lt == 'neidist':
            criterion += [{"name":lt, "function":distance_neighbor_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]

        elif lt == 'enceye':
            encoded_criterion += [{"name":lt, "function":correye, "weight": torchfloat(lt_w)}]
        elif lt == 'encdist':
            encoded_criterion += [{"name":lt, "function":distance_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]
        elif lt == 'encneidist':
            encoded_criterion += [{"name":lt, "function":distance_neighbor_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]
        elif lt == 'encdot':
            encoded_criterion += [{"name":lt, "function":dotproduct_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]
        elif lt == 'encneidot':
            encoded_criterion += [{"name":lt, "function":dotproduct_neighbor_loss, "pass_margins":True, "weight": torchfloat(lt_w)}]

        elif lt_group=='encoded_meta':
            pass

        else:
            raise Exception("Unknown losstype: %s" % (lt_item))
    
    if do_target_encoding and mse_weight > 0:
        encoded_criterion += [{"name":lt, "function":nn.MSELoss(), "weight": torchfloat(mse_weight)}]

    latentnorm_loss_weight_torch=torchfloat(losstype_dict['latentnormloss']['weight'])
    latent_maxrad_weight_torch=torchfloat(losstype_dict['latentmaxradloss']['weight'])
    latent_maxrad_torch=torchfloat(latent_maxrad)
    latentsim_loss_weight_torch=torchfloat(losstype_dict['latentsimloss']['weight'])

    
    loss_str="_"+loss_dict_to_string(losstype_dict)
    if do_fixed_target_encoding:
        loss_str+="+fixlatent"
    elif do_target_encoding:
        loss_str+="+targlatent"

    train_string="%depoch_%s%s%s%s%s" % (nbepochs,lrstr,loss_str,optimstr,optimname_str,zgstr)
    if do_roundtrip:
        if do_use_existing_net:
            train_string+="_addroundtrip"
        else:
            train_string+="_roundtrip"
    
    optimizer = makeoptim(optimizer_name)
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
    
    explainedvar_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    explainedvar_val = np.nan*np.zeros((len(trainpath_list),nbepochs))

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

    #identifiability for each TRUE output -> closest PREDICTED output
    corrloss_OrigScale_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossN_OrigScale_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossRank_OrigScale_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    
    #identifiability for each PREDICTED output -> closest TRUE output
    corrloss_OrigScale_pred2true_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossN_OrigScale_pred2true_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossRank_OrigScale_pred2true_train = np.nan*np.zeros((len(trainpath_list),nbepochs))    
    corrloss_OrigScale_pred2true_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossN_OrigScale_pred2true_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    corrlossRank_OrigScale_pred2true_val = np.nan*np.zeros((len(trainpath_list),nbepochs))

    explainedvar_OrigScale_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    explainedvar_OrigScale_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    
    avgcorr_resid_OrigScale_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    avgcorr_resid_OrigScale_other_train = np.nan*np.zeros((len(trainpath_list),nbepochs))
    avgcorr_resid_OrigScale_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    avgcorr_resid_OrigScale_other_val = np.nan*np.zeros((len(trainpath_list),nbepochs))
    
    topN=2
    if trainpath_list[0]['trainloader'] is None:
        batchsize=training_params['batchsize']
    else:
        batchsize=trainpath_list[0]['trainloader'].batch_size
    
    ################
    #make subject index dataloader for latentsimloss
    numsubjects_train=len(trainpath_list[0]['subjidx_train'])
    
    tmp_latent_batchsize=numsubjects_train
    
    latentsimloss_subjidx_dataloader=data_utils.DataLoader(np.arange(numsubjects_train), batch_size=tmp_latent_batchsize, shuffle=True, drop_last=True)

    if 'starting_point_file' in training_params and training_params['starting_point_file'] and nbepochs<=1:
        starting_point_base=os.path.join(os.path.split(output_file_prefix)[0],os.path.split(training_params['starting_point_file'])[1])
        starting_point_base=re.sub(r"\.pt$","",starting_point_base)
        recordfile=starting_point_base.replace("_chkpt_","_trainrecord_")+".mat"
        imgfile=starting_point_base.replace("_chkpt_","_loss_")+".png"
        imgfile_heatmap=starting_point_base.replace("_chkpt_","_heatmap_")+".png"
        #keep checkpoint_filebase so we don't overwrite anything. shouldn't be saved in the nbepochs=0 case anyway
        checkpoint_filebase="%s_chkpt_%s_%s_%s_%s" % (output_file_prefix,data_string,network_string,train_string,timestamp_suffix)
        
    else:
        recordfile="%s_trainrecord_%s_%s_%s_%s.mat" % (output_file_prefix,data_string,network_string,train_string,timestamp_suffix)
        imgfile="%s_loss_%s_%s_%s_%s.png" % (output_file_prefix,data_string,network_string,train_string,timestamp_suffix)
        imgfile_heatmap="%s_heatmap_%s_%s_%s_%s.png" % (output_file_prefix,data_string,network_string,train_string,timestamp_suffix)
        checkpoint_filebase="%s_chkpt_%s_%s_%s_%s" % (output_file_prefix,data_string,network_string,train_string,timestamp_suffix)
    
    if logger is not None:
        if logger.is_auto():
            logfile=recordfile.replace("_trainrecord_","_log_")[:-4]+".txt"
            logger=logger.transfer(logfile)
        print("Log file:",logger.logfile)
    
    #if saving transformer info (eg: precomputed PCA weights), need to remove the transformer OBJECT with embedded functions etc
    # from the list. Just save params
    input_transformer_file=""
    if precomputed_transformer_info_list and save_input_transforms:
        input_transformer_file="%s_ioxfm_%s_%s_%s_%s.npy" % (output_file_prefix,data_string,network_string,train_string,timestamp_suffix)
        transformer_params_to_save={}
        for k_iox in precomputed_transformer_info_list.keys():
            transformer_params_to_save[k_iox]=precomputed_transformer_info_list[k_iox]["params"]
            for kk,kv in precomputed_transformer_info_list[k_iox]["params"].items():
                if torch.is_tensor(kv):
                    transformer_params_to_save[k_iox][kk]=kv.cpu().numpy()
                else:
                    transformer_params_to_save[k_iox][kk]=kv
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
    trainrecord['krakencoder_version']=get_version(include_date=True)
    trainrecord['trainpath_shuffle']=do_trainpath_shuffle
    trainrecord['roundtrip']=do_roundtrip
    trainrecord['train_string']=train_string
    trainrecord['nbepochs']=nbepochs
    trainrecord['learningrate']=lr
    trainrecord['batchsize']=batchsize
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
    trainrecord['model_description']=network_description_string
    trainrecord['total_parameter_count']=network_parameter_count
    trainrecord['origscalecorr_epochs']=origscalecorr_epochs
    trainrecord['origscalecorr_inputtype']='inverse'
    trainrecord['saved_input_transformer_file']=input_transformer_file
    trainrecord['target_encoding']=do_target_encoding
    trainrecord['fixed_target_encoding']=do_fixed_target_encoding
    trainrecord['meantarget_latentsim']=do_meantarget_latentsim
    trainrecord['recordfile']=recordfile
    
    for extra_train_key,extra_train_val in extra_trainrecord_dict.items():
        trainrecord[extra_train_key]=extra_train_val
    
    if 'starting_point_file' in training_params:
        trainrecord['starting_point_file']=training_params['starting_point_file']
    
    if 'starting_point_epoch' in training_params:
        trainrecord['starting_point_epoch']=training_params['starting_point_epoch']
        
    if data_origscale_list is not None:
        trainrecord['origscalecorr_inputtype']='original'
    
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
    networkinfo['dropout_schedule']=dropout_schedule_list    
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
        if trainpath['train_outputs'] is not None:
            train_outputs_list_np[c2]=trainpath['train_outputs'].cpu().detach().numpy()
        else:
            train_outputs_list_np[c2]=data_optimscale_list['traindata'][c2]


        if trainpath['val_outputs'] is not None:
            val_outputs_list_np[c2]=trainpath['val_outputs'].cpu().detach().numpy()
        else:
            val_outputs_list_np[c2]=data_optimscale_list['valdata'][c2]
            
    starttime=time.time()
    timestamp_last_display=starttime    
    
    skipped_epoch_counter=0
    do_compute_performance_AFTER_latentsim=True #this way recorded performance matches checkpoint

    #dropout_per_epoch=np.linspace(dropout_init,dropout_final,nbepochs)
    #dropout_per_epoch=np.linspace(0,dropout,nbepochs)

    try:
        dropout_per_epoch=np.interp(xp=np.linspace(0,1,len(dropout_schedule_list)),fp=dropout_schedule_list,
                                    x=np.linspace(0,1,nbepochs))
        intergroup_dropout_per_epoch=np.interp(xp=np.linspace(0,1,len(intergroup_dropout_schedule_list)),fp=intergroup_dropout_schedule_list,
                                x=np.linspace(0,1,nbepochs))
    except:
        dropout_per_epoch=dropout*np.ones(nbepochs)
        if intergroup_dropout is None:
            intergroup_dropout_per_epoch=[None]*nbepochs
        else:
            intergroup_dropout_per_epoch=intergroup_dropout*np.ones(nbepochs)
        
    #######################
    for epoch in range(nbepochs):
        
        allpath_train_enc=[None]*len(trainpath_list)
        allpath_val_enc=[None]*len(trainpath_list)
        
        if do_trainpath_shuffle:
            trainpath_order=np.argsort(np.random.random_sample(len(trainpath_list)))
        else:
            trainpath_order=np.arange(len(trainpath_list))
        
        #for itp,trainpath in enumerate(trainpath_list):
        for itp in trainpath_order:
            trainpath=trainpath_list[itp]
            
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

            #output_margin_torch=torchfloat(trainpath['train_marginmean_outputs'])
            output_margin_torch=torchfloat(trainpath['train_marginmin_outputs'])
            

            transcoder_list=None
            if do_roundtrip:
                transcoder_list=[decoder_index_torch]
                decoder_index_torch=encoder_index_torch
            
            if do_eval_only:
                trainloops=0
            

            criterion_latent_mse=nn.MSELoss()
            
            #net.set_dropout(dropout_per_epoch[epoch])
            net.set_dropout(dropout_per_epoch[epoch],intergroup_dropout_per_epoch[epoch])
            
            net.train()
                
            #loop that allows multiple passes for certain paths (default = 1)
            for iloop in range(trainloops):
                train_running_loss = 0

                for batch_idx, train_data in enumerate(trainloader):
                    
                    if do_target_encoding:
                        #pulls out <batchsize> at a time
                        conn_inputs, conn_targets, conn_encoded_targets = train_data
                        

                        if do_fixed_target_encoding:
                            #1. input->latent and loss(predictedlatent,fixedlatent)
                            #2. fixedlatent->output and loss(output,predictedoutput)

                            #first compute current encoding and backprop encoder loss
                            optimizer.zero_grad(set_to_none=do_zerograd_none)
                                                
                            conn_encoded = net(conn_inputs, encoder_index_torch, neg1_index)
                            
                            #loss = criterion_latent_mse(conn_encoded,conn_encoded_targets) #where can we use this?
                            
                            loss = compute_path_loss(conn_encoded=conn_encoded, conn_encoded_targets=conn_encoded_targets, encoded_criterion=encoded_criterion, encoder_margin=encoder_margin_torch, 
                                                     latentnorm_loss_weight=latentnorm_loss_weight_torch, latent_maxrad_weight=latent_maxrad_weight_torch, latent_maxrad=latent_maxrad_torch)
                            
                            loss.backward()
                            optimizer.step()

                            train_running_loss += loss.item() 
                            
                            #Then compute predicted output and backprop decoder loss
                            optimizer.zero_grad(set_to_none=do_zerograd_none)
                            _ , conn_predicted = net(conn_encoded_targets, neg1_index, decoder_index_torch)
                        
                            loss = compute_path_loss(conn_predicted=conn_predicted, conn_targets=conn_targets, criterion=criterion, output_margin=output_margin_torch)
                            loss.backward()
                            optimizer.step()

                            train_running_loss += loss.item() 
                        else:
                            #input->latent->output, then loss(predictedlatent,fixedlatent) and loss(output,predictedoutput)

                            optimizer.zero_grad(set_to_none=do_zerograd_none)

                            conn_encoded, conn_predicted = net(conn_inputs,encoder_index_torch,decoder_index_torch)

                            loss = compute_path_loss(conn_predicted=conn_predicted, conn_targets=conn_targets, conn_encoded=conn_encoded, conn_encoded_targets=conn_encoded_targets, 
                                                     criterion=criterion, encoded_criterion=encoded_criterion, 
                                                     output_margin=output_margin_torch, encoder_margin=encoder_margin_torch, latentnorm_loss_weight=latentnorm_loss_weight_torch, 
                                                     latent_maxrad_weight=latent_maxrad_weight_torch, latent_maxrad=latent_maxrad_torch)
                            loss.backward()
                            optimizer.step()

                            train_running_loss += loss.item() 

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
                    
                if do_target_encoding:
                    #pulls out <batchsize> at a time
                    conn_inputs, conn_targets, conn_encoded_targets = val_data
                    
                    if do_fixed_target_encoding:
                        with torch.no_grad():
                            conn_encoded = net(conn_inputs, encoder_index_torch, neg1_index)
                            _ , conn_predicted = net(conn_encoded_targets, neg1_index, decoder_index_torch)
                            
                        #loss = criterion_latent_mse(conn_encoded,conn_encoded_targets) #where can we use this?
                        loss = compute_path_loss(conn_encoded=conn_encoded, conn_encoded_targets=conn_encoded_targets, encoded_criterion=encoded_criterion, encoder_margin=encoder_margin_torch, 
                                        latentnorm_loss_weight=latentnorm_loss_weight_torch, latent_maxrad_weight=latent_maxrad_weight_torch, latent_maxrad=latent_maxrad_torch)
                        
                        val_running_loss += loss.item() 

                        loss = compute_path_loss(conn_predicted=conn_predicted, conn_targets=conn_targets, criterion=criterion, output_margin=output_margin_torch)

                        val_running_loss += loss.item() 
                    else:
                        with torch.no_grad():
                            conn_encoded, conn_predicted = net(conn_inputs,encoder_index_torch,decoder_index_torch, transcoder_index_list=transcoder_list)

                        # loss terms (validation)
                        loss = compute_path_loss(conn_predicted=conn_predicted, conn_targets=conn_targets, conn_encoded=conn_encoded, conn_encoded_targets=conn_encoded_targets, 
                                                 criterion=criterion, encoded_criterion=encoded_criterion, 
                                                 output_margin=output_margin_torch, encoder_margin=encoder_margin_torch, latentnorm_loss_weight=latentnorm_loss_weight_torch, 
                                                 latent_maxrad_weight=latent_maxrad_weight_torch, latent_maxrad=latent_maxrad_torch)
                        val_running_loss += loss.item() 
                    
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
            
            #tpidx = [first trainingpath for encoder=0, first training path for encoder=1, etc...]
            encidx , tpidx=np.unique([tp['encoder_index'] for tp in trainpath_list],return_index=True)
            
            loss=0
            
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
                        if intergroup:
                            do_latentsim_on_intergroup=True
                            if do_latentsim_on_intergroup:
                                #new mode starting 2/29/2024, do latentsim on all groups after transformation
                                #this will translate each jenc to the same group as ienc and compute the latentsim loss
                                loss+=sum([criterion_latentsim(conn_encoded,net.intergroup_transform_latent(x,jenc,ienc)) for jenc,x in enumerate(batchinput_train_encoded) if ienc != jenc])
                            else:
                                #old mode used prior to 2/29/2024, only do latentsim on same group
                                
                                #if encoder jenc is in the same group as encoder ienc, then compute latentsim loss
                                loss+=sum([criterion_latentsim(conn_encoded,x) for jenc,x in enumerate(batchinput_train_encoded) if ienc > jenc 
                                        and intergroup_inputgroup_list[ienc]==intergroup_inputgroup_list[jenc]])*2
                        else:
                            #loss+=sum([criterion_latentsim(conn_encoded,x) for jenc,x in enumerate(batchinput_train_encoded) if ienc != jenc])
                            loss+=sum([criterion_latentsim(conn_encoded,x) for jenc,x in enumerate(batchinput_train_encoded) if ienc > jenc])*2 #changed 2/12/2024
                    #maybe also add some version of this that isn't sensitive to overall scale?
            
                loss=loss*latentsim_loss_weight_torch
                loss.backward()
                optimizer.step()
            
            net.eval()
        ############################
        #computing performance AFTER latentsim
        trainrecord['compute_performance_after_latentsim']=do_compute_performance_AFTER_latentsim
        if do_compute_performance_AFTER_latentsim:
            
            net.eval()
            for itp in trainpath_order:
                trainpath=trainpath_list[itp]
            
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
                #output_margin_torch=torchfloat(trainpath['train_marginmean_outputs'])
                output_margin_torch=torchfloat(trainpath['train_marginmin_outputs'])

                transcoder_list=None
                if do_roundtrip:
                    transcoder_list=[decoder_index_torch]
                    decoder_index_torch=encoder_index_torch

                #compute full train set identifiability
                with torch.no_grad():
                    if do_fixed_target_encoding:
                        conn_encoded = net(train_inputs, encoder_index_torch, neg1_index)
                        _,conn_predicted = net(train_encoded, neg1_index, decoder_index_torch)
                    else:
                        #for regular mode or 'target' mode, just get latent and predicted the normal way
                        #only do input->fixedlatent->output for fixed_target mode
                        conn_encoded, conn_predicted = net(train_inputs, encoder_index_torch, decoder_index_torch, transcoder_index_list=transcoder_list)

                #fc_preds = conn_predicted.cpu().detach().numpy()
                train_predicted=conn_predicted.cpu()
        
                allpath_train_enc[itp]=conn_encoded #store these encoded outputs to check across paths
                train_cc=xycorr(train_outputs,conn_predicted)
        
                corrloss_train[itp,epoch]=corrtop1acc(cc=train_cc)
                corrlossN_train[itp,epoch]=corrtopNacc(cc=train_cc,topn=topN)
                corrlossRank_train[itp,epoch]=corravgrank(cc=train_cc)
                avgcorr_train[itp,epoch],avgcorr_other_train[itp,epoch]=corr_ident_parts(cc=train_cc)
                explainedvar_train[itp,epoch]=explained_variance_ratio(train_outputs,conn_predicted)

                #compute full val set identifiability
                with torch.no_grad():
                    if do_fixed_target_encoding:
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
                explainedvar_val[itp,epoch]=explained_variance_ratio(val_outputs,conn_predicted)

                #this was negligible for non-PCA transform (norm only)
                #added about 33% more time to each epoch for PCA
                origscale_this_epoch=(epoch==nbepochs-1) or (origscalecorr_epochs>0 and (epoch % origscalecorr_epochs == 0))
                
                if origscale_this_epoch and output_transformer is not None:
                    if data_origscale_list is not None:
                        valOrig_outputs=data_origscale_list['valdata_origscale'][trainpath['output_name']]
                    else:
                        #valOrig_outputs=output_transformer.inverse_transform(val_outputs.cpu())
                        valOrig_outputs=output_transformer.inverse_transform(val_outputs)
                    #valOrig_predicted=output_transformer.inverse_transform(conn_predicted.cpu())
                    valOrig_predicted=output_transformer.inverse_transform(conn_predicted)
            
                    #if requested, look at the difference between mean(traindata) and mean(restore(train predicted))
                    #and shift the origscale restore(val predicted) by that amount 
                    do_adjust_train_restore_mean=False
                    if do_adjust_train_restore_mean:
                        if data_origscale_list is not None:
                            trainOrig_outputs=data_origscale_list['traindata_origscale'][trainpath['output_name']]
                        else:
                            #trainOrig_outputs=output_transformer.inverse_transform(train_outputs.cpu())
                            trainOrig_outputs=output_transformer.inverse_transform(train_outputs)
                        trainOrig_predicted=output_transformer.inverse_transform(train_predicted)
                
                        trainOrig_output_mean=trainOrig_outputs.mean(axis=0,keepdims=True)
                        trainOrig_predicted_mean=trainOrig_predicted.mean(axis=0,keepdims=True)
                        trainOrig_meandiff=trainOrig_predicted_mean-trainOrig_output_mean
                
                        valOrig_predicted-=trainOrig_meandiff
                
                    #################
                    valOrig_outputs=torchfloat(valOrig_outputs)
                    valOrig_predicted=torchfloat(valOrig_predicted)
                    
                    #for even less clear reason, these are sometimes double() instead of float
                    valOrig_cc=xycorr(valOrig_outputs.float(),valOrig_predicted.float())
                    
                    if data_origscale_list is not None and 'traindata_origscale_mean' in data_origscale_list:
                        trainOrig_mean=torchfloat(data_origscale_list['traindata_origscale_mean'][trainpath['output_name']])
                        valOrig_cc_resid=xycorr(valOrig_outputs.float()-trainOrig_mean,valOrig_predicted.float()-trainOrig_mean)
                        avgcorr_resid_OrigScale_val[itp,epoch],avgcorr_resid_OrigScale_other_val[itp,epoch]=corr_ident_parts(cc=valOrig_cc_resid)
                    
                    corrloss_OrigScale_val[itp,epoch]=corrtop1acc(cc=valOrig_cc)
                    corrlossN_OrigScale_val[itp,epoch]=corrtopNacc(cc=valOrig_cc,topn=topN)
                    corrlossRank_OrigScale_val[itp,epoch]=corravgrank(cc=valOrig_cc)
                    avgcorr_OrigScale_val[itp,epoch],avgcorr_OrigScale_other_val[itp,epoch]=corr_ident_parts(cc=valOrig_cc)

                    corrloss_OrigScale_pred2true_val[itp,epoch]=corrtop1acc(cc=valOrig_cc.T)
                    corrlossN_OrigScale_pred2true_val[itp,epoch]=corrtopNacc(cc=valOrig_cc.T,topn=topN)
                    corrlossRank_OrigScale_pred2true_val[itp,epoch]=corravgrank(cc=valOrig_cc.T)
                    

                    explainedvar_OrigScale_val[itp,epoch]=explained_variance_ratio(valOrig_outputs,valOrig_predicted)
                        
                
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
                
                
        if display_on_this_loop:
            print(net)
            tmp_time=time.time()
            if not display_seconds or (tmp_time-timestamp_last_display)>=display_seconds:
                timestamp_last_display=tmp_time
            
                curtime=time.time()-starttime
                print("trainrecord: %s" % (recordfile))
                print('epoch: %d, %.2f seconds (%.2f sec/epoch)' % (epoch,curtime,curtime/(epoch+1)))

                tmp_pathstr=format_columns(column_data=[
                    [itp,[trainpath_list[itp]['encoder_index'],trainpath_list[itp]['decoder_index']]] 
                        for itp in range(len(trainpath_list))
                    ], column_format_list=["path%d","%d->%d"],delimiter=":")

                tmp_column_data=[
                    [
                        tmp_pathstr[itp],
                        loss_train[itp,epoch], 
                        loss_val[itp,epoch], 
                        [corrloss_train[itp,epoch]*train_outputs_np.shape[0], train_outputs_np.shape[0], corrloss_train[itp,epoch]], 
                        [corrloss_val[itp,epoch]*val_outputs_np.shape[0],val_outputs_np.shape[0], corrloss_val[itp,epoch]],
                        corrlossN_train[itp,epoch],
                        corrlossN_val[itp,epoch],
                        corrlossRank_train[itp,epoch],
                        corrlossRank_val[itp,epoch],
                        [corrlossRank_val[itp,:epoch+1].max(), np.argmax(corrlossRank_val[itp,:epoch+1])],
                        explainedvar_train[itp,epoch],
                        explainedvar_val[itp,epoch],
                        explainedvar_OrigScale_val[itp,epoch],
                        [np.nanmax(explainedvar_OrigScale_val[itp,:epoch+1]), nanargmax_safe(explainedvar_OrigScale_val[itp,:epoch+1],nanval=-1)]
                    ] 
                    for itp in range(len(trainpath_list))]
                
                #add an extra row with mean across all paths
                tmp_column_mean=[[np.mean([r[i] for r in tmp_column_data],axis=0) 
                                  if not isinstance(tmp_column_data[0][i],str) else "mean" 
                                  for i in range(len(tmp_column_data[0]))]]
                
                tmp_column_data+=tmp_column_mean

                tmp_column_str=format_columns(column_data=tmp_column_data,
                                column_headers=['path','tr.loss','v.loss','tr.top1acc','v.top1acc',
                                                'tr.topNacc','v.topNacc','tr.rank','v.rank','max(v.rank)','tr.R2','v.R2','v.R2(orig)','max(v.R2(orig))'],
                                column_format_list=["%s","%.6f","%.6f","%.0f/%d=%.6f","%.0f/%d=%.6f",
                                                    "%.6f","%.6f","%.3f","%.3f","%.3f/%d","%.3f","%.3f","%.3f","%.3f/%d"],
                                header_separator="-")
                [print(s) for s in tmp_column_str[:-1]];
                
                if len(trainpath_list)>1:
                    print(tmp_column_str[1]) #print column separator again before mean row
                    print(tmp_column_str[-1])
                    print(tmp_column_str[1]) #print column separator again before repeating COLUMN HEADER row
                    print(tmp_column_str[0])
                
                if skipped_epoch_counter > 0:
                    print('  skipped all paths for %d/%d epochs' % (skipped_epoch_counter,display_epochs))
                print('epoch: %d, %.2f seconds (%.2f sec/epoch)' % (epoch,curtime,curtime/(epoch+1)))

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
            trainrecord['explainedvar_train']=explainedvar_train
            trainrecord['explainedvar_val']=explainedvar_val
            trainrecord['avgcorr_OrigScale_train']=avgcorr_OrigScale_train
            trainrecord['avgcorr_OrigScale_other_train']=avgcorr_OrigScale_other_train
            trainrecord['avgcorr_OrigScale_val']=avgcorr_OrigScale_val
            trainrecord['avgcorr_OrigScale_other_val']=avgcorr_OrigScale_other_val
            trainrecord['avgcorr_resid_OrigScale_train']=avgcorr_resid_OrigScale_train
            trainrecord['avgcorr_resid_OrigScale_other_train']=avgcorr_resid_OrigScale_other_train
            trainrecord['avgcorr_resid_OrigScale_val']=avgcorr_resid_OrigScale_val
            trainrecord['avgcorr_resid_OrigScale_other_val']=avgcorr_resid_OrigScale_other_val
            trainrecord['corrloss_OrigScale_train']=corrloss_OrigScale_train
            trainrecord['corrloss_OrigScale_val']=corrloss_OrigScale_val
            trainrecord['corrlossN_OrigScale_train']=corrlossN_OrigScale_train
            trainrecord['corrlossN_OrigScale_val']=corrlossN_OrigScale_val
            trainrecord['corrlossRank_OrigScale_train']=corrlossRank_OrigScale_train
            trainrecord['corrlossRank_OrigScale_val']=corrlossRank_OrigScale_val
            trainrecord['explainedvar_OrigScale_train']=explainedvar_OrigScale_train
            trainrecord['explainedvar_OrigScale_val']=explainedvar_OrigScale_val
            trainrecord['corrloss_OrigScale_pred2true_train']=corrloss_OrigScale_pred2true_train
            trainrecord['corrloss_OrigScale_pred2true_val']=corrloss_OrigScale_pred2true_val
            trainrecord['corrlossN_OrigScale_pred2true_train']=corrlossN_OrigScale_pred2true_train
            trainrecord['corrlossN_OrigScale_pred2true_val']=corrlossN_OrigScale_pred2true_val
            trainrecord['corrlossRank_OrigScale_pred2true_train']=corrlossRank_OrigScale_pred2true_train
            trainrecord['corrlossRank_OrigScale_pred2true_val']=corrlossRank_OrigScale_pred2true_val
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
            trainrecord['topN']=topN
            trainrecord['maxthreads']=maxthreads
            trainrecord['timestamp']=timestamp_suffix
            trainrecord['epoch_timestamp']=epoch_timestamp
            trainrecord['trainduration']=trainduration
            trainrecord['current_epoch']=epoch
            trainrecord['seconds_per_epoch']=trainduration/(epoch+1)

            savemat(recordfile,trainrecord,format='5',do_compression=True)
            print("Ep %d) Saved %s" % (epoch, recordfile))

            display_kraken_heatmap(trainrecord,metrictype=['top1acc','topNacc','avgrank','avgcorr_resid'],origscale=True,single_epoch=True,
                                    colormap='magma2',outputimagefile={'file':imgfile_heatmap,'dpi':200})
            
            print("Ep %d) Saved %s" % (epoch, imgfile_heatmap))
            
        if checkpoint_on_this_loop:
            if update_single_checkpoint:
                statefile=checkpoint_filebase+".pt"
            else:
                if epoch == nbepochs-1:
                    statefile=checkpoint_filebase+"_ep%06d.pt" % (nbepochs)
                else:
                    statefile=checkpoint_filebase+"_ep%06d.pt" % (epoch)
            checkpoint={"epoch": epoch}
            
            #copy network description fields into checkpoint
            for k in networkinfo.keys():
                checkpoint[k]=networkinfo[k]
                
            checkpoint['training_params']=trainrecord_params
            
            if save_optimizer_params and (epoch == nbepochs-1 or exit_on_this_loop):
                #include optimizer in final checkpoint (so we could resume training)
                checkpoint['optimizer']=optimizer.state_dict()
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
    trainrecord['explainedvar_train']=explainedvar_train
    trainrecord['explainedvar_val']=explainedvar_val
    trainrecord['avgcorr_OrigScale_train']=avgcorr_OrigScale_train
    trainrecord['avgcorr_OrigScale_other_train']=avgcorr_OrigScale_other_train
    trainrecord['avgcorr_OrigScale_val']=avgcorr_OrigScale_val
    trainrecord['avgcorr_OrigScale_other_val']=avgcorr_OrigScale_other_val
    trainrecord['avgcorr_resid_OrigScale_train']=avgcorr_resid_OrigScale_train
    trainrecord['avgcorr_resid_OrigScale_other_train']=avgcorr_resid_OrigScale_other_train
    trainrecord['avgcorr_resid_OrigScale_val']=avgcorr_resid_OrigScale_val
    trainrecord['avgcorr_resid_OrigScale_other_val']=avgcorr_resid_OrigScale_other_val
    trainrecord['corrloss_OrigScale_train']=corrloss_OrigScale_train
    trainrecord['corrloss_OrigScale_val']=corrloss_OrigScale_val
    trainrecord['corrlossN_OrigScale_train']=corrlossN_OrigScale_train
    trainrecord['corrlossN_OrigScale_val']=corrlossN_OrigScale_val
    trainrecord['corrlossRank_OrigScale_train']=corrlossRank_OrigScale_train
    trainrecord['corrlossRank_OrigScale_val']=corrlossRank_OrigScale_val
    trainrecord['explainedvar_OrigScale_train']=explainedvar_OrigScale_train
    trainrecord['explainedvar_OrigScale_val']=explainedvar_OrigScale_val
    trainrecord['corrloss_OrigScale_pred2true_train']=corrloss_OrigScale_pred2true_train
    trainrecord['corrloss_OrigScale_pred2true_val']=corrloss_OrigScale_pred2true_val
    trainrecord['corrlossN_OrigScale_pred2true_train']=corrlossN_OrigScale_pred2true_train
    trainrecord['corrlossN_OrigScale_pred2true_val']=corrlossN_OrigScale_pred2true_val
    trainrecord['corrlossRank_OrigScale_pred2true_train']=corrlossRank_OrigScale_pred2true_train
    trainrecord['corrlossRank_OrigScale_pred2true_val']=corrlossRank_OrigScale_pred2true_val
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
    trainrecord['topN']=topN
    trainrecord['maxthreads']=maxthreads
    trainrecord['timestamp']=timestamp_suffix
    trainrecord['epoch_timestamp']=epoch_timestamp
    trainrecord['trainduration']=trainduration
    trainrecord['current_epoch']=epoch
    trainrecord['seconds_per_epoch']=trainduration/(epoch+1)
    
    savemat(recordfile,trainrecord,format='5',do_compression=True)
    print("Saved %s" % (recordfile))

    display_kraken_heatmap(trainrecord,metrictype=['top1acc','topNacc','avgrank','avgcorr_resid'],origscale=True,single_epoch=True,
                            colormap='magma2',outputimagefile={'file':imgfile_heatmap,'dpi':200})
    
    print("Ep %d) Saved %s" % (epoch, imgfile_heatmap))
    
    return net, trainrecord

def run_network(net, trainpath_list, maxthreads=1, fusionmode=False, pathfinder_list=[], fusionmode_search_list=[], fusionmode_do_not_normalize=False):
    """
    Evaluate a network on a list of trainpaths
    
    For now, run_model.py is NOT calling this function. It was used for testing and debugging earlier in development. It might need some fixes.
    """
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
    
    if fusionmode:
        fusionmode_trainpath_mask=np.ones(len(trainpath_list))>0
        if len(fusionmode_search_list)>0:
            fusionmode_trainpath_mask=[any([x in tp["input_name"] for x in fusionmode_search_list]) for tp in trainpath_list]
            for itp, trainpath in enumerate(trainpath_list):
                print(fusionmode_trainpath_mask[itp],trainpath["input_name"],trainpath["output_name"])
        
        if train_inputs is not None:
            traindata_encoded_mean=torch.mean(torch.stack([tp['train_inputs_encoded'] for itp,tp in enumerate(trainpath_list) if fusionmode_trainpath_mask[itp]]),axis=0)
        valdata_encoded_mean=torch.mean(torch.stack([tp['val_inputs_encoded'] for itp,tp in enumerate(trainpath_list) if fusionmode_trainpath_mask[itp]]),axis=0)
        
        if net.latent_normalize and not fusionmode_do_not_normalize:
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
