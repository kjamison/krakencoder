"""
Functions related to loading and manipulating input data and creating input/output transformations (eg: PCA, demean, etc...)

Includes some hard-coded paths to HCP data files, which may need to be updated for your system.
"""

from .utils import *
from .fetch import get_fetchable_data_list, model_data_folder, replace_data_folder_placeholder, load_flavor_database
from ._resources import resource_path
from scipy.io import loadmat
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import FunctionTransformer

import os
import re
import json
import pandas as pd
from copy import deepcopy

def clean_subject_list(subjects):
    """
    make all subject lists = str type
    (for cases where subjects are float(1234.), convert to int first to make sure they are "1234" and not "1234.0")
    """
    if isinstance(subjects,str):
        subjects=[subjects]
    try:
        newsubjects=[int(x) for x in subjects]
    except:
        newsubjects=[x for x in subjects]
    newsubjects=np.array([str(x).strip() for x in newsubjects])
    return newsubjects

def load_hcp_subject_list(numsubj=993):
    """
    Returns a list of subjects and familyids for the HCP dataset, as read from a hardcoded file.
    
    Parameters:
    numsubj: int, number of subjects to load (420, 993, or 997)
    
    Returns:
    subjects: np.array([subject ids] as str)
    familyid: np.array([family id] as int)
    """
    if os.path.isdir('/Users/kwj5'):
        #datafolder='/Users/kwj5/Box/HCP_SC_FC_new997'
        datafolder='/Users/kwj5/Research/HCP'
        studyfolder='/Users/kwj5/Research/HCP'
    elif os.path.isdir('/home/kwj2001/colossus_shared/HCP'):
        studyfolder='/home/kwj2001/colossus_shared/HCP'
        datafolder='/home/kwj2001/colossus_shared/HCP'
    elif os.path.isdir('/midtier/sablab/scratch/kwj2001'):
        studyfolder='/midtier/sablab/scratch/kwj2001/HCP'
        datafolder='/midtier/sablab/scratch/kwj2001/HCP'
    elif os.path.isdir('/home/ubuntu'):
        studyfolder='/home/ubuntu'
        datafolder='/home/ubuntu'

    familyidx=np.loadtxt('%s/subjects_famidx_rfMRI_dMRI_complete_997.txt' % (studyfolder))
    subj997=np.loadtxt('%s/subjects_rfMRI_dMRI_complete_997.txt' % (studyfolder))
    
    if numsubj==420:
        subjects=np.loadtxt('%s/subjects_unrelated420_scfc.txt' % (studyfolder))
        familyidx=np.array([familyidx[i] for i,s in enumerate(subj997) if s in subjects])
    elif numsubj==993:
        subjects=np.loadtxt('%s/subjects_rfMRI_dMRI_complete_993_minus4.txt' % (studyfolder))
        familyidx=np.array([familyidx[i] for i,s in enumerate(subj997) if s in subjects])
    elif numsubj==997:
        subjects=subj997
    else:
        raise Exception("Unknown numsubj %s" % (numsubj))
    
    subjects=clean_subject_list(subjects)
    
    return subjects, familyidx

def get_hcp_data_flavors_OLD(roi_list=["fs86","shen268","coco439"], 
                         sc_type_list=["ifod2act_volnorm","sdstream_volnorm"], 
                         fc_type_list=["FCcov","FCcovgsr","FCpcorr"], 
                         fc_filter_list=["hpf","bpf","nofilt"],
                         sc=True,
                         fc=True):
    """
    Returns a list of data flavors based on the input lists of roi_list, sc_type_list, fc_type_list, and fc_filter_list
    """
    if not roi_list:
        roi_list=[]
    if not sc_type_list:
        sc_type_list=[]
    if not fc_type_list:
        fc_type_list=[]
    if not fc_filter_list:
        fc_filter_list=[]
    
    if isinstance(roi_list,str):
        roi_list=[roi_list]
    if isinstance(sc_type_list,str):
        sc_type_list=[sc_type_list]
    if isinstance(fc_type_list,str):
        fc_type_list=[fc_type_list]
    if isinstance(fc_filter_list,str):
        fc_filter_list=[fc_filter_list]
    
    if not sc:
        sc_type_list=[]
    
    if not fc:
        fc_filter_list=[]
    
    conntype_list=[]
    for r in roi_list:
        for sc in sc_type_list:
            conntype_list+=["%s_%s" % (r,sc)]
            
        for f in fc_filter_list:
            for fc in fc_type_list:
                fctmp=fc.replace("_gsr","").replace("gsr","")
                c="%s_%s_%s" % (fctmp,r,f)
                if "gsr" in fc:
                    c+="gsr"
                conntype_list+=[c]
    return conntype_list


def canonical_data_flavor_OLD(conntype, only_if_brackets=False, return_groupname=False, accept_unknowns=False):
    """
    Returns the canonical data flavor name for a (possibly) non-canonical input name. 
    eg: canonical_data_flavor('fs86_hpf_FCcov') -> 'FCcorr_fs86_hpf_FC'
    
    Parameters:
    conntype: str, input data flavor name
    only_if_brackets: bool (optional, default=False), if True, will only change the name if the input is in the form "[name]", otherwise return name as-is
    return_groupname: bool (optional, default=False), if True, will return the group name ("SC", "FC", or "encoded") as a second return value
    accept_unknowns: bool (optional, default=False), if True, will not raise an exception if an unknown input is provided
    
    Returns:
    conntype_canonical: str, canonical data flavor name
    groupname: str, group name ("SC", "FC", or "encoded"), if return_groupname=True
    """
    groupname=None
    
    if only_if_brackets:
        #special mode that leaves inputs intact unless they are in the form "[name]", in which case
        #it will return the canonical version of "name"
        if not re.match(r".*\[.+\].*",conntype.lower()):
            if return_groupname:
                return conntype, groupname
            else:
                return conntype
        conntype=re.sub(r"^.*\[(.+)\].*$",r'\1',conntype)
    
    if conntype.lower() == "encoded":
        if return_groupname:
            return "encoded", groupname
        else:
            return "encoded"
    
    if conntype.lower().startswith("fusion"):
        if return_groupname:
            return conntype, groupname
        else:
            return conntype
    
    if conntype.lower() == "mean":
        if return_groupname:
            return conntype, groupname
        else:
            return conntype
    
    # parse user-specified input data type
    input_atlasname=""
    input_flavor=""
    input_fcfilt=""
    input_fcgsr=""
    input_scproc="volnorm"
    
    input_conntype_lower=conntype.lower()
    if "fs86" in input_conntype_lower:
        input_atlasname="fs86"
    elif "shen268" in input_conntype_lower:
        input_atlasname="shen268"
    elif "coco439" in input_conntype_lower or "cocommpsuit439" in input_conntype_lower:
        input_atlasname="coco439"
    else:
        if not accept_unknowns:
            raise Exception("Unknown atlas name for input type: %s" % (conntype))
    
    if "fccov" in input_conntype_lower and "gsr" in input_conntype_lower:
        input_flavor="FCcov"
        input_fcgsr="gsr"
    elif "fccov" in input_conntype_lower:
        input_flavor="FCcov"
    elif "pcorr" in input_conntype_lower:
        input_flavor="FCpcorr"
    elif "sdstream" in input_conntype_lower:
        input_flavor="sdstream"
    elif "ifod2act" in input_conntype_lower:
        input_flavor="ifod2act"
    else:
        if not accept_unknowns:
            raise Exception("Unknown data flavor for input type: %s" % (conntype))
    
    #FC: FCcorr_<atlas>_<fcfilt>[gsr?]_FC, FCpcorr_<atlas>_<fcfilt>_FC
    #SC: <atlas>_sdstream_volnorm, <atlas>_ifod2act_volnorm
    
    if "FC" in input_flavor:
        if "hpf" in input_conntype_lower:
            input_fcfilt="hpf"
        elif "bpf" in input_conntype_lower:
            input_fcfilt="bpf"
        elif "nofilt" in input_conntype_lower:
            input_fcfilt="nofilt"
        else:
            if not accept_unknowns:
                raise Exception("Unknown FC filter for input type: %s" % (conntype))
    
    if "FC" in input_flavor:
        groupname="FC"
        conntype_canonical="%s_%s_%s%s_FC" % (input_flavor,input_atlasname,input_fcfilt,input_fcgsr) #orig style with FC<flavor>_<atlas>_<filt><gsr>_FC
        #conntype_canonical="%s_%s_%s%s" % (input_flavor,input_atlasname,input_fcfilt,input_fcgsr) #new style with FC<flavor>_<atlas>_<filt><gsr>
    else:
        groupname="SC"
        conntype_canonical="%s_%s_%s" % (input_atlasname,input_flavor,input_scproc) #orig style
        #conntype_canonical="SC%s_%s_%s" % (input_flavor,input_atlasname,input_scproc) #new style with SC<algo>_<atlas>_volnorm

    if return_groupname:
        return conntype_canonical, groupname
    else:
        return conntype_canonical

def get_hcp_data_flavors_OLD(roi_list=["fs86","shen268","coco439"], 
                         sc_type_list=["ifod2act_volnorm","sdstream_volnorm"], 
                         fc_type_list=["FCcov","FCcovgsr","FCpcorr"], 
                         fc_filter_list=["hpf","bpf","nofilt"],
                         sc=True,
                         fc=True):
    """
    DEPRECATED old version from before first release. Some older checkpoints use this.
    ################################################
    Returns a list of data flavors based on the input lists of roi_list, sc_type_list, fc_type_list, and fc_filter_list
    """
    if not roi_list:
        roi_list=[]
    if not sc_type_list:
        sc_type_list=[]
    if not fc_type_list:
        fc_type_list=[]
    if not fc_filter_list:
        fc_filter_list=[]
    
    if isinstance(roi_list,str):
        roi_list=[roi_list]
    if isinstance(sc_type_list,str):
        sc_type_list=[sc_type_list]
    if isinstance(fc_type_list,str):
        fc_type_list=[fc_type_list]
    if isinstance(fc_filter_list,str):
        fc_filter_list=[fc_filter_list]
    
    if not sc:
        sc_type_list=[]
    
    if not fc:
        fc_filter_list=[]
    
    conntype_list=[]
    for r in roi_list:
        for sc in sc_type_list:
            conntype_list+=["%s_%s" % (r,sc)]
            
        for f in fc_filter_list:
            for fc in fc_type_list:
                fctmp=fc.replace("_gsr","").replace("gsr","")
                c="%s_%s_%s" % (fctmp,r,f)
                if "gsr" in fc:
                    c+="gsr"
                conntype_list+=[c]

    return conntype_list


def canonical_data_flavor_OLD(conntype, only_if_brackets=False, return_groupname=False, accept_unknowns=False):
    """
    DEPRECATED old version from before first release. Some older checkpoints use this.
    ################################################
    Returns the canonical data flavor name for a (possibly) non-canonical input name. 
    eg: canonical_data_flavor('fs86_hpf_FCcov') -> 'FCcv_fs86_hpf_FC'
    
    Parameters:
    conntype: str, input data flavor name
    only_if_brackets: bool (optional, default=False), if True, will only change the name if the input is in the form "[name]", otherwise return name as-is
    return_groupname: bool (optional, default=False), if True, will return the group name ("SC", "FC", or "encoded") as a second return value
    accept_unknowns: bool (optional, default=False), if True, will not raise an exception if an unknown input is provided
    
    Returns:
    conntype_canonical: str, canonical data flavor name
    groupname: str, group name ("SC", "FC", or "encoded"), if return_groupname=True
    """
    groupname=None
    
    if only_if_brackets:
        #special mode that leaves inputs intact unless they are in the form "[name]", in which case
        #it will return the canonical version of "name"
        if not re.match(r".*\[.+\].*",conntype.lower()):
            if return_groupname:
                return conntype, groupname
            else:
                return conntype
        conntype=re.sub(r"^.*\[(.+)\].*$",r'\1',conntype)
    
    if conntype.lower() == "encoded":
        if return_groupname:
            return "encoded", groupname
        else:
            return "encoded"
    
    if conntype.lower().startswith("fusion"):
        if return_groupname:
            return conntype, groupname
        else:
            return conntype
    
    if conntype.lower() == "mean":
        if return_groupname:
            return conntype, groupname
        else:
            return conntype
    
    # parse user-specified input data type
    input_atlasname=""
    input_flavor=""
    input_fcfilt=""
    input_fcgsr=""
    input_scproc="volnorm"
    
    input_conntype_lower=conntype.lower()
    if "fs86" in input_conntype_lower:
        input_atlasname="fs86"
    elif "shen268" in input_conntype_lower:
        input_atlasname="shen268"
    elif "coco439" in input_conntype_lower or "cocommpsuit439" in input_conntype_lower:
        input_atlasname="coco439"
    else:
        if not accept_unknowns:
            raise Exception("Unknown atlas name for input type: %s" % (conntype))
    
    if ("fccorr" in input_conntype_lower or "fccov" in input_conntype_lower) and "gsr" in input_conntype_lower:
        input_flavor="FCcov"
        input_fcgsr="gsr"
    elif "fccorr" in input_conntype_lower or "fccov" in input_conntype_lower:
        input_flavor="FCcov"
    elif "pcorr" in input_conntype_lower:
        input_flavor="FCpcorr"
    elif "sdstream" in input_conntype_lower:
        input_flavor="sdstream"
    elif "ifod2act" in input_conntype_lower:
        input_flavor="ifod2act"
    else:
        if not accept_unknowns:
            raise Exception("Unknown data flavor for input type: %s" % (conntype))
    
    #FC: FCcorr_<atlas>_<fcfilt>[gsr?]_FC, FCpcorr_<atlas>_<fcfilt>_FC
    #SC: <atlas>_sdstream_volnorm, <atlas>_ifod2act_volnorm
    
    if "FC" in input_flavor:
        if "hpf" in input_conntype_lower:
            input_fcfilt="hpf"
        elif "bpf" in input_conntype_lower:
            input_fcfilt="bpf"
        elif "nofilt" in input_conntype_lower:
            input_fcfilt="nofilt"
        else:
            if not accept_unknowns:
                raise Exception("Unknown FC filter for input type: %s" % (conntype))
    
    if "FC" in input_flavor:
        groupname="FC"
        conntype_canonical="%s_%s_%s%s_FC" % (input_flavor,input_atlasname,input_fcfilt,input_fcgsr) #orig style with FC<flavor>_<atlas>_<filt><gsr>_FC
        #conntype_canonical="%s_%s_%s%s" % (input_flavor,input_atlasname,input_fcfilt,input_fcgsr) #new style with FC<flavor>_<atlas>_<filt><gsr>
    else:
        groupname="SC"
        conntype_canonical="%s_%s_%s" % (input_atlasname,input_flavor,input_scproc) #orig style
        #conntype_canonical="SC%s_%s_%s" % (input_flavor,input_atlasname,input_scproc) #new style with SC<algo>_<atlas>_volnorm

    if return_groupname:
        return conntype_canonical, groupname
    else:
        return conntype_canonical


def get_hcp_data_flavors(roi_list=["fs86","shen268","coco439"], 
                         sc_type_list=["SCifod2act_volnorm","SCsdstream_volnorm"], 
                         fc_type_list=["FCcorr","FCcorrgsr","FCpcorr"], 
                         fc_filter_list=["hpf","bpf","nofilt"],
                         sc=True,
                         fc=True):
    """
    Returns a list of data flavors based on the input lists of roi_list, sc_type_list, fc_type_list, and fc_filter_list
    """
    if not roi_list:
        roi_list=[]
    if not sc_type_list:
        sc_type_list=[]
    if not fc_type_list:
        fc_type_list=[]
    if not fc_filter_list:
        fc_filter_list=[]
    
    if isinstance(roi_list,str):
        roi_list=[roi_list]
    if isinstance(sc_type_list,str):
        sc_type_list=[sc_type_list]
    if isinstance(fc_type_list,str):
        fc_type_list=[fc_type_list]
    if isinstance(fc_filter_list,str):
        fc_filter_list=[fc_filter_list]
    
    if not sc:
        sc_type_list=[]
    
    if not fc:
        fc_filter_list=[]
    
    conntype_list=[]
    for r in roi_list:
        for sc in sc_type_list:
            c="%s_%s" % (sc,r)
            if '_volnorm' in c:
                c=c.replace('_volnorm','')+"_volnorm"
            
            conntype_list+=[c]
            
        for f in fc_filter_list:
            for fc in fc_type_list:
                fctmp=fc.replace("_gsr","").replace("gsr","")
                c="%s_%s_%s" % (fctmp,r,f)
                if "gsr" in fc:
                    c+="gsr"
                conntype_list+=[c]

    conntype_list=[canonical_data_flavor(c) for c in conntype_list]
    return conntype_list


def canonical_data_flavor(conntype, only_if_brackets=False, return_groupname=False, accept_unknowns=False):
    """
    Returns the canonical data flavor name for a (possibly) non-canonical input name. 
    eg: canonical_data_flavor('fs86_hpf_FCcorr') -> 'FCcorr_fs86_hpf'
    
    Parameters:
    conntype: str, input data flavor name
    only_if_brackets: bool (optional, default=False), if True, will only change the name if the input is in the form "[name]", otherwise return name as-is
    return_groupname: bool (optional, default=False), if True, will return the group name ("SC", "FC", or "encoded") as a second return value
    accept_unknowns: bool (optional, default=False), if True, will not raise an exception if an unknown input is provided
    
    Returns:
    conntype_canonical: str, canonical data flavor name
    groupname: str, group name ("SC", "FC", or "encoded"), if return_groupname=True
    """
    groupname=None
    
    if only_if_brackets:
        #special mode that leaves inputs intact unless they are in the form "[name]", in which case
        #it will return the canonical version of "name"
        if not re.match(r".*\[.+\].*",conntype.lower()):
            if return_groupname:
                return conntype, groupname
            else:
                return conntype
        conntype=re.sub(r"^.*\[(.+)\].*$",r'\1',conntype)
    
    if conntype.lower() == "encoded":
        if return_groupname:
            return "encoded", groupname
        else:
            return "encoded"
    
    if conntype.lower().startswith("fusion"):
        if return_groupname:
            return conntype, groupname
        else:
            return conntype
    
    if conntype.lower() == "mean":
        if return_groupname:
            return conntype, groupname
        else:
            return conntype
    
    # parse user-specified input data type
    input_atlasname=""
    input_flavor=""
    input_fcfilt=""
    input_fcgsr=""
    input_scproc="volnorm"
    
    is_unknown=False
    
    input_conntype_lower=conntype.lower()
    if "fs86" in input_conntype_lower:
        input_atlasname="fs86"
    elif "shen268" in input_conntype_lower:
        input_atlasname="shen268"
    elif "coco439" in input_conntype_lower or "cocommpsuit439" in input_conntype_lower:
        input_atlasname="coco439"
    else:
        is_unknown=True
        if not accept_unknowns:
            raise Exception("Unknown atlas name for input type: %s" % (conntype))
    
    if ("fccorr" in input_conntype_lower or "fccov" in input_conntype_lower) and "gsr" in input_conntype_lower:
        input_flavor="FCcorr"
        input_fcgsr="gsr"
    elif "fccorr" in input_conntype_lower or "fccov" in input_conntype_lower:
        input_flavor="FCcorr"
    elif "pcorr" in input_conntype_lower:
        input_flavor="FCpcorr"
    elif "sdstream" in input_conntype_lower:
        input_flavor="sdstream"
    elif "ifod2act" in input_conntype_lower:
        input_flavor="ifod2act"
    else:
        is_unknown=True
        if not accept_unknowns:
            raise Exception("Unknown data flavor for input type: %s" % (conntype))
    
    #FC: FCcorr_<atlas>_<fcfilt>[gsr?], FCpcorr_<atlas>_<fcfilt>
    #SC: SCsdstream_<atlas>_volnorm, SCifod2act_<atlas>_volnorm
    
    if "FC" in input_flavor:
        if "hpf" in input_conntype_lower:
            input_fcfilt="hpf"
        elif "bpf" in input_conntype_lower:
            input_fcfilt="bpf"
        elif "nofilt" in input_conntype_lower:
            input_fcfilt="nofilt"
        else:
            is_unknown=True
            if not accept_unknowns:
                raise Exception("Unknown FC filter for input type: %s" % (conntype))
    else:
        if "sift2volnorm" in input_conntype_lower or "sift2_volnorm" in input_conntype_lower:
            input_scproc="sift2volnorm"
        elif "volnorm" in input_conntype_lower:
            input_scproc="volnorm"
        elif "sift2count" in input_conntype_lower or "sift2_count" in input_conntype_lower or input_conntype_lower.endswith("_sift2"):
            input_scproc="sift2count"
        elif "_count" in input_conntype_lower:
            input_scproc="count"
        else:
            is_unknown=True
            if not accept_unknowns:
                raise Exception("Unknown SC processing type for input type: %s" % (conntype))
        #handle special volnormicv case
        if "volnormicv" in input_conntype_lower or "volnorm_icv" in input_conntype_lower:
            input_scproc=input_scproc.replace("volnorm","volnormicv")
    
    if "FC" in input_flavor:
        groupname="FC"
        conntype_canonical="%s_%s_%s%s" % (input_flavor,input_atlasname,input_fcfilt,input_fcgsr) #new style with FC<flavor>_<atlas>_<filt><gsr>
    else:
        groupname="SC"
        conntype_canonical="SC%s_%s_%s" % (input_flavor,input_atlasname,input_scproc) #new style with SC<algo>_<atlas>_volnorm

    if is_unknown:
        conntype_canonical=conntype
        
    if return_groupname:
        return conntype_canonical, groupname
    else:
        return conntype_canonical

def atlas_from_flavors(conntype_list):
    if isinstance(conntype_list,str):
        return atlas_from_flavors([conntype_list])[0]
    
    atlas_list=[]
    for k_in in conntype_list:
        k_in_atlas="unknown"
        if "fs86" in k_in.lower():
            k_in_atlas="fs86"
        elif "shen268" in k_in.lower():
            k_in_atlas="shen268"
        elif "coco439" in k_in.lower() or "cocommpsuit439" in k_in.lower():
            k_in_atlas="coco439"
        elif "coco" in k_in:
            #for cocoyeo, cocolaus
            k_in_atlas=re.sub(r"^.*(coco[^_]+).*$",r"\1",k_in)
        atlas_list+=[k_in_atlas]
        
    return atlas_list

def get_subjects_from_conndata(conndata, subjects=None, remove_subjects=None):
    """
    Returns a new conndata dict with only the subjects specified in the subjects list or with subjects removed as specified in the remove_subjects list.
    
    Parameters:
    conndata: dict, input conndata dict with 'subjects' and 'data' fields (as returned from load_hcp_data or load_input_data)
    subjects: list(str) (optional), list of subject IDs to keep
    remove_subjects: list(str) (optional), list of subject IDs to remove
    
    Returns:
    conndata_new: dict, new conndata dict with only the specified subjects
    """
    if not isinstance(conndata,dict):
        raise Exception("conndata must be a dict")
    
    if not ('subjects' in conndata and 'data' in conndata):
        #if provided a multi-flavor input, loop over the flavors
        #(make deep copy)
        conndata_new={}
        for k in conndata.keys():
            conndata_new[k]=get_subjects_from_conndata(conndata[k],subjects=subjects, remove_subjects=remove_subjects)
        return conndata_new
    
    if remove_subjects is not None:
        newsubj_idx=np.where([s not in remove_subjects for s in conndata['subjects']])[0]

    elif subjects is not None:
        newsubj_idx=[np.where(conndata['subjects']==s)[0] for s in subjects]
        newsubj_idx=[x[0] for x in newsubj_idx if len(x)>0]
    
    else:
        newsubj_idx=np.arange(len(conndata['subjects']))

    #make new deep copy of data
    conndata_new={}
    for k in conndata.keys():
        if k == 'subjects':
            conndata_new['subjects']=np.array(conndata['subjects'])[newsubj_idx]
        elif k == 'data':
            conndata_new['data']=conndata['data'][newsubj_idx,:]
        else:
            #check if conndata[k] is a tuple:
            
            if np.size(conndata[k])<=1:
                conndata_new[k]=conndata[k]
            elif isinstance(conndata[k],tuple):
                conndata_new[k]=deepcopy(conndata[k])
            else:
                conndata_new[k]=conndata[k].copy()
    
    return conndata_new

def merge_conndata_subjects(conndata_list):
    """
    Merge a list of conndata dicts, each with the same 'subjects' field, into a single conndata dict with all subjects concatenated
    
    Parameters:
    conndata_list: list of conndata dicts with 'subjects' and 'data' fields
    
    Returns:
    conndata_new: dict, new conndata dict with all subjects concatenated
    """
    if not ('subjects' in conndata_list[0] and 'data' in conndata_list[0]):
        #if provided a multi-flavor input, loop over the flavors
        #(make deep copy)
        conndata_new={}
        for conntype in conndata_list[0].keys():
            conndata_new[conntype]=merge_conndata_subjects([c[conntype] for c in conndata_list])
        return conndata_new
    
    #make new deep copy of data
    conndata_new={}
    for k in conndata_list[0].keys():
        if np.size(conndata_list[0][k])<=1:
            conndata_new[k]=conndata_list[0][k]
        else:
            conndata_new[k]=np.concatenate([c[k] for c in conndata_list],axis=0)
    return conndata_new

def load_hcp_data(subjects=[], conn_name_list=[], load_retest=False, quiet=False, keep_diagonal=False):
    """
    Load HCP data from a set of subjects and a list of data types, using hardcoded input paths
    
    Parameters:
    subjects: list(str) of subject IDs to load
    conn_name_list: list(str) of data types to load
    load_retest: bool (optional, default=False), if True, will load retest data instead of main data
    quiet: bool (optional, default=False), if True, will suppress print statements
    keep_diagonal: bool (optional, default=False), if True, will keep the diagonal of the connectome matrices
    
    Returns: (tuple)
    subjects: list(str) of subject IDs
    conndata_alltypes: dict['conntype'] with ['data'] as subj x edges
    """

    #conn_name_list = explicit and complete list of datatypes to load (ignore all other flavor info)

    if os.path.isdir('/Users/kwj5'):
        #datafolder='/Users/kwj5/Box/HCP_SC_FC_new997'
        datafolder='/Users/kwj5/Research/HCP'
        studyfolder='/Users/kwj5/Research/HCP'
    elif os.path.isdir('/home/kwj2001/colossus_shared/HCP'):
        studyfolder='/home/kwj2001/colossus_shared/HCP'
        datafolder='/home/kwj2001/colossus_shared/HCP'
    elif os.path.isdir('/midtier/sablab/scratch/kwj2001'):
        studyfolder='/midtier/sablab/scratch/kwj2001/HCP'
        datafolder='/midtier/sablab/scratch/kwj2001/HCP'        
    elif os.path.isdir('/home/ubuntu'):
        studyfolder='/home/ubuntu'
        datafolder='/home/ubuntu'

    subjects_orig_input=subjects
    if subjects is None or len(subjects)==0:
        subjects,_=load_hcp_subject_list(numsubj=993)

    #scfile_fields=['orig','volnorm']
    #scfile_fields=['volnorm']

    #consider: log transform on SC inputs?
    # also maybe try using orig instead of just volnorm? orig is easier to log transform (all pos vals)
    # log10(volnorm) has a lot of negatives (counts<1)
    # is there any reasonable way to justify log10(connectome)./log10(volumes) to scale?
    #  then it would be positive but scaled by the (logscaled) volumes
    # main reason this matters is that we have 0s for SC which log can't handle, and we cant just set those to
    #  log10(0)->0 because then log10(.1)-> -1 would look < 0 

    pretrained_transformer_file={}
    pretrained_transformer_file['SCifod2act_fs86_volnorm']=None
    pretrained_transformer_file['SCifod2act_shen268_volnorm']=None
    pretrained_transformer_file['SCsdstream_shen268_volnorm']=None
    pretrained_transformer_file['SCifod2act_coco439_volnorm']=None
    pretrained_transformer_file['SCsdstream_coco439_volnorm']=None
    pretrained_transformer_file['FCcorr_fs86_hpf']=None
    pretrained_transformer_file['FCcorr_fs86_hpfgsr']=None
    pretrained_transformer_file['FCpcorr_fs86_hpf']=None
    pretrained_transformer_file['FCcorr_shen268_hpf']=None
    pretrained_transformer_file['FCcorr_shen268_hpfgsr']=None
    pretrained_transformer_file['FCpcorr_shen268_hpf']=None
    pretrained_transformer_file['FCcorr_coco439_hpf']=None
    pretrained_transformer_file['FCcorr_coco439_hpfgsr']=None
    pretrained_transformer_file['FCpcorr_coco439_hpf']=None

    #build list of possijble HCP data files to load
    
    if len(conn_name_list)==0:
        fc_filter_list=["hpf","bpf","nofilt"]
    
    connfile_info=[]
    datagroup='SC'
    
    connfile_info.append({'name':'SCifod2act_fs86_volnorm','file':'%s/sc_fs86_ifod2act_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'SCsdstream_fs86_volnorm','file':'%s/sc_fs86_sdstream_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'SCifod2act_shen268_volnorm','file':'%s/sc_shen268_ifod2act_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'SCsdstream_shen268_volnorm','file':'%s/sc_shen268_sdstream_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'SCifod2act_coco439_volnorm','file':'%s/sc_cocommpsuit439_ifod2act_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'SCsdstream_coco439_volnorm','file':'%s/sc_cocommpsuit439_sdstream_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})

    datagroup='FC'
    #consider: do np.arctanh for FC inputs?

    #hpf
    connfile_info.append({'name':'FCcorr_fs86_hpf','file':'%s/fc_fs86_FCcov_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcorr_fs86_hpfgsr','file':'%s/fc_fs86_FCcov_hpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_fs86_hpf','file':'%s/fc_fs86_FCpcorr_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})

    connfile_info.append({'name':'FCcorr_shen268_hpf','file':'%s/fc_shen268_FCcov_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcorr_shen268_hpfgsr','file':'%s/fc_shen268_FCcov_hpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_shen268_hpf','file':'%s/fc_shen268_FCpcorr_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})

    connfile_info.append({'name':'FCcorr_coco439_hpf','file':'%s/fc_cocommpsuit439_FCcov_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcorr_coco439_hpfgsr','file':'%s/fc_cocommpsuit439_FCcov_hpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_coco439_hpf','file':'%s/fc_cocommpsuit439_FCpcorr_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})

    #bpf
    connfile_info.append({'name':'FCcorr_fs86_bpf','file':'%s/fc_fs86_FCcov_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcorr_fs86_bpfgsr','file':'%s/fc_fs86_FCcov_bpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_fs86_bpf','file':'%s/fc_fs86_FCpcorr_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    
    connfile_info.append({'name':'FCcorr_shen268_bpf','file':'%s/fc_shen268_FCcov_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcorr_shen268_bpfgsr','file':'%s/fc_shen268_FCcov_bpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_shen268_bpf','file':'%s/fc_shen268_FCpcorr_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    
    connfile_info.append({'name':'FCcorr_coco439_bpf','file':'%s/fc_cocommpsuit439_FCcov_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcorr_coco439_bpfgsr','file':'%s/fc_cocommpsuit439_FCcov_bpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_coco439_bpf','file':'%s/fc_cocommpsuit439_FCpcorr_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    
    #nofilt (no compcor, hp2000)
    connfile_info.append({'name':'FCcorr_fs86_nofilt','file':'%s/fc_fs86_FCcov_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcorr_fs86_nofiltgsr','file':'%s/fc_fs86_FCcov_nofilt_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_fs86_nofilt','file':'%s/fc_fs86_FCpcorr_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})

    connfile_info.append({'name':'FCcorr_shen268_nofilt','file':'%s/fc_shen268_FCcov_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcorr_shen268_nofiltgsr','file':'%s/fc_shen268_FCcov_nofilt_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_shen268_nofilt','file':'%s/fc_shen268_FCpcorr_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    
    connfile_info.append({'name':'FCcorr_coco439_nofilt','file':'%s/fc_cocommpsuit439_FCcov_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcorr_coco439_nofiltgsr','file':'%s/fc_cocommpsuit439_FCcov_nofilt_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_coco439_nofilt','file':'%s/fc_cocommpsuit439_FCpcorr_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    
    if load_retest:
        for i in range(len(connfile_info)):
            connfile_info[i]['file']=connfile_info[i]['file'].replace("_993subj.mat","_retest42.mat")
        if subjects_orig_input is None or len(subjects_orig_input)==0:
            subjects=None
        else:
            subjects=[s+"_Retest" if not s.endswith("_Retest") else s for s in subjects]
            

    conndata_alltypes={}

    if len(conn_name_list)==0:
        conn_name_list=[c['name'] for c in connfile_info]

    conn_names_available=[x['name'] for x in connfile_info]
    for i,cname in enumerate(conn_name_list):
        if cname.endswith("_volnorm"):
            connfield="volnorm"
            connsuffix="_volnorm"
            cname=cname.replace("_volnorm","")
        elif cname.endswith("_sift2"):
            connfield="sift2"
            connsuffix="_sift2count"
            cname=cname.replace("_sift2","")
        elif cname.endswith("_sift2volnorm"):
            connfield="sift2volnorm"
            connsuffix="_sift2volnorm"
            cname=cname.replace("_sift2volnorm","")
        elif cname.endswith("_count"):
            connfield="orig"
            connsuffix="_count"
            cname=cname.replace("_count","")
        elif cname.endswith("_orig"):
            connfield="orig"
            connsuffix="_orig"
            cname=cname.replace("_orig","")
        elif cname.endswith("_FC"):
            connfield="FC"
            connsuffix=""
            cname=cname.replace("_FC","")
        elif cname.startswith("FC"):
            connsuffix=""
            connfield="FC" #FC but didn't add the "_FC" to end
        elif "ifod2act" in cname or "sdstream" in cname:
            connfield="orig"
            connsuffix="_orig"
        else:
            raise Exception("Unknown data flavor for %s" % (cname))
        
        connsearch='%s%s' % (cname,connsuffix)
        connsearch=connsearch.replace("_FC","")
        ci=[x for x in connfile_info if x['name']==connsearch]
        if len(ci)==0:
            print("%s not found in file_info. Available names: %s" % (connsearch, conn_names_available))
        ci=ci[0]
        
        if not quiet:
            print("Loading %d/%d: %s" % (i+1,len(conn_name_list),ci['file']))
        Cdata=loadmat(ci['file'],simplify_cells=True)
        subjmissing=Cdata['ismissing']>0
        subjects997=Cdata['subject'][~subjmissing]
        subjects997=clean_subject_list(subjects997)
        if subjects is None:
            subjects=subjects997

        connfield_list=[connfile_info[i]['fieldname'], "SC","FC"]
        for cf in connfield_list:
            if cf in Cdata:
                connfield=cf
                break
        nroi=Cdata[connfield][0].shape[0]
        if keep_diagonal:
            trimask=np.triu_indices(nroi,0) #note: this is equivalent to tril(X,0) in matlab
        else:
            trimask=np.triu_indices(nroi,1) #note: this is equivalent to tril(X,-1) in matlab
        npairs=trimask[0].shape[0]

        Ctriu=[x[trimask] for x in Cdata[connfield][~subjmissing]]
        #restrict to 420 unrelated subjects
        Ctriu=[x for i,x in enumerate(Ctriu) if subjects997[i] in subjects]
        conn_name='%s%s' % (cname,connsuffix)
        
        transformer_file=None
        if conn_name in pretrained_transformer_file:
            transformer_file=pretrained_transformer_file[conn_name]

        conndata_alltypes[conn_name]={'data':np.vstack(Ctriu),'numpairs':npairs,'numroi':nroi,'group':ci['group'],'transformer_file':transformer_file,'subjects':subjects, 'trimask':trimask}
            
            
    nsubj=conndata_alltypes[list(conndata_alltypes.keys())[0]]['data'].shape[0]

    if not quiet:
        print("Norm scale for input data:")
        for conntype in conndata_alltypes:
            normscale=np.linalg.norm(conndata_alltypes[conntype]['data'])
            normscale/=np.sqrt(conndata_alltypes[conntype]['data'].size)
            print("%s: %.6f" % (conntype,normscale))
    
        for conntype in conndata_alltypes:
            print(conntype,conndata_alltypes[conntype]['data'].shape)
    
    return subjects, conndata_alltypes


def load_input_data(inputfile, group=None, inputfield=None, keep_diagonal=False):
    """
    Load input data from a file, with optional selection of a specific field to load
    Input field should contain a [subjects x region x region] array of square matrices for each subject
        * OR it can be [subjects x edges] if it also includes a 'trimask' field, which must be either a 
            [region x region] mask or a tuple or [2 x edges] with (rowidxs,colidxs)
    Input data CAN be an output from run_model with a 'predicted_alltypes' field, but only if there is a single input type and output type.
    Optional group name (eg: 'SC', 'FC') is useful if training regime treats intra-modal and inter-modal differently
    
    Parameters:
    inputfile: str, input file path (.mat file from matlab or scipy.io.savemat, with a field containing the data to load)
    group: str (optional), group name (default=None. Could be 'SC', 'FC', 'encoded')
    inputfield: str (optional), field name to load from the input file 
        (default=None, will try to find field in ['FC','SC','C','encoded','volnorm'] if not provided)
    keep_diagonal: bool (optional, default=False), if True, will keep the diagonal of the connectome matrices
    

    Returns:
    conndata: dict, with: 
        'data' as subj x edges
        'numpairs' as number of edges retained in the upper triangle of the square matrix
        'numroi' as number of ROIs
        'fieldname' as field name
        'group' as group name
        'subjects' as list of subject IDs
        'trimask' as edge mask (for upper triangle of square matrix)
    """
    inputfield_default_search=['data','encoded','FC','SC','C','volnorm'] #,'sift2volnorm','sift2','orig']

    Cdata=loadmat(inputfile,simplify_cells=True)
    
    if 'predicted_alltypes' in Cdata:
        #If input file is an output from a model, it will have data stored in Cdata['predicted_alltypes'][intype][outtype].
        #We can't handle multiple types in one file, but IF there is only one intype and only one outtype, we can handle it.
        #If that outtype is 'encoded', create Cdata['encoded'] = ... Otherwise, Cdata['data'] = ...
        predicted_alltypes_keys=list(Cdata['predicted_alltypes'].keys())
        predicted_alltypes_keys0_keys=list(Cdata['predicted_alltypes'][predicted_alltypes_keys[0]].keys())
        
        if len(predicted_alltypes_keys)==1 and len(predicted_alltypes_keys0_keys)==1:
            if predicted_alltypes_keys0_keys[0] == 'encoded':
                connfield='encoded'
            else:
                connfield='data'
            Cdata={connfield:Cdata['predicted_alltypes'][predicted_alltypes_keys[0]][predicted_alltypes_keys0_keys[0]]}
        else:
            raise Exception("Found multi-input or multi-output data in 'predicted_alltypes' for input file %s. Can only read single intype+output" % (inputfile))
    
    subjmissing=[]
    subjects=[]
    trimask=None
    
    if 'ismissing' in Cdata:
        subjmissing=Cdata['ismissing']>0
        
    if 'subject' in Cdata:
        subjects=Cdata['subject']
    elif 'subjects' in Cdata:
        subjects=Cdata['subjects']
    
    if 'trimask' in Cdata:
        trimask=Cdata['trimask']
    
    subjects=clean_subject_list(subjects)

    connfield=inputfield
    if not connfield:
        for itest in inputfield_default_search:
            if itest in Cdata:
                connfield=itest
                break
    
    if connfield is None:
        print("None of the following fields were found in the input file %s:" % (inputfile),inputfield_default_search)
        raise Exception("Input type not found")
    
    if len(Cdata[connfield][0].shape)<=1:
        #single matrix was in file
        Cmats=[np.atleast_2d(Cdata[connfield])]
    else:
        Cmats=Cdata[connfield]
    
    if connfield == "encoded":
        nroi=1
        npairs=Cmats[0].shape[1]
        Cdata=Cmats[0].copy()
        trimask=None
    elif trimask is not None:
        Cdata=Cmats[0].copy()
        if isinstance(trimask, tuple):
            trimask=(trimask[0],trimask[1])
        elif isinstance(trimask, np.ndarray) and trimask.shape[0]==trimask.shape[1]:
            trimask=np.where(trimask>0)
        elif len(trimask)==2:
            trimask=(trimask[0],trimask[1])
        else:
            raise Exception("Invalid trimask input. Must be [roi x roi] mask, a tuple of (rowidxs,colidxs) or a [2 x edges] array of (rowidxs,colidxs).")
        nroi=max(max(trimask[0]),max(trimask[1]))+1
        npairs=len(trimask[0])
    elif Cmats[0].shape[0] != Cmats[0].shape[1]:
        if Cmats[0].shape[1] == len(subjects):
            Cdata=Cmats[0].T
        else:
            Cdata=Cmats[0].copy()
        nroi=1
        npairs=Cdata.shape[1]
        trimask=None
    else:
        nroi=Cmats[0].shape[0]
        if keep_diagonal:
            trimask=np.triu_indices(nroi,0) #note: this is equivalent to tril(X,0) in matlab
        else:
            trimask=np.triu_indices(nroi,1) #note: this is equivalent to tril(X,-1) in matlab
        npairs=trimask[0].shape[0]
        if len(subjmissing)==0:
            subjmissing=np.zeros(len(Cmats))>0
        
        if len(subjects)>0:
            subjects=subjects[~subjmissing]
        
        Ctriu=[x[trimask] for i,x in enumerate(Cmats) if not subjmissing[i]]
        
        Cdata=np.vstack(Ctriu)
    
    #conndata_alltypes[conn_name]={'data':np.vstack(Ctriu),'numpairs':npairs,'group':ci['group'],'transformer_file':transformer_file}
    conndata={'data':Cdata,'numpairs':npairs,'numroi':nroi,'fieldname':connfield,'group':group,'subjects':subjects,'trimask':trimask}
    
    return conndata

#################################
#################################
def generate_adapt_transformer(input_data, target_data, adapt_mode='meanfit+meanshift',input_data_fitsubjmask=None, target_data_fitsubjmask=None, return_fit_info=False,
                               quiet=False, input_source_name="input"):
    """
    Generate a transformer to adapt input data domain to match training data
    
    Parameters:
    input_data: np.array, input data to use for computing fit. Could be a [1 x Nfeat] mean or [Nsubj x Nfeat] data
    target_data: np.array, target data to use for computing fit. Could be a [1 x Nfeat] mean or [Nsubj x Nfeat] data
        Or it could be a dict from generate_transformer (with a 'params' field containing a 'input_mean' or 'pca_input_mean' field)
    adapt_mode: str (optional, default='meanfit+meanshift'), mode to use for adapting input data to target data
        Options: 'meanfit', 'meanshift', 'meanfit+meanshift'
    input_data_fitsubjmask: np.array (optional), boolean mask for input data subjects to use for fitting
    target_data_fitsubjmask: np.array (optional), boolean mask for target data subjects to use for fitting
    
    Returns:
    transformer: transformer object (with .fit() and .fit_transform() methods, etc...)   
    """
    
    adaptfit_cc=None
    beta=None
    input_data_fitsubjmask=None
    target_data_fitsubjmask=None
    num_input_fitsubj=None
    num_target_fitsubj=None
    
    if adapt_mode is not None:
        #test if target_data is a dict with field 'params'
        if isinstance(target_data,dict):
            if 'params' in target_data and 'input_mean' in target_data['params']:
                target_data=np.atleast_2d(numpyvar(target_data['params']['input_mean']))
            elif 'params' in target_data and 'pca_input_mean' in target_data['params']:
                target_data=np.atleast_2d(numpyvar(target_data['params']['pca_input_mean']))
            elif 'input_mean' in target_data:
                target_data=np.atleast_2d(numpyvar(target_data['input_mean']))
            elif 'pca_input_mean' in target_data:
                target_data=np.atleast_2d(numpyvar(target_data['pca_input_mean']))
        
        if input_data_fitsubjmask is None:
            input_data_fitsubjmask=np.ones(input_data.shape[0])>0
        if target_data_fitsubjmask is None:
            target_data_fitsubjmask=np.ones(target_data.shape[0])>0
        
        num_input_fitsubj=np.sum(input_data_fitsubjmask)
        num_target_fitsubj=np.sum(target_data_fitsubjmask)
        if np.any(input_data_fitsubjmask.astype(bool)!=input_data_fitsubjmask):
            num_input_fitsubj=len(input_data_fitsubjmask)
        if np.any(target_data_fitsubjmask.astype(bool)!=target_data_fitsubjmask):
            num_target_fitsubj=len(target_data_fitsubjmask)
        
        input_data_mean=np.atleast_2d(np.mean(input_data[input_data_fitsubjmask,:],axis=0,keepdims=True))
        target_data_mean=np.atleast_2d(np.mean(target_data[target_data_fitsubjmask,:],axis=0,keepdims=True))
    
    if adapt_mode is None:
        transformer=FunctionTransformer(func=lambda x:torchfloat(x),
                                        inverse_func=lambda x:torchfloat(x))
    elif adapt_mode.lower() == 'meanshift':
        transformer=FunctionTransformer(func=lambda x:torchfloat(x - input_data_mean + target_data_mean),
                                        inverse_func=lambda x:torchfloat(x - target_data_mean + input_data_mean))
        adaptfit_cc=np.corrcoef(input_data_mean,target_data_mean)[0,1]
        print("\tShifting %s data mean to transformer mean" % (input_source_name))
        print("\tInput data mean for adapt has %d subjects." % (num_input_fitsubj))
        print("\tAdapted fit R2: %.3f" % (adaptfit_cc**2))
        
    elif adapt_mode.lower() == 'meanfit+meanshift' or adapt_mode.lower() == 'meanfitshift':
        A=np.vstack((input_data_mean,np.ones(input_data_mean.shape)))
        beta=np.linalg.lstsq(A.T,target_data_mean.T,rcond=None)[0].flatten()
        adaptfit_cc=np.corrcoef(beta.T@A,target_data_mean)[0,1]
        print("\tFitting %s data mean to transformer mean: modeldata=inputdata*%.3f + %.3f" % (input_source_name, beta[0], beta[1]))
        print("\tInput data mean for adapt has %d subjects." % (num_input_fitsubj))
        print("\tAdapted fit R2: %.3f" % (adaptfit_cc**2))
        #print("\tShifting input data mean to transformer mean: %s" % (x))
        transformer=FunctionTransformer(func=lambda x:torchfloat(x*beta[0] - input_data_mean*beta[0] + target_data_mean),
                                        inverse_func=lambda x:torchfloat(x/beta[0] - target_data_mean/beta[0] + input_data_mean))
        
    elif adapt_mode.lower() == 'meanfit':
        #use np.linalg.lstsq to least squares fit of actual_data_mean to transformer_data_mean
        A=np.vstack((input_data_mean,np.ones(input_data_mean.shape)))
        beta=np.linalg.lstsq(A.T,target_data_mean.T,rcond=None)[0].flatten()
        adaptfit_cc=np.corrcoef(beta.T@A,target_data_mean)[0,1]
        print("\tFitting %s data mean to transformer mean: modeldata=inputdata*%.3f + %.3f" % (input_source_name, beta[0], beta[1]))
        print("\tInput data mean for adapt has %d subjects." % (num_input_fitsubj))
        print("\tAdapted fit R2: %.3f" % (adaptfit_cc**2))
        transformer=FunctionTransformer(func=lambda x:torchfloat(x*beta[0] + beta[1]),
                                        inverse_func=lambda x:torchfloat(x/beta[0]-beta[1]/beta[0]))
    else:
        raise Exception("Unknown adapt_mode: %s" % (adapt_mode))

    if return_fit_info:
        fit_info={"adaptfit_cc":adaptfit_cc,    
                  "beta":beta,
                  "input_data_fitsubjmask":input_data_fitsubjmask,
                  "target_data_fitsubjmask":target_data_fitsubjmask,
                  "input_data_fitsubjmask_count":num_input_fitsubj,
                  "target_data_fitsubjmask_count":num_target_fitsubj,
                  "adapt_mode":adapt_mode}
        return transformer, fit_info
    else:
        return transformer
        
    
#################################
#################################
#create data input/output transformers, datasets/dataloaders, and TRAINING PATHS

def generate_transformer(traindata=None, transformer_type=None, transformer_param_dict=None, precomputed_transformer_params=None, return_components=True):
    """
    Generate an sklearn transformer object for input/output data, with fit(), fit_transform(), and inverse_transform() methods
    Many transformations were implemented during testing, but 'pca', 'cfeat', and 'none' are the types we focus on
    
    NOTE: to regenerate a transformer from the output of this function:
    transformer, transformer_info = generate_transformer(transformer_type=transformer_info["params"]["type"], precomputed_transformer_params=transformer_info["params"])
    
    Parameters:
    traindata: np.array (optional), training data to fit the transformer on
    transformer_type: str, type of transformer to use (eg: 'pca', 'cfeat', 'none')
    transformer_param_dict: dict (optional), dictionary of parameters for the transformer (eg: {'reduce_dimension':256})
    precomputed_transformer_params: dict (optional), dictionary of precomputed transformer parameters
    return_components: bool (optional, default=True), if True, will return the components of the transformer (eg: PCA weights)
    
    Returns:
    transformer: transformer object (with .fit() and .fit_transform() methods, etc...)
    transformer_info: dict, with 'type' and 'params' fields
    """
    if transformer_param_dict:
        transformer_info=transformer_param_dict
    else:
        transformer_info={}
    
    if precomputed_transformer_params and precomputed_transformer_params["type"] != transformer_type:
        raise Exception("Precomputed transformer was %s, expected %s" % (precomputed_transformer_params["type"],transformer_type))
    
    input_dimension=None
    output_dimension=None
    if traindata is not None:
        input_dimension=traindata.shape[1]
        output_dimension=traindata.shape[1]
        
    if transformer_type == "none":
        transformer=FunctionTransformer(func=lambda x:torchfloat(x),
                                        inverse_func=lambda x:torchfloat(x))
        transformer_info["type"]="none"
        transformer_info["params"]={}
    
    elif transformer_type == "pca":
        #use PCA to reduce dimensionality of input data
        
        if precomputed_transformer_params:
            if transformer_param_dict and precomputed_transformer_params["reduce_dimension"] != transformer_param_dict['reduce_dimension']:
                raise Exception("Precomputed transformer dimension was %d, expected %d" % (precomputed_transformer_params["reduce_dimension"],
                    transformer_param_dict['reduce_dimension']))
            data_normscale=precomputed_transformer_params['input_normscale']
            normscale=precomputed_transformer_params['output_normscale']
            pca_components=precomputed_transformer_params['pca_components']
            pca_input_mean=precomputed_transformer_params['pca_input_mean']
            pca_dim=precomputed_transformer_params["reduce_dimension"]
            if 'pca_explained_variance_ratio' in precomputed_transformer_params:
                pca_explained_variance_ratio=precomputed_transformer_params['pca_explained_variance_ratio']
            else:
                pca_explained_variance_ratio=np.nan

        else:
            pca_xform=PCA(n_components=transformer_param_dict['reduce_dimension'],random_state=0).fit(traindata)
            data_normscale=np.linalg.norm(pca_xform.transform(traindata))
            normscale=100
            
            pca_components=pca_xform.components_
            pca_input_mean=pca_xform.mean_
            pca_dim=transformer_param_dict['reduce_dimension']
            pca_explained_variance_ratio=np.sum(pca_xform.explained_variance_ratio_)

        #just make it explicitly a lambda function instead of using the PCA.transform() so we can be certain it's reproducible during
        #training and later evaluation
        #Xpc = np.dot(X-pca_input_mean,pca_components.T)
        #Xnew = np.dot(Xpc,pca_components)+pca_input_mean
        
        input_dimension=pca_components.shape[1]
        output_dimension=pca_components.shape[0]
            
        data_normscale=torchfloat(data_normscale)
        normscale=torchfloat(normscale)
        pca_input_mean=torchfloat(pca_input_mean)
        pca_components=torchfloat(pca_components)
        
        transformer=FunctionTransformer(func=lambda x:normscale*(torch.mm(torchfloat(x)-pca_input_mean,pca_components.T)/data_normscale),
                                        inverse_func=lambda x:torch.mm((torchfloat(x)/normscale)*data_normscale,pca_components)+pca_input_mean)
        transformer_info["type"]="pca"
        transformer_info["params"]={"reduce_dimension":pca_dim,
                                    "input_normscale":data_normscale, 
                                    "output_normscale":normscale,
                                    "pca_explained_variance_ratio": pca_explained_variance_ratio}
        
        transformer_info["params"]["pca_input_mean"]=pca_input_mean
        if return_components:
            transformer_info["params"]["pca_components"]=pca_components
            
            
    elif transformer_type == "tsvd":
        #TruncatedSVD is similar to PCA but may work better for sparse inputs (eg: SC matrices)
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

        input_dimension=tsvd_components.shape[1]
        output_dimension=tsvd_components.shape[0]
        
        data_normscale=torchfloat(data_normscale)
        normscale=torchfloat(normscale)
        tsvd_components=torchfloat(tsvd_components)
        
        #use lambda with component data instead of TSVD.transform() to be certain training and evaluation are the same
        #Xtsvd = np.dot(X,tsvd_components.T)
        #Xnew = np.dot(Xtsvd,tsvd_components)
        transformer=FunctionTransformer(func=lambda x:normscale*(torch.mm(torchfloat(x),tsvd_components.T)/data_normscale),
                                        inverse_func=lambda x:(torch.mm(torchfloat(x),tsvd_components)/normscale)*data_normscale)
        
        transformer_info["type"]="tsvd"
        transformer_info["params"]={"reduce_dimension":tsvd_dim,
                                    "input_normscale":data_normscale, 
                                    "output_normscale":normscale}
        if return_components:
            transformer_info["params"]["tsvd_components"]=tsvd_components
            
    elif transformer_type == "norm":
        #normalize input data by the total L2 norm of the [trainsubj x pairs] for that type, and scale output by a constant
        #so that all input types have the same overall scale
        if precomputed_transformer_params:
            data_normscale=precomputed_transformer_params['input_normscale']
            normscale=precomputed_transformer_params['output_normscale']
        else:
            #normalize each SC matrix by the total L2 norm of the [trainsubj x pairs] for that type
            data_normscale=np.linalg.norm(traindata)
            data_normscale/=np.sqrt(traindata.size)

            normscale=100
            
        data_normscale=torchfloat(data_normscale)
        normscale=torchfloat(normscale)
        
        transformer=FunctionTransformer(func=lambda x:normscale*(torchfloat(x)/data_normscale),
                                        inverse_func=lambda x:(torchfloat(x)/normscale)*data_normscale)
        
        transformer_info["type"]="norm"
        transformer_info["params"]={"input_normscale":data_normscale, 
                                    "output_normscale":normscale}
    
    elif transformer_type == "varnorm":
        #normalize variance across each feature (column, SC/FC edge), summed across all columns
        if precomputed_transformer_params:
            data_var=precomputed_transformer_params['input_variance']
        else:
            #variance across each feature (column), summed across all columns
            data_var=np.sum(np.var(traindata,axis=0))

        data_var=torchfloat(data_var)
        transformer=FunctionTransformer(func=lambda x:torchfloat(x)/data_var,
                                        inverse_func=lambda x:torchfloat(x)*data_var)
        
        transformer_info["type"]="varnorm"
        transformer_info["params"]={"input_variance":data_var}
    
    elif transformer_type == "zscore":
        #z-score each input dataset by the overall mean and L2 norm of the entire [trainsubj x pairs] for that input flavor
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
            data_stdev=precomputed_transformer_params['input_stdev']
        else:
            data_mean=np.mean(traindata)
            data_stdev=np.std(traindata)

        data_mean=torchfloat(data_mean)
        data_stdev=torchfloat(data_stdev)
        
        transformer=FunctionTransformer(func=lambda x:(torchfloat(x)-data_mean)/data_stdev,
                                        inverse_func=lambda x:(torchfloat(x)*data_stdev)+data_mean)
        
        transformer_info["type"]="zscore"
        transformer_info["params"]={"input_mean":data_mean, 
                                    "input_stdev":data_stdev}
                                    
    elif transformer_type == "zfeat":
        #z-score each FEATURE (column, SC/FC edge) separately
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
            data_stdev=precomputed_transformer_params['input_stdev']
        else:
            data_mean=np.mean(traindata,axis=0,keepdims=True)
            data_stdev=np.std(traindata,axis=0,keepdims=True)
            data_stdev[data_stdev==0]=1.0
        
        input_dimension=data_mean.shape[1]
        output_dimension=data_mean.shape[1]
        
        data_mean=torchfloat(data_mean)
        data_stdev=torchfloat(data_stdev)
        
        transformer=FunctionTransformer(func=lambda x:(torchfloat(x)-data_mean)/data_stdev,
                                        inverse_func=lambda x:(torchfloat(x)*data_stdev)+data_mean)
        
        transformer_info["type"]="zfeat"
        transformer_info["params"]={"input_mean":data_mean, 
                                    "input_stdev":data_stdev}
        
    elif transformer_type == "cfeat":
        #demean each FEATURE (column, SC/FC edge) based on training data, but don't rescale
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
        else:
            
            data_mean=np.mean(traindata,axis=0,keepdims=True)
        
        input_dimension=data_mean.shape[1]
        output_dimension=data_mean.shape[1]
        
        data_mean=torchfloat(data_mean)
        transformer=FunctionTransformer(func=lambda x:torchfloat(x)-data_mean,
                                        inverse_func=lambda x:torchfloat(x)+data_mean)
        
        transformer_info["type"]="cfeat"
        transformer_info["params"]={"input_mean":data_mean}
        
    elif transformer_type == "cfeat+norm":
        #demean each FEATURE (column, SC/FC edge) based on training, then normalize by the total L2 norm of the entire [trainsubj x pairs] for that type
        #so that all features are centered, and the overall scale is consistent across input flavors
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
            data_rownorm_mean=precomputed_transformer_params['input_rownorm_mean']
        else:
            #demean each FEATURE (column) based on training
            data_mean=np.mean(traindata,axis=0,keepdims=True)
            data_rownorm_mean=np.mean(np.sqrt(np.sum((traindata-data_mean)**2,axis=1)))
        
        input_dimension=data_mean.shape[1]
        output_dimension=data_mean.shape[1]
        
        data_mean=torchfloat(data_mean)
        data_rownorm_mean=torchfloat(data_rownorm_mean)
        
        transformer=FunctionTransformer(func=lambda x:(torchfloat(x)-data_mean)/data_rownorm_mean,
                                        inverse_func=lambda x:torchfloat(x)*data_rownorm_mean+data_mean)
        
        transformer_info["type"]="cfeat+norm"
        transformer_info["params"]={"input_mean":data_mean,"input_rownorm_mean":data_rownorm_mean}

    elif transformer_type == "cfeat+varnorm":
        if precomputed_transformer_params:
            data_mean=precomputed_transformer_params['input_mean']
            data_var=precomputed_transformer_params['input_variance']
        else:
            #demean each FEATURE (column) based on training
            data_mean=np.mean(traindata,axis=0,keepdims=True)
            data_var=np.sum(np.var(traindata-data_mean,axis=0))

        input_dimension=data_mean.shape[1]
        output_dimension=data_mean.shape[1]
        
        data_mean=torchfloat(data_mean)
        data_var=torchfloat(data_var)
        
        transformer=FunctionTransformer(func=lambda x:(torchfloat(x)-data_mean)/data_var,
                                        inverse_func=lambda x:torchfloat(x)*data_var+data_mean)
        
        transformer_info["type"]="cfeat+varnorm"
        transformer_info["params"]={"input_mean":data_mean,"input_variance":data_var}
    
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
            
        data_mean=torchfloat(data_mean)
        data_stdev=torchfloat(data_stdev)
        data_z_rownorm_mean=torchfloat(data_z_rownorm_mean)
        
        transformer=FunctionTransformer(func=lambda x:(torchfloat(x)-data_mean)/(data_stdev*data_z_rownorm_mean),
                                        inverse_func=lambda x:(torchfloat(x)*data_stdev*data_z_rownorm_mean)+data_mean)
        
        transformer_info["type"]="zscore+rownorm"
        transformer_info["params"]={"input_mean":data_mean, 
                                    "input_stdev":data_stdev,
                                    "input_rownorm_mean":data_z_rownorm_mean}
        
    elif transformer_type == "lognorm+rownorm":
        #map data x->log(x), scaling mean/std (excluding zeros) to 0.5, .1 to make SC SOMEWHAT closer to FC
        if precomputed_transformer_params:
            min_nonzero=precomputed_transformer_params['input_minimum_nonzero']
            logmean_nonzero=precomputed_transformer_params['input_logmean_nonzero']
            logstd_nonzero=precomputed_transformer_params['input_logstdev_nonzero']
            logmean_new=precomputed_transformer_params['output_logmean_nonzero']
            logstd_new=precomputed_transformer_params['output_logstdev_nonzero']
            data_rownorm_mean=precomputed_transformer_params['input_rownorm_mean']
        else:
            min_nonzero=np.min(traindata[traindata>0])
        
            logmean_nonzero=np.mean(np.log10(traindata[traindata>min_nonzero]))
            logstd_nonzero=np.std(np.log10(traindata[traindata>min_nonzero]))

            logmean_new=.5
            logstd_new=.1

            lognorm_func = lambda x: (((np.log10(np.clip(x,min_nonzero,None))-logmean_nonzero)/logstd_nonzero)*logstd_new+logmean_new)*(x>=min_nonzero)

            data_rownorm_mean=np.mean(np.sqrt(np.sum(lognorm_func(traindata)**2,axis=1)))
        
        min_nonzero=torchfloat(min_nonzero)
        logmean_nonzero=torchfloat(logmean_nonzero)
        logstd_nonzero=torchfloat(logstd_nonzero)
        logmean_new=torchfloat(logmean_new)
        logstd_new=torchfloat(logstd_new)
        data_rownorm_mean=torchfloat(data_rownorm_mean)
        
        lognorm_func = lambda x: (((torch.log10(torch.clip(torchfloat(x),min_nonzero,None))-logmean_nonzero)/logstd_nonzero)*logstd_new+logmean_new)*(torchfloat(x)>=min_nonzero)/data_rownorm_mean
        
        #for setting x<min_nonzero to 0... how?
        lognorm_inv_func = lambda y: torch.clip(10**(((torchfloat(y)*data_rownorm_mean-logmean_new)/logstd_new)*logstd_nonzero+logmean_nonzero),0,None)
        
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
    transformer_info["params"]["input_dimension"]=input_dimension
    transformer_info["params"]["output_dimension"]=output_dimension
    
    
    return transformer, transformer_info

def load_transformers_from_file(input_transform_file_list, 
                                input_names=None, 
                                quiet=False):
    """
    Load precomputed input transformers from a list of files
    
    Parameters:
    input_transform_file_list: list of str, list of input transformer files (kraken_ioxfm_*.npy) to load
    input_names: list of str (optional), list of input names to load (default=None, load all)
    
    Returns (tuple):
    transformer_list: dict, with input names as keys and transformer objects as values
    transformer_info_list: dict, with input names as keys and transformer information/params as values
    """
    if isinstance(input_transform_file_list,str):
        input_transform_file_list=[input_transform_file_list]
    transformer_list={}
    transformer_info_list={}
    for ioxfile in input_transform_file_list:
        ioxfile=replace_data_folder_placeholder(ioxfile)
        if not quiet:
            print("Loading precomputed input transformations: %s" % (ioxfile))
        if ioxfile.lower().endswith('.npy'):
            ioxtmp=np.load(ioxfile,allow_pickle=True).item()
        elif ioxfile.lower().endswith('.npz'):
            ioxtmp=np.load(ioxfile,allow_pickle=True)["transform_params"].item()
        elif ioxfile.lower().endswith('.h5'):
            ioxtmp=load_h5_to_dict(ioxfile)
        else:
            raise ValueError("Unsupported file format (accepts .npy, .npz, .h5): %s" % (ioxfile))
        
        for conntype in ioxtmp:
            if input_names is not None and conntype not in input_names:
                continue
            transformer_list[conntype], transformer_info_list[conntype] = generate_transformer(transformer_type=ioxtmp[conntype]['type'],  
                                                                                               precomputed_transformer_params=ioxtmp[conntype], 
                                                                                               return_components=True)
            transformer_info_list[conntype]['filename']=ioxfile.split(os.sep)[-1]
            transformer_info_list[conntype]['filepath']=os.path.abspath(ioxfile)
            transformer_info_list[conntype]['fromfile']=True
    return transformer_list, transformer_info_list

def save_transformers_to_file(input_transformer_file, 
                              transformer_info_list, 
                              input_names=None, 
                              extra_params={}, 
                              h5_compression="gzip",
                              save_even_if_fromfile=False,
                              quiet=False):
        """
        Save the transformer information to a file
        
        Parameters:
        input_transformer_file: str, path to the input transformer file
        transformer_info_list: dict, transformer information to save
        input_names: list of str (optional), list of input names to save (default=None, save all)
        extra_params: dict, additional parameters to save
        h5_compression: str, compression method for HDF5 files (default="gzip")
        save_even_if_fromfile: bool, whether to save even if the transformer was loaded from a file (default=False)
        """
        transformer_params_to_save={}
        for k_iox in transformer_info_list.keys():
            if input_names is not None and k_iox not in input_names:
                continue
            if 'fromfile' in transformer_info_list[k_iox] and transformer_info_list[k_iox]['fromfile'] and not save_even_if_fromfile:
                #don't save these, since they are already saved to a file we loaded
                continue
            transformer_params_to_save[k_iox]=transformer_info_list[k_iox]["params"]
            for kk,kv in transformer_info_list[k_iox]["params"].items():
                if torch.is_tensor(kv):
                    transformer_params_to_save[k_iox][kk]=kv.cpu().numpy()
                else:
                    transformer_params_to_save[k_iox][kk]=kv
            transformer_params_to_save[k_iox]["type"]=transformer_info_list[k_iox]["type"]
            for kk,kv in extra_params.items():
                transformer_params_to_save[k_iox][kk]=kv
        
        if len(transformer_params_to_save) == 0:
            return
        
        if input_transformer_file.lower().endswith(".npy"):
            np.save(input_transformer_file, transformer_params_to_save)
        elif input_transformer_file.lower().endswith(".npz"):
            np.savez_compressed(input_transformer_file, transform_params=transformer_params_to_save)
        elif input_transformer_file.lower().endswith(".h5"):
            save_dict_to_h5(input_transformer_file,transformer_params_to_save,compression=h5_compression)
        else:
            del transformer_params_to_save #clear up memory
            raise ValueError("Unsupported file format (accepts .npy, .npz, .h5): %s" % (input_transformer_file))
        
        del transformer_params_to_save #clear up memory
        
        if not quiet:
            print("Saved transforms: %s" % (input_transformer_file))