"""
Functions related to loading and manipulating input data WITHOUT requiring pytorch
"""

from ._resources import resource_path
import numpy as np
from scipy.io import loadmat, savemat

import os

import re
import json
import pandas as pd
from copy import deepcopy
import io
import zipfile
from tqdm.auto import tqdm


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

def flavor_to_bids_string(flavor):
    atlasname = ""
    participant_id = ""
    subject = ""
    flavor_prefix = ""
    flavor_suffix = ""
    
    meas_str = ""
    desc_str = ""

    if "fccorr" in flavor.lower():
        meas_str += "FCcorr"
    elif "fcpcorr" in flavor.lower():
        meas_str += "FCpcorr"
    elif "ifod2act" in flavor.lower():
        meas_str = "SCifod2act"
    elif "sdstream" in flavor.lower():
        meas_str = "SCsdstream"
    else:
        raise Exception("Unknown meas: %s" % (flavor))

    if "hpf" in flavor.lower():
        meas_str += "HPF"
    elif "bpf" in flavor.lower():
        meas_str += "BPF"
    elif "nofilt" in flavor.lower():
        meas_str += "NF"

    if "gsr" in flavor.lower():
        meas_str += "GSR"

    if "sift2count" in flavor.lower() or "sift2_count" in flavor.lower() or flavor.lower().endswith("sift2"):
        meas_str += "SiftCount"
    elif "sift2volnorm" in flavor.lower() or "sift2_volnorm" in flavor.lower():
        meas_str += "SiftVN"
    elif "volnorm" in flavor.lower():
        meas_str += "VN"
    elif "count" in flavor.lower():
        meas_str += "Count"

    atlasname=flavor.split("_")[1].lower()
    
    bids_str = "atlas-%s_meas-%s" % (atlasname, meas_str)
    if desc_str:
        bids_str += "_desc-%s" % (desc_str)

    return bids_str


def parse_bids_string(bids_str):
    bids_str = os.path.basename(bids_str)
    bids_str = bids_str.split(".")[0]
    bids_parts = bids_str.split("_")

    atlasname = ""
    participant_id = ""
    subject = ""
    flavor_prefix = ""
    flavor_suffix = ""

    for s in bids_parts:
        if "-" in s:
            s_name = s.split("-")[0].lower()
            s_val = "-".join(s.split("-")[1:])
            s_val_lower = s_val.lower()
            if s_name == "sub":
                participant_id = s
                subject = s_val

            elif s_name == "atlas" or s_name.lower() == "seg":
                atlasname = s_val.lower()

            elif s_name == "meas":
                if any([s_val_lower.startswith(p) for p in ["fccorr", "fccov"]]):
                    flavor_prefix = "FCcorr_"
                elif s_val_lower.startswith("fcpcorr"):
                    flavor_prefix = "FCpcorr_"
                elif s_val_lower.startswith("scifod2act"):
                    flavor_prefix = "SCifod2act_"
                elif s_val_lower.startswith("scsdstream"):
                    flavor_prefix = "SCsdstream_"
                else:
                    raise Exception("Unknown meas-: %s" % (s_val))

                if flavor_prefix.startswith("FC"):
                    if "hpf" in s_val_lower:
                        flavor_suffix = "_hpf"
                    elif "bpf" in s_val_lower:
                        flavor_suffix = "_bpf"
                    elif s_val_lower.endswith("nf") or s_val_lower.endswith("nfgsr"):
                        flavor_suffix = "_nofilt"
                    if "gsr" in s_val_lower:
                        flavor_suffix += "gsr"
                elif flavor_prefix.startswith("SC"):
                    if "siftvn" in s_val_lower:
                        flavor_suffix = "_sift2volnorm"
                    elif "vn" in s_val_lower:
                        flavor_suffix = "_volnorm"
                    elif "siftcount" in s_val_lower:
                        flavor_suffix = "_sift2"
                    elif "count" in s_val_lower:
                        flavor_suffix = "_count"
                        

    flavorname = flavor_prefix + atlasname + flavor_suffix

    return {
        "participant_id": participant_id,
        "subject": subject,
        "inputtype": flavorname,
    }


def save_data_zip(
    filename, conndata_squaremats, participants_info, bids_desc=None, verbose=False, filetype='tsv'
):
    desc_str = ""
    if bids_desc is not None:
        desc_str = "_desc-%s" % (bids_desc)
    zipargs = {"compression": zipfile.ZIP_DEFLATED, "compresslevel": 6}
    filecount=0
    totalfilecount=len(conndata_squaremats) * len(participants_info)
    pbar=tqdm(total=totalfilecount, desc="Saving data to zip")
    with zipfile.ZipFile(filename, "w", **zipargs) as zip_ref:
        outfile = io.BytesIO()
        participants_info.to_csv(outfile, sep="\t", index=None)
        outfile.seek(0)
        zip_ref.writestr("participants.tsv", outfile.getvalue())
        for conntype in conndata_squaremats:
            for i, conndata in enumerate(conndata_squaremats[conntype]):
                outfile = io.BytesIO()
                if filetype.lower() == 'mat':
                    savemat(outfile, {'data': conndata.astype(np.float32)}, format="5", do_compression=True)
                    conn_ext='.mat'
                elif filetype.lower() == 'tsv':
                    np.savetxt(outfile, conndata, fmt="%.6f", delimiter="\t")
                    conn_ext='.tsv'
                else:
                    raise ValueError("Unsupported file type: %s" % (filetype))
                outfile.seek(0)
                subjid = participants_info["participant_id"][i]
                if subjid.startswith("sub-"):
                    subjid = subjid[4:]
                conndata_filename = "sub-%s_%s%s_relmat.dense%s" % (
                    subjid,
                    flavor_to_bids_string(conntype),
                    desc_str,
                    conn_ext
                )
                filecount+=1
                #if verbose:
                #    tqdm.write("Adding %s" % (conndata_filename))
                pbar.set_description("Adding %s" % (conndata_filename))
                pbar.update(1)
                zip_ref.writestr(conndata_filename, outfile.getvalue())
    pbar.close()


def load_data_zip(filename, filebytes=None, allowed_extensions=None, conntypes_to_load=None, just_read_info=False):
    # output: conndata_squaremats['conntype']=list([roi x roi])
    #         participants_info
    conndata_squaremats = {}
    conndata_participants = {}
    participants_info = None
    
    if filebytes is not None:
        filename_or_filebytes = filebytes
    else:
        filename_or_filebytes = filename

    if allowed_extensions is None:
        allowed_extensions_in_zip = None
    else:
        allowed_extensions_in_zip = set(allowed_extensions) - set(["zip"])
        if len(allowed_extensions_in_zip) == 0:
            allowed_extensions_in_zip = None
    
    with zipfile.ZipFile(filename_or_filebytes, "r") as zip_ref:
        # if a bids-style participants info file was included, read this in separately
        participants_tmp = [
            z for z in zip_ref.namelist() if os.path.basename(z) == "participants.tsv"
        ]
        participants_info = None

        if len(participants_tmp) > 0:
            participants_info = pd.read_table(
                zip_ref.open(participants_tmp[0]), delimiter="\t"
            )

        if participants_info is None:
            participants_list = [
                parse_bids_string(z)["participant_id"] for z in zip_ref.namelist()
            ]
            participants_list = np.unique(
                [s for s in participants_list if s is not None]
            )
            participants_info = pd.DataFrame({"participant_id": participants_list})

        zip_namelist=zip_ref.namelist()
        zipnames_conntype=[parse_bids_string(zfile)['inputtype'] for zfile in zip_namelist]
        
        if conntypes_to_load is not None:
            zip_namelist=[zfile for conntype,zfile in zip(zipnames_conntype,zip_namelist) if conntype in conntypes_to_load]
        
        totalfiles=len(zip_namelist)
        
        if not just_read_info:
            pbar=tqdm(total=totalfiles, dynamic_ncols=True, leave=False, position=0)
        for zfile in zip_namelist:
            if os.path.basename(zfile) == "participants.tsv":
                # skip participants info in main data loop
                continue
            if allowed_extensions_in_zip is not None:
                if not any(
                    [zfile.lower().endswith(ext) for ext in allowed_extensions_in_zip]
                ):
                    continue
            bids_result = parse_bids_string(zfile)
            conntype = bids_result["inputtype"]
            subject = bids_result["participant_id"]
            
            if conntypes_to_load is not None and not conntype in conntypes_to_load:
                #skip this file because it isn't one of the requested conntypes
                continue
            
            if just_read_info:
                data_tmp=[]
            else:
                with zip_ref.open(zfile) as zfile_bytes:
                    if zfile.lower().endswith(".csv"):
                        data_tmp = np.loadtxt(zfile_bytes, delimiter=",",comments=['#','!','%'])

                    elif zfile.lower().endswith(".tsv"):
                        data_tmp = np.loadtxt(zfile_bytes, delimiter="\t",comments=['#','!','%'])

                    elif zfile.lower().endswith(".mat"):
                        matdata = loadmat(zfile_bytes, simplify_cells=True)
                        matfields = ["data", "C", "SC", "FC"]
                        for m in matfields:
                            if m in matdata:
                                data_tmp = matdata[m]
                                break
                    else:
                        # unrecognized file format in zip (could be a random OS file like .DS_Store)
                        continue
            if conntype not in conndata_squaremats:
                conndata_squaremats[conntype] = []
                conndata_participants[conntype] = []
            conndata_squaremats[conntype].append(data_tmp)
            conndata_participants[conntype].append(subject)
            if not just_read_info:
                pbar.update(1)
        if not just_read_info:
            pbar.close()
        
        # now reorder all conndata entries to the same subject order
        participants_info = participants_info.drop_duplicates(
            subset=["participant_id"]
        ).reset_index(drop=True)
        participants_list = participants_info["participant_id"].values

        conndata_subjidx = {}
        for conntype in conndata_squaremats:
            sidx = [
                np.where(np.array(conndata_participants[conntype]) == s)[0][0]
                for s in participants_list
            ]
            conndata_participants[conntype] = [
                conndata_participants[conntype][i] for i in sidx
            ]
            conndata_squaremats[conntype] = [
                conndata_squaremats[conntype][i] for i in sidx
            ]

    return conndata_squaremats, participants_info

def load_data_square2tri(conndata_squaremats, subjects=None, group=None, keep_diagonal=False):
    subjmissing=[]
    connfield='C'
    if len(conndata_squaremats.shape)<=1:
        #single matrix was in file
        Cmats=[np.atleast_2d(conndata_squaremats)]
    else:
        Cmats=conndata_squaremats

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