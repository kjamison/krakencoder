"""
Miscellaneous utility functions
"""

import torch
import numpy as np
import random
import os
import scipy.interpolate
import h5py

from ._version import __version__, __version_date__

try:
    from importlib.resources import files as resource_files  # Py ≥ 3.9
except ImportError:
    from importlib_resources import files as resource_files  # backport for 3.8

def getscriptdir():
    """Return the directory that contains this script"""
    return str(resource_files(__package__))

def set_random_seed(seed):
    """Set random seed for torch, numpy, and random modules"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def numpyvar(x):
    """Convert a torch tensor to a numpy array, or return the input if it is not a tensor"""
    if not torch.is_tensor(x):
        return x
    return x.cpu().detach().numpy()

def torchvar(x, astype=None, requires_grad=False):
    """cast variable to torch (use cuda if available), with optional type conversion"""
    if torch.is_tensor(x):
        if astype is int:
            return x.clone().detach().int().requires_grad_(requires_grad)
        elif astype is float:
            return x.clone().detach().float().requires_grad_(requires_grad)
        else:
            return x.clone().detach().requires_grad_(requires_grad)
    
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
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    if np.issubdtype(astype,np.integer):
        torchtype = torch.int
        if np.issubdtype(input_type, np.unsignedinteger):
            x=x.astype(int)
    else:
        torchtype = torch.float
    
    if islist:
        return torch.tensor(x, dtype=torchtype, device=device, requires_grad=requires_grad)
    else:
        return torch.tensor([x], dtype=torchtype, device=device, requires_grad=requires_grad)

def torchfloat(x,requires_grad=False):
    return torchvar(x,astype=float,requires_grad=requires_grad)

def torchint(x,requires_grad=False):
    return torchvar(x,astype=int,requires_grad=requires_grad)

def explained_variance_ratio(x_true,x_predicted,axis=0, var_true=None):
    """
    compute R2 the same way we would for PCA recon:
    sum(variance of the residual of each FEATURE) / sum(variance of the each feature in TRUE)
    """
    
    if torch.is_tensor(x_true):
        if var_true is None:
            var_true=torch.sum(torch.var(x_true,axis=axis))
        var_resid=torch.sum(torch.var(x_true-x_predicted,axis=axis))
    else:
        if var_true is None:
            var_true=np.sum(np.var(x_true,axis=axis))
        var_resid=np.sum(np.var(x_true-x_predicted,axis=axis))
    return 1-var_resid/var_true

def nanargmax_safe(x,nanval=-1,**kwargs):
    """Return the index of the maximum value in the array, ignoring NaNs, or return nanval if all elements are NaNs"""
    try:
        return np.nanargmax(x, **kwargs)
    except ValueError:
        if nanval is not None:
            nanval=np.int64(nanval)
        # Handle the case where all elements are NaNs
        if 'axis' in kwargs:
            # Calculate the shape of the expected output
            shape = list(x.shape)
            del shape[kwargs['axis']]
            return np.full(shape, nanval)
        else:
            return nanval

def naninterp(x,outliermat=None,axis=0,fill_value=0):
    """linearly interpolate segments of data with nans (or outliermat!=0)
    
    Parameters:
    x: 2d array
    outliermat: 2d array of same shape as x, with 1 for outlier, 0 for normal, default=None (only interp nans)
    axis: axis along which to interpolate, default=0
    fill_value: value to use for out-of-bounds values, default=0
    
    Returns:
    xnew: 2d array with interpolated values
    """
    if axis == 0:
        do_transpose=False
    elif axis == 1:
        do_transpose=True
        x=x.T
    else:
        raise Exception("Only 2d supported for now")
    
    allnan_full=np.all(np.isnan(x),axis=0)
    x=x[:,~allnan_full]
    notnan=~np.any(np.isnan(x),axis=1)
    if outliermat is not None:
        notnan[np.sum(np.abs(outliermat),axis=1)>0]=False
    notnanidx=np.where(notnan)[0]
    
    if len(notnanidx)==0:
        xnew=np.ones(x.shape)*fill_value
    elif len(notnanidx)==1:
        xnew=np.ones(x.shape)*fill_value
        xnew[notnanidx,:]=x[notnanidx,:]
    else:
        xnew=scipy.interpolate.interp1d(notnanidx,x[notnanidx,:],axis=0,bounds_error=False,fill_value=fill_value)(np.arange(x.shape[0]))
    
    if np.any(allnan_full):
        xnew_full=fill_value*np.ones((xnew.shape[0],len(allnan_full)))
        xnew_full[:,~allnan_full]=xnew
        xnew=xnew_full
    
    if do_transpose:
        xnew=xnew.T
    
    return xnew
    
def format_columns(column_data=[],column_headers=[],column_format_list=[],delimiter=", ",align="right", header_separator=None, print_result=False, 
                   truncate_length=None, truncate_endlength=None, truncate_string="..."):
    """
    Format data into columns for nicer output printing with specified alignment and delimiter
    """
    numcolumns=max([len(x) for x in column_data])

    if column_headers and len(column_headers) != numcolumns:
        raise Exception("column_headers must have one entry for each column")
    
    if column_format_list and isinstance(column_format_list,str):
        column_format_list=[column_format_list]
    
    if column_format_list and len(column_format_list) != numcolumns:
        raise Exception("column_format_list must have one entry for each column")
    
    column_maxlen=[0]*numcolumns
    
    #convert all data to strings and find longest width for each column
    column_data_str=[]
    separator_row=None
    if column_headers:
        column_data_str+=[column_headers]
        column_maxlen=[len(s) for s in column_headers]
        if header_separator:
            column_data_str+=[[(header_separator*len(s))[:len(s)] for s in column_headers]]
            separator_row=len(column_data_str)-1
    for irow,r in enumerate(column_data):
        row_str_list=[]
        for icol,c in enumerate(r):
            is_iterable=False
            if isinstance(c,str):
                pass
            else:
                try:
                    if len(c)>1:
                        is_iterable=True
                except:
                    pass
            if column_format_list:
                fmt=column_format_list[icol]
            else:
                fmt="%s"
            if fmt.startswith("{") and fmt.endswith("}"):
                if is_iterable:
                    s=fmt.format(*c)
                else:
                    s=fmt.format(c)
            else:
                if is_iterable:
                    s=fmt % tuple(c)
                else:
                    s=fmt % (c)
            row_str_list+=[s]
            column_maxlen[icol]=max(len(s),column_maxlen[icol])
        column_data_str+=[row_str_list]
    
    if align=="right":
        formatstr=delimiter.join(["%"+str(d)+"s" for d in column_maxlen])
    else:
        formatstr=delimiter.join(["%-"+str(d)+"s" for d in column_maxlen])
    
    column_result=[formatstr % tuple(r) for r in column_data_str]


    if truncate_length is not None:
        column_result_new=[]
        for i,r in enumerate(column_result):
            if align=='left':
                r=r.rstrip()
            if len(r)<=truncate_length:
                pass
            else:
                if truncate_endlength is None:
                    r=r[:truncate_length-len(truncate_string)]+truncate_string
                else:
                    r=r[:truncate_length-truncate_endlength-len(truncate_string)]+truncate_string+r[-truncate_endlength:]
            column_result_new+=[r]
        column_result=column_result_new
    
    if separator_row is not None:
        column_result[separator_row]=column_result[separator_row].replace(delimiter," "*len(delimiter))
    
    if print_result:
        for r in column_result:
            print(r)
    else:
        return column_result
        
#take a dictionary where each field value is a list
#and generate a new list of dictionaries with every combination of fields
def dict_combination_list(listdict, reverse_field_order=False):
    """
    Take a dictionary where each field value is a list and generate a new list of dictionaries with every combination of fields
    """
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
    """
    Flatten a list of lists (useful for argparse lists)
    """
    if l is None:
        return []
    lnew=[]
    for i in l:
        if isinstance(i,str):
            lnew+=[i]
        else:
            try:
                iter(i)
                lnew+=i
            except:
                lnew+=[i]
    #return [x for y in l for x in y]
    return lnew

def unique_preserve_order(seq):
    """
    Return a list of unique elements in the same order as the input list
    """
    u,idx=np.unique(seq, return_index=True)
    return u[np.argsort(idx)]

def triu_indices_torch(n,k=0):
    """pytorch triu_indices doesn't work the same way so use custom function that will"""
    ia,ib=torch.triu_indices(n,n,offset=k)
    return ia,ib

def square2tri(C, tri_indices=None, k=1, return_indices=False):
    """
    Convert a single square matrix to a triangular matrix
    """
    
    if tri_indices is None:
        if torch.is_tensor(C):
            tri_indices=triu_indices_torch(C.shape[0],k=k)
        else:
            tri_indices=np.triu_indices(C.shape[0],k=k)
    else:
        if not type(tri_indices) is tuple:
            #convert to tuple, since it might have been converted to a 2xEdges numpy array
            tri_indices=(tri_indices[0],tri_indices[1])
    if return_indices:
        return C[tri_indices], tri_indices
    else:
        return C[tri_indices]

def tri2square(Ctri, tri_indices=None, numroi=None, k=1, diagval=0):
    """
    Convert a 1d vectorized matrix to a square symmetrical matrix
    
    Example applying to a Nsubj x edges:
    C_list=[tri2square(Ctri[i,:],tri_indices=triu) for i in range(Ctri.shape[0])]
    or
    C_3D=np.stack([tri2square(Ctri[i,:],tri_indices=triu) for i in range(Ctri.shape[0])])
    """
    if tri_indices is None and numroi is None:
        raise Exception("Must provide either tri_indices or numroi")
    
    if tri_indices is None:
        if torch.is_tensor(Ctri):
            tri_indices=triu_indices_torch(numroi,k=k)
        else:
            tri_indices=np.triu_indices(numroi,k=k)
    else:
        if not type(tri_indices) is tuple:
            #convert to tuple, since it might have been converted to a 2xEdges numpy array
            tri_indices=(tri_indices[0],tri_indices[1])
        numroi=np.array(max(max(tri_indices[0]),max(tri_indices[1])))+1
    if torch.is_tensor(Ctri):
        C=torch.zeros(numroi,numroi,dtype=Ctri.dtype,device=Ctri.device)+torch.tensor(diagval,dtype=Ctri.dtype,device=Ctri.device)
    else:
        C=np.zeros((numroi,numroi),dtype=Ctri.dtype)+diagval
    
    C[tri_indices]=Ctri
    C[tri_indices[1],tri_indices[0]]=Ctri
    return C

def common_prefix(strlist):
    """Find the common prefix of a list of strings"""
    strlen=[len(s) for s in strlist]
    result=""
    for i in range(1,min(strlen)):
        if all([s[:i]==strlist[0][:i] for s in strlist]):
            result=strlist[0][:i]
        else:
            break
    return result

def common_suffix(strlist):
    """Find the common suffix of a list of strings"""
    strlen=[len(s) for s in strlist]
    result=""
    for i in range(1,min(strlen)):
        if all([s[-i:]==strlist[0][-i:] for s in strlist]):
            result=strlist[0][-i:]
        else:
            break
    return result

def trim_string(s,left=0,right=0):
    """Trim left=X and right=Y characters from a string. e.g., for removing common_prefix and common_suffix"""
    snew=s
    if left>0:
        snew=snew[left:]
    if right>0:
        snew=snew[:-right]
    return snew

def justfilename(pathstr):
    """Return the filename from a path string (e.g., /path/to/file.txt -> file.txt)"""
    if isinstance(pathstr,str):
        is_iterable=False
        pathstr=[pathstr]
    newstr=[x.split(os.path.sep)[-1] for x in pathstr]
    if not is_iterable:
        newstr=newstr[0]
    return newstr

def data_to_cell_array(data, as2d=False):
    if as2d:
        data_new=np.empty([len(data),1],dtype=object)
        data_new[:,0]=[C for C in data]
    else:
        data_new=np.empty(len(data),dtype=object)
        data_new[:]=[C for C in data]
    return data_new

def clean_args(args, arg_defaults={}, flatten=True):
    """
    Clean up an argparse namespace by copying default values for missing arguments and flattening list-based arguments
    """
    #copy defaults when not provided
    for k,v in vars(args).items():
        if k in arg_defaults:
            if v is None:
                setattr(args,k,arg_defaults[k])

    if flatten:
        #flatten list-based arguments
        for k,v in vars(args).items():
            if isinstance(v,str):
                #str are iterable but not lists
                continue
            try:
                iter(v)
                setattr(args,k,flatlist(v))
            except:
                continue
    return args

def load_h5_to_dict(filename):
    """
    Load an HDF5 file created by Option 1 into a nested dict of dicts.
    Groups become dicts, datasets become numpy arrays, attributes become scalars.
    """
    def _recurse(h5obj):
        result = {}
        # Load attributes
        for k, v in h5obj.attrs.items():
            result[k] = v

        # Load groups and datasets
        for k, v in h5obj.items():
            if isinstance(v, h5py.Group):
                result[k] = _recurse(v)
            elif isinstance(v, h5py.Dataset):
                result[k] = v[()]  # read dataset into numpy array or scalar
        return result

    with h5py.File(filename, "r") as h5f:
        return {k: _recurse(v) for k, v in h5f.items()}

def save_dict_to_h5(
    filename,
    data,
    compression=None,         # e.g. "gzip", "lzf", or None
    compression_opts=None,    # e.g. 9 for gzip level
    overwrite=True
):
    """
    Save a dict-of-dicts to HDF5 using a structured layout:
      - top-level keys -> groups
      - inner dict keys:
          * array-like -> datasets (optionally compressed)
          * scalar (int/float/bool/str/bytes/np scalar) -> attributes

    Parameters
    ----------
    data : dict
        Nested dictionary (values may be dict, arrays/lists, or scalars).
    filename : str
        Output .h5 path.
    compression : str or None
        "gzip", "lzf", or None.
    compression_opts : int or None
        For gzip, 1..9 (9 = max compression).
    overwrite : bool
        If True, open file with mode="w" (overwrite). Else "x" (fail if exists).
    """
    mode = "w" if overwrite else "x"

    def _is_scalar(x):
        # Accept basic Python scalars, numpy scalars, and strings/bytes
        return (
            isinstance(x, (int, float, bool, np.number, np.bool_))
            or isinstance(x, (str, bytes, np.str_, np.bytes_))
        )

    def _to_dataset(h5group, key, value):
        """Create a dataset for value under h5group[key], with compression settings."""
        ds_kwargs = {}
        if compression is not None:
            ds_kwargs["compression"] = compression
            if compression_opts is not None:
                ds_kwargs["compression_opts"] = compression_opts

        # Strings need special handling: variable-length UTF-8
        if isinstance(value, (list, tuple)) and all(isinstance(s, (str, np.str_)) for s in value):
            dt = h5py.string_dtype(encoding="utf-8")
            h5group.create_dataset(str(key), data=np.asarray(value, dtype=dt), dtype=dt, **ds_kwargs)
        elif isinstance(value, np.ndarray) and value.dtype.kind in ("U", "O"):
            # If it's a numpy array of strings/objects, try to coerce to vlen UTF-8 strings
            # (object arrays must be all strings)
            arr = value
            if arr.dtype.kind == "O":
                if not all(isinstance(s, (str, np.str_)) for s in arr.ravel()):
                    raise TypeError(f"Object array at '{key}' contains non-string elements.")
                arr = arr.astype("U")  # to unicode
            dt = h5py.string_dtype(encoding="utf-8")
            h5group.create_dataset(str(key), data=arr.astype(dt), dtype=dt, **ds_kwargs)
        else:
            # General numeric/boolean arrays or lists
            arr = np.asarray(value)
            h5group.create_dataset(str(key), data=arr, **ds_kwargs)

    def _recurse(h5obj, d):
        if not isinstance(d, dict):
            raise TypeError("All non-leaf nodes must be dicts.")
        for k, v in d.items():
            k = str(k)
            if isinstance(v, dict):
                subgrp = h5obj.create_group(k)
                _recurse(subgrp, v)
            else:
                if _is_scalar(v):
                    # Attributes can't be compressed; store scalars here
                    # h5py handles str/bytes/numeric scalars as attributes.
                    h5obj.attrs[k] = v
                else:
                    # Treat anything array-like (including lists) as a dataset
                    _to_dataset(h5obj, k, v)

    with h5py.File(filename, mode) as h5f:
        # create one group per top-level key
        for top_k, top_v in data.items():
            grp = h5f.create_group(str(top_k))
            if isinstance(top_v, dict):
                _recurse(grp, top_v)
            else:
                # If a top-level value is not a dict, decide where to put it:
                if _is_scalar(top_v):
                    grp.attrs["__value__"] = top_v
                else:
                    _to_dataset(grp, "__value__", top_v)

def get_version(include_date=False):
    """Return the version of this package"""
    if include_date:
        return __version__+" ("+__version_date__+")"
    else:
        return __version__
