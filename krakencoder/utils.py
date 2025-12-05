"""
Miscellaneous utility functions
"""

import torch
import numpy as np
import random
import os
import scipy.interpolate

from .utils_notorch import *

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
