"""
Loss functions and other metrics for training and evaluation
"""

import torch
import torch.nn as nn
import numpy as np

from .utils import *

def xycorr(x,y,axis=1):
    """
    Compute correlation between all pairs of rows in x and y (or columns if axis=0)
    
    x: torch tensor or numpy array (Nsubj x M), generally the measured data for N subjects
    y: torch tensor or numpy array (Nsubj x M), generally the predicted data for N subjects
    axis: int (optional, default=1), 1 for row-wise, 0 for column-wise
    
    Returns: torch tensor or numpy array (Nsubj x Nsubj)
    
    NOTE: in train.py we always call cc=xycorr(Ctrue, Cpredicted)
    which means cc[i,:] is cc[true subject i, predicted for all subjects]
    and thus top1acc, which uses argmax(xycorr(true,predicted),axis=1) is:
    for every TRUE output, which subject's PREDICTED output is the best match
    """
    if torch.is_tensor(x):
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/torch.sqrt(torch.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/torch.sqrt(torch.sum(cy ** 2,keepdims=True,axis=axis))
        cc=torch.matmul(cx,cy.t())
    else:
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/np.sqrt(np.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/np.sqrt(np.sum(cy ** 2,keepdims=True,axis=axis))
        cc=np.matmul(cx,cy.T)
    return cc

def xycosine(x,y,axis=1):
    """
    Compute cosine distance between all pairs of rows in x and y (or columns if axis=0)
    
    x: torch tensor or numpy array (Nsubj x M), generally the measured data for N subjects
    y: torch tensor or numpy array (Nsubj x M), generally the predicted data for N subjects
    axis: int (optional, default=1), 1 for row-wise, 0 for column-wise
    
    Returns: torch tensor or numpy array (Nsubj x Nsubj)
    
    SEE: xycorr() for notes on argument order
    """
    if torch.is_tensor(x):
        cx=x/torch.sqrt(torch.sum(x ** 2,keepdims=True,axis=axis))
        cy=y/torch.sqrt(torch.sum(y ** 2,keepdims=True,axis=axis))
        cc=torch.matmul(cx,cy.t())
    else:
        cx=x/np.sqrt(np.sum(x ** 2,keepdims=True,axis=axis))
        cy=y/np.sqrt(np.sum(y ** 2,keepdims=True,axis=axis))
        cc=np.matmul(cx,cy.T)
    return cc

def corravgrank(x=None, y=None ,cc=None, sort_descending=True, return_ranklist=False):
    """
    Compute average rank of each row in xycorr(x,y).
    Perfect match is 1.0, meaning every row i in x has the best match with row i in y
    Chance is 0.5, meaning every row i in x has a random match with row i in y
    
    Inputs: either x and y must be provided, or cc must be provided
    x: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
    y: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
    cc: torch tensor or numpy array (Nsubj x Nsubj), (optional precomputed cc matrix) 
    sort_descending: bool, (optional, default=True), use True for correlation, False for distance
    
    Returns: float (or FloatTensor), average rank percentile (0.0-1.0)
    """
    if cc is None:
        cc=xycorr(x,y)
    if torch.is_tensor(cc):
        sidx=torch.argsort(cc,axis=1,descending=sort_descending)
        selfidx=torch.atleast_2d(torch.arange(cc.shape[0],device=sidx.device)).t()
        srank=torch.argmax((sidx==selfidx).double(),axis=1).double()
        #return np.mean(srank+1) #1-based rank
        ranklist=1-srank/cc.shape[0]
        avgrank=1-torch.mean(srank)/cc.shape[0] #percentile
    else:
        if sort_descending:
            sidx=np.argsort(cc,axis=1)[:,::-1]
        else:
            sidx=np.argsort(cc,axis=1)
        selfidx=np.atleast_2d(np.arange(cc.shape[0])).T
        srank=np.argmax(sidx==selfidx,axis=1)
        #return np.mean(srank+1) #1-based rank
        ranklist=1-srank/cc.shape[0]
        avgrank=1-np.mean(srank)/cc.shape[0] #percentile
    if return_ranklist:
        return avgrank,ranklist
    else:
        return avgrank


def distavgrank(x=None, y=None, d=None, return_ranklist=False):
    """
    Return avgrank using distance instead of correlation (See corravgrank)
    
    Inputs: either x and y must be provided, or d must be provided
    x: torch tensor or numpy array (Nsubj x M) (ignored if d is provided)
    y: torch tensor or numpy array (Nsubj x M) (ignored if d is provided)
    d: torch tensor or numpy array (Nsubj x Nsubj), (optional precomputed distance matrix)
    
    Returns: float (or FloatTensor), average rank percentile (0.0-1.0)
    """
    if d is None:
        d=torch.cdist(x,y)
    return corravgrank(cc=d,sort_descending=False, return_ranklist=return_ranklist)

def corrtrace(x,y):
    """Loss function: negative mean of correlation between row i in x and row i in y"""
    cc=xycorr(x,y)
    loss=-(torch.trace(cc)/cc.shape[0]-torch.mean(cc))
    return loss

def correye(x,y):
    """
    Loss function: mean squared error between pairwise correlation matrix for xycorr(x,y) and identity matrix
    (i.e., want diagonal to be near 1, off-diagonal to be near 0)
    """
    cc=xycorr(x,y)
    #need keepdim for some reason now that correye and enceye are separated
    loss=torch.norm(cc-torch.eye(cc.shape[0],device=cc.device),keepdim=True)
    return loss

def var_match_loss(xpred,xtrue,axis=0,relative_to_true=True):
    """
    Loss function: squared difference between variance of xpred and xtrue
    """
    xtrue_var=torch.mean((xtrue-xtrue.mean(axis=axis))**2)
    xpred_var=torch.mean((xpred-xpred.mean(axis=axis))**2)
    if relative_to_true:
        loss=((xtrue_var-xpred_var)/xtrue_var)**2
    else:
        loss=(xtrue_var-xpred_var)**2
    return loss

def distance_loss(x,y, margin=None, neighbor=False):
    """
    Loss function: difference between self-distance and other-distance for x and y, with optional margin
    If neighbor=True, reconstruction loss applies only to nearest neighbor distance, otherwise to mean distance between all
        off-diagonal pairs.
    
    Inputs:
    x: torch tensor (Nsubj x M), generally the measured data
    y: torch tensor (Nsubj x M), generally the predicted data
    margin: float, optional margin for distance loss (distance above margin is penalized, below is ignored)
    neighbor: bool, (optional, default=False), True for maximizing nearest neighbor distance, False for maximizing mean distance
    
    Returns: 
    loss: torch FloatTensor, difference between self-distance and other-distance
    """
    
    d=torch.cdist(x,y)
    dtrace=torch.trace(d)
    dself=dtrace/d.shape[0] #mean predicted->true distance
    
    if neighbor:
        dnei=d+torch.eye(d.shape[0],device=d.device)*d.max()
        #mean of row-wise min and column-wise min
        dother=torch.mean((dnei.min(axis=0)[0]+dnei.min(axis=1)[0])/2)
    else:
        dother=(torch.sum(d)-dtrace)/(d.shape[0]*(d.shape[0]-1)) #mean predicted->other distance
    
    if margin is not None:
        #dother=torch.min(dother,margin)
        #dother=-torch.nn.ReLU()(dother-margin) #pre 4/5/2024
        #if dother<margin, penalize (lower = more penalty).
        #if dother>=margin, ignore
        #standard triplet loss: torch.nn.ReLU()(dself-dother+margin) or torch.clamp(dself-dother+margin,min=0)
        dother=-torch.nn.ReLU()(margin-dother) #new 4/5/2024
    
    loss=dself-dother
    return loss

def distance_neighbor_loss(x,y, margin=None):
    """Loss function wrapper for distance_loss(x,y,margin,neighbor=True)"""
    return distance_loss(x,y, margin=margin, neighbor=True)

def dotproduct_loss(x,y,margin=None, neighbor=False):
    #for normalized (unit sphere) inputs, x.y = corr(x,y) so 1=perfect, -1=opposite
    #so 1-x.y, diag should be 0 like with distance metric
    d=1-x@y.T
    dtrace=torch.trace(d)
    dself=dtrace/d.shape[0] #mean predicted->true distance
    if neighbor:
        dnei=d+torch.eye(d.shape[0],device=d.device)*d.max()
        dother=torch.mean((dnei.min(axis=0)[0]+dnei.min(axis=1)[0])/2)
    else:
        dother=torch.sum(d-dtrace)/(d.shape[0]*(d.shape[0]-1)) #diag is all zeros by definition anyway
    
    if margin is not None:
        #dother=torch.min(dother,margin)
        dother=-torch.nn.ReLU()(dother-margin)
    
    loss=dself-dother
    return loss

def dotproduct_neighbor_loss(x,y,margin=None):
    """Loss function wrapper for dotproduct_loss(x,y,margin,neighbor=True)"""
    return dotproduct_loss(x,y,margin=margin, neighbor=True)

def corr_ident_parts(x=None, y=None ,cc=None):
    """
    Compute average self-correlation (diagonal) and average other-correlation (off-diagonal) for xycorr(x,y)
    
    Inputs: either x and y must be provided, or cc must be provided
    x: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
    y: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
    cc: torch tensor or numpy array (Nsubj x Nsubj), (optional precomputed cc matrix)
    
    Returns: tuple of two floats (or FloatTensors), average self-correlation and average other-correlation
    """
    if cc is None:
        cc=xycorr(x,y)
    cc_self=cc.trace()/cc.shape[0]
    if torch.is_tensor(cc):
        #cc_other=cc[torch.triu_indices(cc.shape[0],cc.shape[1],offset=1)].mean()
        cc_other=cc[triu_indices_torch(cc.shape[0],k=1)].mean()
    else:
        cc_other=cc[np.triu_indices(cc.shape[0],k=1)].mean()
    
    return cc_self,cc_other

def corrmatch(x,y):
    """Loss function: minimize matrix norm of xycorr(x,x)-xycorr(x,y) 
    (match prediction->meas correlation to intersubject correlation of measured data)"""
    cc_input=xycorr(x,x)
    cc_output=xycorr(x,y)
    loss=torch.norm(cc_output-cc_input)
    return loss

def disttop1acc(x=None, y=None ,d=None):
    """Top-1 accuracy but using distance (best d=0). See corrtop1acc"""
    if d is None:
        d=torch.cdist(x,y)
    if torch.is_tensor(d):
        s1idx=torch.argmin(d,axis=1)
        dmatch=s1idx==torch.arange(len(s1idx),device=s1idx.device)
        dmatch=dmatch.double()
    else:
        s1idx=np.argmin(d,axis=1)
        dmatch=s1idx==np.arange(len(s1idx))

    return dmatch.mean()

def disttopNacc(x=None, y=None, d=None, topn=1):
    """
    Compute top-N accuracy for cdist(x,y). See corrtop1acc.
    """
    if d is None:
        d=torch.cdist(x,y)
    #topidx=np.argsort(np.abs(cc),axis=1)[:,-topn:]
    if torch.is_tensor(d):
        topidx=torch.argsort(-d,axis=1,descending=True)[:,:topn]
        selfidx=torch.atleast_2d(torch.arange(d.shape[0],device=topidx.device)).t()
        dmatch=torch.any(topidx==selfidx,axis=1).double()
    else:
        topidx=np.argsort(-d,axis=1)[:,-topn:]
        selfidx=np.atleast_2d(np.arange(d.shape[0])).T
        dmatch=np.any(topidx==selfidx,axis=1)
    return dmatch.mean()

def corrtop1acc(x=None, y=None ,cc=None):
    """
    Compute top-1 accuracy for xycorr(x=meas,y=predicted)
    i.e., argmax(axis=1): for every subject (row) in x=meas, which subject (column) in y=predicted is closest match
    
    Inputs: either x and y must be provided, or cc must be provided
    x: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
    y: torch tensor or numpy array (Nsubj x M) (ignored if cc is provided)
    cc: torch tensor or numpy array (Nsubj x Nsubj), (optional precomputed cc matrix)
    
    Returns: float (or FloatTensor), top-1 accuracy (0.0-1.0)
    """
    
    if cc is None:
        cc=xycorr(x,y)
    #s1idx=np.argsort(np.abs(cc),axis=1)[:,-1]
    if torch.is_tensor(cc):
        s1idx=torch.argmax(cc,axis=1)
        ccmatch=s1idx==torch.arange(len(s1idx),device=s1idx.device)
        ccmatch=ccmatch.double()
    else:
        s1idx=np.argmax(cc,axis=1)
        ccmatch=s1idx==np.arange(len(s1idx))

    #s1idx=np.argsort(cc,axis=1)[:,-1]
    return ccmatch.mean()

def corrtopNacc(x=None, y=None, cc=None, topn=1):
    """
    Compute top-N accuracy for xycorr(x,y). See corrtop1acc.
    """
    if cc is None:
        cc=xycorr(x,y)
    #topidx=np.argsort(np.abs(cc),axis=1)[:,-topn:]
    if torch.is_tensor(cc):
        topidx=torch.argsort(cc,axis=1,descending=True)[:,:topn]
        selfidx=torch.atleast_2d(torch.arange(cc.shape[0],device=topidx.device)).t()
        ccmatch=torch.any(topidx==selfidx,axis=1).double()
    else:
        topidx=np.argsort(cc,axis=1)[:,-topn:]
        selfidx=np.atleast_2d(np.arange(cc.shape[0])).T
        ccmatch=np.any(topidx==selfidx,axis=1)
    return ccmatch.mean()

def columncorr(x,y,axis=1):
    """
    Compute correlation(x[:,i],y[:,i]) for each column i in x and y (or rows if axis=0)
    """
    if torch.is_tensor(x):
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/torch.sqrt(torch.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/torch.sqrt(torch.sum(cy ** 2,keepdims=True,axis=axis))
        cc=torch.sum(cx*cy,axis=axis)
    else:
        cx=x-x.mean(keepdims=True,axis=axis)
        cy=y-y.mean(keepdims=True,axis=axis)
        cx=cx/np.sqrt(np.sum(cx ** 2,keepdims=True,axis=axis))
        cy=cy/np.sqrt(np.sum(cy ** 2,keepdims=True,axis=axis))
        cc=np.sum(cx*cy,axis=axis)
    return cc

def mseloss(x,y):
    return torch.FloatTensor(nn.MSELoss()(x,y))
