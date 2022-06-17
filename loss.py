import torch
import torch.nn as nn
import numpy as np

# loss functions

#NOTE: in train.py we always call cc=xycorr(Ctrue, Cpredicted)
#   which means cc[i,:] is cc[true subject i, predicted for all subjects]
#   and thus top1acc, which uses argmax(xycorr(true,predicted),axis=1) is:
#   for every TRUE output, which subject's PREDICTED output is the best match

def xycorr(x,y,axis=1):
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

def mseloss(x,y):
    return torch.FloatTensor(nn.MSELoss()(x,y))

def triu_indices_torch(n,k=0):
    ia,ib=torch.triu_indices(n,n,offset=k)
    return [ia,ib]

def corravgrank(x=None, y=None ,cc=None, sort_descending=True):
    if cc is None:
        cc=xycorr(x,y)
    if torch.is_tensor(cc):
        sidx=torch.argsort(cc,axis=1,descending=sort_descending)
        selfidx=torch.atleast_2d(torch.arange(cc.shape[0],device=sidx.device)).t()
        srank=torch.argmax((sidx==selfidx).double(),axis=1).double()
        #return np.mean(srank+1) #1-based rank
        return 1-torch.mean(srank)/cc.shape[0] #percentile
    else:
        if sort_descending:
            sidx=np.argsort(cc,axis=1)[:,::-1]
        else:
            sidx=np.argsort(cc,axis=1)
        selfidx=np.atleast_2d(np.arange(cc.shape[0])).T
        srank=np.argmax(sidx==selfidx,axis=1)
        #return np.mean(srank+1) #1-based rank
        avgrank=1-np.mean(srank)/cc.shape[0] #percentile
    return avgrank

def distavgrank(x=None, y=None, d=None):
    if d is None:
        d=torch.cdist(x,y)
    return corravgrank(cc=d,sort_descending=False)

def corrtrace(x,y):
    cc=xycorr(x,y)
    loss=-(torch.trace(cc)/cc.shape[0]-torch.mean(cc))
    return loss

def correye(x,y):
    cc=xycorr(x,y)
    #need keepdim for some reason now that correye and enceye are separated
    loss=torch.norm(cc-torch.eye(cc.shape[0],device=cc.device),keepdim=True)
    return loss


def correye_encodedeye(x,y,xe,w):
    cc=xycorr(x,y)
    loss1=torch.norm(cc-torch.eye(cc.shape[0],device=cc.device))
    cc_enc=xycorr(xe,xe)
    loss2=torch.norm(cc_enc-torch.eye(cc_enc.shape[0],device=cc_enc.device))
    loss=loss1+w*loss2
    return loss

def dist_encodeddist(x,y,xe,w,margin=None,encoder_margin=None,neighbor=False, encode_dot=False):
    #main x,y distance:
    d=torch.cdist(x,y)
    dtrace=torch.trace(d)
    dself=dtrace/d.shape[0] #mean predicted->true distance
    if neighbor:
        dnei=d+torch.eye(d.shape[0],device=d.device)*d.max()
        dother=torch.mean((dnei.min(axis=0)[0]+dnei.min(axis=1)[0])/2)
    else:
        dother=(torch.sum(d)-dtrace)/(d.shape[0]*(d.shape[0]-1)) #mean predicted->other distance
    
    ##########
    #encoder xe,xe distance
    if encode_dot:
        #for dot, perfect=1, orthog=0, opposite=-1
        #so 1-dot, perfeoct=0, orthog=1, opposite=2
        #corr and dot are identical if norm=1
        #dist and dot are monotonic if norm=1
        d=1-xe.dot(xe.T)
    else:
        d=torch.cdist(xe,xe)
    
    
    if neighbor:
        dnei=d+torch.eye(d.shape[0],device=d.device)*d.max()
        denc_other=torch.mean((dnei.min(axis=0)[0]+dnei.min(axis=1)[0])/2)
    else:
        denc_other=torch.sum(d)/(d.shape[0]*(d.shape[0]-1)) #diag is all zeros by definition anyway
    
    ###########
    if margin is not None:
        #dother=torch.min(dother,margin)
        dother=-torch.nn.ReLU()(dother-margin)
    if encoder_margin is not None:
        #denc_other=torch.min(denc_other,encoder_margin)
        denc_other=-torch.nn.ReLU()(denc_other-encoder_margin)
    loss=(dself-dother)-w*denc_other
    return loss

def distance_loss(x,y, margin=None, neighbor=False):
    d=torch.cdist(x,y)
    dtrace=torch.trace(d)
    dself=dtrace/d.shape[0] #mean predicted->true distance
    
    if neighbor:
        dnei=d+torch.eye(d.shape[0],device=d.device)*d.max()
        dother=torch.mean((dnei.min(axis=0)[0]+dnei.min(axis=1)[0])/2)
    else:
        dother=(torch.sum(d)-dtrace)/(d.shape[0]*(d.shape[0]-1)) #mean predicted->other distance
    
    if margin is not None:
        #dother=torch.min(dother,margin)
        dother=-torch.nn.ReLU()(dother-margin)
    
    loss=dself-dother
    return loss

def distance_neighbor_loss(x,y, margin=None):
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
    return dotproduct_loss(x,y,margin=margin, neighbor=True)

def corr_ident_parts(x=None, y=None ,cc=None):
    if cc is None:
        cc=xycorr(x,y)
    cc_self=cc.trace()/cc.shape[0]
    if torch.is_tensor(cc):
        #cc_other=cc[torch.triu_indices(cc.shape[0],cc.shape[1],offset=1)].mean()
        #pytorch triu_indices doesn't work the same way so use custom function that will
        cc_other=cc[triu_indices_torch(cc.shape[0],k=1)].mean()
    else:
        cc_other=cc[np.triu_indices(cc.shape[0],k=1)].mean()
    
    return cc_self,cc_other

def corrmatch(x,y):
    cc_input=xycorr(x,x)
    cc_output=xycorr(x,y)
    loss=torch.norm(cc_output-cc_input)
    return loss

def disttop1acc(x=None, y=None ,d=None):
    #same as corrtop1acc but euclidean distance and argmin (best d=0)
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

def corrtop1acc(x=None, y=None ,cc=None):
    #argmax(axis=1): for every subject (row) in x, which subject (row) in y is closest match
    
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