
import torch
import numpy as np
import random

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
        if astype is int:
            return x.int()
        elif astype is float:
            return x.float()
        else:
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

def clean_args(args, arg_defaults={}):
    #copy defaults when not provided
    for k,v in vars(args).items():
        if k in arg_defaults:
            if v is None:
                setattr(args,k,arg_defaults[k])

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

def clean_subject_list(subjects):
    #make all subject lists = str type
    #(for cases where subjects are float(1234.), convert to int first to make sure they are "1234" and not "1234.0")
    try:
        newsubjects=[int(x) for x in subjects]
    except:
        newsubjects=[x for x in subjects]
    newsubjects=np.array([str(x) for x in newsubjects])

    return newsubjects