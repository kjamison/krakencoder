#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from scipy.io import loadmat, savemat

class CustomFormatter_optlambda(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass

def argument_parse_optlambda(argv):
    parser=argparse.ArgumentParser(description='Convert covariance FC to regularized precision or partial correlation',formatter_class=CustomFormatter_optlambda,
        epilog="""
Example usage:

Option 1: Regularization target is the mean unregularized inverse of the input subjects. 
    Output is a set of PARTIAL CORRELATION matrices.

regularize_fc_precision.py --input newstudy_fs86_FCcov_hpf_100subj.mat \\
    --output newstudy_fs86_FCpcorr_hpf_100subj.mat \\
    --partialcorr \\
    --outputfigure newstudy_fs86_FCpcorr_hpf_100subj_regularization.png

Option 2:  Compute regularization target mean from separate training data, with lambda=0, then optimize input data to target
    Target file is a PRECISION matrix. Final output is a set of PARTIAL CORRELATION matrices

regularize_fc_precision.py --input traindata_fs86_FCcov_hpf_993subj.mat \\
    --subjectsplitfile subject_splits_993subj_683train_79val_196test_retestInTest.mat --subjectsplitname train \\
    --applylambda 0 --outputmean regtarget_fs86_hpf_683trainsubj.mat 
    
regularize_fc_precision.py --input newstudy_fs86_FCcov_hpf_100subj.mat \\
    --target regtarget_fs86_hpf_683trainsubj.mat \\
    --output newstudy_fs86_FCpcorr_hpf_100subj.mat \\
    --partialcorr \\
    --outputfigure newstudy_fs86_FCpcorr_hpf_100subj_regularization.png
    """)
    
    parser.add_argument('--input',action='store',dest='inputfile', help='.mat file containing variable "C" with a cell array/list of square FC matrices',required=True)
    parser.add_argument('--output',action='store',dest='outputfile', help='Output file (.mat)')
    parser.add_argument('--outputmean',action='store',dest='outputfile_mean', help='Output name for the MEAN result across subjects in the input.')
    parser.add_argument('--target',action='store',dest='targetfile', help='Optional .mat file containing a precision matrix "C" that will be the optimzation target\n(otherwise use mean of unregularized inverse of all inputs)')
    parser.add_argument('--partialcorr',action='store_true',dest='partialcorr', help='Transform outputs to partial correlation, instead of precision')
    parser.add_argument('--outputfigure',action='store',dest='outputfigure', help='Filename to save a summary figure of the regularization (.png)')
    
    lambda_search_parsegroup=parser.add_argument_group('Lambda search options')
    lambda_search_parsegroup.add_argument('--applylambda',action='store',dest='applylambda', type=float, help='Apply this lambda to create new output (NO search)')
    lambda_search_parsegroup.add_argument('--roundlambda',action='store',dest='roundlambda', default=2, type=int, help='Decimal places to keep in optimized lambda')
    lambda_search_parsegroup.add_argument('--lambdagridloops',action='store',dest='lambdagridloops',default=3,type=int, help='Number of iterative loops in grid search')
    lambda_search_parsegroup.add_argument('--lambdagridsize',action='store',dest='lambdagridsize',default=10,type=int, help='Number of lambda within each loop')
    lambda_search_parsegroup.add_argument('--lambdarange',action='store',dest='lambdarange',default=[0,1],type=float, nargs=2, help='Lambda search range')
    
    subject_split_parsegroup=parser.add_argument_group('Subject split options')
    subject_split_parsegroup.add_argument('--subjectsplitfile','--subjectfile',action='store',dest='subject_split_file', help='.mat file containing pre-saved "subjects","subjidx_train","subjidx_val","subjidx_test" fields')
    subject_split_parsegroup.add_argument('--subjectsplitname',action='store',dest='subject_split_name', default='all', help='Which data split to evaluate: "all", "train", "test", "val", "retest", etc...')
    subject_split_parsegroup.add_argument('--outputsubjectsplitname',action='store',dest='output_subject_split_name', default=None, help='Which data split to OUTPUT after finding lambda (default=splitname): "all", "train", "test", "val", "retest", etc...')
    
    return parser.parse_args(argv)

def precision_to_partialcorr(Xprec):
    """
    Normalize a precision matrix to partial correlation matrix (and set diag=0)
    """
    D=np.diag(1/np.sqrt(np.diag(np.abs(Xprec))))
    Xpc=D@(-Xprec)@D
    Xpc[np.eye(Xpc.shape[0])>0]=0
    return Xpc

def unregularized_precision_mean(FClist, quiet=False):
    """
    Compute unregularized inverse (or pinv if fails due to sparsity/collinearity) of each FC in input, then return mean
    """
    try:
        FCprec_mean=np.mean(np.stack([np.linalg.inv(x) for x in FClist],axis=-1),axis=-1)
    except np.linalg.LinAlgError:
        if not quiet:
            print("inv(x) failed on inputs for initial unreg step. Using pinv(x) instead")
        FCprec_mean=np.mean(np.stack([np.linalg.pinv(x) for x in FClist],axis=-1),axis=-1)
    return FCprec_mean

def invtikh(X, lam, quiet=True):
    """
    Compute tikhonov-regularized inverse of X with specified lambda
    """
    if lam==0:
        #only for no-regularization case, try pinv if inv fails
        try:
            Xinv=np.linalg.inv(X)
        except np.linalg.LinAlgError:
            if not quiet:
                print("inv(x) failed on inputs for initial unreg step. Using pinv(x) instead")
            Xinv=np.linalg.pinv(X)
    else:
        Xinv=np.linalg.inv(X+lam*np.trace(X)/X.shape[0]*np.eye(X.shape[0]))
    
    return Xinv

def find_optimal_precision_lambda(FClist, FCprec_target=None, 
    lambda_range=[0,1], lambda_loops=3, lambda_gridcount=10,
    drawplot=False, plotfilename=None):
    """
    Perform grid search to identify lambda that minimizes the average difference between each regularized 
    inverse and a 'target' precision matrix. This target can be provided, otherwise use the average unregularized 
    inverse.
    
    Parameters:
    FClist: list of square input covariance/correlation FC matrices
    FCprec_target: target square PRECISION FC for regularization (if None, use mean unreg inverse of inputs)
    lambda_range, lambda_loops, lambda_gridcount: grid search parameters
    drawplot: True = plot grid search summary figure but don't save
    plotfilename: filename (eg: ending in .png) to save grid search summary figure
    
    Returns:
    optlambda: scalar lambda value identified from grid search
    """
    
    if FCprec_target is None:
        FCprec_target=unregularized_precision_mean(FClist)
    
    mask=np.triu_indices(FCprec_target.shape[0],1) #skip diag when computing similarity
    
    #invtikh=lambda x,lam: np.linalg.inv(x+lam*np.trace(x)/x.shape[0]*np.eye(x.shape[0]))
    
    lambda_full=np.empty(0)
    reg_err_full=np.empty(0)
    
    for iloop in range(lambda_loops):
        lam=np.linspace(lambda_range[0],lambda_range[1],lambda_gridcount)
        reg_err=np.zeros([len(FClist),len(lam)])
        
        for i,l in enumerate(lam):
            FCprec=[invtikh(x,l) for x in FClist]
            FCprec_reg_err=[np.linalg.norm(x[mask]-FCprec_target[mask]) for x in FCprec]
            reg_err[:,i]=FCprec_reg_err
        
        reg_err_mean=np.mean(reg_err,axis=0)
        midx=np.argmin(reg_err_mean)
        if midx==0:
            lambda_range=[0,lam[1]]
        elif midx==len(lam)-1:
            lambda_range=[lam[-2],1]
        else:
            lambda_range=[lam[midx-1],lam[midx+1]]
            
        lambda_full=np.concatenate([lambda_full,lam])
        reg_err_full=np.concatenate([reg_err_full,reg_err_mean])
    
    sidx=np.argsort(lambda_full)
    lambda_full=lambda_full[sidx]
    reg_err_full=reg_err_full[sidx]
    
    midx=np.argmin(reg_err_full)
    optlambda=lambda_full[midx]
    
    if drawplot or plotfilename:
        #dont bother importing this unless we actually use it (takes time sometimes)
        import matplotlib.pyplot as plt
        fig=plt.figure()
        plt.plot(lambda_full,reg_err_full,'-+')
        plt.plot(lambda_full[midx],reg_err_full[midx],'ro',markersize=10,markerfacecolor='none')
        plt.title('optlambda = %f' % (optlambda))
        
        if plotfilename:
            fig.savefig(plotfilename,dpi=100)
        else:
            plt.show()
    
    return optlambda

def run_optlambda():
    args=argument_parse_optlambda(sys.argv[1:])
    
    fields_to_search=['C','FC']
    
    inputfile=args.inputfile
    targetfile=args.targetfile
    outputfile=args.outputfile
    outputfile_mean=args.outputfile_mean
    output_figure=args.outputfigure
    do_partialcorr=args.partialcorr
    
    applylambda=args.applylambda
    lambda_rounding_places=args.roundlambda
    lambda_loops=args.lambdagridloops
    lambda_gridsize=args.lambdagridsize
    lambda_range=args.lambdarange
    
    splitfile=args.subject_split_file
    splitname=args.subject_split_name
    outsplitname=args.output_subject_split_name
    if outsplitname is None:
        outsplitname=splitname
    
    M=loadmat(inputfile,simplify_cells=True)
    fields_found=[f for f in fields_to_search if f in M]
    assert len(fields_found)>0, f"Input have one of the following fields: {fields_to_search}"
    datafield=fields_found[0]
    FClist=M[datafield]
    
    fields_to_keep=['subject','subjects','ismissing','is_missing']
    M_out={}
    for f in fields_to_keep:
        if f in M:
            M_out[f]=M[f]
    
    if splitfile is not None:
        subject_field=['subject','subjects']
        subject_field=[f for f in subject_field if f in M]
        assert len(subject_field)>0, "Input data must contain 'subject' or 'subjects' fields if subject_split_file was provided"
        subject_field=subject_field[0]
        subjects=M[subject_field]
        subjects=[str(s) for s in subjects]
        assert len(subjects)==len(FClist), "len(%s) must match len(%s) in input file" % (subject_field,datafield)
        
        subjsplits=loadmat(splitfile,simplify_cells=True)
        subjects_fromsplit=np.array([str(x) for x in subjsplits['subjects']])
        if splitname.lower() == 'all':
            pass
        elif "subjidx_"+splitname in subjsplits:
            subjects_fromsplit=subjects_fromsplit[subjsplits["subjidx_"+splitname]]
        else:
            raise Exception("Split name %s not found in subject_split_file %s" % (splitname,splitfile))
        
        FClist=[x for i,x in enumerate(FClist) if subjects[i] in subjects_fromsplit]
        
        #if we are using a subset of subjects, reset the subject info in the output file
        subjects=[s for s in subjects if s in subjects_fromsplit]
        subjcell=np.empty([len(subjects),1],dtype=object)
        subjcell[:,0]=subjects
        M_out={}
        M_out[subject_field]=subjcell
        
        print("Loaded %d (%dx%d) matrices from split=%s in %s" % (len(FClist),FClist[0].shape[0],FClist[0].shape[1],splitname,inputfile))
    else:
        print("Loaded %d (%dx%d) matrices from %s" % (len(FClist),FClist[0].shape[0],FClist[0].shape[1],inputfile))
    
    if targetfile is None:
        FCprec_target=None
    else:
        Mtarg=loadmat(targetfile,simplify_cells=True)
        fields_found=[f for f in fields_to_search if f in Mtarg]
        assert len(fields_found)>0, f"Target file have one of the following fields: {fields_to_search}"
        targetfield=fields_found[0]
        FCprec_target=Mtarg[targetfield]
        print("Search will match inputs to data from target file:",targetfile)
    
    if applylambda is not None:
        optlambda=applylambda
        print("Applying user-specified lambda: %f" % (optlambda))
    else:
        optlambda=find_optimal_precision_lambda(FClist, 
            FCprec_target=FCprec_target, 
            lambda_range=lambda_range, lambda_loops=lambda_loops, lambda_gridcount=lambda_gridsize, 
            plotfilename=output_figure)
        
        print("Found optimal lambda: %f" % (optlambda))
    
    if not (outputfile or outputfile_mean):
        exit(0)
    
    lambda_raw=optlambda
    optlambda=np.round(lambda_raw,lambda_rounding_places) #round to the nearest 0.01
    
    #########
    if splitfile is not None and outsplitname != splitname:
        splitname=outsplitname
        
        #if we are saving a different subset of subjects than are used to find lambda
        FClist=M[datafield]
        
        fields_to_keep=['subject','subjects','ismissing','is_missing']
        M_out={}
        for f in fields_to_keep:
            if f in M:
                M_out[f]=M[f]
        
        subjects=M[subject_field]
        subjects=[str(s) for s in subjects]
        
        subjects_fromsplit=np.array([str(x) for x in subjsplits['subjects']])
        if splitname.lower() == 'all':
            pass
        elif "subjidx_"+splitname in subjsplits:
            subjects_fromsplit=subjects_fromsplit[subjsplits["subjidx_"+splitname]]
        else:
            raise Exception("Split name %s not found in subject_split_file %s" % (splitname,splitfile))
        
        FClist=[x for i,x in enumerate(FClist) if subjects[i] in subjects_fromsplit]
        
        #if we are using a subset of subjects, reset the subject info in the output file
        subjects=[s for s in subjects if s in subjects_fromsplit]
        subjcell=np.empty([len(subjects),1],dtype=object)
        subjcell[:,0]=subjects
        M_out={}
        M_out[subject_field]=subjcell
        
    #########
    M_out['lambda_raw']=lambda_raw
    M_out['lambda']=optlambda
    
    if do_partialcorr:
        FCprec=[precision_to_partialcorr(invtikh(x, optlambda)) for x in FClist]
        M_out['conntype']='partialcorr'
    else:
        FCprec=[invtikh(x, optlambda) for x in FClist]
        M_out['conntype']='precision'
    
    if outputfile:
        FCout=np.empty([len(FCprec),1],dtype=object)
        FCout[:,0]=FCprec
        savemat(outputfile,{'C':FCout,**M_out},format='5',do_compression=True)
        print("Saved %d (%dx%d) matrices to %s" % (len(FCout),FCout[0,0].shape[0],FCout[0,0].shape[1],outputfile))
    
    if outputfile_mean:
        FCout=np.mean(np.stack(FCprec,axis=-1),-1)
        savemat(outputfile_mean,{'C':FCout,**M_out},format='5',do_compression=True)
        print("Saved 1 (%dx%d) matrix to %s" % (FCout.shape[0],FCout.shape[1],outputfile_mean))

if __name__ == "__main__":
    run_optlambda()
