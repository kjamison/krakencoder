"""
Command-line script to collect input files for applying Krakencoder to a dataset
"""

import os
import sys
import argparse
import re
import pandas as pd
from scipy.io import loadmat, savemat
import numpy as np
import warnings

import krakencoder.jupyter_functions as kjf

def argument_parse_collectdata(argv):
    parser=argparse.ArgumentParser(description='Collect connectome data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--subjectfile',action='store',dest='subjectfile', help='Plain-text file with subject names (one per line)')
    parser.add_argument('--output',action='store',dest='outputfile', help='Output file (.mat or .zip)',required=True)
    parser.add_argument('--inputdata',action='store',dest='inputdata',nargs='*',help='Can be <flavor>=<filepat> or <filepat>. Use "{SUBJECT}" to insert subject name')
    parser.add_argument('--inputdatafield',action='store',dest='inputdatafield', help='Field name in input data files to use (default = "data")')
    parser.add_argument('--bidsdesc',action='store',dest='bidsdesc', help='Extra string to add to .tsv filenames in BIDS .zip (*_desc-{BIDSDESC}_*.tsv)')
    parser.add_argument('--canonical',action='store_true',dest='canonical', help='Transform input flavors to canonical format (only works for known flavors)')
    
    return parser.parse_args(argv)

def data_to_cell_array(data):
    data_new=np.empty(len(data),dtype=object)
    data_new[:]=[C for C in data]
    return data_new

def load_single_connectome_from_file(filename, datafields=[]):
    if not os.path.exists(filename):
        raise FileNotFoundError("File not found: %s" % filename)
    #load data from file, either .mat, .csv, space-separated .txt/.tsv
    C=None
    if filename.lower().endswith('.mat'):
        M=loadmat(filename,simplify_cells=True)
        for f in datafields:
            if f in M:
                C=M[f]
                break
    else:
        try:
            C=np.loadtxt(filename,delimiter=',',comments=['#','!','%'])
        except ValueError:
            C=np.loadtxt(filename,comments=['#','!','%'])
    return C

def run_collectdata(argv=None):
    if argv is None:
        argv=sys.argv[1:]
    #read in command-line inputs
    args=argument_parse_collectdata(argv)
    
    subjects=None
    outputfile=args.outputfile
    bids_desc_str=args.bidsdesc
    do_canonical=args.canonical
    inputdatafield=args.inputdatafield
    
    if do_canonical:
        #this import requires torch, so only import if needed
        from krakencoder.data import canonical_data_flavor
    else:
        canonical_data_flavor=lambda x: x
    
    if inputdatafield is not None:
        datafields=[inputdatafield]
    else:
        datafields=['C','SC','FC','data']
    
    if args.subjectfile:
        if not os.path.exists(args.subjectfile):
            sys.exit('Error: subject file does not exist')
        if args.subjectfile.endswith('.mat'):
            Msubj=loadmat(args.subjectfile,simplify_cells=True)
            if 'subjects' in Msubj:
                subjects=Msubj['subjects']
            elif 'subject' in Msubj:
                subjects=[Msubj['subject']]
            else:
                sys.exit('Error: subject file does not contain "subjects" or "subject"')
        else:
            with open(args.subjectfile) as f:
                subjects = [s.strip() for s in f.readlines()]
    
    conndata_alltypes={}
    
    for inputfile_info in args.inputdata:
        if '=' in inputfile_info:
            conntype,filepat=inputfile_info.split('=')
        else:
            filepat=inputfile_info
            conntype='unknown'
        #canonicalize input flavors if requested:
        if do_canonical:
            conntype=canonical_data_flavor(conntype,accept_unknowns=True)
        filepat=re.sub('\{(s|subj|subject)\}','{SUBJECT}',filepat,flags=re.IGNORECASE)
        
        conndata=[]
        if '{SUBJECT}' in filepat:
            print("%s: Loading %d subject-specific input files: %s" % (conntype,len(subjects),filepat))
            if subjects is None:
                sys.exit('Error: subject file required for input file with "{SUBJECT}"')
            for s in subjects:
                filepat_subj=filepat.format(SUBJECT=s)
                filepat_subj=os.path.expanduser(filepat_subj)
                if not os.path.exists(filepat_subj):
                    sys.exit('Error: input file does not exist for subject %s: %s' % (filepat_subj,s))
                    
                M=load_single_connectome_from_file(filepat_subj,datafields=datafields)
                conndata.append(M)
        else:
            filepat=os.path.expanduser(filepat)
            if not os.path.exists(filepat):
                sys.exit('Error: input file does not exist')
            M=load_single_connectome_from_file(filepat,datafields=datafields)
            conndata=M
        conndata_alltypes[conntype]=conndata
    
    #check that all subjects are the same shape
    # (can be smaller if the last ROI is missing)
    for conntype in conndata_alltypes:
        sz=[C.shape for C in conndata_alltypes[conntype]]
        if not all([s==sz[0] for s in sz]):
            maxsz=np.max(np.stack(sz),axis=0)
            warnings.warn("Warning: not all data shapes are the same for %s. Padding to %s" % (conntype,maxsz))
            #pad with zeros to maxsz
            conndata=[]
            for C in conndata_alltypes[conntype]:
                Cnew=np.zeros(maxsz)
                Cnew[:C.shape[0],:C.shape[1]]=C
                conndata.append(Cnew)
            conndata_alltypes[conntype]=conndata
    
    for conntype in conndata_alltypes:
        print("%s: %s" % (conntype,kjf.data_shape_string(conndata_alltypes[conntype])))
    
    #savemat(outputfile,{'predicted_alltypes':conndata_alltypes},format='5',do_compression=True)
    
    if outputfile.lower().endswith(".zip"):
        #make a dummy participants_info file with participant_id=sub-####, and all subjects marked as "training" (to be used for adaptation)
        subjsplit=['train' for s in subjects]
        participants_info=pd.DataFrame({
            'participant_id':['sub-%04d' % (i+1) for i in range(len(subjects))],
            'subject':subjects, 
            'train_val_test':subjsplit
        })
        
        print("Writing tsv to .zip ...")
        kjf.save_data_zip(outputfile, conndata_alltypes, participants_info, bids_desc=bids_desc_str, verbose=False)
        print("Saved data to %s (%s)" % (outputfile,kjf.humanize_filesize(os.path.getsize(outputfile),binary=True)))
    elif outputfile.lower().endswith(".mat"):
        
        #raise NotImplementedError("Saving to .mat not yet implemented")
        for conntype in conndata_alltypes:
            outfile_thistype=re.sub('\{(t|f|type|flav|flavor)\}','{FLAVOR}',outputfile,flags=re.IGNORECASE)
            if '{FLAVOR}' in outfile_thistype:
                outfile_thistype=outfile_thistype.format(FLAVOR=conntype)
            else:
                outfile_thistype=outfile_thistype.replace(".mat","_%s.mat" % (conntype))
            savemat(outfile_thistype,{'subjects':subjects, 'C':data_to_cell_array(conndata_alltypes[conntype])},format='5',do_compression=True)
            print("Saved data to %s (%s)" % (outfile_thistype,kjf.humanize_filesize(os.path.getsize(outfile_thistype),binary=True)))

if __name__ == "__main__":
    if len(sys.argv)<=1:
        argument_parse_collectdata(['-h'])
        sys.exit(0)
    run_collectdata(sys.argv[1:])
