#!/usr/bin/env python3

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
import zipfile

import krakencoder.jupyter_functions as kjf

def argument_parse_collectdata(argv):
    parser=argparse.ArgumentParser(description='Collect connectome data',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--subjectfile',action='store',dest='subjectfile', help='Plain-text file with subject names (one per line)')
    parser.add_argument('--subjects',action='store',dest='subjectlist', nargs='*',help='List of subject names (overrides subjectfile)')
    parser.add_argument('--output',action='store',dest='outputfile', help='Output file (.mat or .zip)',required=True)
    parser.add_argument('--inputdata',action='store',dest='inputdata',nargs='*',help='Can be <flavor>=<filepat> or <filepat>. Use "{SUBJECT}" to insert subject name')
    parser.add_argument('--inputdatafield',action='store',dest='inputdatafield', help='Field name in input data files to use (default = "data")')
    parser.add_argument('--bidsdesc',action='store',dest='bidsdesc', help='Extra string to add to .tsv filenames in BIDS .zip (*_desc-{BIDSDESC}_*.tsv)')
    parser.add_argument('--bidsifysubjects','--bidsify_subjects',action='store_true',dest='bidsify_subjects', help='If set, will convert subject names to BIDS-friendly ("sub-"+remove all non-alphanumeric characters)')
    parser.add_argument('--canonical',action='store_true',dest='canonical', help='Transform input flavors to canonical format (only works for known flavors)')
    parser.add_argument('--ziptype',action='store',dest='ziptype', default='tsv', choices=['tsv','mat'], help='Type of files to save in .zip (default: tsv, can be mat)')
    
    return parser.parse_args(argv)

def data_to_cell_array(data, as2d=False):
    if as2d:
        data_new=np.empty([len(data),1],dtype=object)
        data_new[:,0]=[C for C in data]
    else:
        data_new=np.empty(len(data),dtype=object)
        data_new[:]=[C for C in data]
    return data_new

def check_file_exists(filename):
    #check based on dir and look in zips
    file_exists=True
    if '.zip'+os.path.sep in filename.lower():
        zipfilename=re.sub(r'\.zip'+os.path.sep+r'.+$','.zip',filename,flags=re.IGNORECASE)
        filename_new=re.sub(r'^.+\.zip'+os.path.sep,'',filename,flags=re.IGNORECASE)
        if not os.path.exists(zipfilename):
            file_exists=False
        else:
            #check if the file exists in the zip
            with zipfile.ZipFile(zipfilename, 'r') as z:
                if not filename_new in z.namelist():
                    file_exists=False
    elif not os.path.exists(filename):
        file_exists=False
    return file_exists

def load_single_connectome_from_file(filename, datafields=[], subjectfields=[],file_bytes=None, return_subjectname=False):
    if datafields is None or len(datafields)==0:
        datafields=['C','SC','FC','data']
    if subjectfields is None or len(subjectfields)==0:
        subjectfields=['subject','subjects']
    
    if file_bytes is None:
        #check if file exists, unless we are given bytes directly
        if not check_file_exists(filename):
            raise FileNotFoundError("File not found: %s" % filename)
        
        if '.zip'+os.path.sep in filename.lower():
            zipfilename=re.sub(r'\.zip'+os.path.sep+r'.+$','.zip',filename,flags=re.IGNORECASE)
            filename_new=re.sub(r'^.+\.zip'+os.path.sep,'',filename,flags=re.IGNORECASE)
            with zipfile.ZipFile(zipfilename, 'r') as z:
                if filename_new not in z.namelist():
                    raise FileNotFoundError("File not found in zip: %s" % filename)
                with z.open(filename_new) as zb:
                    return load_single_connectome_from_file(filename_new, datafields=datafields, file_bytes=zb)
    
    file_to_load=filename if file_bytes is None else file_bytes
    
    #load data from file, either .mat, .csv, space-separated .txt/.tsv
    C=None
    subjectname=None
    if filename.lower().endswith('.mat'):
        M=loadmat(file_to_load,simplify_cells=True)
        for f in datafields:
            if f in M:
                C=M[f]
                break
        for sf in subjectfields:
            if sf in M:
                subjectname=M[sf]
                break
    else:
        try:
            C=np.loadtxt(file_to_load,delimiter=',',comments=['#','!','%'])
        except ValueError:
            C=np.loadtxt(file_to_load,comments=['#','!','%'])
    
    if return_subjectname:
        return C, subjectname
    else:
        return C

def run_collectdata(argv=None):
    if argv is None:
        argv=sys.argv[1:]
    #read in command-line inputs
    args=argument_parse_collectdata(argv)
    
    subjects=args.subjectlist
    outputfile=args.outputfile
    bids_desc_str=args.bidsdesc
    do_canonical=args.canonical
    do_bidsify_subjects=args.bidsify_subjects
    inputdatafield=args.inputdatafield
    ziptype=args.ziptype
    
    if do_canonical:
        #this import requires torch, so only import if needed
        from krakencoder.data import canonical_data_flavor
    else:
        canonical_data_flavor=lambda x: x
    
    if inputdatafield is not None:
        datafields=[inputdatafield]
    else:
        datafields=['C','SC','FC','data']
    
    subjects=[]
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
                if not check_file_exists(filepat_subj):
                    sys.exit('Error: input file does not exist for subject %s: %s' % (filepat_subj,s))
                M=load_single_connectome_from_file(filepat_subj,datafields=datafields)
                conndata.append(M)
        else:
            filepat=os.path.expanduser(filepat)
            if not check_file_exists(filepat):
                sys.exit('Error: input file does not exist: %s' % (filepat))
            
            M,Msubj=load_single_connectome_from_file(filepat,datafields=datafields, return_subjectname=True)
            if Msubj is not None:
                subjects+=[x for x in Msubj]
            conndata=M
        if conntype in conndata_alltypes:
            conndata_alltypes[conntype]+=[x for x in conndata]
        else:
            conndata_alltypes[conntype]=[x for x in conndata]
    
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
        if do_bidsify_subjects:
            #convert subjects to BIDS-friendly format
            #check if all subjects are already sub-[A-Za-z0-9]+:
            if all(re.match(r'sub-[A-Za-z0-9]+', s) for s in subjects):
                bids_subjects = [s for s in subjects]
                print("Subjects already in BIDS format, no conversion needed.")
            else:
                #remove all non-alphanumeric characters and convert to sub-<subject>
                bids_subjects = ['sub-' + re.sub(r'[^a-zA-Z0-9]', '', s) for s in subjects]
                print("Converted subjects to BIDS format  (See participants.tsv): %s" % (bids_subjects))
        else:
            bids_subjects = ['sub-%04d' % (i+1) for i in range(len(subjects))]
            print("Creating numerical sub-#### BIDS subject IDs (See participants.tsv): %s" % (bids_subjects))
            
        participants_info=pd.DataFrame({
            'participant_id':bids_subjects,
            'subject':subjects, 
            'train_val_test':subjsplit
        })
        
        print("Writing %s to .zip ..." % (ziptype))
        kjf.save_data_zip(outputfile, conndata_alltypes, participants_info, bids_desc=bids_desc_str, verbose=False, filetype=ziptype)
        print("Saved data to %s (%s)" % (outputfile,kjf.humanize_filesize(os.path.getsize(outputfile),binary=True)))
        
    elif outputfile.lower().endswith(".mat"):
        for conntype in conndata_alltypes:
            outfile_thistype=re.sub('\{(t|f|type|flav|flavor)\}','{FLAVOR}',outputfile,flags=re.IGNORECASE)
            if '{FLAVOR}' in outfile_thistype:
                outfile_thistype=outfile_thistype.format(FLAVOR=conntype)
            else:
                if len(conndata_alltypes)>1 and conntype != 'unknown':
                    outfile_thistype=outfile_thistype.replace(".mat","_%s.mat" % (conntype))
            savemat(outfile_thistype,{'subjects':data_to_cell_array(subjects, True),
                                      'C':data_to_cell_array(conndata_alltypes[conntype],True)},format='5',do_compression=True)
            print("Saved data to %s (%s)" % (outfile_thistype,kjf.humanize_filesize(os.path.getsize(outfile_thistype),binary=True)))

if __name__ == "__main__":
    if len(sys.argv)<=1:
        argument_parse_collectdata(['-h'])
        sys.exit(0)
    run_collectdata(sys.argv[1:])
