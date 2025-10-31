#!/usr/bin/env python3

"""
Command-line script to merge a list of Krakencoder checkpoints (.pt files) into a new single model
"""

from krakencoder.model import *
from krakencoder.merge import merge_model_files, print_merged_model
from krakencoder.utils import get_version
from krakencoder.fetch import get_fetchable_data_list, fetch_model_data

import os
import sys
import argparse
import warnings

def argument_parse_mergemodel(argv):
    parser=argparse.ArgumentParser(description='Merge encoder+decoders from multiple Krakencoder checkpoints into a single model',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--checkpointlist',action='store',dest='checkpoint_list', help='Checkpoint files (.pt) to merge', nargs='*', required=True)
    parser.add_argument('--output',action='store',dest='output', help='file to save merged model (.pt)', required=True)
    parser.add_argument('--canonical',action='store_true',dest='canonical', help='Canonicalize all input flavors before merging')
    parser.add_argument('--version', action='version',version='Krakencoder v{version}'.format(version=get_version(include_date=True)))
    
    args=parser.parse_args(argv)
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    return args

def run_merge_models(argv):
    args=argument_parse_mergemodel(argv)
    checkpointfile_list=args.checkpoint_list
    outputcheckpointfile=args.output
    canonicalize_input_names=args.canonical
    
    warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization")

    checkpointfile_list_resolved=[]
    for c in checkpointfile_list:
        if not os.path.exists(c) and c in get_fetchable_data_list(filenames_only=True):
            c=fetch_model_data(files_to_fetch=c, force_download=False)
        checkpointfile_list_resolved+=[c]
    checkpointfile_list=checkpointfile_list_resolved
    net, checkpoint_info=merge_model_files(checkpoint_filename_list=checkpointfile_list, canonicalize_input_names=canonicalize_input_names)
    
    print("Merged model info:")
    print_merged_model(checkpoint_info)
    
    # save merged model
    net.save_checkpoint(outputcheckpointfile, checkpoint_info)
    print(f'Merged model saved to {outputcheckpointfile}')
    
if __name__ == "__main__":
    run_merge_models(sys.argv[1:])
