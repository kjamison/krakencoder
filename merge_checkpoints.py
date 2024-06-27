"""
Command-line script to merge a list of Krakencoder checkpoints (.pt files) into a new single model
"""

from krakencoder.model import *
from krakencoder.merge import merge_model_files
from krakencoder.utils import format_columns, get_version
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

    net, checkpoint_info=merge_model_files(checkpoint_filename_list=checkpointfile_list, canonicalize_input_names=canonicalize_input_names)
    
    #######
    # print info about merged model
    print("Merged model info:")
    
    output_info_columns=[]
    for i, conn_name in enumerate(checkpoint_info['input_name_list']):
        output_info_columns.append(['%d)' % (i+1), 
                                    conn_name, 
                                    '(Sx%d)' % (checkpoint_info['orig_input_size_list'][i]), 
                                    ' from: '+checkpoint_info['merged_checkpointfile_list'][checkpoint_info['merged_source_net_idx'][i]]])
        
    _=format_columns(column_data=output_info_columns, column_format_list=['%s','%s','%s','%s'],
                     delimiter=" ",align="left", print_result=True, truncate_length=130, truncate_endlength=30)
    
    print("Total: %d inputs. %d paths" % (len(checkpoint_info['input_name_list']), len(checkpoint_info['trainpath_encoder_index_list'])))
    
    # save merged model
    net.save_checkpoint(outputcheckpointfile, checkpoint_info)
    print(f'Merged model saved to {outputcheckpointfile}')
    
if __name__ == "__main__":
    run_merge_models(sys.argv[1:])
