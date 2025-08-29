#!/usr/bin/env python3

"""
Command-line script to print information about a saved Krakencoder checkpoint (.pt) file
"""

from krakencoder.model import *
from krakencoder.plotfigures import display_kraken_heatmap
from krakencoder.utils import get_version, format_columns

from collections.abc import Iterable
import os
import sys
import argparse
import warnings
from scipy.io import loadmat

def argument_parse_describecheckpoint(argv):
    parser=argparse.ArgumentParser(description='Describe krakencoder checkpoint',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--checkpoint',action='store',dest='checkpoint', help='Checkpoint file (.pt)')
    parser.add_argument('--listtrainingpaths',action='store_true',dest='list_training_paths',help='List training paths for checkpoint')
    parser.add_argument('--printmodel',action='store_true',dest='print_model',help='Print model architecture')
    parser.add_argument('--heatmap_record',action='store',dest='heatmap_record_file',help='Generate 2x2 heatmap grid from this record file')
    parser.add_argument('--heatmap_perpath',action='store_true',dest='heatmap_per_path',help='Heatmap uses best per-path epoch instead of whole-model')
    parser.add_argument('--heatmap_output',action='store',dest='heatmap_output_file',help='(default = from record name)')
    parser.add_argument('--heatmap_metrics',action='store',dest='heatmap_metric_names',nargs='*',help='(default = top1acc topnacc avgrank r2)')
    parser.add_argument('--heatmap_showepoch',action='store_true',dest='heatmap_show_epoch',help='Show optimal per-path epoch number (default = false)')
    parser.add_argument('--heatmap_epoch',action='store',dest='heatmap_epoch',type=int,help='Show heatmap for specified epoch')
    parser.add_argument('--heatmap_fraction_perpath',action='store_true',dest='heatmap_fraction_epoch',help='Show performance as a fraction of the BEST epoch for each path')
    parser.add_argument('--heatmap_training',action='store_true',dest='heatmap_training',help='Show training performance instead of validation performance')
    parser.add_argument('--heatmap_dpi',action='store',dest='heatmap_dpi',type=int,default=200,help='(output figure dpi. default = 200)')
    parser.add_argument('--heatmap_extrashortnames',action='store_true',dest='heatmap_extrashortnames',help='Extra short names for publication figs')
    
    parser.add_argument('--version', action='version',version='Krakencoder v{version}'.format(version=get_version(include_date=True)))
    
    return parser.parse_args(argv)

def run_describe_checkpoint(argv=None):
    if argv is None:
        argv=sys.argv[1:]
    #read in command-line inputs
    args=argument_parse_describecheckpoint(argv)
    
    if args.heatmap_record_file:
        recordfile=args.heatmap_record_file
        trainrecord=loadmat(recordfile,simplify_cells=True)
        trainrecord['recordfile']=recordfile
        if args.heatmap_output_file:
            imgfile_heatmap=args.heatmap_output_file
        else:
            imgfile_heatmap=recordfile.replace("_trainrecord_","_heatmap_").replace(".mat",".png")
        do_single_epoch=not args.heatmap_per_path
        if args.heatmap_metric_names is None or len(args.heatmap_metric_names)==0:
            heatmap_metric_names=['top1acc','topNacc','avgrank','r2']
        else:
            heatmap_metric_names=args.heatmap_metric_names
        heatmap_show_epoch=args.heatmap_show_epoch
        heatmap_explicit_epoch=args.heatmap_epoch
        heatmap_fraction_perpath=args.heatmap_fraction_epoch
        heatmap_training=args.heatmap_training
        heatmap_dpi=args.heatmap_dpi
        heatmap_extrashortnames=args.heatmap_extrashortnames
        display_kraken_heatmap(trainrecord,metrictype=heatmap_metric_names,origscale=True,single_epoch=do_single_epoch,best_epoch_fraction=heatmap_fraction_perpath,
                               training=heatmap_training, extra_short_names=heatmap_extrashortnames,
                               show_epoch=heatmap_show_epoch, explicit_epoch=heatmap_explicit_epoch,outputimagefile={'file':imgfile_heatmap,'dpi':heatmap_dpi})
    if args.checkpoint is None:
        return
        
    ptfile=args.checkpoint
    #load model and input transformers
    warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization")

    net, checkpoint=Krakencoder.load_checkpoint(ptfile)
    conn_names=checkpoint['input_name_list']
    trainpath_pairs = [[conn_names[i],conn_names[j]] for i,j in zip(checkpoint['trainpath_encoder_index_list'], checkpoint['trainpath_decoder_index_list'])]
    
    fields_to_skip=['trainpath_encoder_index_list','trainpath_decoder_index_list','optimizer','training_params','merged_training_params_list','merged_checkpoint_info_list']
    
    print("Model information:")
    for k in checkpoint:
        if k in fields_to_skip:
            continue
        print("%s:" % (k),checkpoint[k])
    
    if 'training_params' in checkpoint:
        for k in checkpoint['training_params']:
            if k in fields_to_skip:
                continue
            if isinstance(checkpoint['training_params'][k],Iterable):
                continue
            print("training_params['%s']:" % (k),checkpoint['training_params'][k])
    
    print("")
    print("Input types (%d):" % (len(conn_names)))
    output_info_columns=[]
    output_column_format_list=['%s','%s','%s']
    if 'merged_checkpointfile_list' in checkpoint:
        output_column_format_list.append('%s')
    
    for i,iname in enumerate(conn_names):
        rowitem=['%d)' % (i+1), iname, '(Sx%d)' % (checkpoint['orig_input_size_list'][i])]
        if 'merged_checkpointfile_list' in checkpoint:
            rowitem.append('from: '+checkpoint['merged_checkpointfile_list'][checkpoint['merged_source_net_idx'][i]])
        output_info_columns.append(rowitem)
    
    _=format_columns(column_data=output_info_columns, column_format_list=output_column_format_list,
                     delimiter=" ",align="left", print_result=True, truncate_length=130, truncate_endlength=30)
    
    if args.print_model:
        print("")
        print(net)
    
    if args.list_training_paths:
        print("")
        print("Training paths (%d):" % (len(trainpath_pairs)))
        for i,[x,y] in enumerate(trainpath_pairs):
            print("%d) %s -> %s" % (i+1,x,y))
    
if __name__ == "__main__":
    if len(sys.argv)<=1:
        argument_parse_describecheckpoint(['-h'])
        sys.exit(0)
    run_describe_checkpoint(sys.argv[1:])
