from krakencoder import *
from collections.abc import Iterable
import os
import sys
import argparse
import warnings

def argument_parse_newdata(argv):
    parser=argparse.ArgumentParser(description='Describe krakencoder checkpoint',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument(action='store',dest='checkpoint', help='Checkpoint file (.pt)')
    parser.add_argument('--listtrainingpaths',action='store_true',dest='list_training_paths',help='List training paths for checkpoint')
    parser.add_argument('--printmodel',action='store_true',dest='print_model',help='Print model architecture')
    
    return parser.parse_args(argv)

def run_describe_model(argv):
    #read in command-line inputs
    args=argument_parse_newdata(argv)
    
    ptfile=args.checkpoint
    
    #load model and input transformers
    warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization")

    net, checkpoint=Krakencoder.load_checkpoint(ptfile)
    conn_names=checkpoint['input_name_list']
    trainpath_pairs = [[conn_names[i],conn_names[j]] for i,j in zip(checkpoint['trainpath_encoder_index_list'], checkpoint['trainpath_decoder_index_list'])]
    
    fields_to_skip=['trainpath_encoder_index_list','trainpath_decoder_index_list','optimizer','training_params']
    
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
            print("training_params[%s]:" % (k),checkpoint['training_params'][k])
    
    print("")
    print("Input types (%d):" % (len(conn_names)))
    for i,iname in enumerate(conn_names):
        print("%d) %s (Sx%d)" % (i+1,iname,checkpoint['orig_input_size_list'][i]))
    
    if args.print_model:
        print("")
        print(net)
    
    if args.list_training_paths:
        print("")
        print("Training paths (%d):" % (len(trainpath_pairs)))
        for i,[x,y] in enumerate(trainpath_pairs):
            print("%d) %s -> %s" % (i+1,x,y))
    
if __name__ == "__main__":
    run_describe_model(sys.argv[1:])
