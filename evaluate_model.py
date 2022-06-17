import os
numthreads=4
os.environ['OPENBLAS_NUM_THREADS'] = str(numthreads)
os.environ['MKL_NUM_THREADS'] = str(numthreads)
os.environ['NUMEXPR_NUM_THREADS']=str(numthreads)

from krakencoder import *
from train import *
from run_training import load_data, load_subject_list

from sklearn.cross_decomposition import PLSRegression #test this

from scipy.io import loadmat, savemat
import re

import sys
import argparse
#need inputs and outputs


def argument_parse_eval(argv):
    parser=argparse.ArgumentParser(description='Evaluate krakencoder checkpoint',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--checkpoint',action='store',dest='checkpoint', help='Checkpoint file (.pt)')
    parser.add_argument('--trainrecord',action='store',dest='trainrecord', default='auto', help='trainrecord.mat file')
    parser.add_argument('--inputxform',action='append',dest='input_transform_file', help='Precomputed transformer file (.npy)',nargs='*')
    parser.add_argument('--burst',action='store_true',dest='burst',help='burst mode eval')
    parser.add_argument('--burstinclude',action='append',dest='burst_include',help='inputnames to include in burst average',nargs='*')
    parser.add_argument('--encodingoutput',action='store',dest='encoding_output', help='file to save model encodings')
    parser.add_argument('--predictionoutput_prefix',action='store',dest='prediction_output_prefix', help='PREFIX file to save model outputs')
    parser.add_argument('--predictionoutput_split',action='store',dest='prediction_output_split', type=int, default=1, help='how many paths per prediction output file?')
    parser.add_argument('--similarity',action='store',dest='similarity',default='corr',help='corr or dist')
    parser.add_argument('--testplsdim',action='store',dest='testplsdim', type=int, help='test PLS mapping')
    parser.add_argument('--fixtrainmean',action='store_true',dest='fixtrainmean', help='adjust origscale by trainmean')
    parser.add_argument('--savetrain',action='store_true',dest='savetrain',help='save training data outputs also')
    parser.add_argument('--optimscale',action='store_true',dest='optimscale',help='keep predictions in optmization scale, NOT origscale')
    parser.add_argument('--pathfinder',action='append',dest='pathfinder_list', help='pathfinder evaluation path names',nargs='*')
    return parser.parse_args(argv)

#1. load subject data
#2. for a given .pt:
#   * load .pt
#   * transform

def run_evaluate_model(argv):
    args=argument_parse_eval(argv)
    burstmode=args.burst
    ptfile=args.checkpoint
    recordfile=args.trainrecord
    
    burstmode_names=flatlist(args.burst_include)
    
    if recordfile == "auto":
        recordfile=ptfile.replace("_checkpoint_","_trainrecord_")
        recordfile=recordfile.replace("_chkpt_","_trainrecord_")
        recordfile=re.sub("_(epoch|ep)[0-9]+\.pt$",".mat",recordfile)
        
    precomputed_transformer_info_list=None
    input_transform_file=None

    if args.input_transform_file == "auto":
        input_transform_file=ptfile.replace("_checkpoint_","_iox_")
        input_transform_file=input_transform_file.replace("_chkpt_","_ioxfm_")
        input_transform_file=re.sub("_(epoch|ep)[0-9]+\.pt$",".npy",input_transform_file)
        input_transform_file_list=[input_transform_file]
    else:
        input_transform_file_list=[]
        if args.input_transform_file and len(args.input_transform_file) > 0:
            tmpxfm=flatlist(args.input_transform_file)
            if len(tmpxfm)>0:
                input_transform_file_list=tmpxfm
    
    precomputed_transformer_info_list=None
    if len(input_transform_file_list)>0:
        precomputed_transformer_info_list={}
        for ioxfile in input_transform_file_list:
            print("Loading precomputed input transformations: %s" % (ioxfile))
            ioxtmp=np.load(ioxfile,allow_pickle=True).item()
            for k in ioxtmp:
                precomputed_transformer_info_list[k]=ioxtmp[k]
        
    outfile_encoding = args.encoding_output
    outfile_prediction_prefix = args.prediction_output_prefix
    outfile_prediction_split = args.prediction_output_split
    
    eval_similarity=args.similarity
    do_savetrain=args.savetrain
    
    
    do_origscale=not args.optimscale
    
    plsdim=args.testplsdim
    fix_train_mean=args.fixtrainmean
    
    pathfinder_list=flatlist(args.pathfinder_list)
    
    record=loadmat(recordfile)
    
    #########################
    
    net, checkpoint=Krakencoder.load_checkpoint(ptfile)
    
    conn_names=checkpoint['input_name_list']
    
    trainpath_pairs = [[conn_names[i],conn_names[j]] for i,j in zip(checkpoint['trainpath_encoder_index_list'], checkpoint['trainpath_decoder_index_list'])]
    
    #########################
    #### load subject lists
    nsubj=len(record['subjects'][0])
    #print("nsubj",nsubj)
    
    subjects, famidx = load_subject_list(nsubj)
    subjects_out, conndata_alltypes = load_data(subjects=subjects, conn_name_list=conn_names, quiet=False)
    if len(subjects) != len(subjects_out) or not all(subjects==subjects_out):
        raise Exception("Subjects did not match expected")

    #load train/val splits directly from training record
    subjidx_train=record['subjidx_train'][0]
    subjidx_val=record['subjidx_val'][0]
    #######################
    
    
    trainpath_list, data_origscale, data_transformer_info_list = generate_training_paths(conndata_alltypes, conn_names, subjects, subjidx_train, subjidx_val, 
                                    trainpath_pairs=trainpath_pairs, trainpath_group_pairs=None, skip_selfs=checkpoint['skip_selfs'],
                                    reduce_dimension=checkpoint['reduce_dimension'], 
                                    use_truncated_svd='use_truncated_svd' in checkpoint and checkpoint['use_truncated_svd'], 
                                    leave_data_alone=checkpoint['reduce_dimension'] is None, 
                                    use_truncated_svd_for_sc='use_truncated_svd_for_sc' in checkpoint and checkpoint['use_truncated_svd_for_sc'],
                                    keep_origscale_data=True, 
                                    use_pretrained_encoder=False, quiet=False,
                                    precomputed_transformer_info_list=precomputed_transformer_info_list)
    
    net, trainpath_list = run_network(net, trainpath_list, burstmode=burstmode, pathfinder_list=pathfinder_list, burstmode_search_list=burstmode_names)
    
    trainpath_encoding_output=[]
    trainpath_prediction_output=[]
    for itp, trainpath in enumerate(trainpath_list):
        
        if do_origscale:
            
            traindata=data_origscale['traindata_origscale'][trainpath['output_name']] 
            valdata=data_origscale['valdata_origscale'][trainpath['output_name']]
        
            #traindata_predicted=trainpath['train_outputs_predicted']
            #valdata_predicted=trainpath['val_outputs_predicted']
            
            traindata_restore=trainpath['output_transformer'].inverse_transform(trainpath['train_outputs_predicted'].cpu())
            valdata_restore=trainpath['output_transformer'].inverse_transform(trainpath['val_outputs_predicted'].cpu())
            
        else:
            traindata=trainpath['train_outputs']
            valdata=trainpath['val_outputs']
            traindata_restore=trainpath['train_outputs_predicted'] #.cpu()
            valdata_restore=trainpath['val_outputs_predicted'] #.cpu()
        
        if outfile_encoding:
            trainpath_encoding_output+=[{}]
            trainpath_encoding_output[itp]['input_name']=trainpath['input_name']
            trainpath_encoding_output[itp]['output_name']=trainpath['output_name']
            trainpath_encoding_output[itp]['val_inputs_encoded']=numpyvar(trainpath['val_inputs_encoded']).copy()
            
        if outfile_prediction_prefix:
            trainpath_prediction_output+=[{}]
            trainpath_prediction_output[itp]['input_name']=trainpath['input_name']
            trainpath_prediction_output[itp]['output_name']=trainpath['output_name']
            trainpath_prediction_output[itp]['val_outputs_predicted']=numpyvar(valdata_restore).copy()
            trainpath_prediction_output[itp]['val_outputs']=numpyvar(valdata).copy()
        
        if outfile_encoding or outfile_prediction_prefix:
            continue
        
        traindata=torchfloat(traindata)
        valdata=torchfloat(valdata)
        traindata_restore=torchfloat(traindata_restore)
        valdata_restore=torchfloat(valdata_restore)
        
        burstmode_str=""
        if burstmode:
            burstmode_str=" (burst mode)"
        shortstr='cc'
        print("")
        print("%s->%s%s" % (trainpath['input_name'],trainpath['output_name'],burstmode_str))
        for mtype in ['orig','proc','orig2proc','proc2orig']:
        #for mtype in ['proc2orig']:
            if mtype == 'orig':
                cc_train=xycorr(traindata,traindata)
                cc_val=xycorr(valdata,valdata)
            elif mtype == 'proc':
                cc_train=xycorr(traindata_restore,traindata_restore)
                cc_val=xycorr(valdata_restore,valdata_restore)
            elif mtype == 'orig2proc':
                cc_train=xycorr(traindata,traindata_restore)
                cc_val=xycorr(valdata,valdata_restore)
            elif mtype == 'proc2orig':
                cc_train=xycorr(traindata_restore,traindata)
                cc_val=xycorr(valdata_restore,valdata)
        
            ccself_train, ccother_train = corr_ident_parts(cc=cc_train)
            ccself_val, ccother_val = corr_ident_parts(cc=cc_val)
            corr_acc_train=corrtop1acc(cc=cc_train)
            corr_acc_val=corrtop1acc(cc=cc_val)
        
            corr_rank_train=corravgrank(cc=cc_train)
            corr_rank_val=corravgrank(cc=cc_val)
        
            print("%10s: %s.t=(%6.3f,%6.3f), %s.v=(%6.3f,%6.3f), top1.t=%.3f, top1.v=%.3f, rank.t=%.3f, rank.v=%.3f" %
                          (mtype,shortstr,ccself_train,ccother_train,shortstr,ccself_val,ccother_val,corr_acc_train,corr_acc_val,corr_rank_train,corr_rank_val))

    output_dict={}
    output_dict['subjects']=subjects
    output_dict['subjidx_train']=subjidx_train
    output_dict['subjidx_val']=subjidx_val
    output_dict['checkpointfile']=ptfile
    output_dict['trainrecordfile']=recordfile
    output_dict['input_transform_file_list']=input_transform_file_list
    output_dict['burstmode']=burstmode
    if burstmode:
        output_dict['burstmode_names']=burstmode_names
    output_dict['origscale']=do_origscale
    
    if outfile_encoding:
        output_dict['trainpath_data']=trainpath_encoding_output
        savemat(outfile_encoding,output_dict,format='5',do_compression=True)
        print("Saved %s" % (outfile_encoding))
    
    if outfile_prediction_prefix:
        outfile_prediction_split=min(max(outfile_prediction_split,1),len(trainpath_prediction_output))
        itp=0
        for i1 in range(0,len(trainpath_prediction_output),outfile_prediction_split):
            i2=i1+outfile_prediction_split
            i2=min(i2,len(trainpath_prediction_output))
            print(i1,i2)
            output_dict['trainpath_data']=trainpath_prediction_output[i1:i2]
            if isinstance(output_dict['trainpath_data'],dict):
                output_dict['trainpath_data']=[output_dict['trainpath_data']] #make sure its a list
            itp=itp+1
            outfile_tmp=outfile_prediction_prefix+"_split%04d.mat" % (itp)
            savemat(outfile_tmp,output_dict,format='5',do_compression=True)
            print("Saved %s" % (outfile_tmp))
        
        #for itp, tp in enumerate(trainpath_prediction_output):
        #    output_dict['trainpath_data']=[trainpath_prediction_output[itp]]
        #    outfile_tmp=outfile_prediction_prefix+"_tp%04d.mat" % (itp)
        #    savemat(outfile_tmp,output_dict,format='5',do_compression=True)
        #    print("Saved %s" % (outfile_tmp))

if __name__ == "__main__":
    run_evaluate_model(sys.argv[1:])
