from krakencoder import *
from train import *
from run_training import load_data, load_subject_list

from scipy.io import loadmat, savemat
import re

import sys
import argparse


def argument_parse_newdata(argv):
    parser=argparse.ArgumentParser(description='Evaluate krakencoder checkpoint',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--checkpoint',action='store',dest='checkpoint', help='Checkpoint file (.pt)')
    parser.add_argument('--trainrecord',action='store',dest='trainrecord', default='auto', help='trainrecord.mat file')
    parser.add_argument('--inputxform',action='append',dest='input_transform_file', help='Precomputed transformer file (.npy)',nargs='*')
    parser.add_argument('--output',action='store',dest='output', help='file to save model outputs')
    parser.add_argument('--inputdata',action='store',dest='input_data_file', help='.mat file containing input data to transform (instead of default HCP validation set)')
    
    parser.add_argument('--inputname',action='append',dest='input_names', help='Name of input data flavors (eg: FCcov_fs86_hpf, SCsdstream_fs86, encoded)',nargs='*')
    parser.add_argument('--outputnames',action='append',dest='output_names', help='List of data flavors from model to predict (default=all)',nargs='*')
    
    #parser.add_argument('--burst',action='store_true',dest='burst',help='burst mode eval')
    #parser.add_argument('--pathfinder',action='append',dest='pathfinder_list', help='pathfinder evaluation path names',nargs='*')
    
    return parser.parse_args(argv)

def load_new_data(inputfile, inputfield=None, quiet=False):
    inputfield_default_search=['encoded','FC','C','volnorm'] #,'sift2volnorm','sift2','orig']

    Cdata=loadmat(inputfile)
    if 'ismissing' in Cdata:
        subjmissing=Cdata['ismissing'][0]>0
    else:
        subjmissing=[]
    if 'subject' in Cdata:
        subjects=Cdata['subject'][0]
    else:
        subjects=[]
    
    conntype=None
    if inputfield:
        conntype=inputfield
    else:
        for itest in inputfield_default_search:
            if itest in Cdata:
                conntype=itest
                break
    
    if conntype is None:
        print("None of the following fields were found in the input file %s:" % (inputfile),inputfield_default_search)
        raise Exception("Input type not found")
    
    if len(Cdata[conntype][0][0].shape)==0:
        #single matrix was in file
        Cmats=[Cdata[conntype]]
    else:
        Cmats=Cdata[conntype][0]
    
    if conntype == "encoded":
        nroi=1
        npairs=Cmats[0].shape[1]
        Cdata=Cmats[0].copy()
    else:
        nroi=Cmats[0].shape[0]
        trimask=np.triu_indices(nroi,1)
        npairs=trimask[0].shape[0]
        if len(subjmissing)==0:
            subjmissing=np.zeros(len(Cmats))>0
        
        if len(subjects)>0:
            subjects=subjects[~subjmissing]
    
        Ctriu=[x[trimask] for i,x in enumerate(Cmats) if not subjmissing[i]]
        Cdata=np.vstack(Ctriu)
        #restrict to 420 unrelated subjects
        #Ctriu=[x for i,x in enumerate(Ctriu) if subjects997[i] in subjects]
    
    conndata={'data':Cdata,'numpairs':npairs,'numroi':nroi,'fieldname':conntype,'subjects':subjects}
    
    return conndata

def canonical_data_flavor(conntype):
    if conntype.lower() == "encoded":
        return "encoded"
    
    # parse user-specified input data type
    input_atlasname=""
    input_flavor=""
    input_fcfilt=""
    input_fcgsr=""
    input_scproc="volnorm"
    
    input_conntype_lower=conntype.lower()
    if "fs86" in input_conntype_lower:
        input_atlasname="fs86"
    elif "shen268" in input_conntype_lower:
        input_atlasname="shen268"
    elif "coco439" in input_conntype_lower:
        input_atlasname="coco439"
    else:
        print("Unknown atlas name for input type: %s" % (conntype))
        sys.exit(1)
    
    if "fccov" in input_conntype_lower and "gsr" in input_conntype_lower:
        input_flavor="FCcov"
        input_fcgsr="gsr"
    elif "fccov" in input_conntype_lower:
        input_flavor="FCcov"
    elif "pcorr" in input_conntype_lower:
        input_flavor="FCpcorr"
    elif "sdstream" in input_conntype_lower:
        input_flavor="sdstream"
    elif "ifod2act" in input_conntype_lower:
        input_flavor="ifod2act"
    else:
        print("Unknown data flavor for input type: %s" % (conntype))
        sys.exit(1)
    
    #FC: FCcov_<atlas>_<fcfilt>[gsr?]_FC, FCpcorr_<atlas>_<fcfilt>_FC
    #SC: <atlas>_sdstream_volnorm, <atlas>_ifod2act_volnorm
    
    if input_flavor.startswith("FC"):
        if "hpf" in input_conntype_lower:
            input_fcfilt="hpf"
        elif "bpf" in input_conntype_lower:
            input_fcfilt="bpf"
        elif "nofilt" in input_conntype_lower:
            input_fcfilt="nofilt"
        else:
            print("Unknown FC filter for input type: %s" % (conntype))
            sys.exit(1)
    
    if input_flavor.startswith("FC"):
        conntype_canonical="%s_%s_%s%s_FC" % (input_flavor,input_atlasname,input_fcfilt,input_fcgsr)
    else:
        conntype_canonical="%s_%s_%s" % (input_atlasname,input_flavor,input_scproc)
    
    return conntype_canonical

def run_model_on_new_data(argv):
    args=argument_parse_newdata(argv)
    
    ptfile=args.checkpoint
    recordfile=args.trainrecord
    
    if recordfile == "auto":
        recordfile=ptfile.replace("_checkpoint_","_trainrecord_")
        recordfile=recordfile.replace("_chkpt_","_trainrecord_")
        recordfile=re.sub("_(epoch|ep)[0-9]+\.pt$",".mat",recordfile)
        
    precomputed_transformer_info_list=None
    input_transform_file=None
    input_transform_file_list=[]
    
    if len(args.input_transform_file)<1 or args.input_transform_file[0] == "auto":
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
    
    if not input_transform_file_list:
        print("Must provide input transform (ioxfm) file")
        sys.exit(1)
        
    input_conntype_list=[]
    tmp_inputtypes=flatlist(args.input_names)
    if len(tmp_inputtypes)>0:
        input_conntype_list=tmp_inputtypes
    
    output_conntype_list=[]
    tmp_outputtypes=flatlist(args.output_names)
    if len(tmp_outputtypes)>0:
        output_conntype_list=tmp_outputtypes
    
    outfile = args.output
    input_file=args.input_data_file
    
    #record=loadmat(recordfile)
    #conn_names=[s.strip() for s in record['input_name_list']]

    #ptfile='connae_chkpt_SCFC_fs86_993subj_pc256_25paths_latent128_0layer_latentunit_drop0.5_750epoch_lr0.0001_correye+enceye.w10+mse.w1+latentsimloss.w5000_adamw.w0.01_skipacc_20220518_234644_ep000750.pt'
    #input_transform_file_list=['connae_ioxfm_SCFC_fs86_993subj_pc256_25paths_20220521.npy']

    precomputed_transformer_info_list={}
    for ioxfile in input_transform_file_list:
        print("Loading precomputed input transformations: %s" % (ioxfile))
        ioxtmp=np.load(ioxfile,allow_pickle=True).item()
        for k in ioxtmp:
            precomputed_transformer_info_list[k]=ioxtmp[k]
        
    transformer_list={}
    for conntype in precomputed_transformer_info_list:
        transformer, transformer_info = generate_transformer(traindata=None, transformer_type=precomputed_transformer_info_list[conntype]["type"], transformer_param_dict=None, 
            precomputed_transformer_params=precomputed_transformer_info_list[conntype], return_components=False)
        transformer_list[conntype]=transformer
    
    net, checkpoint=Krakencoder.load_checkpoint(ptfile)

    conn_names=checkpoint['input_name_list']

    trainpath_pairs = [[conn_names[i],conn_names[j]] for i,j in zip(checkpoint['trainpath_encoder_index_list'], checkpoint['trainpath_decoder_index_list'])]



    encoded_alltypes={}
    predicted_alltypes={}

    ######################
    # parse user-specified input data type
    #input_conntype_canonical=canonical_data_flavor(input_conntype)
    #inputtype_list=[input_conntype_canonical]
    inputtype_list=[canonical_data_flavor(x) for x in input_conntype_list]
    
    if len(output_conntype_list)==0 or output_conntype_list[0]=="all": 
        output_conntype_list=conn_names.copy()
    else:
        output_conntype_list=[canonical_data_flavor(x) for x in output_conntype_list]
    ######################
    
    conndata_alltypes={}
    conndata_alltypes[inputtype_list[0]]=load_new_data(inputfile=input_file, inputfield=None, quiet=False)
    
    neg1_torch=torchint(-1)
            
    net.eval()
    for intype in inputtype_list:
        if intype == "encoded":
            conn_encoded=torchfloat(conndata_alltypes[intype]['data'])
            encoded_alltypes[intype]=conn_encoded.numpy()
        else:
            if not intype in conn_names:
                raise Exception("Input type %s not found in model" % (intype))
            encoder_index=[idx for idx,c in enumerate(conn_names) if c==intype][0]
    
            encoder_index_torch=torchint(encoder_index)
    
            inputdata_origscale=conndata_alltypes[intype]['data']
            inputdata=torchfloat(transformer_list[intype].transform(inputdata_origscale))
    
            with torch.no_grad():
                conn_encoded = net(inputdata, encoder_index_torch, neg1_torch)
    
            encoded_alltypes[intype]=conn_encoded.numpy()
    
        predicted_alltypes[intype]={}
        for outtype in output_conntype_list:
            if outtype == "encoded":
                continue
            if not outtype in conn_names:
                raise Exception("Output type %s not found in model" % (outtype))
            decoder_index=[idx for idx,c in enumerate(conn_names) if c==outtype][0]
            decoder_index_torch=torchint(decoder_index)
            with torch.no_grad():
                _ , conn_predicted = net(conn_encoded, neg1_torch, decoder_index_torch)
            conn_predicted_origscale=transformer_list[outtype].inverse_transform(conn_predicted.cpu())
            predicted_alltypes[intype][outtype]=conn_predicted_origscale

    #conndata={'data':np.vstack(Ctriu),'numpairs':npairs,'numroi':nroi,'fieldname':conntype,'subjects':subjects}
    savemat(outfile,{'inputfile':input_file,'inputtypes':inputtype_list,'outputtypes':output_conntype_list,
        'encoded_alltypes':encoded_alltypes,'predicted_alltypes':predicted_alltypes,
        'subjects':conndata_alltypes[inputtype_list[0]]['subjects']},format='5',do_compression=True)
    print("Saved %s" % (outfile))

if __name__ == "__main__":
    run_model_on_new_data(sys.argv[1:])
