from krakencoder import *
from train import *
from run_training import load_input_data, load_hcp_data, load_hcp_subject_list, canonical_data_flavor
from utils import *

from scipy.io import loadmat, savemat
import re

import os
import sys
import argparse
import warnings

def argument_parse_newdata(argv):
   #for list-based inputs, need to specify the defaults this way, otherwise the argparse append just adds to them
    arg_defaults={}
    arg_defaults['burst_include']=[]
    arg_defaults['input_names']=[]
    arg_defaults['output_names']=[]
    arg_defaults['input_data_file']=[]
    arg_defaults['input_transform_file']=["auto"]

    parser=argparse.ArgumentParser(description='Evaluate krakencoder checkpoint',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--checkpoint',action='store',dest='checkpoint', help='Checkpoint file (.pt)')
    parser.add_argument('--trainrecord',action='store',dest='trainrecord', default='auto', help='trainrecord.mat file')
    parser.add_argument('--inputxform',action='append',dest='input_transform_file', help='Precomputed transformer file (.npy)',nargs='*')
    parser.add_argument('--output',action='store',dest='output', help='file to save model outputs. Can include "{input}" and/or "{output}" in name to save separate files for each input/output combo (or just group inputs and outputs)')
    parser.add_argument('--inputdata',action='append',dest='input_data_file', help='.mat file(s) containing input data to transform (instead of default HCP validation set). Can be "name=file"', nargs='*')
    
    parser.add_argument('--inputname','--inputnames',action='append',dest='input_names', help='Name(s) of input data flavors (eg: FCcov_fs86_hpf, SCsdstream_fs86, encoded)',nargs='*')
    parser.add_argument('--outputname','--outputnames',action='append',dest='output_names', help='List of data flavors from model to predict (default=all)',nargs='*')
    
    parser.add_argument('--burstinclude',action='append',dest='burst_include',help='inputnames to include in burst average',nargs='*')
    parser.add_argument('--burst',action='store_true',dest='burst',help='burst mode eval')
    parser.add_argument('--burstnorm',action='store_true',dest='burstnorm',help='re-normalize latent vectors after averaging')
    #parser.add_argument('--pathfinder',action='append',dest='pathfinder_list', help='pathfinder evaluation path names',nargs='*')

    testing_group=parser.add_argument_group('Testing options')
    testing_group.add_argument('--savetransformedinputs',action='store_true',dest='save_transformed_inputs', help='Transform inputs and save them as "encoded" values (for testing PCA/transformed space data)')
    testing_group.add_argument('--untransformedoutputs',action='store_true',dest='save_untransformed_outputs', help='Keep outputs in PCA/transformed space (for testing)')

    args=parser.parse_args(argv)
    args=clean_args(args,arg_defaults)
    return args

def search_flavors(searchstring_list,full_list):
    if isinstance(searchstring_list,str):
        searchstring_list=[searchstring_list]
    
    if len(searchstring_list)==0:
        new_list=full_list.copy()
    else:
        new_list=[]
        for s in searchstring_list:
            try:
                s_canonical=canonical_data_flavor(s)
            except:
                s_canonical=None
            
            if s.lower() == 'all':
                new_list+=full_list
            elif s.lower() in ['encoded','burst','transformed']:
                new_list+=[s.lower()]
            elif s in full_list:
                new_list+=[s]
            elif s_canonical is not None and s_canonical in full_list:
                new_list+=[s_canonical]
            elif s.lower()=='sc':
                new_list+=[intype for intype in full_list if "sdstream" in intype or "ifod2act" in intype]
            elif s.lower()=='fc':
                new_list+=[intype for intype in full_list if intype.startswith("FC")]
            else:
                new_list+=[intype for intype in full_list if s in intype]
    
    #return unique elements in list, in their original order
    return [str(s) for s in np.array(new_list)[np.sort(np.unique(new_list,return_index=True)[1])]]

def run_model_on_new_data(argv):
    #read in command-line inputs
    args=argument_parse_newdata(argv)
    
    ptfile=args.checkpoint
    recordfile=args.trainrecord
    burstmode=args.burst
    burstnorm=args.burstnorm
    input_burstmode_names=args.burst_include
    do_save_transformed_inputs=args.save_transformed_inputs
    outputs_in_model_space=args.save_untransformed_outputs
    outfile = args.output
    input_transform_file_list=args.input_transform_file
    input_conntype_list=args.input_names
    output_conntype_list=args.output_names
    input_file_list=args.input_data_file

    #note: don't actually use trainrecord during model evaluation
    #checkpoint includes all info about data flavors and model design
    #but doesn't contain any info about loss functions, training schedule, etc...
    #consider: in save_checkpoint, just add the trainrecord keys that aren't the 
    #big per-epoch loss values (ie: we have networkinfo[], make traininfo[] with those params)
    if recordfile == "auto":
        recordfile=ptfile.replace("_checkpoint_","_trainrecord_")
        recordfile=recordfile.replace("_chkpt_","_trainrecord_")
        recordfile=re.sub("_(epoch|ep)[0-9]+\.pt$",".mat",recordfile)

    if len(input_transform_file_list)>0 and input_transform_file_list[0] == "auto":
        input_transform_file=ptfile.replace("_checkpoint_","_iox_")
        input_transform_file=input_transform_file.replace("_chkpt_","_ioxfm_")
        input_transform_file=re.sub("_(epoch|ep)[0-9]+\.pt$",".npy",input_transform_file)
        input_transform_file_list=[input_transform_file]
    
    if len(input_file_list)>0:
        if(all(["=" in x for x in input_file_list])):
            tmp_inputfiles=input_file_list
            input_conntype_list=[]
            input_file_list=[]
            for x in tmp_inputfiles:
                input_conntype_list+=[x.split("=")[0]]
                input_file_list+=[x.split("=")[-1]]
    
    if len(input_conntype_list)==0:
        #try to figure out conntypes from filenames
        print("--inputname not provided. Guessing input type from filenames:")
        for x in input_file_list:
            xc=canonical_data_flavor(x.split(os.path.sep)[-1])
            input_conntype_list+=[xc]
            print("%s = %s" % (xc,x))
    
    #handle some shortcuts for the input/output filenames
    outfile_template=outfile
    if re.search("{.+}",outfile_template):
        instrlist=["i","in","input","s","src","source"]
        outstrlist=["o","out","output","t","trg","targ","target"]
        for s in instrlist:
            outfile_template=outfile_template.replace("{"+s+"}","{input}")
        for s in outstrlist:
            outfile_template=outfile_template.replace("{"+s+"}","{output}")
    
    ##############
    #load model and input transformers
    warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization")
    
    net, checkpoint=Krakencoder.load_checkpoint(ptfile)
    conn_names=checkpoint['input_name_list']
    trainpath_pairs = [[conn_names[i],conn_names[j]] for i,j in zip(checkpoint['trainpath_encoder_index_list'], checkpoint['trainpath_decoder_index_list'])]

    if not input_transform_file_list and checkpoint['input_transformation_info'].upper()!='NONE':
        print("Must provide input transform (ioxfm) file")
        sys.exit(1)
    
    precomputed_transformer_info_list={}
    for ioxfile in input_transform_file_list:
        print("Loading precomputed input transformations: %s" % (ioxfile))
        ioxtmp=np.load(ioxfile,allow_pickle=True).item()
        for k in ioxtmp:
            precomputed_transformer_info_list[k]=ioxtmp[k]
        
    transformer_list={}
    for conntype in precomputed_transformer_info_list:
        transformer, transformer_info = generate_transformer(traindata=None, 
            transformer_type=precomputed_transformer_info_list[conntype]["type"], transformer_param_dict=None, 
            precomputed_transformer_params=precomputed_transformer_info_list[conntype], return_components=False)
        transformer_list[conntype]=transformer

    ######################
    # parse user-specified input data type
    #conn_names = full list of input names from model checkpoint
    #input_conntype_canonical=canonical_data_flavor(input_conntype)
    #input_conntype_list=[input_conntype_canonical]
    input_conntype_list=search_flavors(input_conntype_list, conn_names)
    
    do_self_only=False
    if "none" in [x.lower() for x in output_conntype_list]:
        output_conntype_list=[]
    elif "self" in [x.lower() for x in output_conntype_list]:
        do_self_only=True
        output_conntype_list=input_conntype_list.copy()
    else:
        output_conntype_list=search_flavors(output_conntype_list,conn_names)
    
    #if user requested TRANSFORMED INPUTS (eg: PC-space inputs) dont produce any predicted outputs
    if do_save_transformed_inputs:
        #only output 'transformed' for this option
        output_conntype_list=['transformed']
    
    print("Input types (%d):" % (len(input_conntype_list)), input_conntype_list)
    print("Output types (%d):" % (len(output_conntype_list)), output_conntype_list)
    
    if burstmode:
        burstmode_names=search_flavors(input_burstmode_names, input_conntype_list)
        print("Burst input types (%d):" % (len(burstmode_names)), burstmode_names)
    else:
        burstmode_names=[]
    
    #build a list of output files (either consolidated, per input/output or per input->output path)
    if burstmode:
        eval_input_conntype_list=['burst']
    else:
        #for non-burst, just copy original input type list for evaluation
        eval_input_conntype_list=input_conntype_list.copy()
    
    outfile_list=[]
    outfile_input_output_list=[]
    
    if "{input}" in outfile_template and "{output}" in outfile:
        for intype in eval_input_conntype_list:
            for outtype in output_conntype_list:
                if do_self_only and outtype != intype:
                    continue
                outfile_tmp=outfile_template.replace("{input}",intype).replace("{output}",outtype)
                outfile_list+=[outfile_tmp]
                outfile_input_output_list+=[{"intype":[intype],"outtype":[outtype]}]
    elif "{input}" in outfile_template:
        for intype in eval_input_conntype_list:
            outfile_tmp=outfile_template.replace("{input}",intype)
            outfile_list+=[outfile_tmp]
            if do_self_only:
                 outfile_input_output_list+=[{"intype":[intype],"outtype":[intype]}]
            else:
                outfile_input_output_list+=[{"intype":[intype],"outtype":output_conntype_list}]
    elif "{output}" in outfile_template:
        for outtype in output_conntype_list:
            outfile_tmp=outfile_template.replace("{output}",outtype)
            outfile_list+=[outfile_tmp]
            if do_self_only:
                outfile_input_output_list+=[{"intype":[outtype],"outtype":[outtype]}]
            else:
                outfile_input_output_list+=[{"intype":eval_input_conntype_list,"outtype":[outtype]}]
    else:
        outfile_list+=[outfile_template]
        outfile_input_output_list+=[{"intype":eval_input_conntype_list,"outtype":output_conntype_list}]
    
    ######################
    #load input data
    
    if len(input_file_list) > 0:
        conndata_alltypes={}
        for i,x in enumerate(input_conntype_list):
            conndata_alltypes[x]=load_input_data(inputfile=input_file_list[i], inputfield=None)
            print(x,conndata_alltypes[x]['data'].shape)
    else:
        input_file="all"
        subjects, famidx = load_hcp_subject_list(numsubj=993)
        subjects_out, conndata_alltypes = load_hcp_data(subjects=subjects, conn_name_list=input_conntype_list, quiet=False)
        for conntype in conndata_alltypes.keys():
            conndata_alltypes[conntype]['subjects']=subjects_out
    
    encoded_alltypes={}
    predicted_alltypes={}
    
    neg1_torch=torchint(-1)
            
    net.eval()
    
    #encode all inputs to latent space
    for intype in input_conntype_list:
        if intype == "encoded" or intype == "burst":
            encoded_name=intype
            conn_encoded=torchfloat(conndata_alltypes[intype]['data'])
            encoded_alltypes[intype]=conn_encoded.numpy()
        else:
            if not intype in conn_names:
                raise Exception("Input type %s not found in model" % (intype))
            encoder_index=[idx for idx,c in enumerate(conn_names) if c==intype][0]
    
            encoder_index_torch=torchint(encoder_index)
    
            inputdata_origscale=conndata_alltypes[intype]['data']
            #reminder for sklearn PCA
            #Xpc=np.dot(X-pca.mean_, pca.components_.T) [ / np.sqrt(pca.explained_variance_) # if whiten, which=False by default]
            #X=np.dot(Xpc, pca.components_) + pca.mean_
            inputdata=torchfloat(transformer_list[intype].transform(inputdata_origscale))
            
            if do_save_transformed_inputs:
                conn_encoded=inputdata
            else:
                with torch.no_grad():
                    conn_encoded = net(inputdata, encoder_index_torch, neg1_torch)
    
            encoded_alltypes[intype]=conn_encoded.numpy()
    
    #burstmode averaging in encoding latent space
    if burstmode:
        print("Burst mode evaluation. Computing mean of input data flavors in latent space.")
        encoded_mean=None
        encoded_inputtype_count=0
        
        for intype in encoded_alltypes.keys():
            if intype == "encoded" or intype == "burst":
                conn_encoded=encoded_alltypes[intype]
            else:
                if not intype in encoded_alltypes:
                    raise Exception("Input type %s not found in model" % (intype))
                    
                if not intype in burstmode_names:
                    #only average encodings from specified input types
                    continue
                
                print("Burst mode evaluation includes: %s" % (intype))
                
                encoded_inputtype_count+=1
                conn_encoded=encoded_alltypes[intype]
                
            if encoded_mean is None:
                encoded_mean=conn_encoded
            else:
                encoded_mean+=conn_encoded
        encoded_mean=encoded_mean/encoded_inputtype_count
        
        encoded_norm=np.sqrt(np.sum(encoded_mean**2,axis=1,keepdims=True))
        if burstnorm:
            print("Mean burst vector length before re-normalization: %.4f (renormed to 1.0)" % (np.mean(encoded_norm)))
            encoded_mean=encoded_mean/encoded_norm
        else:
            print("Mean burst vector length: %.4f" % (np.mean(encoded_norm)))
        
        encoded_alltypes={'burst': encoded_mean.copy()}
    
    #now decode encoded inputs to requested outputs
    for intype in encoded_alltypes.keys():
        outtypes_for_this_input=np.unique([x['outtype'] for x in outfile_input_output_list if intype in x['intype']])
        
        if len(outtypes_for_this_input) == 0:
            continue
        
        predicted_alltypes[intype]={}
        conn_encoded=torchfloat(encoded_alltypes[intype])
        
        for outtype in outtypes_for_this_input:
            if outtype == "encoded" or outtype == "burst" or outtype=='transformed':
                continue
            if not outtype in conn_names:
                raise Exception("Output type %s not found in model" % (outtype))
                
            if do_self_only and intype != outtype:
                continue
            decoder_index=[idx for idx,c in enumerate(conn_names) if c==outtype][0]
            decoder_index_torch=torchint(decoder_index)
            with torch.no_grad():
                _ , conn_predicted = net(conn_encoded, neg1_torch, decoder_index_torch)
            
            if outputs_in_model_space:
                #for testing, keep outputs in the compressed/PCA model space
                conn_predicted_origscale=conn_predicted.cpu().detach().numpy()
            else:
                conn_predicted_origscale=transformer_list[outtype].inverse_transform(conn_predicted.cpu().detach().numpy())
            predicted_alltypes[intype][outtype]=conn_predicted_origscale
            print("Output %s->%s: %dx%d" % (intype,outtype,conn_predicted_origscale.shape[0],conn_predicted_origscale.shape[1]))

    #now save each output file with the input->output paths requested
    for i_out,outfile in enumerate(outfile_list):
        intype=outfile_input_output_list[i_out]['intype']
        outtype=outfile_input_output_list[i_out]['outtype']
        outdict={'checkpoint':ptfile,'recordfile':recordfile,'inputfiles':input_file_list,'subjects':conndata_alltypes[input_conntype_list[0]]['subjects']}
        
        if intype == 'all':
            outdict['inputtypes']=eval_input_conntype_list
        else:
            outdict['inputtypes']=intype
        
        if outtype == 'all':
            outdict['outputtypes']=output_conntype_list
        else:
            outdict['outputtypes']=outtype
        
        outdict['predicted_alltypes']={k_in:{k_out:v_out for k_out,v_out in v_in.items() if k_out in outdict['outputtypes']} for k_in,v_in in predicted_alltypes.items() if k_in in outdict['inputtypes']}
        
        if burstmode:
            outdict['burstmode_inputtypes']=burstmode_names
        
        for encout in ['encoded','burst','transformed']:
            if encout in outdict['outputtypes']:
                for k_in in outdict['inputtypes']:
                    outdict['predicted_alltypes'][k_in]={encout:encoded_alltypes[k_in]}
        
        savemat(outfile,outdict,format='5',do_compression=True)
        print("Saved %d/%d: %s" % (i_out+1,len(outfile_list),outfile))
        for k_in in outdict['predicted_alltypes'].keys():
            for k_out in outdict['predicted_alltypes'][k_in].keys():
                print("\t%s->%s (%dx%d)" % (k_in,k_out,*outdict['predicted_alltypes'][k_in][k_out].shape))

if __name__ == "__main__":
    run_model_on_new_data(sys.argv[1:])
