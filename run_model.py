"""
Command-line script for evaluating a krakencoder model checkpoint on new data
Must provide a checkpoint file (.pt) and input transform files (*ioxfm*.npy), unless model was trained on non-transformed data
Outputs can include predicted connectomes of different types, latent vectors, and/or just heatmaps and records of model performance metrics on new data

Main functions it calls, after parsing args:

- krakencoder/model.py/Krakencoder.load_checkpoint()
- krakencoder/data.py/load_hcp_data()
    or data.py/load_input_data()
- krakencoder/data.py/generate_transformer()
    using information from specified saved transformer files
- krakencoder/model.py/Krakencoder()
    - to run forward predictions on data

Examples:
#1. Evaluate checkpoint on held-out "test" split from HCP data, using precomputed PCA input transformers, and save performance metrics and heatmaps
python run_model.py --inputdata test --subjectfile subject_splits_958subj_683train_79val_196test_retestInTest.mat \
    --checkpoint kraken_chkpt_SCFC_20240406_022034_ep002000.pt \
    --inputxform kraken_ioxfm_SCFC_coco439_993subj_pc256_25paths_710train_20220527.npy \
        kraken_ioxfm_SCFC_fs86_993subj_pc256_25paths_710train_20220527.npy \
        kraken_ioxfm_SCFC_shen268_993subj_pc256_25paths_710train_20220527.npy \
    --newtrainrecord hcp_20240406_022034_ep002000_mse.w1000_newver_test.mat \
    --heatmap hcp_20240406_022034_ep002000_mse.w1000_newver_test.png \
    --heatmapmetrics top1acc topNacc avgrank avgcorr_resid \
    --fusion --fusioninclude fusion=all fusionSC=SC fusionFC=FC --fusionnoself --fusionnoatlas

#2. To generate predicted connectomes, add:
    --outputname all --output mydata_20240406_022034_ep002000_{output}.mat
# which will generate an file predictions of each connectivity flavor in the model, named like:
#   mydata_20240406_022034_ep002000_FCcorr_shen268_hpf_FC.mat
# which will contain the predicted FCcorr_shen268_hpf_FC from every input type provided

#3. To generate predicted connectomes from only fusion inputs:
    --output 'mydata_20240406_022034_ep002000_{input}.mat' --fusion --onlyfusioninputs
# or for multiple fusion types (i.e., fusionSC=only using SC inputs):
    --output 'mydata_20240406_022034_ep002000_{input}.mat' --fusion --onlyfusioninputs \
    --fusioninclude fusion=all fusionSC=SC fusionFC=FC --fusionnoself --fusionnoatlas

#4. To generate the latent space outputs for this input data, add:
    --outputname encoded --output mydata_20240406_022034_ep002000_{output}.mat"

#5. To use your own non-HCP input data, provide a .mat file for each input type, with a 'data' field containing the [subjects x region x region] 
# connectivity data. Then include the filenames and connectivity names using:
    --inputdata '[SCsdstream_fs86_volnorm]=mydata_fs86_sdstream_volnorm.mat' \
        '[SCifod2act_fs86_volnorm]=mydata_fs86_ifod2act_volnorm.mat' \
        '[SCsdstream_shen268_volnorm]=mydata_shen268_sdstream_volnorm.mat' \
        '[SCifod2act_shen268_volnorm]=mydata_shen268_ifod2act_volnorm.mat' \
        '[SCsdstream_coco439_volnorm]=mydata_coco439_sdstream_volnorm.mat' \
        '[SCifod2act_coco439_volnorm]=mydata_coco439_ifod2act_volnorm.mat' \
    --adaptmode meanfit+meanshift
#where "--adaptmode meanfit+meanshift" uses minimal approach for domain shift by linearly mapping the population mean of your input data to the 
# population mean of the training data. This is a simple way to adapt the input data to the model, but may not be sufficient for all cases."

"""

from krakencoder.model import *
from krakencoder.train import *
from krakencoder.data import *
from krakencoder.utils import *

from scipy.io import loadmat, savemat
import re

import os
import sys
import argparse
import warnings

def argument_parse_newdata(argv):
    #for list-based inputs, need to specify the defaults this way, otherwise the argparse append just adds to them
    arg_defaults={}
    arg_defaults['fusion_include']=[]
    arg_defaults['input_names']=[]
    arg_defaults['output_names']=[]
    arg_defaults['input_data_file']=[]
    arg_defaults['input_transform_file']=["auto"]
    arg_defaults['heatmap_metrictype_list']=['top1acc','topNacc','avgrank','avgcorr_resid']
    
    parser=argparse.ArgumentParser(description='Evaluate krakencoder checkpoint',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--checkpoint',action='store',dest='checkpoint', help='Checkpoint file (.pt)')
    parser.add_argument('--innercheckpoint',action='store',dest='innercheckpoint', help='Inner checkpoint file (.pt): eg: if checkpoint is an adaptation layer')
    parser.add_argument('--trainrecord',action='store',dest='trainrecord', default='auto', help='trainrecord.mat file')
    parser.add_argument('--inputxform',action='append',dest='input_transform_file', help='Precomputed transformer file (.npy)',nargs='*')
    parser.add_argument('--output',action='store',dest='output', help='file to save model outputs. Can include "{input}" and/or "{output}" in name to save separate files for each input/output combo (or just group inputs and outputs)')
    parser.add_argument('--inputdata',action='append',dest='input_data_file', help='.mat file(s) containing input data to transform (instead of default HCP validation set). Can be "name=file"', nargs='*')
    
    parser.add_argument('--inputname','--inputnames',action='append',dest='input_names', help='Name(s) of input data flavors (eg: FCcorr_fs86_hpf, SCsdstream_fs86, encoded)',nargs='*')
    parser.add_argument('--outputname','--outputnames',action='append',dest='output_names', help='List of data flavors from model to predict (default=all)',nargs='*')
    
    parser.add_argument('--adaptmode',action='store',dest='adapt_mode',default='none',help='How do adapt new data to fit model (default: none)')
    
    parser.add_argument('--fusioninclude',action='append',dest='fusion_include',help='inputnames to include in fusion average',nargs='*')
    parser.add_argument('--fusion',action ='store_true',dest='fusion',help='fusion mode eval')
    parser.add_argument('--fusionnoself','--fusion.noself',action ='store_true',dest='fusion_noself',help='Add .noself version to fusion outputs (excludes latent from same input)')
    parser.add_argument('--fusionnoatlas','--fusion.noatlas',action ='store_true',dest='fusion_noatlas',help='Add .noatlas version to fusion outputs (excludes latent from same atlas)')
    parser.add_argument('--fusionnorm',action='store_true',dest='fusionnorm',help='re-normalize latent vectors after averaging')
    parser.add_argument('--onlyfusioninputs',action='store_true',dest='only_fusion_inputs',help='Only predict outputs from fusion inputs (not from individual flavors)')
    
    misc_group=parser.add_argument_group('Misc')
    misc_group.add_argument('--heatmap',action='store',dest='heatmap_file', help='Save heatmap image')
    misc_group.add_argument('--heatmapmetrics',action='store',dest='heatmap_metrictype_list', help='List of metric types for heatmap',nargs='*')
    misc_group.add_argument('--heatmap_colormap',action='store',dest='heatmap_colormap', default='magma', help='Colormap name for heatmap')
    misc_group.add_argument('--newtrainrecord',action='store',dest='new_train_record_file', help='Save a "fake" trainrecord file')
    misc_group.add_argument('--subjectsplitfile','--subjectfile',action='store',dest='subject_split_file', help='.mat file containing pre-saved "subjects","subjidx_train","subjidx_val","subjidx_test" fields (or "trainrecord" to use from training record)')
    misc_group.add_argument('--subjectsplitname',action='store',dest='subject_split_name', help='Which data split to evaluate: "all", "train", "test", "val", "retest", etc... (overrides --inputdata for hardcoded HCP)')
    
    testing_group=parser.add_argument_group('Testing options')
    testing_group.add_argument('--savetransformedinputs',action='store_true',dest='save_transformed_inputs', help='Transform inputs and save them as "encoded" values (for testing PCA/transformed space data)')
    testing_group.add_argument('--untransformedoutputs',action='store_true',dest='save_untransformed_outputs', help='Keep outputs in PCA/transformed space (for testing)')
    testing_group.add_argument('--hackcortex',action='store_true',dest='hack_cortex', help='Hack to only use cortex for eval (for fs86 and coco439) (for testing)')
    testing_group.add_argument('--ccmat',action='store_true',dest='ccmat', help='Save full SUBJxSUBJ ccmat for every prediction path (large record file)')
    
    parser.add_argument('--version', action='version',version='Krakencoder v{version}'.format(version=get_version(include_date=True)))
    
    args=parser.parse_args(argv)
    args=clean_args(args,arg_defaults)
    return args

def search_flavors(searchstring_list,full_list, return_index=False):
    if isinstance(searchstring_list,str):
        searchstring_list=[searchstring_list]
    
    if len(searchstring_list)==0:
        new_list=full_list.copy()
        new_list_index=[0]*len(full_list)
    else:
        new_list=[]
        new_list_index=[]
        for i,s in enumerate(searchstring_list):
            try:
                s_canonical=canonical_data_flavor(s)
            except:
                s_canonical=None
            
            s_new=[s]
            if s.lower() == 'all':
                s_new=full_list
            elif s.lower() in ['encoded','fusion','transformed']:
                s_new=[s.lower()]
            elif s in full_list:
                s_new=[s]
            elif s_canonical is not None and s_canonical in full_list:
                s_new=[s_canonical]
            elif s.lower()=='sc':
                s_new=[intype for intype in full_list if intype.startswith("SC") or "sdstream" in intype or "ifod2act" in intype]
            elif s.lower()=='fc':
                s_new=[intype for intype in full_list if intype.startswith("FC")]
            else:
                s_new=[intype for intype in full_list if s in intype]
                
            new_list+=s_new
            new_list_index+=[i]*len(s_new)
                
    #return unique elements in list, in their original order
    uidx=np.sort(np.unique(new_list,return_index=True)[1])
    new_list=[str(new_list[i]) for i in uidx]
    new_list_index=[new_list_index[i] for i in uidx]
    
    if return_index:
        return new_list,new_list_index
    else:
        return new_list

def run_model_on_new_data(argv=None):
    if argv is None:
        argv=sys.argv[1:]
    #read in command-line inputs
    args=argument_parse_newdata(argv)
    
    ptfile=args.checkpoint
    innerptfile=args.innercheckpoint
    recordfile=args.trainrecord
    fusionmode=args.fusion
    fusionnorm=args.fusionnorm
    fusion_noself=args.fusion_noself
    fusion_noatlas=args.fusion_noatlas
    input_fusionmode_names=args.fusion_include
    only_fusion_mode=args.only_fusion_inputs
    do_save_transformed_inputs=args.save_transformed_inputs
    outputs_in_model_space=args.save_untransformed_outputs
    outfile = args.output
    input_transform_file_list=args.input_transform_file
    input_conntype_list=args.input_names
    output_conntype_list=args.output_names
    input_file_list=args.input_data_file
    heatmapfile=args.heatmap_file
    heatmap_metrictype_list=args.heatmap_metrictype_list
    heatmap_colormap=args.heatmap_colormap
    new_train_recordfile=args.new_train_record_file
    input_subject_split_file=args.subject_split_file
    input_subject_split_name=args.subject_split_name
    
    adapt_mode=args.adapt_mode
    save_ccmat_in_record=args.ccmat
    
    hack_cortex=args.hack_cortex
    if hack_cortex:
        trimask86=np.triu_indices(86,1) 
        trimask439=np.triu_indices(439,1) 
        cortex86=np.ones((86,86),dtype=bool)
        cortex439=np.ones((439,439),dtype=bool)
        cortex86[:18,:]=False
        cortex86[:,:18]=False
        cortex439[:81,:]=False
        cortex439[:,:81]=False
        hack_cortex_mask={'fs86':cortex86[trimask86],'coco439':cortex439[trimask439]}
        
    if adapt_mode.lower()=='none':
        adapt_mode=None
    
    ##############
    #load model checkpoint
    warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization")
    
    outerptfile=None
    if innerptfile is not None:
        outerptfile=ptfile
        ptfile=innerptfile
        print("Loading inner model from %s" % (ptfile))
        
    net, checkpoint=Krakencoder.load_checkpoint(ptfile)
    
    ##########
    #handle special case for OLD checkpoints before we updated the connectivity flavors
    #if the checkpoint uses the old style of flavor names, convert them to the new style
    try:
        all_old_names=all([canonical_data_flavor_OLD(x)==x for x in checkpoint['input_name_list']])
    except:
        all_old_names=False
    
    if all_old_names:
        print("This checkpoint uses old style of flavor names")
        checkpoint['input_name_list']=[canonical_data_flavor(x) for x in checkpoint['input_name_list']]
        checkpoint['training_params']['trainpath_names']=['%s->%s' % 
                                                        (canonical_data_flavor(x.split("->")[0]),canonical_data_flavor(x.split("->")[1])) 
                                                        for x in checkpoint['training_params']['trainpath_names']]
        checkpoint['training_params']['trainpath_names_short']=['%s->%s' % 
                                                                (canonical_data_flavor(x.split("->")[0]),canonical_data_flavor(x.split("->")[1])) 
                                                                for x in checkpoint['training_params']['trainpath_names_short']]
    ########
    
    conn_names=checkpoint['input_name_list']
    trainpath_pairs = [[conn_names[i],conn_names[j]] for i,j in zip(checkpoint['trainpath_encoder_index_list'], checkpoint['trainpath_decoder_index_list'])]

    #############
    
    #for fusion mode, check whether we are doing multiple fusion types
    if any(['=' in b for b in input_fusionmode_names]):
        orig_input_fusionmode_names=input_fusionmode_names
        input_fusionmode_names={}
        for b in orig_input_fusionmode_names:
            bname=b.split("=")[0]
            blist=flatlist(b.split("=")[-1].split(","))
            input_fusionmode_names[bname]=blist
    else:
        bname='fusion'
        blist=flatlist([b.split(",") for b in input_fusionmode_names])
        input_fusionmode_names={bname:blist}
        
    #note: don't actually use trainrecord during model evaluation
    #checkpoint includes all info about data flavors and model design
    #but doesn't contain any info about loss functions, training schedule, etc...
    #consider: in save_checkpoint, just add the trainrecord keys that aren't the 
    #big per-epoch loss values (ie: we have networkinfo[], make traininfo[] with those params)
    #note: we might use this to get training/testing/val subject info
    if recordfile == "auto":
        recordfile=ptfile.replace("_checkpoint_","_trainrecord_")
        recordfile=recordfile.replace("_chkpt_","_trainrecord_")
        recordfile=re.sub(r"_(epoch|ep)[0-9]+\.pt$",".mat",recordfile)

    if len(input_transform_file_list)>0 and input_transform_file_list[0] == "auto":
        input_transform_file=ptfile.replace("_checkpoint_","_iox_")
        input_transform_file=input_transform_file.replace("_chkpt_","_ioxfm_")
        input_transform_file=re.sub(r"_(epoch|ep)[0-9]+\.pt$",".npy",input_transform_file)
        input_transform_file_list=[input_transform_file]
    
    if input_subject_split_file and input_subject_split_file.lower() == "trainrecord":
        if os.path.exists(recordfile):
            print("Using subject splits from training record: %s" % (recordfile))
            input_subject_split_file=recordfile
        else:
            raise Exception("Training record not found. Cannot use 'trainrecord' subject split option. %s" % (recordfile))         
           
    input_subject_splits=None
    subjects_train=None
    subjects_val=None
    subjects_test=None
    
    subjects_to_eval=None #this might end iup being subjects_test, subjects_val, etc...
    
    if input_subject_split_file:
        subjects_to_eval_splitname='all'
        if input_subject_split_name:
            subjects_to_eval_splitname=input_subject_split_name.lower()
        print("Loading subject splits from %s" % (input_subject_split_file))
        input_subject_splits=loadmat(input_subject_split_file,simplify_cells=True)
        for f in ["subjects", "subjidx_train", "subjidx_val", "subjidx_test"]:
            if not f in input_subject_splits:
                input_subject_splits[f]=[]
            print("\t%d %s" % (len(input_subject_splits[f]),f))
        
        subjects=input_subject_splits['subjects']
        subjects=clean_subject_list(subjects)
        subjects_train=[s for i,s in enumerate(subjects) if i in input_subject_splits['subjidx_train']]
        subjects_val=[s for i,s in enumerate(subjects) if i in input_subject_splits['subjidx_val']]
        subjects_test=[s for i,s in enumerate(subjects) if i in input_subject_splits['subjidx_test']]
    
        if subjects_to_eval_splitname == 'all':
            subjects_to_eval=subjects
        elif 'subjidx_' + subjects_to_eval_splitname in input_subject_splits:
            subjects_to_eval=[subjects[i] for i in input_subject_splits['subjidx_' + subjects_to_eval_splitname]]
        else:
            raise Exception("Invalid subject split name: %s" % (subjects_to_eval_splitname))
        
    if len(input_file_list)>0:
        if input_file_list[0] in ['all','test','train','val','retest']:
            #use HCP data
            pass
        elif(all(["=" in x for x in input_file_list])):
            tmp_inputfiles=input_file_list
            input_conntype_list=[]
            input_file_list=[]
            for x in tmp_inputfiles:
                input_conntype_list+=[x.split("=")[0]]
                input_file_list+=[x.split("=")[-1]]
        else:
            #try to figure out conntypes from filenames
            print("--inputname not provided. Guessing input type from filenames:")
            input_conntype_list=[]
            for x in input_file_list:
                xc=canonical_data_flavor(justfilename(x))
                input_conntype_list+=[xc]
                print("  %s = %s" % (xc,x))
    
    #if input "file list" is an HCP data split name, and we provided a subject split name argument, override the "file list" argument
    if len(input_file_list) > 0 and input_file_list[0].lower() in ['all','test','train','val','retest']:
        if input_subject_split_name in ['all','test','train','val','retest']:
            input_file_list[0]=input_subject_split_name.lower()
    
    ######################
    # parse user-specified input data type
    #conn_names = full list of input names from model checkpoint
    #input_conntype_canonical=canonical_data_flavor(input_conntype)
    #input_conntype_list=[input_conntype_canonical]
    input_conntype_list, input_conntype_idx =search_flavors(input_conntype_list, conn_names, return_index=True)
    
    if len(input_file_list) > 0 and not input_file_list[0] in ['all','test','train','val','retest']:
        #if we provided multiple input files, the conntype list may have been reordered during search_flavors
        # or only a subset were used (if the model checkpoint only accepts certain inputs),
        # so reorder the input files
        input_file_list = [input_file_list[i] for i in input_conntype_idx]
        print("Final conntype = filename mapping:")
        for xc,x in zip(input_conntype_list,input_file_list):
            print("  %s = %s" % (xc,x))
    
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
    
    if fusionmode:
        fusionmode_names_dict={k:search_flavors(v, input_conntype_list) for k,v in input_fusionmode_names.items()}
        
        original_fusionmode_names_dict=fusionmode_names_dict.copy()
        if fusion_noself:
            #note: include all flavors as inputs for this, because the selection is done at the decoding stage
            new_fusionmode_names_dict=fusionmode_names_dict.copy()
            for k in original_fusionmode_names_dict:
                new_fusionmode_names_dict[k+".noself"]=fusionmode_names_dict[k].copy()
            fusionmode_names_dict=new_fusionmode_names_dict
            
        if fusion_noatlas:
            #note: include all flavors as inputs for this, because the selection is done at the decoding stage
            new_fusionmode_names_dict=fusionmode_names_dict.copy()
            for k in original_fusionmode_names_dict:
                new_fusionmode_names_dict[k+".noatlas"]=fusionmode_names_dict[k].copy()
            fusionmode_names_dict=new_fusionmode_names_dict
            
        for k in fusionmode_names_dict:
            print("fusion mode '%s' input types (%d):" % (k,len(fusionmode_names_dict[k])), fusionmode_names_dict[k])
            
    else:
        fusionmode_names_dict={}
    
    #build a list of output files (either consolidated, per input/output or per input->output path)
    if only_fusion_mode and fusionmode_names_dict:
        eval_input_conntype_list=list(fusionmode_names_dict.keys())
    elif fusionmode_names_dict:
        eval_input_conntype_list=input_conntype_list.copy()+list(fusionmode_names_dict.keys())
    else:
        eval_input_conntype_list=input_conntype_list.copy()
    
    
    
    #handle some shortcuts for the input/output filenames
    do_save_output_data=True
    if outfile is None:
        do_save_output_data=False
        outfile="dummy_{input}_{output}.mat"

    outfile_template=outfile
    if re.search("{.+}",outfile_template):
        instrlist=["i","in","input","s","src","source"]
        outstrlist=["o","out","output","t","trg","targ","target"]
        for s in instrlist:
            outfile_template=outfile_template.replace("{"+s+"}","{input}")
        for s in outstrlist:
            outfile_template=outfile_template.replace("{"+s+"}","{output}")
            
    outfile_list=[]
    outfile_input_output_list=[]
    
    if "{input}" in outfile_template and "{output}" in outfile_template:
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
        
    ##############
    #load input transformers
    if not input_transform_file_list and checkpoint['input_transformation_info'].upper()!='NONE':
        print("Must provide input transform (ioxfm) file")
        sys.exit(1)
    
    precomputed_transformer_info_list={}
    if checkpoint['input_transformation_info']=='none':
        for conntype in conn_names:
            precomputed_transformer_info_list[conntype]={'type':'none'}
    else:
        for ioxfile in input_transform_file_list:
            print("Loading precomputed input transformations: %s" % (ioxfile))
            ioxtmp=np.load(ioxfile,allow_pickle=True).item()
            for k in ioxtmp:
                precomputed_transformer_info_list[k]=ioxtmp[k]
        
    transformer_list={}
    transformer_info_list={}
    for conntype in precomputed_transformer_info_list:
        transformer, transformer_info = generate_transformer(traindata=None, 
            transformer_type=precomputed_transformer_info_list[conntype]["type"], transformer_param_dict=None, 
            precomputed_transformer_params=precomputed_transformer_info_list[conntype], return_components=True)
        transformer_list[conntype]=transformer
        transformer_info_list[conntype]=transformer_info
    
    ##########
    #handle special case for OLD saved transformers before we updated the connectivity flavors
    #if the transformers use the old style of flavor names, convert them to the new style
    try:
        all_old_names=all([canonical_data_flavor_OLD(x)==x for x in transformer_list])
    except:
        all_old_names=False
    
    if all_old_names:
        print("The transformers use old style of flavor names")
        transformer_list={canonical_data_flavor(x):transformer_list[x] for x in transformer_list}
        transformer_info_list={canonical_data_flavor(x):transformer_info_list[x] for x in transformer_info_list}
    ########
    
    ######################
    #load input data
    
    output_subject_splits=input_subject_splits

    if len(input_file_list) > 0 and not input_file_list[0] in ['all','test','train','val','retest']:
        #load data from files specified in command line
        
        if subjects_to_eval is not None:
            output_subject_splits=None
        
        conndata_alltypes={}
        for i,x in enumerate(input_conntype_list):
            
            conndata_alltypes[x]=load_input_data(inputfile=input_file_list[i], inputfield=None)
            if 'subjects' in conndata_alltypes[x]:
                conndata_alltypes[x]['subjects']=clean_subject_list(conndata_alltypes[x]['subjects'])

            #how should we adapt the input data to the model?
            if subjects_train is not None:
                subjidx_adapt=np.array([i for i,s in enumerate(conndata_alltypes[x]['subjects']) if s in subjects_train])
            else:
                subjidx_adapt=np.arange(conndata_alltypes[x]['data'].shape[0])
            
            adxfm=generate_adapt_transformer(input_data=conndata_alltypes[x]['data'],
                                        target_data=transformer_info_list[x],
                                        adapt_mode=adapt_mode,
                                        input_data_fitsubjmask=subjidx_adapt)
            conndata_alltypes[x]['data']=adxfm.transform(conndata_alltypes[x]['data'])
            
            if subjects_to_eval is not None:
                subjidx_for_eval=np.array([i for i,s in enumerate(conndata_alltypes[x]['subjects']) if s in subjects_to_eval])
                conndata_alltypes[x]['data']=conndata_alltypes[x]['data'][subjidx_for_eval]
                
            print(x,conndata_alltypes[x]['data'].shape)
    else:
        #load hardcoded HCP data files
        input_file="all"
        if len(input_file_list) > 0:
            if any([x in input_file_list for x in ["train","val","test"]]) and input_subject_splits is None:
                raise Exception("Must provide --subjectsplitfile for train, val, test options")
            subjects_out=None
            conndata_alltypes=None
            
            if any([x in input_file_list for x in ["train","val","test","retest"]]):
                #any time we request ONLY a specific split, after reading in that split's data, we need
                # to set the split dict to None so that the trainrecord/heatmap creation later uses all data
                # from the requested split
                output_subject_splits=None

            for i in input_file_list:
                if i=="all":
                    subjects_tmp, _ = load_hcp_subject_list(numsubj=993)
                    subjects_out_tmp, conndata_alltypes_tmp = load_hcp_data(subjects=subjects_tmp, conn_name_list=input_conntype_list, quiet=False)
                elif i=="retest":
                    subjects_out_tmp, conndata_alltypes_tmp = load_hcp_data(subjects=None, conn_name_list=input_conntype_list, load_retest=True,quiet=False)
                elif i=="train":
                    subjects_out_tmp, conndata_alltypes_tmp = load_hcp_data(subjects=subjects_train, conn_name_list=input_conntype_list, quiet=False)
                elif i=="val":
                    subjects_out_tmp, conndata_alltypes_tmp = load_hcp_data(subjects=subjects_val, conn_name_list=input_conntype_list, quiet=False)
                elif i=="test":
                    subjects_out_tmp, conndata_alltypes_tmp = load_hcp_data(subjects=subjects_test, conn_name_list=input_conntype_list, quiet=False)
                
                if subjects_out is None:
                    subjects_out=subjects_out_tmp
                else:
                    if any([s in subjects_out for s in subjects_out_tmp]):
                        raise Exception("Duplicate subject for in input data list: %s" % (", ".join(input_file_list)))
                    
                    subjects_out=np.concatenate((subjects_out,subjects_out_tmp))
                
                if conndata_alltypes is None:
                    conndata_alltypes=conndata_alltypes_tmp
                else:
                    for conntype in conndata_alltypes.keys():
                        conndata_alltypes[conntype]['data']=np.vstack((conndata_alltypes[conntype]['data'],conndata_alltypes_tmp[conntype]['data']))
                        conndata_alltypes[conntype]['subjects']=subjects_out

   
        else:
            subjects, famidx = load_hcp_subject_list(numsubj=993)
            subjects_out, conndata_alltypes = load_hcp_data(subjects=subjects, conn_name_list=input_conntype_list, quiet=False)
        
        for conntype in conndata_alltypes.keys():
            conndata_alltypes[conntype]['subjects']=clean_subject_list(subjects_out)
    
    #if we have an inner and outer checkpoint, we loaded the inner first, now load outer
    #cant do this until we have all of the data transformers built
    if outerptfile is not None:
        print("Loading outer model from %s" % (outerptfile))
        inner_net=net
        inner_checkpoint_dict=checkpoint
        
        data_transformer_list=[]
        data_inputsize_list=[]
        
        none_transformer, none_transformer_info = generate_transformer(transformer_type="none")
        
        
        for i_conn, conn_name in enumerate(conn_names):
            if conn_name in transformer_info_list:
                transformer, transformer_info = generate_transformer(transformer_type=transformer_info_list[conn_name]["params"]["type"], precomputed_transformer_params=transformer_info_list[conn_name]["params"])
            else:
                transformer=none_transformer
                transformer_info=none_transformer_info
            
            if conn_name in conndata_alltypes:
                inputsize=conndata_alltypes[conn_name]['data'].shape[1]
                #inputsize=1
            else:
                inputsize=1
            
            data_transformer_list+=[transformer]
            data_inputsize_list+=[inputsize]
        
        net, checkpoint = KrakenAdapter.load_checkpoint(filename=outerptfile, inner_model=inner_net, data_transformer_list=data_transformer_list, inner_model_extra_dict=inner_checkpoint_dict)
                  
        
        #now that we've built the outer model with the data transformers built in, we need to reset the data transformers
        #objects to "none"
        transformer_list={}
        transformer_info_list={}  
        
        for conn_name in conn_names:
            transformer_list[conn_name]=none_transformer
            transformer_info_list[conn_name]=none_transformer_info
            
        
    encoded_alltypes={}
    predicted_alltypes={}
    
    neg1_torch=torchint(-1)
            
    net.eval()
    
    #encode all inputs to latent space
    for intype in input_conntype_list:
        if intype == "encoded" or intype.startswith("fusion"):
            encoded_name=intype
            conn_encoded=torchfloat(conndata_alltypes[intype]['data'])
            encoded_alltypes[intype]=numpyvar(conn_encoded)
        else:
            if not intype in conn_names:
                raise Exception("Input type %s not found in model" % (intype))
            encoder_index=[idx for idx,c in enumerate(conn_names) if c==intype][0]
    
            encoder_index_torch=torchint(encoder_index)
    
            inputdata_origscale=conndata_alltypes[intype]['data']
            
            inputdata=torchfloat(transformer_list[intype].transform(inputdata_origscale))
            
            if do_save_transformed_inputs:
                conn_encoded=inputdata
            else:
                with torch.no_grad():
                    conn_encoded = net(inputdata, encoder_index_torch, neg1_torch)
            
            encoded_alltypes[intype]=numpyvar(conn_encoded)
    
    #fusionmode averaging in encoding latent space
    for fusiontype, fusionmode_names in fusionmode_names_dict.items():
        print("%s: fusion mode evaluation. Computing mean of input data flavors in latent space." % (fusiontype))
        print("%s: input types " % (fusiontype),fusionmode_names)
        encoded_mean=None
        encoded_inputtype_count=0
        
        for intype in encoded_alltypes.keys():
            if intype == fusiontype:
                print("fusion type '%s' evaluation already computed in input" % (fusiontype))
                encoded_mean=encoded_alltypes[intype].copy()
                encoded_inputtype_count=1
                break
            else:
                if not intype in fusionmode_names:
                    #only average encodings from specified input types
                    continue
                
                print("fusion type '%s' evaluation includes: %s" % (fusiontype,intype))
                conn_encoded=encoded_alltypes[intype].copy()
            
            if encoded_mean is None:
                encoded_mean=conn_encoded
            else:
                encoded_mean+=conn_encoded
            encoded_inputtype_count+=1
        
        encoded_mean=encoded_mean/encoded_inputtype_count
        encoded_norm=np.sqrt(np.sum(encoded_mean**2,axis=1,keepdims=True))
        
        if fusionnorm:
            print("Mean '%s' vector length before re-normalization: %.4f (renormed to 1.0)" % (fusiontype,np.mean(encoded_norm)))
            encoded_mean=encoded_mean/encoded_norm
        else:
            print("Mean '%s' vector length: %.4f" % (fusiontype,np.mean(encoded_norm)))

        encoded_alltypes[fusiontype]=encoded_mean.copy()
    
    #now decode encoded inputs to requested outputs
    for intype in encoded_alltypes.keys():
        outtypes_for_this_input=np.unique([x['outtype'] for x in outfile_input_output_list if intype in x['intype']])
        if len(outtypes_for_this_input) == 0:
            continue
        
        ############## intergroup
        #use this so we know if its intergroup
        if intype in conn_names:
            encoder_index=[idx for idx,c in enumerate(conn_names) if c==intype][0]
        else:
            encoder_index=-1
        encoder_index_torch=torchint(encoder_index)
            
        ############## end intergroup
        
        predicted_alltypes[intype]={}

        for outtype in outtypes_for_this_input:
            if outtype == "encoded" or outtype=='transformed' or outtype in fusionmode_names_dict:
                continue
            
            if not outtype in conn_names:
                raise Exception("Output type %s not found in model" % (outtype))
                
            if do_self_only and intype != outtype:
                continue
                            
            decoder_index=[idx for idx,c in enumerate(conn_names) if c==outtype][0]
            decoder_index_torch=torchint(decoder_index)
            
            ############# intergroup
            conn_encoded=None
            if intype in fusionmode_names_dict and (".noself" in intype or ".noatlas" in intype):
                pass
            else:
                conn_encoded=torchfloat(encoded_alltypes[intype])
            
            if net.intergroup:
                #net.inputgroup_list has items, but no neames, but net.inputgroup_list should correspond to encoder_index and decoder_index
                #conntype_group_dict=net.inputgroup_list
                conntype_encoder_index_dict={k:i for i,k in enumerate(conn_names)}
                encoded_alltypes_thisgroup={}
                for k_in in encoded_alltypes:
                    if k_in in conntype_encoder_index_dict:
                        encidx=conntype_encoder_index_dict[k_in]
                    else:
                        encidx=-1
                    encoded_alltypes_thisgroup[k_in]=net.intergroup_transform_latent(torchfloat(encoded_alltypes[k_in]),torchint(encidx),decoder_index_torch).cpu().detach().numpy()
                
                if conn_encoded is not None:
                    conn_encoded=net.intergroup_transform_latent(conn_encoded,encoder_index_torch,decoder_index_torch)
            else:
                encoded_alltypes_thisgroup=encoded_alltypes
            ############# end intergroup
            
            noself_str=''
            if intype in fusionmode_names_dict and ".noself" in intype:
                #"noself" fusion mode has to compute a new average of all inputs except self
                noself_intype=[k_in for k_in in fusionmode_names_dict[intype] if k_in != outtype]
                conn_encoded=np.add.reduce([encoded_alltypes_thisgroup[k_in] for k_in in noself_intype])/len(noself_intype)
                if fusionnorm:
                    encoded_norm=np.sqrt(np.sum(conn_encoded**2,axis=1,keepdims=True))
                    conn_encoded=conn_encoded/encoded_norm
                conn_encoded=torchfloat(conn_encoded)
                noself_str=' (%d encoded inputs)' % (len(noself_intype))
                
            elif intype in fusionmode_names_dict and ".noatlas" in intype:
                #"noatlas" fusion mode has to compute a new average of all inputs except its own atlas
                out_atlas=atlas_from_flavors(outtype)
                noself_intype=[k_in for k_in in fusionmode_names_dict[intype] if atlas_from_flavors(k_in) != out_atlas]
                conn_encoded=np.add.reduce([encoded_alltypes_thisgroup[k_in] for k_in in noself_intype])/len(noself_intype)
                if fusionnorm:
                    encoded_norm=np.sqrt(np.sum(conn_encoded**2,axis=1,keepdims=True))
                    conn_encoded=conn_encoded/encoded_norm
                conn_encoded=torchfloat(conn_encoded)
                noself_str=' (%d encoded inputs)' % (len(noself_intype))
                
            elif intype in fusionmode_names_dict and net.intergroup:
                #for intergroup, we have to re-compute averages for each output type, even for fusion mode
                conn_encoded=np.add.reduce([encoded_alltypes_thisgroup[k_in] for k_in in fusionmode_names_dict[intype]])/len(fusionmode_names_dict[intype])
                if fusionnorm:
                    encoded_norm=np.sqrt(np.sum(conn_encoded**2,axis=1,keepdims=True))
                    conn_encoded=conn_encoded/encoded_norm
                conn_encoded=torchfloat(conn_encoded)
                
            if do_save_transformed_inputs:
                conn_predicted=conn_encoded
            else:
                with torch.no_grad():
                    _ , conn_predicted = net(conn_encoded, neg1_torch, decoder_index_torch)
            
            if outputs_in_model_space:
                #for testing, keep outputs in the compressed/PCA model space
                conn_predicted_origscale=conn_predicted.cpu().detach().numpy()
            else:
                conn_predicted_origscale=transformer_list[outtype].inverse_transform(conn_predicted.cpu().detach().numpy())
            predicted_alltypes[intype][outtype]=conn_predicted_origscale.cpu().detach().numpy()
            print("Output %s->%s: %dx%d%s" % (intype,outtype,conn_predicted_origscale.shape[0],conn_predicted_origscale.shape[1],noself_str))

    #now save each output file with the input->output paths requested
    if do_save_output_data:
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

            if fusionmode_names_dict:
                outdict['fusionmode_inputtypes']=fusionmode_names_dict
            
            for encout in ['encoded','transformed']+list(fusionmode_names_dict.keys()):
                if encout in outdict['outputtypes']:
                    for k_in in outdict['inputtypes']:
                        outdict['predicted_alltypes'][k_in]={encout:encoded_alltypes[k_in]}
            
            savemat(outfile,outdict,format='5',do_compression=True)
            print("Saved %d/%d: %s" % (i_out+1,len(outfile_list),outfile))
            for k_in in outdict['predicted_alltypes'].keys():
                for k_out in outdict['predicted_alltypes'][k_in].keys():
                    print("\t%s->%s (%dx%d)" % (k_in,k_out,*outdict['predicted_alltypes'][k_in][k_out].shape))

    if new_train_recordfile is not None or heatmapfile is not None:
        #by default, "all" and provide splitfile, then
        # we computed on all (did not use splitfile), and now we use splitfile to pull out just the subjidx_val for trainrecord/heatmap
        #
        #if we said "test" and provide splitfile, then
        # we computed on just subjidx_test (to potentially save output data), and now we DON'T
        # want to use splitfile for saving trainrecord/heatmap
        #
        #if we provide custom data AND a splitfile, then 
        # we compute on all of the provided data and now use splitfile to pull out just the subjidx_val for trainrecord/heatmap

        if 'topN' in checkpoint:
            topN=checkpoint['topN']
        elif 'training_params' in checkpoint and 'topN' in checkpoint['training_params']:
            topN=checkpoint['training_params']['topN']
        else:
            topN=2

        newrecord={}
        newrecord['recordfile']=ptfile
        newrecord['trainpath_names']=[]
        newrecord['input_name_list']=input_conntype_list
        newrecord['nbepochs']=checkpoint['epoch']
        newrecord['current_epoch']=checkpoint['epoch']
        newrecord['starting_point_file']=ptfile
        newrecord['starting_point_epoch']=checkpoint['epoch']
        newrecord['topN']=topN

        #using whatever subject list was provided as "val" for fake training record
        
        if output_subject_splits:
            for tv in ['subjidx_train','subjidx_val','subj_test']:
                if tv in output_subject_splits:
                    newrecord[tv]=output_subject_splits[tv]
        else:
            subjidx=np.arange(conndata_alltypes[input_conntype_list[0]]['data'].shape[0])
            newrecord['subjidx_val']=subjidx
            newrecord['numsubjects_val']=len(subjidx)
        
        z=np.nan*np.ones((len(predicted_alltypes.keys())*len(output_conntype_list),1))

        metriclist=['corrloss','corrlossN','corrlossRank','avgcorr','explainedvar','avgcorr_resid']
        for m in metriclist:
            for tv in ['train','val']:
                for o in ['','_OrigScale']:
                    newrecord['%s%s_%s' % (m,o,tv)]=z.copy()
                    if m=='avgcorr':
                        newrecord['%s%s_other_%s' % (m,o,tv)]=z.copy()
        
        if save_ccmat_in_record:
            newrecord['ccmat_val']=[np.nan]*(len(predicted_alltypes.keys())*len(output_conntype_list))
        
        newrecord_trainpath_list=[]
        for i_out,outfile in enumerate(outfile_list):
            intype=outfile_input_output_list[i_out]['intype']
            outtype=outfile_input_output_list[i_out]['outtype']
            
            if intype == 'all':
                intype=eval_input_conntype_list
            
            if outtype == 'all':
                outtype=output_conntype_list

            for k_in in intype:
                for k_out in outtype:
                    tpname="%s->%s" % (k_in,k_out)
                    if tpname in newrecord_trainpath_list:
                        continue
                    newrecord_trainpath_list+=[tpname]
                    itp=len(newrecord_trainpath_list)-1

                    for o in ['_OrigScale']:
                        for tv in ['train','val']:

                            if not k_out in conndata_alltypes:
                                #some output flavors of the model were not in the input, so cant test them
                                continue
                            
                            if 'subjidx_'+tv in newrecord:
                                subjidx=newrecord['subjidx_'+tv]
                            else:
                                continue
                                
                            x_true=torchfloat(conndata_alltypes[k_out]['data'][subjidx,:])
                            x_pred=predicted_alltypes[k_in][k_out][subjidx,:]
                            
                            x_pred=torchfloat(x_pred)
                            
                            print("new record for %d: %s (%d %s subj)" % (itp,tpname,len(subjidx),tv))

                            if hack_cortex:
                                if 'fs86' in k_out:
                                    x_true=x_true[:,hack_cortex_mask['fs86']]
                                    x_pred=x_pred[:,hack_cortex_mask['fs86']]
                                elif 'coco439' in k_out:
                                    x_true=x_true[:,hack_cortex_mask['coco439']]
                                    x_pred=x_pred[:,hack_cortex_mask['coco439']]
                                print("Hack cortex: xtrue=(%dx%d), pred=(%dx%d)" % (x_true.shape[0],x_true.shape[1],x_pred.shape[0],x_pred.shape[1]))
                                
                            cc=xycorr(x_true,x_pred)
                            
                            x_true_mean=x_true.mean(axis=0,keepdims=True)
                            cc_resid=xycorr(x_true-x_true_mean, x_pred-x_true_mean)
                            
                            if save_ccmat_in_record and tv=='val':
                                newrecord['ccmat_val'][itp]=np.array(cc).copy()
                                
                            for m in metriclist:
                                metricfield='%s%s_%s' % (m,o,tv)
                                metricfield2=None
                                v=None
                                v2=None
                                if m == 'corrloss':
                                    v=corrtop1acc(cc=cc)
                                elif m == 'corrlossN':
                                    v=corrtopNacc(cc=cc,topn=topN)
                                elif m == 'corrlossRank':
                                    v=corravgrank(cc=cc)
                                elif m == 'avgcorr':
                                    metricfield2='%s%s_other_%s' % (m,o,tv)
                                    v,v2=corr_ident_parts(cc=cc)
                                elif m == 'avgcorr_resid':
                                    v,_=corr_ident_parts(cc=cc_resid)
                                elif m == 'explainedvar':
                                    v=explained_variance_ratio(x_true,x_pred,axis=0)
                                
                                newrecord[metricfield][itp,:] = numpyvar(v)
                                if metricfield2 is not None:
                                    newrecord[metricfield2][itp,:] = numpyvar(v2)
                                
        trainpath_count=len(newrecord_trainpath_list)
        for m in metriclist:
            for tv in ['train','val']:
                for o in ['','_OrigScale']:
                    metricfield='%s%s_%s' % (m,o,tv)
                    newrecord[metricfield]=newrecord[metricfield][:trainpath_count,:]

        newrecord['trainpath_names']=newrecord_trainpath_list
        if new_train_recordfile:
            savemat(new_train_recordfile,newrecord,format='5',do_compression=True)
            print("Saved %s" % (new_train_recordfile))

        if heatmapfile:
            do_single_epoch=True
            display_kraken_heatmap(newrecord,metrictype=heatmap_metrictype_list,origscale=True,single_epoch=do_single_epoch,
                                   colormap=heatmap_colormap,
                                   outputimagefile={'file':heatmapfile,'dpi':200})
            
if __name__ == "__main__":
    if len(sys.argv)<=1:
        argument_parse_newdata(['-h'])
        sys.exit(0)
    run_model_on_new_data(sys.argv[1:])
