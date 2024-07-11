"""
Command-line script for training krakencoder. 

This script takes arguments to specify everything from datasets, flavors, architecture, training params, etc.
It loads data, creates the network, and initiates training.

Main functions it calls, after parsing args:

- krakencoder/data.py/load_hcp_data()
    or data.py/load_input_data()
- krakencoder/train.py/generate_training_paths()
- krakencoder/train.py/train_network()

Example:

python run_training.py --subjectfile subject_splits_993subj_710train_80val_203test_retestInTest.mat \
    --roinames fs86 shen268 coco439 --datagroups SCFC --latentsize 128 --transformation pca256  --latentunit --dropout 0.5 \
    --losstype correye+enceye.w10+neidist+encdist.w10+mse.w1000+latentsimloss.w10000 \
    --epochs 2000 --checkpointepochsevery 500 --displayepochs 25 \
    --inputxform kraken_ioxfm_SCFC_coco439_993subj_pc256_25paths_710train_20220527.npy \
        kraken_ioxfm_SCFC_fs86_993subj_pc256_25paths_710train_20220527.npy \
        kraken_ioxfm_SCFC_shen268_993subj_pc256_25paths_710train_20220527.npy \
   
"""

if __name__ == "__main__":
    #for running in command line on AWS, need to restrict threads so it doesn't freeze during PCA sometimes
    import os
    if os.getenv('USER') == 'ubuntu':
        numthreads=3
        os.environ['OPENBLAS_NUM_THREADS'] = str(numthreads)
        os.environ['MKL_NUM_THREADS'] = str(numthreads)
        os.environ['NUMEXPR_NUM_THREADS']=str(numthreads)

#######################

from krakencoder.model import *
from krakencoder.train import *
from krakencoder.utils import *
from krakencoder.data import *
from krakencoder.log import Logger

import re
import os
import sys
import argparse

def argument_parse_runtraining(argv):
    #for list-based inputs, need to specify the defaults this way, otherwise the argparse append just adds to them
    arg_defaults={}
    arg_defaults['roinames']=["fs86+shen268+coco439"]
    arg_defaults['dataflavors']=["SCifod2act","SCsdstream","FCcorr","FCcorrgsr","FCpcorr"]
    arg_defaults['fcfilt']=["hpf"]
    arg_defaults['pathgroups']=['all']
    arg_defaults['losstype']=['correye+enceye']
    arg_defaults['dropout']=[0]
    arg_defaults['latent_sim_weight']=[5000]
    arg_defaults['explicit_checkpoint_epochs']=[]
    arg_defaults['hiddenlayersizes']=[]
    arg_defaults['dropout_schedule']=None

    parser=argparse.ArgumentParser(description='Train krakencoder', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_arg_group=parser.add_argument_group('Input data options')
    input_arg_group.add_argument('--subjectcount',action='store',dest='subjectcount',type=int, default=993, help='HCPTRAIN: Which HCP subject set? 993 (default) or 420')
    input_arg_group.add_argument('--dataflavors',action='append',dest='dataflavors',help='HCPTRAIN: SCifod2act,SCsdstream,FCcorr,FCcorrgsr,FCpcorr (default=%s)' % (arg_defaults["dataflavors"]),nargs='*')
    input_arg_group.add_argument('--roinames',action='append',dest='roinames',help='HCPTRAIN: fs86,shen268,coco439... (default=%s)' % (arg_defaults["roinames"]),nargs='*')
    input_arg_group.add_argument('--fcfilt',action='append',dest='fcfilt',help='list of hpf, bpf, nofilt (default=%s)' % (arg_defaults["fcfilt"]),nargs='*')
    input_arg_group.add_argument('--inputdata',action='append',dest='input_data_file', help='name=file, name=file, (or name@group=file)... Override HCPTRAIN: .mat file(s) containing input data to transform (instead of default HCP set).', nargs='*')
    input_arg_group.add_argument('--trainvalsplitfrac',action='store',dest='trainval_split_frac',type=float, default=0.8, help='Fraction of subjects for training+val')
    input_arg_group.add_argument('--valsplitfrac',action='store',dest='val_split_frac',type=float, default=0.1, help='Fraction *OF TRAIN+VAL* subjects for validation')
    input_arg_group.add_argument('--subjectsplitfile','--subjectfile',action='store',dest='subject_split_file', help='OVERRIDE trainsplit,valsplit: .mat file containing pre-saved "subjects","subjidx_train","subjidx_val","subjidx_test" fields')

    #data transformation options
    xfm_arg_group=parser.add_argument_group('Input transformation options')
    xfm_arg_group.add_argument('--pcadim',action='store',dest='pcadim', type=int, default=256, help='pca dimensionality reduction (default=256. 0=No PCA)')
    xfm_arg_group.add_argument('--tsvd',action='store_true',dest='use_tsvd', help='use truncated SVD instead of PCA')
    xfm_arg_group.add_argument('--sclognorm',action='store_true',dest='sc_lognorm', help='non-PCA runs use log transform for SC')
    xfm_arg_group.add_argument('--sctsvd',action='store_true',dest='sc_tsvd', help='use Truncated SVD for SC')
    xfm_arg_group.add_argument('--transformation',action='store',dest='transformation', help='transformation type string (eg: "pca256", overrides pcadim, tsvd,etc)')
    xfm_arg_group.add_argument('--inputxform',action='append',dest='input_transform_file', help='Precomputed transformer files (.npy)',nargs='*')

    #model options
    model_arg_group=parser.add_argument_group('Model architecture options')
    model_arg_group.add_argument('--hiddenlayersizes',action='append',dest='hiddenlayersizes',type=int,help='hidden layer sizes',nargs='*')
    model_arg_group.add_argument('--latentsize',action='store',dest='latentsize', type=int, default=128, help='latent space dimension')
    model_arg_group.add_argument('--leakyrelu',action='store',dest='leakyrelu_negative_slope', type=float, default=0., help='Leaky ReLU negative slope (0=ReLU). For deep networks only')
    model_arg_group.add_argument('--startingpoint',action='store',dest='starting_point_file', help='.pt file to START with')
    model_arg_group.add_argument('--pathgroups','--datagroups',action='append',dest='pathgroups',help='Training path groups: list of SCFC (all), FC, SC, FC2SC, etc...',nargs='*')
    model_arg_group.add_argument('--trainpaths',action='append',dest='trainpaths',help='Specific paths to train. eg \"SCifod2act_fs86->FCpcorr_fs86_hpf\"',nargs='*')
    model_arg_group.add_argument('--domainadapt',action='store_true',dest='domain_adaptation', help='Add outer domain adapation transforms (for out-of-sample data)')
    model_arg_group.add_argument('--domainadaptpoly',action='store',dest='domain_adaptation_polynomial',type=int, default=1, help='Polynomial order for domain adapation transforms (default = 1)')
    
    loss_arg_group=parser.add_argument_group('Loss function parameters')
    loss_arg_group.add_argument('--losstype',action='append',dest='losstype',help='list of correye+enceye, dist+encdist, etc...',nargs='*')
    loss_arg_group.add_argument('--latentunit',action='store_true',dest='latentunit', help='latent all normalzed to shell of unit sphere')
    loss_arg_group.add_argument('--latentradweight',action='store',dest='latentradweight', type=float, default=10, help='weight to apply to keeping latent rad<1')
    loss_arg_group.add_argument('--mseweight',action='store',dest='mseweight', type=float, default=1, help='Weight to apply to true->predicted MSE')
    loss_arg_group.add_argument('--latentinnerweight',action='store',dest='latent_inner_loss_weight', type=float, default=10, help='Weight to apply to latent-space inner loop loss (enceye,encdist, etc...)')
    loss_arg_group.add_argument('--latentsimweight',action='append',dest='latent_sim_weight',type=float,help='list of latentsimloss weights to try . default=5000',nargs='*')

    train_arg_group=parser.add_argument_group('Training parameters')
    train_arg_group.add_argument('--epochs',action='store',dest='epochs',type=int, default=5000, help='number of epochs (0=evaluate existing network only)')
    train_arg_group.add_argument('--batchsize',dest='batch_size',type=int,default=41,help='main batch size. default=41 (no batch)')
    train_arg_group.add_argument('--dropout',action='append',dest='dropout',type=float,help='list of dropouts to try',nargs='*')
    train_arg_group.add_argument('--dropout_schedule',action='store',dest='dropout_schedule',type=float,help='pair of init, final dropout fractions',nargs='*')
    train_arg_group.add_argument('--dropout_final_layer',action='store',dest='dropout_final_layer',type=float,help='use different dropout for final decoder layer')
    train_arg_group.add_argument('--adamdecay',action='store',dest='adam_decay',type=float, default=0.01, help='Adam weight decay')
    train_arg_group.add_argument('--learningrate',action='store',dest='learning_rate',type=float, default=1e-4, help='Learning rate')
    train_arg_group.add_argument('--skipself',action='store_true',dest='skipself', help='Skip A->A paths during training')
    train_arg_group.add_argument('--roundtrip',action='store_true',dest='roundtrip', help='roundtrip training paths A->B->A')
    train_arg_group.add_argument('--addroundtripepochs',action='store',dest='add_roundtrip_epochs', type=int, default=0, help='add roundtrip training paths A->B->A AFTER normal training')
    train_arg_group.add_argument('--addmeanlatentepochs',action='store',dest='add_meanlatent_epochs', type=int, default=0, help='add meanlatent training paths AFTER normal training')
    train_arg_group.add_argument('--trainblocks',action='store',dest='trainblocks', type=int, default=1, help='How many total times perform normal training + (roundtrip or meanlatent) set? (optimizer resets each block)')
    train_arg_group.add_argument('--randseed',action='store',dest='random_seed',type=int, default=0, help='Specify random seed for initialization')
    
    fixed_arg_group=parser.add_argument_group('Target-encoding options (Train new data to match pre-trained latent representation)')
    fixed_arg_group.add_argument('--encodedinputfile',action='store',dest='encoded_input_file', help='.mat file containing latent space data')
    fixed_arg_group.add_argument('--targetencoding',action='store_true',dest='target_encoding', help='Train encoders/decoders while trying to match latent->target')
    fixed_arg_group.add_argument('--fixedtargetencoding',action='store_true',dest='fixed_target_encoding', help='Just train encoders/decoders to match FIXED (input->fixed, fixed->output) --encodinginputfile')
    fixed_arg_group.add_argument('--onlyselfpathtargetencoding',action='store_true',dest='only_self_target_encoding', help='Only train each input to itself (no cross-flavors)')
    fixed_arg_group.add_argument('--targetencodingname',action='store',dest='target_encoding_name', help='Encoding type for target-encoding ("self" for per-flavor latent space input, "fusion", or specific flavor)')
    fixed_arg_group.add_argument('--addfixedencodingepochsafter',action='store',dest='add_fixed_encoding_epochs_after', type=int, default=0, help='Add fixedencoding epochs AFTER normal epochs')
    fixed_arg_group.add_argument('--addfixedencodingepochsbefore',action='store',dest='add_fixed_encoding_epochs_before', type=int, default=0, help='Add fixedencoding epochs BEFORE normal epochs')

    misc_arg_group=parser.add_argument_group('Other options')
    misc_arg_group.add_argument('--checkpointepochsevery','--checkpointsevery',action='store',dest='checkpoint_epochs_every', type=int, default=1000, help='How often to save checkpoints')
    misc_arg_group.add_argument('--explicitcheckpointepochs',action='append',dest='explicit_checkpoint_epochs', type=int, help='Explicit list of epochs at which to save checkpoints',nargs='*')
    misc_arg_group.add_argument('--displayepochs',action='store',dest='display_epochs', type=int, default=100, help='How often to print training progress')
    misc_arg_group.add_argument('--optimizercheckpoint',action='store_true',dest='optimizer_in_checkpoint',help='Include optimizer params in checkpoint (allows resumed training)')
    misc_arg_group.add_argument('--maxthreads',action='store',dest='max_threads', type=int, default=10, help='How many CPU threads to use')
    misc_arg_group.add_argument('--outputprefix',action='store',dest='output_file_prefix', default="kraken", help='Prefix for output files')
    misc_arg_group.add_argument('--logfile',action='store',dest='logfile', default='auto',help='Optional file to print outputs to (along with stdout). "auto"=<prefix>_log_*.txt')

    misc_arg_group.add_argument('--intergroup',action='store_true',dest='intergroup', help='Do separate inter-group mapping (experimental)')
    misc_arg_group.add_argument('--intergroup_extra_layer_count',action='store',dest='intergroup_extra_layer_count',type=int,default=0,help='How many extra layers for inter-group? (experimental)')
    misc_arg_group.add_argument('--intergroup_skip_relu',action='store_true',dest='intergroup_skip_relu',help='No ReLU in inter-group mapping (experimental)')
    misc_arg_group.add_argument('--intergroup_dropout',action='store',dest='intergroup_dropout',type=float,default=None,help='Dropout for inter-group mapping (experimental)')
    
    misc_arg_group.add_argument('--dropout_final_layer_dict',action='store',dest='dropout_final_layer_dict',type=str,nargs='*',help='use different dropout for final decoder layer... flavor list hack')
    
    misc_arg_group.add_argument('--discard_origscale',action='store_true',dest='discard_origscale',help='Throw out origscale data (for memory). OrigScale performance computed on reconstructed invPCA(PCA(input))')
    
    misc_arg_group.add_argument('--version', action='version',version='Krakencoder v{version}'.format(version=get_version(include_date=True)))
    
    args=parser.parse_args(argv)
    args=clean_args(args,arg_defaults)
    return args

#######################################################
#######################################################
#### 

def run_training_command(argv=None):
    if argv is None:
        argv=sys.argv[1:]
    #add this so we can redirect to "| tee" if necessary
    sys.stdout.reconfigure(line_buffering=True)

    args=argument_parse_runtraining(argv)

    ##################
    trainthreads=args.max_threads
    input_epochs=args.epochs
    input_roundtrip=args.roundtrip
    input_learningrate=args.learning_rate
    input_adamdecay=args.adam_decay
    input_pcadim=args.pcadim
    input_use_tsvd=args.use_tsvd
    do_use_lognorm_for_sc=args.sc_lognorm
    do_use_tsvd_for_sc=args.sc_tsvd
    input_transform_file_list=args.input_transform_file
    transformation_type_string=args.transformation
    add_roundtrip_epochs=args.add_roundtrip_epochs
    add_meanlatent_epochs=args.add_meanlatent_epochs
    input_trainblocks=args.trainblocks
    checkpoint_epochs=args.checkpoint_epochs_every
    explicit_checkpoint_epoch_list=args.explicit_checkpoint_epochs
    do_skipself=args.skipself
    latentsize=args.latentsize
    input_latentradweight = args.latentradweight
    input_latentunit = args.latentunit
    input_lossnames=args.losstype
    input_dropout_list=args.dropout
    input_latentsimweight_list=args.latent_sim_weight
    input_hiddenlayers=args.hiddenlayersizes
    input_leakyrelu=args.leakyrelu_negative_slope
    input_mse_weight=args.mseweight
    input_latent_inner_loss_weight=args.latent_inner_loss_weight
    input_batchsize=args.batch_size
    input_encodingfile=args.encoded_input_file
    do_fixed_target_encoding=args.fixed_target_encoding
    do_target_encoding=args.target_encoding
    do_only_self_target_encoding=args.only_self_target_encoding
    fixed_encoding_input_name=args.target_encoding_name
    add_fixed_encoding_epochs_after=args.add_fixed_encoding_epochs_after
    add_fixed_encoding_epochs_before=args.add_fixed_encoding_epochs_before
    starting_point_file=args.starting_point_file
    trainval_split_frac=args.trainval_split_frac
    val_split_frac=args.val_split_frac
    output_file_prefix=args.output_file_prefix
    display_epochs=args.display_epochs
    optimizer_in_checkpoint=args.optimizer_in_checkpoint
    logfile=args.logfile
    random_seed_val=args.random_seed
    do_domain_adaptation=args.domain_adaptation
    domain_adaptation_polynomial=args.domain_adaptation_polynomial

    if do_domain_adaptation and starting_point_file is None:
        raise Exception("Must specify starting point file to use domain adaptation")
    
    dropout_schedule_list=args.dropout_schedule
    input_dropout_init=None
    if dropout_schedule_list is not None:
        input_dropout_init=dropout_schedule_list[0]
    
    if input_dropout_init is not None:
        input_dropout_list=[input_dropout_init]
    
    dropout_final_layer=args.dropout_final_layer
    dropout_final_layer_dict_arg=args.dropout_final_layer_dict
    try:
        dropout_final_layer_dict={canonical_data_flavor(x.split('=')[0]):float(x.split('=')[1]) for x in dropout_final_layer_dict_arg}
    except:
        dropout_final_layer_dict=None
    
    keep_origscale_data=not args.discard_origscale
    
    ############### intergroup
    intergroup=args.intergroup
    intergroup_extra_layer_count=args.intergroup_extra_layer_count
    intergroup_skip_relu=args.intergroup_skip_relu
    intergroup_dropout=args.intergroup_dropout
    ############## end intergroup
    
    print("") #print blank to space out console output but not logfile
    
    log=None
    if logfile is not None:
        #this will override print() to go to stdout and logfile
        log=Logger(logfile=logfile,append=False)

    print("Command: ", " ".join(sys.argv))
    print("")
    
    #################################

    #load input data

    #ultimately, need:
    #subjects, familyidx, conndata_alltypes
    #conndata_alltypes[conn_name]={'data':np.vstack(Ctriu),'numpairs':npairs,'group':ci['group'],'transformer_file':transformer_file}
    #
    #subjidx_train, subjidx_val (computed below at beginning )
    
    #path specification:
    #option 1: args.trainpaths = ["name1->name2","name1->name3","name2->name3",...]
    #   explicit list of exact input names (keys in conndata_alltypes)
    #option 2: args.pathgroups = ["SCFC","FC", "SC","FC2SC","SC2FC"]
    #   groups that are specified in conndata_alltypes["name1"]["group"]="SC", etc..
    #   SCFC=all paths
    #   SC or FC: just within one modality
    #   FC2SC: within modality and FC->SC
    #   SC2SC: within modality and SC->FC

    #if hcpdata and subjsplitfile:
    #   subjects=load_subj_split_file
    #   conndata_alltypes=load_hcp_data(subjects)
    #
    #if hcpdata and NOT subjsplitfile:
    #   subjects=load_hcp_subject_list(numsubj)
    #   conndata_alltypes=load_hcp_data(subjects)
    #
    #if newdata and subjsplitfile:
    #   subjects=load_split_file
    #   conndata_alltypes=load_input_data(subjects,...) #can do subset IF 'subjects' in conndata_alltypes
    #
    #if newdata and NOT subjsplitfile:
    #   conndata_alltypes=load_input_data(...)
    #   subjects=np.arange(numsubj)
    #
    #if no subjects, subjects=np.arange(numsubj)
    #always: go through all conndata_alltypes and confirm subjects match
    
    input_data_file_list=args.input_data_file
    input_subject_split_file=args.subject_split_file

    input_subject_splits=None
    subjects=None
    familyidx=None #only used when computing test/train splits (and not used if subject_split_file provided)

    subj_str=""
    if input_subject_split_file:
        print("Loading subject splits from %s" % (input_subject_split_file))
        input_subject_splits=loadmat(input_subject_split_file,simplify_cells=True)
        for f in ["subjects", "subjidx_train", "subjidx_val", "subjidx_test"]:
            print("\t%d %s" % (len(input_subject_splits[f]),f))

        if "familyidx" in input_subject_splits:
            familyidx=input_subject_splits["familyidx"]
        
        subjects=input_subject_splits['subjects']
        subjects=clean_subject_list(subjects)

        subj_str="%dsubj" % (len(subjects))

        #string to designate if we are using alternate train/val/test split with all retest subjects in held-out TEST
        if "710train_80val_203test_retestInTest" in input_subject_split_file:
            subj_str="%dsubjB" % (len(subjects))

    conndata_alltypes={}
    if input_data_file_list:
        #load input data from files given as command-line arguments
        input_file_list=[]
        input_conntype_list=[]
        if not all(["=" in f for f in input_data_file_list]):
            print("Input data flavors not specified. Guessing from filenames")

        for f in input_data_file_list:
            fsplit=f.split("=")
            if len(fsplit)==1:
                xc=fsplit[0]
                input_file=fsplit[0]
            else:
                xc=fsplit[0]
                input_file=fsplit[1]
            if "@" in xc:
                xc,input_groupname=xc.split("@")
            else:
                input_groupname=None
            xc,groupname=canonical_data_flavor(xc, only_if_brackets=True, return_groupname=True)
            if input_groupname is not None:
                groupname=input_groupname

            #check groupname to see if it is include in any of the "pathgroups"
            if (args.pathgroups is None or len(args.pathgroups)==0 
                or any([x.lower() == 'all' for x in args.pathgroups]) 
                or any([x.lower() == 'self' for x in args.pathgroups])):
                #ignore groupnames for "all" or "self"
                pass
            elif groupname is not None:
                xc_in_pathgroups=any(groupname in x for x in args.pathgroups)
                if not xc_in_pathgroups:
                    print("Skipping input %s: Group %s for not found in pathgroup input %s: %s" % (xc,groupname,args.pathgroups, f))
                    #dont load this input file
                    continue
            
            conndata_alltypes[xc]=load_input_data(input_file,group=groupname)

            print("%s@%s=%s" % (xc,groupname,input_file))
        input_conntype_list=conndata_alltypes.keys()
        roilist_str="%dflav" % (len(input_conntype_list)) #use a shorter string tro avoid filename issues
    else:
        #load hardcoded HCP data paths
        input_nsubj=args.subjectcount

        if subjects is None or len(subjects)==0:
            subjects, familyidx = load_hcp_subject_list(input_nsubj)
    
        input_roilist=flatlist([x.split("+") for x in args.roinames])

        roilist_str="+".join(input_roilist)
        
        fcfilt_types=args.fcfilt
        #input_dataflavors=[canonical_data_flavor(xc,return_groupname=False) for xc in args.dataflavors]
        input_dataflavors=args.dataflavors
        #input_dataflavors = "FCpcorr" or "SCifod2act" (no roi or fcfilt info)
        
        filttypes_full=["hpf","bpf","nofilt"]
        sctypes_full=["ifod2act","sdstream"]
        fctypes_full=["FCcorr","FCcorrgsr","FCpcorr"]
        #fcfilt_types=[x for x in filttypes_full if any([x in y for y in input_dataflavors])]
        sctypes=[x+"_volnorm" for x in sctypes_full if any([x in y for y in input_dataflavors])]
        #fctypes=[x for x in fctypes_full if any([x in y for y in input_dataflavors])]
        fctypes=[]
        if any(["FCcorr" in y and not "gsr" in y for y in input_dataflavors]):
            fctypes+=["FCcorr"]
        if any(["FCcorr" in y and "gsr" in y for y in input_dataflavors]):
            fctypes+=["FCcorrgsr"]
        if any(["FCpcorr" in y for y in input_dataflavors]):
            fctypes+=["FCpcorr"]

        #dont load FC inputs if only training SC paths, etc
        sc_in_pathgroups=True
        fc_in_pathgroups=True
        if args.pathgroups is not None and len(args.pathgroups)>0 and not any([x.lower() == 'all' for x in args.pathgroups]):
            fc_in_pathgroups=any("FC" in x for x in args.pathgroups)
            sc_in_pathgroups=any("SC" in x for x in args.pathgroups)
        
        input_conn_name_list=get_hcp_data_flavors(roi_list=input_roilist, fc_type_list=fctypes ,sc_type_list=sctypes, fc_filter_list=fcfilt_types, 
                                                  sc=sc_in_pathgroups, fc=fc_in_pathgroups)
        
        subjects_out, conndata_alltypes = load_hcp_data(subjects=subjects, conn_name_list=input_conn_name_list)
    
    nsubj=conndata_alltypes[list(conndata_alltypes.keys())[0]]['data'].shape[0]

    input_conn_name_list=list(conndata_alltypes.keys())

    input_pathgroups=args.pathgroups

    input_trainpath_pairs=[]

    if args.trainpaths is not None and len(args.trainpaths)>0:
        input_pathgroups=['all']
        if not all(["->" in x for x in args.trainpaths]):
            raise Exception("Training paths must be specified as 'x->y'")
        tmp_trainpaths=[x.split("->") for x in args.trainpaths]
        for x0,y0 in tmp_trainpaths:
            x=canonical_data_flavor(x0,only_if_brackets=True)
            y=canonical_data_flavor(y0,only_if_brackets=True)
            if x not in input_conn_name_list or y not in input_conn_name_list:
                raise Exception("Could not find inputs '%s' or '%s' in data" % (x,y))
            input_trainpath_pairs+=[[x,y]]

    if subjects is None or len(subjects)==0:
        if "subjects" in conndata_alltypes[list(conndata_alltypes.keys())[0]]:
            subjects=conndata_alltypes[list(conndata_alltypes.keys())[0]]["subjects"]
        else:
            subjects=np.arange(conndata_alltypes[list(conndata_alltypes.keys())[0]]["data"].shape[0])
    
    for k in conndata_alltypes:
        if conndata_alltypes[k]["data"].shape[0] != len(subjects):
            raise Exception("Number of subjects not consistent for '%s': %d expected, but %d found" 
                            % (k,len(subjects),conndata_alltypes[k]["data"].shape[0]))
        
        if "subjects" in conndata_alltypes[k] and not all(subjects==conndata_alltypes[k]["subjects"]):
            raise Exception("Subjects in data are not identical to expected for '%s'" % (k))
    if not subj_str:
        subj_str="%dsubj" % (len(subjects))
    

    conn_names=list(conndata_alltypes.keys())

    ##################################
    #load checkpoint file for initial starting point, if given
    checkpoint_pretrained=None
    if starting_point_file:
        print("Loading pretrained network as starting point: %s" % (starting_point_file))
        _, checkpoint_pretrained=Krakencoder.load_checkpoint(starting_point_file)

        #if using startingpoint, replace any inputs/defaults related to network architecture
        # and input transformations with information from checkpoint
        #conn_names=[c for c in checkpoint_pretrained['input_name_list'] if c in conn_names]
        conn_names=checkpoint_pretrained['input_name_list']
        input_hiddenlayers=checkpoint_pretrained['hiddenlayers']
        latentsize=checkpoint_pretrained['latentsize']
        input_leakyrelu=checkpoint_pretrained['leakyrelu_negative_slope']
        input_use_tsvd=checkpoint_pretrained['use_truncated_svd']
        input_pcadim=checkpoint_pretrained['reduce_dimension']
        transformation_type_string=checkpoint_pretrained['input_transformation_info']
        do_use_lognorm_for_sc=checkpoint_pretrained['use_lognorm_for_sc']
        do_use_tsvd_for_sc=checkpoint_pretrained['use_truncated_svd_for_sc']
        input_latentunit=checkpoint_pretrained['latent_normalize']


    ##################################
    precomputed_transformer_info_list={}
    if input_transform_file_list is not None and len(input_transform_file_list)>0:
        precomputed_transformer_info_list={}
        for ioxfile in input_transform_file_list:
            print("Loading precomputed input transformations: %s" % (ioxfile))
            ioxtmp=np.load(ioxfile,allow_pickle=True).item()
            for k in ioxtmp:
                precomputed_transformer_info_list[k]=ioxtmp[k]
                precomputed_transformer_info_list[k]['filename']=ioxfile.split(os.sep)[-1]


    if input_latentunit:
        input_latentradweight=0

    training_params_listdict={}
    training_params_listdict['latentsize']=[latentsize]
    training_params_listdict['losstype']=input_lossnames
    training_params_listdict['latent_inner_loss_weight']=[input_latent_inner_loss_weight]
    training_params_listdict['hiddenlayers']=[ input_hiddenlayers ]
    training_params_listdict['dropout']=input_dropout_list
    training_params_listdict['dropout_schedule_list']=[dropout_schedule_list]
    training_params_listdict['dropout_final_layer']=[dropout_final_layer]
    training_params_listdict['batchsize']=[input_batchsize]
    training_params_listdict['latentsim_loss_weight']=input_latentsimweight_list
    training_params_listdict['adam_decay']=[input_adamdecay]
    training_params_listdict['mse_weight']=[input_mse_weight]
    training_params_listdict['learningrate']=[input_learningrate]
    training_params_listdict['nbepochs']=[input_epochs]
    training_params_listdict['skip_relu']=[False]
    training_params_listdict['optimizer_name']=["adamw"]
    training_params_listdict['zerograd_none']=[True]
    training_params_listdict['relu_tanh_alternate']=[False]
    training_params_listdict['leakyrelu_negative_slope']=[input_leakyrelu]
    training_params_listdict['origscalecorr_epochs']=[display_epochs]
    
    if input_pcadim == 0:
        training_params_listdict['reduce_dimension']=[None]
    else:
        training_params_listdict['reduce_dimension']=[input_pcadim]
    
    training_params_listdict['use_truncated_svd']=[input_use_tsvd]
    training_params_listdict['trainpath_shuffle']=[True]
    training_params_listdict['latent_maxrad_weight']=[input_latentradweight]
    training_params_listdict['latent_normalize']=[input_latentunit]
    training_params_listdict['target_encoding']=[do_target_encoding]
    training_params_listdict['fixed_target_encoding']=[do_fixed_target_encoding]
    training_params_listdict['meantarget_latentsim']=[False]
    training_params_listdict['trainblocks']=[input_trainblocks]
    training_params_listdict['roundtrip']=[input_roundtrip]
    training_params_listdict['trainval_split_frac']=[trainval_split_frac]
    training_params_listdict['val_split_frac']=[val_split_frac]

    ############## intergroup
    training_params_listdict['intergroup']=[intergroup]
    training_params_listdict['intergroup_extra_layer_count']=[intergroup_extra_layer_count]
    training_params_listdict['intergroup_relu']=[not intergroup_skip_relu]
    training_params_listdict['intergroup_dropout']=[intergroup_dropout]
    ############## end intergroup
    
    ############# dropout_final_layer_scale_dict
    if dropout_final_layer_dict is not None:
        dropout_final_layer_list=[dropout_final_layer_dict[k] if k in dropout_final_layer_dict else dropout_final_layer for k in conn_names]
        if not all([x is None for x in dropout_final_layer_list]):
            print("Using input-specific dropout_final_layer:")
            for i,c in enumerate(conn_names):
                print("%d: %s = %g" % (i,c,dropout_final_layer_list[i]))
            training_params_listdict['dropout_final_layer_list']=[dropout_final_layer_list]
    ##############
    if checkpoint_pretrained is not None:
        training_params_listdict['starting_point_file']=[starting_point_file]
        training_params_listdict['starting_point_epoch']=[checkpoint_pretrained['epoch']]
    
    #generate a new list of dictionaries with every combination of fields (order we built the dict matters)
    training_params_list = dict_combination_list(training_params_listdict, reverse_field_order=True)    
 
    crosstrain_repeats=1 #crosstrain_repeats (non-self paths)
    reduce_dimension_default=256
    
    extra_trainrecord_dict={}
    extra_trainrecord_dict['command']=" ".join(sys.argv)
    ######################

    for training_params in training_params_list:
        # if only a single set of training parameters was specified (no lists), this will only loop one time
        # In general, it's probably better to do it one-at-a-time from a script so you can see each model's
        # log separately
        
        ###################
        #copy over some params each time through the loop in case they were
        #modified in the loop
        batchsize=training_params['batchsize']
        if 'reduce_dimension' in training_params:
            reduce_dimension=training_params['reduce_dimension']
        else:
            reduce_dimension=reduce_dimension_default
        
        if 'use_truncated_svd' in training_params:
            use_truncated_svd=training_params['use_truncated_svd']
        else:
            use_truncated_svd=False
        
        ####################

        if input_subject_splits:
            subjidx_train=input_subject_splits['subjidx_train'].copy()
            subjidx_val=input_subject_splits['subjidx_val'].copy()
            subjidx_test=input_subject_splits['subjidx_test'].copy()
        else:
            trainval_test_seed=0
            train_val_seed=0
            #if subj>420, first train_frac=0.8, then 0.9 (val=0.1)
            #if subj<=420, first train_frac=0.8, then 0.875 (val=.125)

            if familyidx is not None and len(familyidx) > 0:
                #if familyidx was provided, use these to make sure subjects with the same familyidx are always in the same split
                subjidx_trainval, subjidx_test, familyidx_trainval, familyidx_test = random_train_test_split_groups(groups=familyidx, numsubj=nsubj, 
                                                                                                                seed=trainval_test_seed,
                                                                                                                train_frac=training_params['trainval_split_frac'])
                #split train/val from within initial trainval
                subjidx_train, subjidx_val, familyidx_train, familyidx_val = random_train_test_split_groups(groups=familyidx_trainval, subjlist=subjidx_trainval,
                                                                                                    seed=train_val_seed, 
                                                                                                    train_frac=1-training_params['val_split_frac'])
            else:
                #if familyidx not provided, use basic split
                subjidx_trainval, subjidx_test= random_train_test_split(numsubj=nsubj,
                                                                        seed=trainval_test_seed,
                                                                        train_frac=training_params['trainval_split_frac'])
                #split train/val from within initial trainval
                subjidx_train, subjidx_val = random_train_test_split(subjlist=subjidx_trainval,
                                                                     seed=train_val_seed, 
                                                                     train_frac=1-training_params['val_split_frac'])
        ####################

        for grouptype in input_pathgroups:
            ###################
 
            trainpath_pairs=[] #only this direction 
            trainpath_group_pairs=[] #all pairs by default

            if len(input_trainpath_pairs) > 0:
                trainpath_pairs = input_trainpath_pairs
                trainpath_group_pairs=[]
                #data_string="+".join(["%s2%s" % (x,y) for x,y in input_trainpath_flavor_pairs])
                data_string="userpath"
            else:
                trainpath_group_pairs=[]
                for x in input_conn_name_list:
                    for y in input_conn_name_list:
                        xg=conndata_alltypes[x]['group']
                        yg=conndata_alltypes[y]['group']
                        if grouptype == 'self':
                            if x!=y:
                                continue #skip all but self
                        if xg is None or yg is None:
                            #if no group provided, assume all paths valid
                            pass
                        elif grouptype == 'all':
                            pass
                        elif grouptype == 'SCFC':
                            pass
                        elif grouptype == 'SC2FC':
                            if xg == 'FC' and yg == 'SC':
                                continue #skip FC->SC (keep SC<->SC, FC<->FC, SC->FC)
                        elif grouptype == 'FC2SC':
                            if xg == 'SC' and yg == 'FC':
                                continue #skip SC->FC (keep SC<->SC, FC<->FC, FC->SC)
                        elif xg==grouptype and yg==grouptype:
                            pass
                        trainpath_pairs+=[[x,y]]
                data_string=grouptype
            
            data_string+="_"+roilist_str
            data_string+="_"+subj_str

            data_string=re.sub("^_+","",data_string)
            
            set_random_seed(random_seed_val)
            print("Random seed: %d" % (random_seed_val))
            
            #generate trainpath info each time so the dataloader batches are reproducible
            encoded_inputs=None
            if input_encodingfile:
                Mtmp=loadmat(input_encodingfile,simplify_cells=True)
                if not 'subjects' in Mtmp:
                    raise Exception("input encoding file must have 'subjects' field")
                
                Mtmp['subjects']=clean_subject_list(Mtmp['subjects'])

                if len(Mtmp['subjects']) != len(subjects):
                    raise Exception("input encoding file must contain the same number of subjects (%d) as input data (%d)", len(Mtmp['subjects']),len(subjects))

                if not all([Mtmp['subjects'][i]==subjects[i] for i in range(len(subjects))]):
                    raise Exception("input encoding subjects must match input data subjects")

                fixed_encoding_input_name_found=fixed_encoding_input_name

                if 'encoded' in Mtmp:
                    encoded_inputs={'*':Mtmp['encoded'].copy()}
                    fixed_encoding_input_name_found='encoded'
                elif 'predicted_alltypes' in Mtmp:
                    if len(Mtmp['predicted_alltypes'].keys())==1 and 'encoded' in Mtmp['predicted_alltypes'][list(Mtmp['predicted_alltypes'].keys())[0]]:
                        #if only a single field is found in Mtmp['predicted_alltypes'], use that
                        tmp_encoding_input_name=list(Mtmp['predicted_alltypes'].keys())[0]
                        tmp_shape=Mtmp['predicted_alltypes'][tmp_encoding_input_name]['encoded'].shape

                        encoded_inputs={'*': Mtmp['predicted_alltypes'][tmp_encoding_input_name]['encoded'].copy()}

                        if tmp_encoding_input_name != fixed_encoding_input_name:
                            print("Loaded only latent-space matrix found in %s: %s" % (input_encodingfile,tmp_encoding_input_name))
                        fixed_encoding_input_name_found=tmp_encoding_input_name
                    elif fixed_encoding_input_name == 'self':
                        encoded_inputs={}
                        for conntype in conndata_alltypes.keys():
                            if conntype in Mtmp['predicted_alltypes'] and 'encoded' in Mtmp['predicted_alltypes'][conntype]:
                                encoded_inputs[conntype]=Mtmp['predicted_alltypes'][conntype]['encoded'].copy()

                    elif fixed_encoding_input_name in Mtmp['predicted_alltypes'] and 'encoded' in Mtmp['predicted_alltypes'][fixed_encoding_input_name]:
                        encoded_inputs={'*': Mtmp['predicted_alltypes'][fixed_encoding_input_name]['encoded'].copy()}

                else:
                    raise Exception("Encoded data not found in %s" % (input_encodingfile))
                
                #check size of encoded_inputs
                encoded_inputs_shape=None
                for conntype in encoded_inputs:
                    if conntype == '*':
                        conn_str=fixed_encoding_input_name_found
                    else:
                        conn_str=conntype
                    encoded_inputs_shape=encoded_inputs[conntype].shape

                    if encoded_inputs[conntype].shape[0]!=len(subjects) or encoded_inputs[conntype].shape[1]!=latentsize:
                        raise Exception("Latent-space data from %s, ['predicted_alltypes']['%s']['encoded'] must be %dx%d, not %dx%d" % input_encodingfile,conn_str, 
                                        len(subjects),latentsize,encoded_inputs[conntype].shape[0],encoded_inputs[conntype].shape[1])

                print("Loaded target latent-space values from %s (%s)" % (input_encodingfile,encoded_inputs_shape))
            
            #create_data_loader=training_params['nbepochs']>1
            create_data_loader=True
            if do_target_encoding or do_fixed_target_encoding:
                ##################
                # this mode means an existing latent-space was provided and we want to train all encoders to project to that, 
                #   and all decoders to decode from that
                
                if encoded_inputs is None:
                    raise Exception("Must provide encoded inputs file")
                
                conndata_alltypes_targetencoding=conndata_alltypes.copy()
                for conntype in conndata_alltypes_targetencoding.keys():
                    if conntype in encoded_inputs:
                        conndata_alltypes_targetencoding[conntype]['encoded']=encoded_inputs[conntype].copy()
                    elif '*' in encoded_inputs:
                        conndata_alltypes_targetencoding[conntype]['encoded']=encoded_inputs['*'].copy()
                    else:
                        raise Exception("Input data type %s not found in latent-space data" % (conntype))

                
                if do_fixed_target_encoding or do_only_self_target_encoding:
                    data_string_targetencoding="self"+"_"+roilist_str+"_"+subj_str
                    init_trainpath_pairs="self"
                    init_trainpath_group_pairs=[]
                else:
                    data_string_targetencoding=grouptype+"_"+roilist_str+"_"+subj_str
                    init_trainpath_pairs=trainpath_pairs
                    init_trainpath_group_pairs=trainpath_group_pairs

                    
                trainpath_list, data_optimscale, data_orig, data_transformer_info_list = generate_training_paths(conndata_alltypes_targetencoding, conn_names, subjects, subjidx_train, subjidx_val, 
                                                trainpath_pairs=init_trainpath_pairs, 
                                                trainpath_group_pairs=init_trainpath_group_pairs, data_string=data_string_targetencoding, 
                                                batch_size=batchsize, skip_selfs=False, crosstrain_repeats=crosstrain_repeats,
                                                reduce_dimension=reduce_dimension,use_pretrained_encoder=False, keep_origscale_data=keep_origscale_data,           
                                                use_lognorm_for_sc=do_use_lognorm_for_sc, 
                                                use_truncated_svd=use_truncated_svd, 
                                                use_truncated_svd_for_sc=do_use_tsvd_for_sc,
                                                input_transformation_info=transformation_type_string,
                                                precomputed_transformer_info_list=precomputed_transformer_info_list, create_data_loader=create_data_loader)
            
            else:
                ##################
                # this is the normal training mode             
                trainpath_list, data_optimscale, data_orig, data_transformer_info_list = generate_training_paths(conndata_alltypes, conn_names, subjects, subjidx_train, subjidx_val, 
                                                    trainpath_pairs=trainpath_pairs, 
                                                    trainpath_group_pairs=trainpath_group_pairs, data_string=data_string, 
                                                    batch_size=batchsize, skip_selfs=do_skipself, crosstrain_repeats=crosstrain_repeats,
                                                    reduce_dimension=reduce_dimension,use_pretrained_encoder=False, keep_origscale_data=keep_origscale_data,           
                                                    use_lognorm_for_sc=do_use_lognorm_for_sc, 
                                                    use_truncated_svd=use_truncated_svd, 
                                                    use_truncated_svd_for_sc=do_use_tsvd_for_sc,
                                                    input_transformation_info=transformation_type_string,
                                                    precomputed_transformer_info_list=precomputed_transformer_info_list, create_data_loader=create_data_loader)
            
            
            #dont save input transforms if we are using precomputed ones
            save_input_transforms=precomputed_transformer_info_list is None or not all([k in precomputed_transformer_info_list for k in data_transformer_info_list])
            
            #load checkpoint file for initial starting point, if given
            if starting_point_file:
                #allow override of dropout amount
                checkpoint_override={}
                checkpoint_override['dropout']=training_params['dropout']
                
                if do_domain_adaptation:
                    print("Adding outer domain-adaptation layers. Resetting OUTER data transformations to 'none'")
                    data_string="adapt"+grouptype+"_"+subj_str
                    inner_net, checkpoint=Krakencoder.load_checkpoint(starting_point_file, checkpoint_override=checkpoint_override)
                    data_transformer_list=[]
                    data_inputsize_list=[]
                    
                    none_transformer, none_transformer_info = generate_transformer(transformer_type="none")
                    
                    new_outer_transformer_info_list={}
                    
                    for i_conn, conn_name in enumerate(conn_names):
                        new_outer_transformer_info_list[conn_name]=none_transformer_info
                        
                        if conn_name in data_transformer_info_list:
                            transformer, transformer_info = generate_transformer(transformer_type=data_transformer_info_list[conn_name]["params"]["type"], precomputed_transformer_params=data_transformer_info_list[conn_name]["params"])
                            
                            do_meanshift=True
                            
                            if do_meanshift:
                                print("doing meanshift!")
                                transformer_data_mean=None
                                if "pca_input_mean" in transformer_info["params"]:
                                    transformer_data_mean=transformer_info["params"]["pca_input_mean"]
                                elif "input_mean" in transformer_info["params"]:
                                    transformer_data_mean=transformer_info["params"]["data_mean"]
                                if transformer_data_mean is not None:
                                    if torch.is_tensor(transformer_data_mean):
                                        transformer_data_mean=transformer_data_mean.detach().cpu().numpy()
                                    transformer_data_mean=np.atleast_2d(transformer_data_mean)
                                    actual_data_mean=np.mean(conndata_alltypes[conn_name]['data'],axis=0,keepdims=True)
                                    new_outer_transformer_info_list[conn_name]={'type':'cfeat', 'input_mean':actual_data_mean-transformer_data_mean}
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
                        
                    net = KrakenAdapter(inner_model=inner_net, inputsize_list=data_inputsize_list, data_transformer_list=data_transformer_list,
                                        linear_polynomial_order=domain_adaptation_polynomial)
                    
                    net.freeze_inner_model(True)
                    
                    #previous: input_transformation_info="NONE"
                    #new: precomputed_transformer_info_list=new_outer_transformer_info_list, 
                    trainpath_list, data_optimscale, data_orig, data_transformer_info_list = generate_training_paths(conndata_alltypes, conn_names, subjects, subjidx_train, subjidx_val, 
                                                        trainpath_pairs=trainpath_pairs, 
                                                        trainpath_group_pairs=trainpath_group_pairs, data_string=data_string, 
                                                        batch_size=batchsize, skip_selfs=do_skipself, crosstrain_repeats=crosstrain_repeats,
                                                        reduce_dimension=None,use_pretrained_encoder=False, keep_origscale_data=keep_origscale_data,           
                                                        precomputed_transformer_info_list=new_outer_transformer_info_list, 
                                                        create_data_loader=create_data_loader)
                            
                else:
                    net, checkpoint=Krakencoder.load_checkpoint(starting_point_file, checkpoint_override=checkpoint_override)

            else:
                net=None
            
            datastring0=trainpath_list[0]['data_string']
            
            trainblocks=training_params['trainblocks']
            
            for blockloop in range(trainblocks):
                if trainblocks > 1:
                    trainpath_list[0]['data_string']=datastring0+"_b%d" % (blockloop+1)
                training_params_tmp=training_params.copy()
                
                net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net, 
                                                 data_optimscale_list=data_optimscale, data_origscale_list=data_orig,
                                                 trainthreads=trainthreads,display_epochs=display_epochs,save_epochs=display_epochs,
                                                 checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False,
                                                 explicit_checkpoint_epoch_list=explicit_checkpoint_epoch_list,
                                                 precomputed_transformer_info_list=data_transformer_info_list,
                                                 save_optimizer_params=optimizer_in_checkpoint,
                                                 save_input_transforms=save_input_transforms, 
                                                 output_file_prefix=output_file_prefix,logger=log,extra_trainrecord_dict=extra_trainrecord_dict)
            
                if not training_params['roundtrip'] and add_roundtrip_epochs > 0:
                    print("Adding %d roundtrip epochs" % (add_roundtrip_epochs))
                    training_params_tmp=training_params.copy()
                    training_params_tmp['roundtrip']=True
                    training_params_tmp['nbepochs']=add_roundtrip_epochs
                    net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net,
                                                     data_optimscale_list=data_optimscale, data_origscale_list=data_orig,
                                                     trainthreads=trainthreads,display_epochs=display_epochs,save_epochs=display_epochs,
                                                     checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False,
                                                     save_optimizer_params=optimizer_in_checkpoint,
                                                     output_file_prefix=output_file_prefix,
                                                     extra_trainrecord_dict=extra_trainrecord_dict)
                                                     
                if not training_params['meantarget_latentsim'] and add_meanlatent_epochs > 0:
                    print("Adding %d meanlatent epochs" % (add_meanlatent_epochs))
                    training_params_tmp=training_params.copy()
                    training_params_tmp['meantarget_latentsim']=True
                    training_params_tmp['nbepochs']=add_meanlatent_epochs
                    net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net, 
                                                     data_optimscale_list=data_optimscale, data_origscale_list=data_orig,
                                                     trainthreads=trainthreads,display_epochs=display_epochs,save_epochs=display_epochs,
                                                     checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False,
                                                     save_optimizer_params=optimizer_in_checkpoint,
                                                     output_file_prefix=output_file_prefix,
                                                     extra_trainrecord_dict=extra_trainrecord_dict)
                                                     
                if not do_fixed_target_encoding and add_fixed_encoding_epochs_after > 0:
                    raise Exception("add_fixed_encoding not yet supported")
                    print("Adding %d fixedencoding epochs" % (add_fixed_encoding_epochs_after))
                    
                    conndata_alltypes_targetencoding=conndata_alltypes.copy()
                    #for conntype in conndata_alltypes_targetencoding.keys():
                        
                        
                    data_string_targetencoding="self"+"_"+roilist_str
                
                    trainpath_list, data_orig, data_transformer_info_list = generate_training_paths(conndata_alltypes_targetencoding, conn_names, subjects, subjidx_train, subjidx_val, 
                                                    trainpath_pairs="self", 
                                                    trainpath_group_pairs=[], data_string=data_string_targetencoding, 
                                                    batch_size=batchsize, skip_selfs=False, crosstrain_repeats=crosstrain_repeats,
                                                    reduce_dimension=reduce_dimension,use_pretrained_encoder=False, keep_origscale_data=keep_origscale_data,           
                                                    use_lognorm_for_sc=do_use_lognorm_for_sc, 
                                                    use_truncated_svd=use_truncated_svd, 
                                                    use_truncated_svd_for_sc=do_use_tsvd_for_sc,
                                                    input_transformation_info=transformation_type_string,
                                                    precomputed_transformer_info_list=precomputed_transformer_info_list,
                                                    extra_trainrecord_dict=extra_trainrecord_dict)
                    
                    if trainblocks > 1:
                        trainpath_list[0]['data_string']=datastring0+"_b%d" % (blockloop+1)
                    
                    training_params_tmp=training_params.copy()
                    training_params_tmp['fixed_encoding']=True
                    training_params_tmp['nbepochs']=add_fixed_encoding_epochs_after
                    net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net, data_origscale_list=data_orig,
                                                     trainthreads=trainthreads,display_epochs=display_epochs,save_epochs=display_epochs,
                                                     checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False,
                                                     save_optimizer_params=optimizer_in_checkpoint,
                                                     output_file_prefix=output_file_prefix,
                                                     extra_trainrecord_dict=extra_trainrecord_dict)
                    
if __name__ == "__main__":
    if len(sys.argv)<=1:
        argument_parse_runtraining(['-h'])
        sys.exit(0)
    run_training_command(sys.argv[1:])
