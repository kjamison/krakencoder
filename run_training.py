if __name__ == "__main__":
    #for running in command line on AWS, need to restrict threads so it doesn't freeze during PCA sometimes
    import os
    if os.getenv('USER') == 'ubuntu':
        numthreads=3
        os.environ['OPENBLAS_NUM_THREADS'] = str(numthreads)
        os.environ['MKL_NUM_THREADS'] = str(numthreads)
        os.environ['NUMEXPR_NUM_THREADS']=str(numthreads)

#######################

from krakencoder import *
from train import *
from utils import *
import re
import os
import sys
import argparse

def argument_parse_runtraining(argv):
    #for list-based inputs, need to specify the defaults this way, otherwise the argparse append just adds to them
    arg_defaults={}
    arg_defaults['roinames']=["fs86+shen268+coco439"]
    arg_defaults['dataflavors']=["SCifod2act","SCsdstream","FCcov","FCcovgsr","FCpcorr"]
    arg_defaults['fcfilt']=["hpf"]
    arg_defaults['pathgroups']=['all']
    arg_defaults['losstype']=['correye+enceye']
    arg_defaults['dropout']=[0]
    arg_defaults['latent_sim_weight']=[5000]
    arg_defaults['explicit_checkpoint_epochs']=[]
    arg_defaults['hiddenlayersizes']=[]
    parser=argparse.ArgumentParser(description='Train krakencoder', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    input_arg_group=parser.add_argument_group('Input data options')
    input_arg_group.add_argument('--subjectcount',action='store',dest='subjectcount',type=int, default=993, help='HCPTRAIN: Which HCP subject set? 993 (default) or 420')
    input_arg_group.add_argument('--dataflavors',action='append',dest='dataflavors',help='HCPTRAIN: SCifod2act,SCsdstream,FCcov,FCcovgsr,FCpcorr (default=%s)' % (arg_defaults["dataflavors"]),nargs='*')
    input_arg_group.add_argument('--roinames',action='append',dest='roinames',help='HCPTRAIN: fs86,shen268,coco439... (default=%s)' % (arg_defaults["roinames"]),nargs='*')
    input_arg_group.add_argument('--fcfilt',action='append',dest='fcfilt',help='list of hpf, bpf, nofilt (default=%s)' % (arg_defaults["fcfilt"]),nargs='*')
    input_arg_group.add_argument('--inputdata',action='append',dest='input_data_file', help='name=file, name=file, ... Override HCPTRAIN: .mat file(s) containing input data to transform (instead of default HCP set).', nargs='*')
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
    
    loss_arg_group=parser.add_argument_group('Loss function parameters')
    loss_arg_group.add_argument('--losstype',action='append',dest='losstype',help='list of correye+enceye, dist+encdist, etc...',nargs='*')
    loss_arg_group.add_argument('--latentunit',action='store_true',dest='latentunit', help='latent all normalzed to shell of unit sphere')
    loss_arg_group.add_argument('--latentradweight',action='store',dest='latentradweight', type=float, default=10, help='weight to apply to keeping latent rad<1')
    loss_arg_group.add_argument('--mseweight',action='store',dest='mseweight', type=float, default=1, help='Weight to apply to true->predicted MSE')
    loss_arg_group.add_argument('--latentinnerweight',action='store',dest='latent_inner_loss_weight', type=float, default=10, help='Weight to apply to latent-space inner loop loss (enceye,encdist, etc...)')
    loss_arg_group.add_argument('--latentsimweight',action='append',dest='latent_sim_weight',type=float,help='list of latentsimloss weights to try . default=5000',nargs='*')

    train_arg_group=parser.add_argument_group('Training parameters')
    train_arg_group.add_argument('--epochs',action='store',dest='epochs',type=int, default=5000, help='number of epochs')
    train_arg_group.add_argument('--batchsize',dest='batch_size',type=int,default=41,help='main batch size. default=41 (no batch)')
    train_arg_group.add_argument('--dropout',action='append',dest='dropout',type=float,help='list of dropouts to try',nargs='*')
    train_arg_group.add_argument('--noskipacc',action='store_true',dest='noskipacc', help='do NOT skip accurate paths during training')
    train_arg_group.add_argument('--noearlystop',action='store_true',dest='noearlystop', help='do NOT stop early if skipacc (keep working latentsim)')
    train_arg_group.add_argument('--skipself',action='store_true',dest='skipself', help='Skip A->A paths during training')
    train_arg_group.add_argument('--roundtrip',action='store_true',dest='roundtrip', help='roundtrip training paths A->B->A')
    train_arg_group.add_argument('--addroundtripepochs',action='store',dest='add_roundtrip_epochs', type=int, default=0, help='add roundtrip training paths A->B->A AFTER normal training')
    train_arg_group.add_argument('--addmeanlatentepochs',action='store',dest='add_meanlatent_epochs', type=int, default=0, help='add meanlatent training paths AFTER normal training')
    train_arg_group.add_argument('--trainblocks',action='store',dest='trainblocks', type=int, default=1, help='How many total times perform normal training + (roundtrip or meanlatent) set? (optimizer resets each block)')
    train_arg_group.add_argument('--latentsimbatchsize',dest='latent_sim_batch_size',type=int,default=0,help='Batch size for latentsimloss. default=0 (no batch)')
    train_arg_group.add_argument('--singleoptimizer',action='store_true',dest='single_optimizer', help='Use single optimizer across all paths and latentsim')
    train_arg_group.add_argument('--adamdecay',action='store',dest='adam_decay',type=float, default=0.01, help='Adam weight decay')

    fixed_arg_group=parser.add_argument_group('Target-encoding options (Train new data to match pre-trained latent representation)')
    fixed_arg_group.add_argument('--encodedinputfile',action='store',dest='encoded_input_file', help='.mat file containing latent space data')
    fixed_arg_group.add_argument('--targetencoding',action='store_true',dest='target_encoding', help='Train encoders/decoders while trying to match latent->target')
    fixed_arg_group.add_argument('--fixedtargetencoding',action='store_true',dest='fixed_target_encoding', help='Just train encoders/decoders to match FIXED (input->fixed, fixed->output) --encodinginputfile')
    fixed_arg_group.add_argument('--targetencodingname',action='store',dest='target_encoding_name', help='Encoding type for target-encoding ("self" for per-flavor latent space input, "burst", or specific flavor)')
    fixed_arg_group.add_argument('--addfixedencodingepochsafter',action='store',dest='add_fixed_encoding_epochs_after', type=int, default=0, help='Add fixedencoding epochs AFTER normal epochs')
    fixed_arg_group.add_argument('--addfixedencodingepochsbefore',action='store',dest='add_fixed_encoding_epochs_before', type=int, default=0, help='Add fixedencoding epochs BEFORE normal epochs')

    misc_arg_group=parser.add_argument_group('Other options')
    misc_arg_group.add_argument('--checkpointepochsevery',action='store',dest='checkpoint_epochs_every', type=int, default=1000, help='How often to save checkpoints')
    misc_arg_group.add_argument('--explicitcheckpointepochs',action='append',dest='explicit_checkpoint_epochs', type=int, help='Explicit list of epochs at which to save checkpoints',nargs='*')
    misc_arg_group.add_argument('--displayepochs',action='store',dest='display_epochs', type=int, default=100, help='How often to print training progress')
    misc_arg_group.add_argument('--maxthreads',action='store',dest='max_threads', type=int, default=10, help='How many CPU threads to use')
    misc_arg_group.add_argument('--outputprefix',action='store',dest='output_file_prefix', default="connae", help='Prefix for output files')

    args=parser.parse_args(argv)
    args=clean_args(args,arg_defaults)
    return args

def load_hcp_subject_list(numsubj=993):
    if os.path.isdir('/Users/kwj5'):
        datafolder='/Users/kwj5/Box/HCP_SC_FC_new997'
        studyfolder='/Users/kwj5/Research/HCP'
    elif os.path.isdir('/home/kwj2001/colossus_shared/HCP'):
        studyfolder='/home/kwj2001/colossus_shared/HCP'
        datafolder='/home/kwj2001/colossus_shared/HCP'
    elif os.path.isdir('/home/ubuntu'):
        studyfolder='/home/ubuntu'
        datafolder='/home/ubuntu'

    familyidx=np.loadtxt('%s/subjects_famidx_rfMRI_dMRI_complete_997.txt' % (studyfolder))
    subj997=np.loadtxt('%s/subjects_rfMRI_dMRI_complete_997.txt' % (studyfolder))
    
    if numsubj==420:
        subjects=np.loadtxt('%s/subjects_unrelated420_scfc.txt' % (studyfolder))
        familyidx=np.array([familyidx[i] for i,s in enumerate(subj997) if s in subjects])
    elif numsubj==993:
        subjects=np.loadtxt('%s/subjects_rfMRI_dMRI_complete_993_minus4.txt' % (studyfolder))
        familyidx=np.array([familyidx[i] for i,s in enumerate(subj997) if s in subjects])
    elif numsubj==997:
        subjects=subj997
    else:
        raise Exception("Unknown numsubj %s" % (numsubj))
    
    subjects=clean_subject_list(subjects)
    
    return subjects, familyidx

def get_hcp_data_flavors(roi_list=["fs86","shen268","coco439"], 
                         sc_type_list=["ifod2act_volnorm","sdstream_volnorm"], 
                         fc_type_list=["FCcov","FCcovgsr","FCpcorr"], 
                         fc_filter_list=["hpf","bpf","nofilt"],
                         sc=True,
                         fc=True):
    if not roi_list:
        roi_list=[]
    if not sc_type_list:
        sc_type_list=[]
    if not fc_type_list:
        fc_type_list=[]
    if not fc_filter_list:
        fc_filter_list=[]
    
    if isinstance(roi_list,str):
        roi_list=[roi_list]
    if isinstance(sc_type_list,str):
        sc_type_list=[sc_type_list]
    if isinstance(fc_type_list,str):
        fc_type_list=[fc_type_list]
    if isinstance(fc_filter_list,str):
        fc_filter_list=[fc_filter_list]
    
    if not sc:
        sc_type_list=[]
    
    if not fc:
        fc_filter_list=[]
    
    conntype_list=[]
    for r in roi_list:
        for sc in sc_type_list:
            conntype_list+=["%s_%s" % (r,sc)]
        for f in fc_filter_list:
            for fc in fc_type_list:
                fctmp=fc.replace("_gsr","").replace("gsr","")
                c="%s_%s_%s" % (fctmp,r,f)
                if "gsr" in fc:
                    c+="gsr"
                conntype_list+=[c]
    return conntype_list

def load_hcp_data(subjects=[], conn_name_list=[], quiet=False):
    #conn_name_list = explicit and complete list of datatypes to load (ignore all other flavor info)

    if os.path.isdir('/Users/kwj5'):
        datafolder='/Users/kwj5/Box/HCP_SC_FC_new997'
        studyfolder='/Users/kwj5/Research/HCP'
    elif os.path.isdir('/home/kwj2001/colossus_shared/HCP'):
        studyfolder='/home/kwj2001/colossus_shared/HCP'
        datafolder='/home/kwj2001/colossus_shared/HCP'
    elif os.path.isdir('/home/ubuntu'):
        studyfolder='/home/ubuntu'
        datafolder='/home/ubuntu'

    if len(subjects)==0 or subjects is None:
        subjects,_=load_hcp_subject_list(numsubj=993)

    #scfile_fields=['orig','volnorm']
    #scfile_fields=['volnorm']

    #consider: log transform on SC inputs?
    # also maybe try using orig instead of just volnorm? orig is easier to log transform (all pos vals)
    # log10(volnorm) has a lot of negatives (counts<1)
    # is there any reasonable way to justify log10(connectome)./log10(volumes) to scale?
    #  then it would be positive but scaled by the (logscaled) volumes
    # main reason this matters is that we have 0s for SC which log can't handle, and we cant just set those to
    #  log10(0)->0 because then log10(.1)-> -1 would look < 0 

    pretrained_transformer_file={}
    pretrained_transformer_file['fs86_ifod2act_volnorm']=None
    pretrained_transformer_file['shen268_ifod2act_volnorm']=None
    pretrained_transformer_file['shen268_sdstream_volnorm']=None
    pretrained_transformer_file['coco439_ifod2act_volnorm']=None
    pretrained_transformer_file['coco439_sdstream_volnorm']=None
    pretrained_transformer_file['FCcov_fs86_hpf']=None
    pretrained_transformer_file['FCcov_fs86_hpfgsr']=None
    pretrained_transformer_file['FCpcorr_fs86_hpf']=None
    pretrained_transformer_file['FCcov_shen268_hpf']=None
    pretrained_transformer_file['FCcov_shen268_hpfgsr']=None
    pretrained_transformer_file['FCpcorr_shen268_hpf']=None
    pretrained_transformer_file['FCcov_coco439_hpf']=None
    pretrained_transformer_file['FCcov_coco439_hpfgsr']=None
    pretrained_transformer_file['FCpcorr_coco439_hpf']=None

    #build list of possijble HCP data files to load

    if len(conn_name_list)==0:
        fc_filter_list=["hpf","bpf","nofilt"]
    
    connfile_info=[]
    datagroup='SC'
    
    connfile_info.append({'name':'fs86_ifod2act_volnorm','file':'%s/sc_fs86_ifod2act_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'fs86_sdstream_volnorm','file':'%s/sc_fs86_sdstream_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'shen268_ifod2act_volnorm','file':'%s/sc_shen268_ifod2act_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'shen268_sdstream_volnorm','file':'%s/sc_shen268_sdstream_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'coco439_ifod2act_volnorm','file':'%s/sc_cocommpsuit439_ifod2act_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})
    connfile_info.append({'name':'coco439_sdstream_volnorm','file':'%s/sc_cocommpsuit439_sdstream_volnorm_993subj.mat' % (datafolder),'fieldname':'SC','group':datagroup})

    datagroup='FC'
    #consider: do np.arctanh for FC inputs?

    #hpf
    connfile_info.append({'name':'FCcov_fs86_hpf','file':'%s/fc_fs86_FCcov_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcov_fs86_hpfgsr','file':'%s/fc_fs86_FCcov_hpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_fs86_hpf','file':'%s/fc_fs86_FCpcorr_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})

    connfile_info.append({'name':'FCcov_shen268_hpf','file':'%s/fc_shen268_FCcov_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcov_shen268_hpfgsr','file':'%s/fc_shen268_FCcov_hpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_shen268_hpf','file':'%s/fc_shen268_FCpcorr_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})

    connfile_info.append({'name':'FCcov_coco439_hpf','file':'%s/fc_cocommpsuit439_FCcov_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcov_coco439_hpfgsr','file':'%s/fc_cocommpsuit439_FCcov_hpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_coco439_hpf','file':'%s/fc_cocommpsuit439_FCpcorr_hpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})

    #bpf
    connfile_info.append({'name':'FCcov_fs86_bpf','file':'%s/fc_fs86_FCcov_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcov_fs86_bpfgsr','file':'%s/fc_fs86_FCcov_bpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_fs86_bpf','file':'%s/fc_fs86_FCpcorr_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    
    connfile_info.append({'name':'FCcov_shen268_bpf','file':'%s/fc_shen268_FCcov_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcov_shen268_bpfgsr','file':'%s/fc_shen268_FCcov_bpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_shen268_bpf','file':'%s/fc_shen268_FCpcorr_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    
    connfile_info.append({'name':'FCcov_coco439_bpf','file':'%s/fc_cocommpsuit439_FCcov_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcov_coco439_bpfgsr','file':'%s/fc_cocommpsuit439_FCcov_bpf_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_coco439_bpf','file':'%s/fc_cocommpsuit439_FCpcorr_bpf_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
        
    #nofilt (no compcor, hp2000)
    connfile_info.append({'name':'FCcov_fs86_nofilt','file':'%s/fc_fs86_FCcov_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcov_fs86_nofiltgsr','file':'%s/fc_fs86_FCcov_nofilt_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_fs86_nofilt','file':'%s/fc_fs86_FCpcorr_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})

    connfile_info.append({'name':'FCcov_shen268_nofilt','file':'%s/fc_shen268_FCcov_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcov_shen268_nofiltgsr','file':'%s/fc_shen268_FCcov_nofilt_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_shen268_nofilt','file':'%s/fc_shen268_FCpcorr_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    
    connfile_info.append({'name':'FCcov_coco439_nofilt','file':'%s/fc_cocommpsuit439_FCcov_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCcov_coco439_nofiltgsr','file':'%s/fc_cocommpsuit439_FCcov_nofilt_gsr_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})
    connfile_info.append({'name':'FCpcorr_coco439_nofilt','file':'%s/fc_cocommpsuit439_FCpcorr_nofilt_993subj.mat' % (datafolder),'fieldname':'FC','group':datagroup})


    #print("connfile_info:",[k["name"] for k in connfile_info])
    conndata_alltypes={}

    if len(conn_name_list)==0:
        conn_name_list=connfile_info.keys()

    conn_names_available=[x['name'] for x in connfile_info]
    for i,cname in enumerate(conn_name_list):
        if cname.endswith("_volnorm"):
            conntype="volnorm"
            cname=cname.replace("_volnorm","")
        elif cname.endswith("_sift2"):
            conntype="sift2"
            cname=cname.replace("_sift2","")
        elif cname.endswith("_sift2volnorm"):
            conntype="sift2volnorm"
            cname=cname.replace("_sift2volnorm","")
        elif cname.endswith("_orig"):
            conntype="orig"
            cname=cname.replace("_orig","")
        elif cname.endswith("_FC"):
            conntype="FC"
            cname=cname.replace("_FC","")
        elif cname.startswith("FC"):
            conntype="FC" #FC but didn't add the "_FC" to end
        elif "ifod2act" in cname or "sdstream" in cname:
            conntype="orig"
        else:
            print("Unknown data flavor for %s" % (cname))
            sys.exit(1)
        
        connsearch='%s_%s' % (cname,conntype)
        connsearch=connsearch.replace("_FC","")
        ci=[x for x in connfile_info if x['name']==connsearch]
        if len(ci)==0:
            print("%s not found in file_info. Available names: %s" % (connsearch, conn_names_available))
        ci=ci[0]
        
        if not quiet:
            print("Loading %d/%d: %s" % (i+1,len(conn_name_list),ci['file']))
        Cdata=loadmat(ci['file'],simplify_cells=True)
        subjmissing=Cdata['ismissing']>0
        subjects997=Cdata['subject'][~subjmissing]
        subjects997=clean_subject_list(subjects997)
        
        connfield_list=[connfile_info[i]['fieldname'], "SC","FC"]
        connfield=conntype
        for cf in connfield_list:
            if cf in Cdata:
                connfield=cf
                break
        nroi=Cdata[connfield][0].shape[0]
        trimask=np.triu_indices(nroi,1) #note: this is equivalent to tril(X,-1) in matlab
        npairs=trimask[0].shape[0]

        Ctriu=[x[trimask] for x in Cdata[connfield][~subjmissing]]
        #restrict to 420 unrelated subjects
        Ctriu=[x for i,x in enumerate(Ctriu) if subjects997[i] in subjects]
        conn_name='%s_%s' % (cname,conntype)
        
        transformer_file=None
        if conn_name in pretrained_transformer_file:
            transformer_file=pretrained_transformer_file[conn_name]

        conndata_alltypes[conn_name]={'data':np.vstack(Ctriu),'numpairs':npairs,'group':ci['group'],'transformer_file':transformer_file,'subjects':subjects}
            
            
    nsubj=conndata_alltypes[list(conndata_alltypes.keys())[0]]['data'].shape[0]

    if not quiet:
        print("Norm scale for input data:")
        for conntype in conndata_alltypes:
            normscale=np.linalg.norm(conndata_alltypes[conntype]['data'])
            normscale/=np.sqrt(conndata_alltypes[conntype]['data'].size)
            print("%s: %.6f" % (conntype,normscale))
    
        for conntype in conndata_alltypes:
            print(conntype,conndata_alltypes[conntype]['data'].shape)
    
    return subjects, conndata_alltypes


def load_input_data(inputfile, group=None, inputfield=None):
    inputfield_default_search=['encoded','FC','SC','C','volnorm'] #,'sift2volnorm','sift2','orig']

    Cdata=loadmat(inputfile,simplify_cells=True)

    if 'ismissing' in Cdata:
        subjmissing=Cdata['ismissing']>0
    else:
        subjmissing=[]
    if 'subject' in Cdata:
        subjects=Cdata['subject']
    else:
        subjects=[]

    subjects=clean_subject_list(subjects)

    connfield=inputfield
    if not connfield:
        for itest in inputfield_default_search:
            if itest in Cdata:
                connfield=itest
                break
    
    if connfield is None:
        print("None of the following fields were found in the input file %s:" % (inputfile),inputfield_default_search)
        raise Exception("Input type not found")
    
    if len(Cdata[connfield][0].shape)==0:
        #single matrix was in file
        Cmats=[Cdata[connfield]]
    else:
        Cmats=Cdata[connfield]
    
    if connfield == "encoded":
        nroi=1
        npairs=Cmats[0].shape[1]
        Cdata=Cmats[0].copy()
    else:
        nroi=Cmats[0].shape[0]
        trimask=np.triu_indices(nroi,1) #note: this is equivalent to tril(X,-1) in matlab
        npairs=trimask[0].shape[0]
        if len(subjmissing)==0:
            subjmissing=np.zeros(len(Cmats))>0
        
        if len(subjects)>0:
            subjects=subjects[~subjmissing]
    
        Ctriu=[x[trimask] for i,x in enumerate(Cmats) if not subjmissing[i]]
        Cdata=np.vstack(Ctriu)
    
    #conndata_alltypes[conn_name]={'data':np.vstack(Ctriu),'numpairs':npairs,'group':ci['group'],'transformer_file':transformer_file}
    conndata={'data':Cdata,'numpairs':npairs,'numroi':nroi,'fieldname':connfield,'group':group,'subjects':subjects}
    
    return conndata

def canonical_data_flavor(conntype, only_if_brackets=False, return_groupname=False):
    groupname=None
    
    if only_if_brackets:
        #special mode that leaves inputs intact unless they are in the form "[name]", in which case
        #it will return the canonical version of "name"
        if not re.match(".*\[.+\].*",conntype.lower()):
            if return_groupname:
                return conntype, groupname
            else:
                return conntype
        conntype=re.sub("^.*\[(.+)\].*$",'\\1',conntype)
    
    if conntype.lower() == "encoded":
        if return_groupname:
            return "encoded", groupname
        else:
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
    elif "coco439" in input_conntype_lower or "cocommpsuit439" in input_conntype_lower:
        input_atlasname="coco439"
    else:
        raise Exception("Unknown atlas name for input type: %s" % (conntype))
    
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
        raise Exception("Unknown data flavor for input type: %s" % (conntype))
    
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
            raise Exception("Unknown FC filter for input type: %s" % (conntype))
    
    if input_flavor.startswith("FC"):
        groupname="FC"
        conntype_canonical="%s_%s_%s%s_FC" % (input_flavor,input_atlasname,input_fcfilt,input_fcgsr) #orig style with FC<flavor>_<atlas>_<filt><gsr>_FC
        #conntype_canonical="%s_%s_%s%s" % (input_flavor,input_atlasname,input_fcfilt,input_fcgsr) #new style with FC<flavor>_<atlas>_<filt><gsr>
    else:
        groupname="SC"
        conntype_canonical="%s_%s_%s" % (input_atlasname,input_flavor,input_scproc) #orig style
        #conntype_canonical="SC%s_%s_%s" % (input_flavor,input_atlasname,input_scproc) #new style with SC<algo>_<atlas>_volnorm

    if return_groupname:
        return conntype_canonical, groupname
    else:
        return conntype_canonical

#######################################################
#######################################################
#### 

def run_training_command(argv):
    args=argument_parse_runtraining(argv)

    ##################
    
    trainthreads=args.max_threads
    input_epochs=args.epochs
    input_roundtrip=args.roundtrip
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
    do_skipaccpath=not args.noskipacc #invert argument
    do_earlystop=not args.noearlystop #invert argument
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
    input_use_separate_optimizer=not args.single_optimizer
    input_batchsize=args.batch_size
    input_latentsimbatchsize=args.latent_sim_batch_size
    input_encodingfile=args.encoded_input_file
    do_fixed_target_encoding=args.fixed_target_encoding
    do_target_encoding=args.target_encoding
    fixed_encoding_input_name=args.target_encoding_name
    add_fixed_encoding_epochs_after=args.add_fixed_encoding_epochs_after
    add_fixed_encoding_epochs_before=args.add_fixed_encoding_epochs_before
    starting_point_file=args.starting_point_file
    trainval_split_frac=args.trainval_split_frac
    val_split_frac=args.val_split_frac
    output_file_prefix=args.output_file_prefix
    display_epochs=args.display_epochs

    if input_latentunit:
        input_latentradweight=0
    
    
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

            if groupname is not None and args.pathgroups is not None and len(args.pathgroups)>0 and not any([x.lower() == 'all' for x in args.pathgroups]):
                xc_in_pathgroups=any(groupname in x for x in args.pathgroups)
                if not xc_in_pathgroups:
                    print("Skipping input %s: Group %s for not found in pathgroup input %s: %s" % (xc,groupname,args.pathgroups, f))
                    #dont load this input file
                    continue
            
            conndata_alltypes[xc]=load_input_data(input_file,group=groupname)

            print("%s@%s=%s" % (xc,groupname,input_file))

    else:
        #HCPTRAIN
        input_nsubj=args.subjectcount

        if subjects is None or len(subjects)==0:
            subjects, familyidx = load_hcp_subject_list(input_nsubj)
    
        input_roilist=flatlist([x.split("+") for x in args.roinames])

        roilist_str="+".join(input_roilist)
        
        fcfilt_types=args.fcfilt
        input_dataflavors=args.dataflavors
        #input_dataflavors = "FCpcorr" or "SCifod2act" (no roi or fcfilt info)
        
        filttypes_full=["hpf","bpf","nofilt"]
        sctypes_full=["ifod2act","sdstream"]
        fctypes_full=["FCcov","FCcovgsr","FCpcorr"]
        #fcfilt_types=[x for x in filttypes_full if any([x in y for y in input_dataflavors])]
        sctypes=[x+"_volnorm" for x in sctypes_full if any([x in y for y in input_dataflavors])]
        fctypes=[x for x in fctypes_full if any([x in y for y in input_dataflavors])]

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
    
    ##################################
    precomputed_transformer_info_list=None
    if input_transform_file_list is not None and len(input_transform_file_list)>0:
        precomputed_transformer_info_list={}
        for ioxfile in input_transform_file_list:
            print("Loading precomputed input transformations: %s" % (ioxfile))
            ioxtmp=np.load(ioxfile,allow_pickle=True).item()
            for k in ioxtmp:
                precomputed_transformer_info_list[k]=ioxtmp[k]
                precomputed_transformer_info_list[k]['filename']=ioxfile.split(os.sep)[-1]


    
    import matplotlib.pyplot as plt
    from IPython import display
    from cycler import cycler

    training_params_listdict={}
    training_params_listdict['latentsize']=[latentsize]
    #training_params_listdict['losstype']='mse'
    #training_params_listdict['losstype']=['corrtrace']
    #training_params_listdict['losstype']=['correye']
    #training_params_listdict['losstype']=['corrmatch']
    training_params_listdict['losstype']=input_lossnames
    #training_params_listdict['losstype']=['neidist+encdist']
    #training_params_listdict['losstype']=['correye+enceye']
    training_params_listdict['latent_inner_loss_weight']=[input_latent_inner_loss_weight]
    #training_params_listdict['hiddenlayers']=[[256,256]]
    #training_params_listdict['hiddenlayers']=[[128]]
    #training_params_listdict['hiddenlayers']=[[128]*3,[128]*7]
    #training_params_listdict['hiddenlayers']=[ [] ]
    training_params_listdict['hiddenlayers']=[ input_hiddenlayers ]
    #training_params_listdict['dropout']=[0,.1,.25,.5]
    #training_params_listdict['dropout']=[0,.5]
    #training_params_listdict['dropout']=[0,.1,.25]
    #training_params_listdict['dropout']=[0]
    training_params_listdict['dropout']=input_dropout_list
    #training_params_listdict['batchsize']=[42,21,14]
    
    training_params_listdict['skip_accurate_paths']=[do_skipaccpath]
    training_params_listdict['early_stopping']=[do_earlystop]
        
    if len(subjects)>420:
        training_params_listdict['batchsize']=[input_batchsize] #avoid leaving out too many for the familyidx case
    else:
        #training_params_listdict['batchsize']=[42,21,14]
        training_params_listdict['batchsize']=[42]
    
    #training_params_listdict['latentsim_loss_weight']=[10]
    training_params_listdict['latentsim_loss_weight']=input_latentsimweight_list
    #training_params_listdict['adam_decay']=[.01, .1, 1, 10]
    #training_params_listdict['adam_decay']=[.01,.1]
    #training_params_listdict['adam_decay']=[.01]
    training_params_listdict['adam_decay']=[input_adamdecay]
    #training_params_listdict['mse_weight']=[1,10]
    training_params_listdict['mse_weight']=[input_mse_weight]
    #training_params_listdict['learningrate']=[0.001] #try faster?
    #training_params_listdict['learningrate']=[0.0001] #1e-4 default unless otherwise specified
    #training_params_listdict['learningrate']=[0.00001] #1e-4 default unless otherwise specified
    training_params_listdict['learningrate']=[1e-4] #1e-4 default unless otherwise specified
    #training_params_listdict['nbepochs']=[100]
    #training_params_listdict['nbepochs']=[1000]
    #training_params_listdict['nbepochs']=[2000]
    training_params_listdict['nbepochs']=[input_epochs]
    #training_params_listdict['nbepochs']=[10000]
    training_params_listdict['skip_relu']=[False]
    training_params_listdict['separate_optimizer']=[input_use_separate_optimizer]
    training_params_listdict['optimizer_name']=["adamw"]
    training_params_listdict['zerograd_none']=[True]

    training_params_listdict['relu_tanh_alternate']=[False]
    training_params_listdict['leakyrelu_negative_slope']=[input_leakyrelu]
    training_params_listdict['origscalecorr_epochs']=[25]
    
    if input_pcadim == 0:
        training_params_listdict['reduce_dimension']=[None]
    else:
        training_params_listdict['reduce_dimension']=[input_pcadim]
    
    training_params_listdict['use_truncated_svd']=[input_use_tsvd]
    
    training_params_listdict['trainpath_shuffle']=[True]
    
    #training_params_listdict['latent_maxrad_weight']=[10]
    #training_params_listdict['latent_normalize']=[True]
    training_params_listdict['latent_maxrad_weight']=[input_latentradweight]
    training_params_listdict['latent_normalize']=[input_latentunit]
    
    #training_params_listdict['latentsim_batchsize']=[0]
    
    training_params_listdict['target_encoding']=[do_target_encoding]
    training_params_listdict['fixed_target_encoding']=[do_fixed_target_encoding]

    training_params_listdict['meantarget_latentsim']=[False]
    
    training_params_listdict['trainblocks']=[input_trainblocks]
    
    #training_params_listdict['batchwise_latentsim']=[False]
    #training_params_listdict['latentsim_loss_weight']=[500,1000,2500,5000,10000,50000]
    
    training_params_listdict['roundtrip']=[input_roundtrip]
    
    training_params_listdict['trainval_split_frac']=[trainval_split_frac]
    training_params_listdict['val_split_frac']=[val_split_frac]

    training_params_list = dict_combination_list(training_params_listdict, reverse_field_order=True)    
    #%matplotlib inline

    #training_params_list=training_params_list[1:] #HACK HACK HACK!!!!!!
    #training_params_list=training_params_list[4:] #HACK HACK HACK!!!!!!

    #training_params_list=[training_params_list[0]] #HACK HACK HACK!!!!!!


    ######################

    conn_names=list(conndata_alltypes.keys())
    crosstrain_repeats=1 #crosstrain_repeats
    reduce_dimension_default=256
           
    for training_params in training_params_list:

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

            set_random_seed(0)
            
            
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
                
            if do_target_encoding or do_fixed_target_encoding:
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

                data_string_targetencoding="self"+"_"+roilist_str
                
                trainpath_list, data_orig, data_transformer_info_list = generate_training_paths(conndata_alltypes_targetencoding, conn_names, subjects, subjidx_train, subjidx_val, 
                                                trainpath_pairs="self", 
                                                trainpath_group_pairs=[], data_string=data_string_targetencoding, 
                                                batch_size=batchsize, skip_selfs=False, crosstrain_repeats=crosstrain_repeats,
                                                reduce_dimension=reduce_dimension,use_pretrained_encoder=False, keep_origscale_data=True,           
                                                use_lognorm_for_sc=do_use_lognorm_for_sc, 
                                                use_truncated_svd=use_truncated_svd, 
                                                use_truncated_svd_for_sc=do_use_tsvd_for_sc,
                                                input_transformation_info=transformation_type_string,
                                                precomputed_transformer_info_list=precomputed_transformer_info_list)
            
            else:
                trainpath_list, data_orig, data_transformer_info_list = generate_training_paths(conndata_alltypes, conn_names, subjects, subjidx_train, subjidx_val, 
                                                    trainpath_pairs=trainpath_pairs, 
                                                    trainpath_group_pairs=trainpath_group_pairs, data_string=data_string, 
                                                    batch_size=batchsize, skip_selfs=do_skipself, crosstrain_repeats=crosstrain_repeats,
                                                    reduce_dimension=reduce_dimension,use_pretrained_encoder=False, keep_origscale_data=True,           
                                                    use_lognorm_for_sc=do_use_lognorm_for_sc, 
                                                    use_truncated_svd=use_truncated_svd, 
                                                    use_truncated_svd_for_sc=do_use_tsvd_for_sc,
                                                    input_transformation_info=transformation_type_string,
                                                    precomputed_transformer_info_list=precomputed_transformer_info_list)
            
            
            #dont save input transforms if we are using precomputed ones
            save_input_transforms=precomputed_transformer_info_list is None or not all([k in precomputed_transformer_info_list for k in data_transformer_info_list])
            
            #load checkpoint file for initial starting point, if given
            if starting_point_file:
                #allow override of dropout amount
                checkpoint_override={}
                checkpoint_override['dropout']=training_params['dropout']
                net, checkpoint=Krakencoder.load_checkpoint(starting_point_file, checkpoint_override=checkpoint_override)
            else:
                net=None
            
            datastring0=trainpath_list[0]['data_string']
            
            trainblocks=training_params['trainblocks']
            
            for blockloop in range(trainblocks):
                if trainblocks > 1:
                    trainpath_list[0]['data_string']=datastring0+"_b%d" % (blockloop+1)
                training_params_tmp=training_params.copy()
                
                net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net, data_origscale_list=data_orig,
                                                 trainthreads=trainthreads,display_epochs=display_epochs,save_epochs=100,
                                                 checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False,
                                                 explicit_checkpoint_epoch_list=explicit_checkpoint_epoch_list,
                                                 precomputed_transformer_info_list=data_transformer_info_list,
                                                 save_input_transforms=save_input_transforms, 
                                                 output_file_prefix=output_file_prefix)
            
                if not training_params['roundtrip'] and add_roundtrip_epochs > 0:
                    print("Adding %d roundtrip epochs" % (add_roundtrip_epochs))
                    training_params_tmp=training_params.copy()
                    training_params_tmp['roundtrip']=True
                    training_params_tmp['nbepochs']=add_roundtrip_epochs
                    net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net, data_origscale_list=data_orig,
                                                     trainthreads=trainthreads,display_epochs=display_epochs,save_epochs=100,
                                                     checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False,
                                                     output_file_prefix=output_file_prefix)
                                                     
                if not training_params['meantarget_latentsim'] and add_meanlatent_epochs > 0:
                    print("Adding %d meanlatent epochs" % (add_meanlatent_epochs))
                    training_params_tmp=training_params.copy()
                    training_params_tmp['meantarget_latentsim']=True
                    training_params_tmp['latentsim_batchsize']=batchsize #maybe?
                    training_params_tmp['nbepochs']=add_meanlatent_epochs
                    net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net, data_origscale_list=data_orig,
                                                     trainthreads=trainthreads,display_epochs=display_epochs,save_epochs=100,
                                                     checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False,
                                                     output_file_prefix=output_file_prefix)
                                                     
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
                                                    reduce_dimension=reduce_dimension,use_pretrained_encoder=False, keep_origscale_data=True,           
                                                    use_lognorm_for_sc=do_use_lognorm_for_sc, 
                                                    use_truncated_svd=use_truncated_svd, 
                                                    use_truncated_svd_for_sc=do_use_tsvd_for_sc,
                                                    input_transformation_info=transformation_type_string,
                                                    precomputed_transformer_info_list=precomputed_transformer_info_list)
                    
                    if trainblocks > 1:
                        trainpath_list[0]['data_string']=datastring0+"_b%d" % (blockloop+1)
                    
                    training_params_tmp=training_params.copy()
                    training_params_tmp['fixed_encoding']=True
                    training_params_tmp['nbepochs']=add_fixed_encoding_epochs_after
                    net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net, data_origscale_list=data_orig,
                                                     trainthreads=trainthreads,display_epochs=display_epochs,save_epochs=100,
                                                     checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False,
                                                     output_file_prefix=output_file_prefix)
                    
if __name__ == "__main__":
    run_training_command(sys.argv[1:])
