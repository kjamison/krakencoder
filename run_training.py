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
import re
import os
import sys
import argparse

def load_subject_list(numsubj=420):
    if os.path.isdir('/Users/kwj5'):
        datafolder='/Users/kwj5/Box/HCP_SC_FC_new997'
        studyfolder='/Users/kwj5/Research/HCP'
    elif os.path.isdir('/home/kwj2001/colossus_shared/HCP'):
        studyfolder='/home/kwj2001/colossus_shared/HCP'
        datafolder='/home/kwj2001/colossus_shared/HCP'
    elif os.path.isdir('/home/ubuntu'):
        studyfolder='/home/ubuntu'
        datafolder='/home/ubuntu'

    famidx=np.loadtxt('%s/subjects_famidx_rfMRI_dMRI_complete_997.txt' % (studyfolder))
    subj997=np.loadtxt('%s/subjects_rfMRI_dMRI_complete_997.txt' % (studyfolder))
    
    if numsubj==420:
        subjects=np.loadtxt('%s/subjects_unrelated420_scfc.txt' % (studyfolder))
        famidx=np.array([famidx[i] for i,s in enumerate(subj997) if s in subjects])
    elif numsubj==993:
        subjects=np.loadtxt('%s/subjects_rfMRI_dMRI_complete_993_minus4.txt' % (studyfolder))
        famidx=np.array([famidx[i] for i,s in enumerate(subj997) if s in subjects])
    elif numsubj==997:
        subjects=subj997
    else:
        raise Exception("Unknown numsubj %s" % (numsubj))
    return subjects, famidx

def load_data(subjects=[], conn_name_list=[], fc_filter_list=["hpf"], quiet=False):
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
        subjects=np.loadtxt('%s/subjects_unrelated420_scfc.txt' % (studyfolder))

    #scfile_fields=['orig','volnorm']
    #scfile_fields=['volnorm']

    connfile_info=[]
    datagroup='SC'
    #consider: log transform on SC inputs?
    # also maybe try using orig instead of just volnorm? orig is easier to log transform (all pos vals)
    # log10(volnorm) has a lot of negatives (counts<1)
    # is there any reasonable way to justify log10(connectome)./log10(volumes) to scale?
    #  then it would be positive but scaled by the (logscaled) volumes
    # main reason this matters is that we have 0s for SC which log can't handle, and we cant just set those to
    #  log10(0)->0 because then log10(.1)-> -1 would look < 0 

    pretrained_transformer_file={}
    pretrained_transformer_file['fs86_ifod2act_volnorm']=None
    #pretrained_transformer_file['fs86_sdstream_volnorm']='%s/connae_checkpoint_self_fs86_sdstream_volnorm_batch42_nonorm_1paths_latent256_0layer_2000epoch_lr0.0001_correye+enceye.w10+mse.w10_adamw.w0.1_20220405_130152675337_epoch002000.pt' % (datafolder)
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
    
    connfile_info.append({'name':'fs86_ifod2act_volnorm','file':'%s/sc_ifod2act_fs86_997subj.mat' % (datafolder),'types':['volnorm'],'group':datagroup})
    connfile_info.append({'name':'fs86_sdstream_volnorm','file':'%s/sc_sdstream_fs86_997subj.mat' % (datafolder),'types':['volnorm'],'group':datagroup})
    connfile_info.append({'name':'shen268_ifod2act_volnorm','file':'%s/sc_ifod2act_shen268_997subj.mat' % (datafolder),'types':['volnorm'],'group':datagroup})
    connfile_info.append({'name':'shen268_sdstream_volnorm','file':'%s/sc_sdstream_shen268_997subj.mat' % (datafolder),'types':['volnorm'],'group':datagroup})
    connfile_info.append({'name':'coco439_ifod2act_volnorm','file':'%s/sc_ifod2act_cocommpsuit439_997subj.mat' % (datafolder),'types':['volnorm'],'group':datagroup})
    connfile_info.append({'name':'coco439_sdstream_volnorm','file':'%s/sc_sdstream_cocommpsuit439_997subj.mat' % (datafolder),'types':['volnorm'],'group':datagroup})

    datagroup='FC'
    #consider: do np.arctanh for FC inputs?
    if "hpf" in fc_filter_list or len(conn_name_list)==0:
        connfile_info.append({'name':'FCcov_fs86_hpf','file':'%s/fc_fs86_FCcov_hpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCcov_fs86_hpfgsr','file':'%s/fc_fs86_FCcov_hpf_gsr_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCpcorr_fs86_hpf','file':'%s/fc_fs86_FCpcorr_hpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})

        connfile_info.append({'name':'FCcov_shen268_hpf','file':'%s/fc_shen268_FCcov_hpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCcov_shen268_hpfgsr','file':'%s/fc_shen268_FCcov_hpf_gsr_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCpcorr_shen268_hpf','file':'%s/fc_shen268_FCpcorr_hpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})

        connfile_info.append({'name':'FCcov_coco439_hpf','file':'%s/fc_cocommpsuit439_FCcov_hpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCcov_coco439_hpfgsr','file':'%s/fc_cocommpsuit439_FCcov_hpf_gsr_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCpcorr_coco439_hpf','file':'%s/fc_cocommpsuit439_FCpcorr_hpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})

    if "bpf" in fc_filter_list or len(conn_name_list)==0:
        connfile_info.append({'name':'FCcov_fs86_bpf','file':'%s/fc_fs86_FCcov_bpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCcov_fs86_bpfgsr','file':'%s/fc_fs86_FCcov_bpf_gsr_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCpcorr_fs86_bpf','file':'%s/fc_fs86_FCpcorr_bpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        
        connfile_info.append({'name':'FCcov_shen268_bpf','file':'%s/fc_shen268_FCcov_bpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCcov_shen268_bpfgsr','file':'%s/fc_shen268_FCcov_bpf_gsr_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCpcorr_shen268_bpf','file':'%s/fc_shen268_FCpcorr_bpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        
        connfile_info.append({'name':'FCcov_coco439_bpf','file':'%s/fc_cocommpsuit439_FCcov_bpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCcov_coco439_bpfgsr','file':'%s/fc_cocommpsuit439_FCcov_bpf_gsr_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCpcorr_coco439_bpf','file':'%s/fc_cocommpsuit439_FCpcorr_bpf_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        
    if "nofilt" in fc_filter_list or len(conn_name_list)==0:
        connfile_info.append({'name':'FCcov_fs86_nofilt','file':'%s/fc_fs86_FCcov_nofilt_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCcov_fs86_nofiltgsr','file':'%s/fc_fs86_FCcov_nofilt_gsr_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCpcorr_fs86_nofilt','file':'%s/fc_fs86_FCpcorr_nofilt_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})

        connfile_info.append({'name':'FCcov_shen268_nofilt','file':'%s/fc_shen268_FCcov_nofilt_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCcov_shen268_nofiltgsr','file':'%s/fc_shen268_FCcov_nofilt_gsr_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCpcorr_shen268_nofilt','file':'%s/fc_shen268_FCpcorr_nofilt_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        
        connfile_info.append({'name':'FCcov_coco439_nofilt','file':'%s/fc_cocommpsuit439_FCcov_nofilt_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCcov_coco439_nofiltgsr','file':'%s/fc_cocommpsuit439_FCcov_nofilt_gsr_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})
        connfile_info.append({'name':'FCpcorr_coco439_nofilt','file':'%s/fc_cocommpsuit439_FCpcorr_nofilt_997subj.mat' % (datafolder),'types':['FC'],'group':datagroup})


    #print("connfile_info:",[k["name"] for k in connfile_info])
    conndata_alltypes={}

    #print("conn_name_list:",conn_name_list)
    if len(conn_name_list)==0:
        for ci_idx,ci in enumerate(connfile_info):
            cname=ci['name']
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
            else:
                print("Unknown data flavor for %s" % (cname))
                sys.exit(1)
            
            if not quiet:
                print("Loading %d/%d: %s" % (ci_idx+1,len(connfile_info),ci['file']))
            Cdata=loadmat(ci['file'])
            subjmissing=Cdata['ismissing'][0]>0
            subjects997=Cdata['subject'][0][~subjmissing].astype(float)
            #for conntype in ci['types']:
            nroi=Cdata[conntype][0][0].shape[0]
            trimask=np.triu_indices(nroi,1)
            npairs=trimask[0].shape[0]
            Ctriu=[x[trimask] for x in Cdata[conntype][0][~subjmissing]]
            #restrict to 420 unrelated subjects
            Ctriu=[x for i,x in enumerate(Ctriu) if subjects997[i] in subjects]
            #conn_name='%s_%s' % (ci['name'],conntype)
            conn_name='%s_%s' % (cname,conntype)
            transformer_file=None
            if conn_name in pretrained_transformer_file:
                transformer_file=pretrained_transformer_file[conn_name]
            conndata_alltypes[conn_name]={'data':np.vstack(Ctriu),'numpairs':npairs,'group':ci['group'],'transformer_file':transformer_file}
    else:
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
            Cdata=loadmat(ci['file'])
            subjmissing=Cdata['ismissing'][0]>0
            subjects997=Cdata['subject'][0][~subjmissing].astype(float)
            
            nroi=Cdata[conntype][0][0].shape[0]
            trimask=np.triu_indices(nroi,1)
            npairs=trimask[0].shape[0]
            Ctriu=[x[trimask] for x in Cdata[conntype][0][~subjmissing]]
            #restrict to 420 unrelated subjects
            Ctriu=[x for i,x in enumerate(Ctriu) if subjects997[i] in subjects]
            #conn_name='%s_%s' % (ci['name'],conntype)
            conn_name='%s_%s' % (cname,conntype)
            
            transformer_file=None
            if conn_name in pretrained_transformer_file:
                transformer_file=pretrained_transformer_file[conn_name]
                
            conndata_alltypes[conn_name]={'data':np.vstack(Ctriu),'numpairs':npairs,'group':ci['group'],'transformer_file':transformer_file}
            
            
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


#######################################################
#######################################################
#### 

def argument_parse(argv):
    parser=argparse.ArgumentParser(description='Train krakencoder', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--subjectcount',action='store',dest='subjectcount',type=int, default=420, help='Which subject set? 420 (default) or 993')
    parser.add_argument('--datagroups',action='append',dest='datagroups',help='list of SCFC, FC, SC, FC2SC, etc...',nargs='*')
    parser.add_argument('--dataflavors',action='append',dest='dataflavors',help='SCifod2act,SCsdstream,FCcov,FCcovgsr,FCpcorr',nargs='*')
    parser.add_argument('--trainpaths',action='append',dest='trainpaths',help='eg SCifod2act->FCpcorr',nargs='*')
    parser.add_argument('--roinames',action='append',dest='roinames',help='fs86,shen268,coco439...',nargs='*')
    parser.add_argument('--fcfilt',action='append',dest='fcfilt',help='list of hpf (default), bpf, nofilt',nargs='*')
    parser.add_argument('--adamdecay',action='store',dest='adam_decay',type=float, default=0.01, help='Adam weight decay')
    parser.add_argument('--epochs',action='store',dest='epochs',type=int, default=5000, help='number of epochs')
    parser.add_argument('--losstype',action='append',dest='losstype',help='list of correye+enceye, dist+encdist, etc...',nargs='*')
    parser.add_argument('--batchsize',dest='batch_size',type=int,default=41,help='main batch size. default=41 (no batch)')
    parser.add_argument('--dropout',action='append',dest='dropout',type=float,help='list of dropouts to try',nargs='*')
    parser.add_argument('--hiddenlayersizes',action='append',dest='hiddenlayersizes',type=int,help='hidden layer sizes',nargs='*')
    parser.add_argument('--leakyrelu',action='store',dest='leakyrelu_negative_slope', type=float, default=0., help='Leaky ReLU negative slope (0=ReLU). For deep networks only')
    parser.add_argument('--latentsize',action='store',dest='latentsize', type=int, default=128, help='latent space dimension')
    parser.add_argument('--latentunit',action='store_true',dest='latentunit', help='latent all normalzed to shell of unit sphere')
    parser.add_argument('--latentradweight',action='store',dest='latentradweight', type=float, default=10, help='weight to apply to keeping latent rad<1')
    parser.add_argument('--noskipacc',action='store_true',dest='noskipacc', help='do NOT skip accurate paths during training')
    parser.add_argument('--noearlystop',action='store_true',dest='noearlystop', help='do NOT stop early if skipacc (keep working latentsim)')
    parser.add_argument('--skipself',action='store_true',dest='skipself', help='Skip A->A paths during training')
    parser.add_argument('--roundtrip',action='store_true',dest='roundtrip', help='roundtrip training paths A->B->A')
    parser.add_argument('--addroundtripepochs',action='store',dest='add_roundtrip_epochs', type=int, default=0, help='add roundtrip training paths A->B->A AFTER normal training')
    parser.add_argument('--addmeanlatentepochs',action='store',dest='add_meanlatent_epochs', type=int, default=0, help='add meanlatent training paths AFTER normal training')
    parser.add_argument('--trainblocks',action='store',dest='trainblocks', type=int, default=1, help='How many total times perform normal training + (roundtrip or meanlatent) set? (optimizer resets each block)')
    parser.add_argument('--pcadim',action='store',dest='pcadim', type=int, default=256, help='pca dimensionality reduction (default=256. 0=No PCA)')
    parser.add_argument('--tsvd',action='store_true',dest='use_tsvd', help='use truncated SVD instead of PCA')
    parser.add_argument('--sclognorm',action='store_true',dest='sc_lognorm', help='non-PCA runs use log transform for SC')
    parser.add_argument('--sctsvd',action='store_true',dest='sc_tsvd', help='use Truncated SVD for SC')
    parser.add_argument('--mseweight',action='store',dest='mseweight', type=float, default=1, help='Weight to apply to true->predicted MSE')
    parser.add_argument('--latentinnerweight',action='store',dest='latent_inner_loss_weight', type=float, default=10, help='Weight to apply to latent-space inner loop loss (enceye,encdist, etc...)')
    parser.add_argument('--latentsimweight',action='append',dest='latent_sim_weight',type=float,help='list of latentsimloss weights to try . default=5000',nargs='*')
    parser.add_argument('--latentsimbatchsize',dest='latent_sim_batch_size',type=int,default=0,help='Batch size for latentsimloss. default=0 (no batch)')
    parser.add_argument('--singleoptimizer',action='store_true',dest='single_optimizer', help='Use single optimizer across all paths and latentsim')
    parser.add_argument('--transformation',action='store',dest='transformation', help='transformation type string (overrides pcadim, tsvd,etc)')
    parser.add_argument('--checkpointepochsevery',action='store',dest='checkpoint_epochs_every', type=int, default=1000, help='How often to save checkpoints')
    parser.add_argument('--explicitcheckpointepochs',action='append',dest='explicit_checkpoint_epochs', type=int, help='Explicit list of epochs at which to save checkpoints',nargs='*')
    parser.add_argument('--inputxform',action='append',dest='input_transform_file', help='Precomputed transformer files (.npy)',nargs='*')
    parser.add_argument('--maxthreads',action='store',dest='max_threads', type=int, default=10, help='How many CPU threads to use')
    parser.add_argument('--subjectfile',action='store',dest='subject_split_file', help='.mat file containing pre-saved "subjects","subjidx_train","subjidx_val","subjidx_test" fields')
    parser.add_argument('--startingpoint',action='store',dest='starting_point_file', help='.pt file to START with')
    parser.add_argument('--encodedinputfile',action='store',dest='encoded_input_file', help='.mat file containing latent space data')
    parser.add_argument('--fixedencoding',action='store_true',dest='fixed_encoding', help='Just train encoders/decoders to match fixed --encodinginputfile')
    parser.add_argument('--addfixedencodingepochsafter',action='store',dest='add_fixed_encoding_epochs_after', type=int, default=0, help='Add fixedencoding epochs AFTER normal epochs')
    parser.add_argument('--addfixedencodingepochsbefore',action='store',dest='add_fixed_encoding_epochs_before', type=int, default=0, help='Add fixedencoding epochs BEFORE normal epochs')
    
    return parser.parse_args(argv)

if __name__ == "__main__":
    ######### parse input arguments
    args=argument_parse(sys.argv[1:])
    
    trainthreads=args.max_threads
    
    input_nsubj=args.subjectcount
    input_datagroups=args.datagroups
    input_epochs=args.epochs
    input_roundtrip=args.roundtrip
    
    input_adamdecay=args.adam_decay
    
    input_pcadim=args.pcadim
    input_use_tsvd=args.use_tsvd
    do_use_lognorm_for_sc=args.sc_lognorm
    do_use_tsvd_for_sc=args.sc_tsvd
    
    input_transform_file_list=[]
    if args.input_transform_file and len(args.input_transform_file) > 0:
        tmpxfm=flatlist(args.input_transform_file)
        if len(tmpxfm)>0:
            input_transform_file_list=tmpxfm
    
    transformation_type_string=args.transformation
    
    add_roundtrip_epochs=args.add_roundtrip_epochs

    add_meanlatent_epochs=args.add_meanlatent_epochs
    
    input_trainblocks=args.trainblocks
    
    checkpoint_epochs=args.checkpoint_epochs_every
    
    explicit_checkpoint_epoch_list=[]
    if args.explicit_checkpoint_epochs and len(args.explicit_checkpoint_epochs)>0:
        tmpepoch=flatlist(args.explicit_checkpoint_epochs)
        if len(tmpepoch)>0:
            explicit_checkpoint_epoch_list=tmpepoch
    
    do_skipaccpath=True
    if args.noskipacc:
        do_skipaccpath=False
    
    do_earlystop=True
    if args.noearlystop:
        do_earlystop=False
    

    do_skipself=args.skipself
    
    latentsize=args.latentsize
    
    input_latentradweight = args.latentradweight;
    input_latentunit = args.latentunit
    if input_latentunit:
        input_latentradweight=0
    
    input_roilist=["fs86+shen268+coco439"]
    if args.roinames and len(args.roinames) > 0:
        input_roilist=flatlist(args.roinames)
    
    input_roilist=flatlist([x.split("+") for x in input_roilist])
    
    input_dataflavors=["SCifod2act","SCsdstream","FCcov","FCcovgsr","FCpcorr"]
    if args.dataflavors and len(args.dataflavors) > 0:
        tmpflavs=flatlist(args.dataflavors)
        if len(tmpflavs)>0:
            input_dataflavors=tmpflavs

    input_datagroups=['SCFC','SC','FC']
    if args.datagroups and len(args.datagroups) > 0:
        tmpgroups=flatlist(args.datagroups)
        if len(tmpgroups)>0:
            input_datagroups=tmpgroups
    
    input_trainpath_pairs=[]
    input_trainpath_flavor_pairs=[]
    if args.trainpaths and len(args.trainpaths) > 0:
        tmppaths=flatlist(args.trainpaths)
        if len(tmppaths)>0:
            input_trainpath_flavor_pairs=[tp.split("->") for tp in tmppaths]
            input_datagroups=["all"]
            input_dataflavors=flatlist(input_trainpath_flavor_pairs)
    
    fcfilt_types=["hpf"]
    if args.fcfilt and len(args.fcfilt) > 0 and len:
        tmpfilt=flatlist(args.fcfilt)
        if len(tmpfilt) > 0:
            fcfilt_types=tmpfilt
    
    #input_lossnames=['correye+enceye','neidist+encdist']
    input_lossnames=['correye+enceye']
    if args.losstype and len(args.losstype) > 0:
        input_lossnames=flatlist(args.losstype)
    
    input_dropout_list=[0]
    if args.dropout and len(args.dropout) > 0:
        input_dropout_list=flatlist(args.dropout)
    
    input_latentsimweight_list=[5000]
    if args.latent_sim_weight and len(args.latent_sim_weight) > 0:
        input_latentsimweight_list=flatlist(args.latent_sim_weight)
    
    input_hiddenlayers=[]
    if args.hiddenlayersizes and len(args.hiddenlayersizes) > 0:
        input_hiddenlayers=flatlist(args.hiddenlayersizes)
    
    input_leakyrelu=args.leakyrelu_negative_slope
    
    input_mse_weight=args.mseweight
    
    input_latent_inner_loss_weight=args.latent_inner_loss_weight

    input_subject_list_file=args.subject_split_file
    
    input_use_separate_optimizer=not args.single_optimizer
    
    input_batchsize=args.batch_size
    input_latentsimbatchsize=args.latent_sim_batch_size
    
    input_encodingfile=args.encoded_input_file
    do_fixed_encoding=args.fixed_encoding
    add_fixed_encoding_epochs_after=args.add_fixed_encoding_epochs_after
    add_fixed_encoding_epochs_before=args.add_fixed_encoding_epochs_before
    
    starting_point_file=args.starting_point_file
    
    #################################3

    #input_nsubj=420
    #input_nsubj=993
    
    #subjects, conndata_alltypes=load_data()
    
    if len(input_roilist) > 0:
        input_dataflavors_upper=[x.upper() for x in input_dataflavors]
        
        roilist_str="+".join(input_roilist)
        input_conn_name_list=[]
        input_trainpath_pairs=[]
        for a in input_roilist:
            name2flavor_dict={}
            name2flavor_dict["%s_ifod2act_volnorm" % (a)]="SCifod2act"
            name2flavor_dict["%s_sdstream_volnorm" % (a)]="SCsdstream"
            for f in fcfilt_types:
                name2flavor_dict["FCcov_%s_%s_FC" % (a,f)]="FCcov"
                name2flavor_dict["FCcov_%s_%sgsr_FC" % (a,f)]="FCcovgsr"
                name2flavor_dict["FCpcorr_%s_%s_FC" % (a,f)]="FCpcorr"
            
            flavor2name=lambda x: [k for k,v in name2flavor_dict.items() if v==x][0]
            
            if set(["SCIFOD2ACT","IFOD2ACT"]) & set(input_dataflavors_upper):
                input_conn_name_list+=[flavor2name("SCifod2act")]
            if set(["SCSDSTREAM","SDSTREAM"]) & set(input_dataflavors_upper):
                input_conn_name_list+=[flavor2name("SCsdstream")]
            for f in fcfilt_types:
                if set(["FCCOV","COV"]) & set(input_dataflavors_upper):
                    input_conn_name_list+=[flavor2name("FCcov")]
                if set(["FCCOVGSR","COVGSR"]) & set(input_dataflavors_upper):
                    input_conn_name_list+=[flavor2name("FCcovgsr")]
                if set(["FCPCORR","PCORR"]) & set(input_dataflavors_upper):
                    input_conn_name_list+=[flavor2name("FCpcorr")]
            
            if input_trainpath_flavor_pairs:
                input_trainpath_pairs+=[[flavor2name(x),flavor2name(y)] for x,y in input_trainpath_flavor_pairs]
                #add selfs by default (removed by generate_training_paths later if requested)
                input_trainpath_pairs+=[[flavor2name(x),flavor2name(x)] for x,y in input_trainpath_flavor_pairs]
                input_trainpath_pairs+=[[flavor2name(y),flavor2name(y)] for x,y in input_trainpath_flavor_pairs]
            #input_conn_name_list=flatlist(input_conn_name_list)
    else:
        roilist_str="fs86+shen268+coco439"
        input_conn_name_list=[]

    precomputed_transformer_info_list=None
    if len(input_transform_file_list)>0:
        precomputed_transformer_info_list={}
        for ioxfile in input_transform_file_list:
            print("Loading precomputed input transformations: %s" % (ioxfile))
            ioxtmp=np.load(ioxfile,allow_pickle=True).item()
            for k in ioxtmp:
                precomputed_transformer_info_list[k]=ioxtmp[k]
                precomputed_transformer_info_list[k]['filename']=ioxfile.split(os.sep)[-1]
    
    input_subject_list=None
    input_subject_list_str="" #string to designate if we are using alternate train/val/test split with all retest subjects in held-out TEST
    if input_subject_list_file:
        print("Loading subject splits from %s" % (input_subject_list_file))
        input_subject_list=loadmat(input_subject_list_file)
        #fields are read in as an array within a single-val array, so just fix that for easier coding
        for f in ["subjects", "subjidx_train", "subjidx_val", "subjidx_test"]:
            input_subject_list[f]=input_subject_list[f][0]
            print("\t%d %s" % (len(input_subject_list[f]),f))
        
        if "710train_80val_203test_retestInTest" in input_subject_list_file:
            input_subject_list_str="B"

    
    subjects, famidx = load_subject_list(input_nsubj)
    subjects_out, conndata_alltypes = load_data(subjects=subjects, conn_name_list=input_conn_name_list, fc_filter_list=fcfilt_types)
    
    if len(subjects) != len(subjects_out) or not all(subjects==subjects_out):
        raise Exception("Subjects did not match expected")

    if input_subject_list and (len(subjects) != len(input_subject_list["subjects"]) or not all(subjects==input_subject_list["subjects"])):
        raise Exception("Subjects in --subjectfile did not match subjects in input data")
    
    nsubj=conndata_alltypes[list(conndata_alltypes.keys())[0]]['data'].shape[0]
    
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
        training_params_listdict['batchsize']=[input_batchsize] #avoid leaving out too many for the famidx case
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
    
    training_params_listdict['fixed_encoding']=[do_fixed_encoding]
    training_params_listdict['meantarget_latentsim']=[False]
    
    training_params_listdict['trainblocks']=[input_trainblocks]
    
    #training_params_listdict['batchwise_latentsim']=[False]
    #training_params_listdict['latentsim_loss_weight']=[500,1000,2500,5000,10000,50000]
    
    training_params_listdict['roundtrip']=[input_roundtrip]
    
    training_params_list = dict_combination_list(training_params_listdict, reverse_field_order=True)    
    #%matplotlib inline

    #training_params_list=training_params_list[1:] #HACK HACK HACK!!!!!!
    #training_params_list=training_params_list[4:] #HACK HACK HACK!!!!!!

    #training_params_list=[training_params_list[0]] #HACK HACK HACK!!!!!!


    ######################
    conn_names=list(conndata_alltypes.keys())
    
    crosstrain_repeats=1 #crosstrain_repeats
    reduce_dimension_default=256

    #trainthreads=24
    #trainthreads=16

    trainval_test_seed=0
    train_val_seed=0
    
    for training_params in training_params_list:
        
        if input_subject_list:
            subjidx_train=input_subject_list['subjidx_train'].copy()
            subjidx_val=input_subject_list['subjidx_val'].copy()
            subjidx_test=input_subject_list['subjidx_test'].copy()
        else:
            if len(subjects)>420:
                subjidx_trainval, subjidx_test, famidx_trainval, famidx_test = random_train_test_split_groups(groups=famidx, numsubj=nsubj, 
                                                                                                          seed=trainval_test_seed,
                                                                                                          train_frac=0.8)
            else:
                subjidx_trainval, subjidx_test = random_train_test_split(numsubj=nsubj, train_frac=0.8, seed=trainval_test_seed)
        
            #split train/val from within initial trainval
            if len(subjects)>420:
                    subjidx_train, subjidx_val, famidx_train, famidx_val = random_train_test_split_groups(groups=famidx_trainval, subjlist=subjidx_trainval,
                                                                                                seed=train_val_seed, 
                                                                                                train_frac=0.9)
            else:
                subjidx_train, subjidx_val = random_train_test_split(subjlist=subjidx_trainval, train_frac=0.875, seed=train_val_seed)
    

        ###################
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

        #for grouptype in ['SC','FC','SCFC','FC2SC']:
        #for grouptype in ['SCFC']:
        for grouptype in input_datagroups:
            ###################
            #data_string=None
            #data_string="FC_fs86+shen268"
            #data_string="SCFC_fs86"
            #data_string="SC_fs86+shen268"
            #data_string=grouptype+"_fs86+shen268"
            #data_string=grouptype+"_fs86"
            
            #data_string=grouptype+"_fs86+shen268+coco439"

            #if batchsize != 6:
            #    data_string+="_batch%d" % (batchsize)
            
            #data_string="fs86+shen268_ifod2act_volnorm"
            #data_string="fs86+shen268"
            trainpath_pairs=[] #only this direction 
            #trainpath_group_pairs=[['FC','SC'],['SC','SC'],['FC','FC']] #skip SC->FC

            if len(input_trainpath_pairs) > 0:
                trainpath_pairs = input_trainpath_pairs
                trainpath_group_pairs=[]
                data_string="+".join(["%s2%s" % (x,y) for x,y in input_trainpath_flavor_pairs])
            else:
                if grouptype == "SC":
                    trainpath_group_pairs=[['SC','SC']]
                elif grouptype == "FC":
                    trainpath_group_pairs=[['FC','FC']]
                elif grouptype == "SC2FC":
                    trainpath_group_pairs=[['SC','FC'],['SC','SC'],['FC','FC']] #skip FC->SC
                elif grouptype == "FC2SC":
                    trainpath_group_pairs=[['FC','SC'],['SC','SC'],['FC','FC']] #skip SC->FC
                elif grouptype == "SCFC":
                    trainpath_group_pairs=[] #all
                
                data_string=grouptype
            
            data_string+="_"+roilist_str
            
            if len(subjects)!=420:
                data_string+="_%dsubj" % (len(subjects))
                data_string+=input_subject_list_str #add extra chars if using alternate training set
            
            set_random_seed(0)
            
            
            #generate trainpath info each time so the dataloader batches are reproducible
            encoded_inputs=None
            if input_encodingfile:
                Mtmp=loadmat(input_encodingfile)
                if not 'subjects' in Mtmp:
                    raise Exception("input encoding file must have 'subjects' field")
                if len(Mtmp['subjects'][0]) != len(subjects):
                    raise Exception("input encoding file must contain the same number of subjects (%d) as input data (%d)", len(Mtmp['subjects'][0]),len(subjects))
                if not all([Mtmp['subjects'][0][i]==subjects[i] for i in range(len(subjects))]):
                    raise Exception("input encoding subjects must match input data subjects")

                encoded_inputs=Mtmp['encoded'].copy()
                print("Loaded target latent-space values from %s (%s)" % (input_encodingfile,encoded_inputs.shape))
                
            if do_fixed_encoding:
                if encoded_inputs is None:
                    raise Exception("Must provide encoded inputs file")
                
                conndata_alltypes_fixedencoding=conndata_alltypes.copy()
                for conntype in conndata_alltypes_fixedencoding.keys():
                    conndata_alltypes_fixedencoding[conntype]['encoded']=encoded_inputs.copy()
                
                data_string_fixedencoding="self"+"_"+roilist_str
                
                trainpath_list, data_orig, data_transformer_info_list = generate_training_paths(conndata_alltypes_fixedencoding, conn_names, subjects, subjidx_train, subjidx_val, 
                                                trainpath_pairs="self", 
                                                trainpath_group_pairs=[], data_string=data_string_fixedencoding, 
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
                                                 trainthreads=trainthreads,display_epochs=100,save_epochs=100,
                                                 checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False,
                                                 explicit_checkpoint_epoch_list=explicit_checkpoint_epoch_list,
                                                 precomputed_transformer_info_list=data_transformer_info_list,
                                                 save_input_transforms=save_input_transforms)
            
                if not training_params['roundtrip'] and add_roundtrip_epochs > 0:
                    print("Adding %d roundtrip epochs" % (add_roundtrip_epochs))
                    training_params_tmp=training_params.copy()
                    training_params_tmp['roundtrip']=True
                    training_params_tmp['nbepochs']=add_roundtrip_epochs
                    net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net, data_origscale_list=data_orig,
                                                     trainthreads=trainthreads,display_epochs=100,save_epochs=100,
                                                     checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False)
                                                     
                if not training_params['meantarget_latentsim'] and add_meanlatent_epochs > 0:
                    print("Adding %d meanlatent epochs" % (add_meanlatent_epochs))
                    training_params_tmp=training_params.copy()
                    training_params_tmp['meantarget_latentsim']=True
                    training_params_tmp['latentsim_batchsize']=batchsize #maybe?
                    training_params_tmp['nbepochs']=add_meanlatent_epochs
                    net, trainrecord = train_network(trainpath_list,training_params_tmp, net=net, data_origscale_list=data_orig,
                                                     trainthreads=trainthreads,display_epochs=100,save_epochs=100,
                                                     checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False)
                                                     
                if not do_fixed_encoding and add_fixed_encoding_epochs_after > 0:
                    raise Exception("add_fixed_encoding not yet supported")
                    print("Adding %d fixedencoding epochs" % (add_fixed_encoding_epochs_after))
                    
                    conndata_alltypes_fixedencoding=conndata_alltypes.copy()
                    #for conntype in conndata_alltypes_fixedencoding.keys():
                        
                        
                    data_string_fixedencoding="self"+"_"+roilist_str
                
                    trainpath_list, data_orig, data_transformer_info_list = generate_training_paths(conndata_alltypes_fixedencoding, conn_names, subjects, subjidx_train, subjidx_val, 
                                                    trainpath_pairs="self", 
                                                    trainpath_group_pairs=[], data_string=data_string_fixedencoding, 
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
                                                     trainthreads=trainthreads,display_epochs=100,save_epochs=100,
                                                     checkpoint_epochs=checkpoint_epochs, update_single_checkpoint=False)