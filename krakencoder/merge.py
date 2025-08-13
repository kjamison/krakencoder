"""
Functions for merging multiple networks into a single network.
For example, if you have trained one network on a set of input flavors, and then 
trained a new network on a different set of input flavors with the same target
latent space, you can use this function to merge the encoders and decoders from
both networks into a single model that can translate between all input flavors.
"""

from krakencoder.model import *
from krakencoder.data import canonical_data_flavor
from krakencoder.utils import format_columns
import os

def merge_models(net_and_checkpoint_dict_list, canonicalize_input_names=False, input_names=None):
    """
    Merge encoders+decoders from multiple networks into a single network.
    
    Provide a list of dictionaries with the following format: 
    [{'net': net1, 'checkpoint': checkpoint1_dict}, {'net': net2, 'checkpoint': checkpoint2_dict}]
    where net1 and net2 are Krakencoder objects, and checkpoint1_dict and checkpoint2_dict are the
    corresponding checkpoint dictionaries stored in checkpoint files, with additional parameters describing
    each model.
    
    This function will create a single list of input names from all the networks (removing duplicates),
    and then create a new model combining the encoders and decoders from all the networks.
    
    Parameters:
    net_and_checkpoint_dict_list : list of dict, each with 'net' and 'checkpoint'. See description above.
    canonicalize_input_names : bool (default False) whether to canonicalize the input names before merging.
        This is useful if the input names are a mix of old and new flavor name conventions
    input_names : list of str (optional). If provided, only merge the encoders/decoders for these input names.
    
    Returns:
    net, checkpoint: Krakencoder object and checkpoint dictionary for the merged model
    """
    
    input_names_to_merge=input_names
    
    # get the unique input names, in order of appearance
    input_name_list=[]
    input_name_list_source_net_idx=[]
    input_name_list_source_input_idx=[]
    for inet, net_chk in enumerate(net_and_checkpoint_dict_list):
        for i,n in enumerate(net_chk['checkpoint']['input_name_list']):
            if input_names_to_merge is not None and n not in input_names_to_merge:
                #skip this input name if it is not in the requested list
                continue
            if 'input_names_to_merge' in net_chk['checkpoint'] and \
               net_chk['checkpoint']['input_names_to_merge'] is not None and n not in net_chk['checkpoint']['input_names_to_merge']:
                #skip this input name if it is not in the requested list FOR THIS MODEL
                #this allows merging parts of models that have the same input names
                continue
            #if canonicalize_input_names is True, convert the input name to the canonical flavor name
            if canonicalize_input_names:
                n=canonical_data_flavor(n)
            if n not in input_name_list:
                input_name_list.append(n)
                input_name_list_source_net_idx.append(inet)
                input_name_list_source_input_idx.append(i)
    
    net=net_and_checkpoint_dict_list[0]['net']
    checkpoint_info=net_and_checkpoint_dict_list[0]['checkpoint'].copy()
    
    for iconn, conn_name in enumerate(input_name_list):
        inet=input_name_list_source_net_idx[iconn]
        iinput=input_name_list_source_input_idx[iconn]
        net_tmp=net_and_checkpoint_dict_list[inet]['net']
        chk_tmp=net_and_checkpoint_dict_list[inet]['checkpoint']
        if inet==0:
            #we already have the first network's encoders/decoders
            continue
        #add the encoders and decoders from the other networks
        net.encoder_list.append(net_tmp.encoder_list[iinput])
        net.decoder_list.append(net_tmp.decoder_list[iinput])
        net.inputsize_list.append(net_tmp.inputsize_list[iinput])
        
    num_inputs_orig0=len(net_and_checkpoint_dict_list[0]['checkpoint']['input_name_list'])
    num_trainpaths_orig0=len(net_and_checkpoint_dict_list[0]['checkpoint']['trainpath_decoder_index_list'])
    
    for k,v in checkpoint_info.items():
        if k in ['input_name_list','trainpath_encoder_index_list','trainpath_decoder_index_list']:
            #skip these fields. we will update them later
            continue
        try:
            if len(v)==num_inputs_orig0:
                #this field is a list of values for each of the original input names.
                #we need to merge and reorder it across all checkpoints to match the new input names
                vnew=[]
                for iconn, conn_name in enumerate(input_name_list):
                    inet=input_name_list_source_net_idx[iconn]
                    iinput=input_name_list_source_input_idx[iconn]
                    chk_tmp=net_and_checkpoint_dict_list[inet]['checkpoint']
                    vnew.append(chk_tmp[k][iinput])
                print("Merging input field checkpoint_info['%s']: %d items" % (k,len(vnew)))
                checkpoint_info[k]=vnew
                
            elif len(v)==num_trainpaths_orig0:
                #this field is a list of values for each of the original trainpaths.
                #we would need to merge and reorder it across all checkpoints to match the new trainpaths
                #but it's quite a bit more complicated than that, so we'll just delete it for now
                print("Removing trainpath field checkpoint_info['%s']" % (k))
                del checkpoint_info[k]
        except:
            pass
    
    
    checkpoint_info['input_name_list']=input_name_list
    checkpoint_info['merged_training_params_list']=[chk['checkpoint']['training_params'].copy() for chk in net_and_checkpoint_dict_list]
    checkpoint_info['merged_checkpoint_info_list']=[chk['checkpoint'].copy() for chk in net_and_checkpoint_dict_list]
    checkpoint_info['merged_source_net_idx']=input_name_list_source_net_idx
    checkpoint_info['merged_source_input_idx']=input_name_list_source_input_idx
    
    checkpoint_info['training_params']['total_parameter_count']=sum([p.numel() for p in net.parameters()]) #count total weights in model
        
    #create new list of trainpath_encoder_index_list and trainpath_decoder_index_list
    #so they point to the updated entries in input_name_list
    checkpoint_info['trainpath_encoder_index_list']=[]
    checkpoint_info['trainpath_decoder_index_list']=[]
    for i, conn1 in enumerate(checkpoint_info['input_name_list']):
        for j, conn2 in enumerate(checkpoint_info['input_name_list']):
            checkpoint_info['trainpath_encoder_index_list'].append(i)
            checkpoint_info['trainpath_decoder_index_list'].append(j)

    return net, checkpoint_info

def merge_model_files(checkpoint_filename_list, canonicalize_input_names=False, input_names=None):
    """
    Load a list of saved Krakencoder models and merge them into a single model.
    
    See merge_models() for more details.
    
    Parameters:
    checkpoint_filename_list : List of .pt files to merge
    canonicalize_input_names : bool (default False) whether to canonicalize the input names before merging.
        This is useful if the input names are a mix of old and new flavor name conventions
    input_names : list of str (optional). If provided, only merge the encoders/decoders for these input names.
    
    Returns:
    net, checkpoint: Krakencoder object and checkpoint dictionary for the merged model
    """
    net_and_checkpoint_dict_list=[]
    for ptfile in checkpoint_filename_list:
        net_i, checkpoint_i=Krakencoder.load_checkpoint(ptfile)
        net_and_checkpoint_dict_list.append({'net':net_i, 'checkpoint':checkpoint_i})
    
    net, checkpoint_info = merge_models(net_and_checkpoint_dict_list=net_and_checkpoint_dict_list, 
                                        canonicalize_input_names=canonicalize_input_names,
                                        input_names=input_names)
    
    checkpoint_info['merged_checkpointfile_list']=[os.path.split(ptfile)[-1] for ptfile in checkpoint_filename_list]
    
    return net, checkpoint_info

def print_merged_model(checkpoint_info):
    output_info_columns=[]
    for i, conn_name in enumerate(checkpoint_info['input_name_list']):
        output_info_columns.append(['%d)' % (i+1), 
                                    conn_name, 
                                    '(Sx%d)' % (checkpoint_info['orig_input_size_list'][i]), 
                                    ' from: '+checkpoint_info['merged_checkpointfile_list'][checkpoint_info['merged_source_net_idx'][i]]])
        
    _=format_columns(column_data=output_info_columns, column_format_list=['%s','%s','%s','%s'],
                     delimiter=" ",align="left", print_result=True, truncate_length=130, truncate_endlength=30)
    
    print("Total: %d inputs. %d paths" % (len(checkpoint_info['input_name_list']), len(checkpoint_info['trainpath_encoder_index_list'])))
    