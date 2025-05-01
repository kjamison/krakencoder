import torch
import torch.nn as nn
import numpy as np

from .utils import get_version

class KrakenAdapter(nn.Module):
    """
    This class takes a pre-trained Krakencoder model and a data transformer
    and adds an additional trainable domain adaptation layer.
    """
    def __init__(self, inner_model, inputsize_list=None, data_transformer_list=None, 
                 linear_polynomial_order=1,
                 leaky_relu=False,
                 dropout=0.0,
                 inner_model_extra_dict={},
                 eval_mode=False):
        super(KrakenAdapter, self).__init__()
        if inputsize_list is None:
            inputsize_list=inner_model.inputsize_list
            
        self.inner_model = inner_model
        self.data_transformer_list=data_transformer_list
        self.dropout=dropout
        self.inputsize_list=inputsize_list
        self.leaky_relu=leaky_relu
        self.linear_polynomial_order=linear_polynomial_order
        self.inner_model_extra_dict=inner_model_extra_dict
        
        self.encoder_outer_layer_list=nn.ModuleList()
        self.decoder_outer_layer_list=nn.ModuleList()
        
        for i,isize in enumerate(self.inputsize_list):
            nnseq=[SimpleLinearTransform(isize, polynomial_order=self.linear_polynomial_order)]
            if leaky_relu:
                nnseq+=[nn.LeakyReLU()]
            if dropout > 0:
                nnseq+=[nn.Dropout(dropout)]
            self.encoder_outer_layer_list.append(nn.Sequential(*nnseq))
            
        for i,isize in enumerate(self.inputsize_list):
            nnseq=[SimpleLinearTransform(isize, polynomial_order=self.linear_polynomial_order)]
            if leaky_relu:
                nnseq+=[nn.LeakyReLU()]
            if dropout > 0:
                nnseq+=[nn.Dropout(dropout)]
            self.decoder_outer_layer_list.append(nn.Sequential(*nnseq))
            
        if eval_mode:
            self.eval()
        
            
    def forward(self, x, encoder_index, decoder_index, **kwargs):
        #if we gave it both encoder and decoder indices:
        #transform x, encode, decode, then inverse transform
        #and return x_enc, x_dec
        if encoder_index >= 0 and decoder_index >= 0:
            x = self.encoder_outer_layer_list[encoder_index](x)
            if not self.data_transformer_list is None:
                x = self.data_transformer_list[encoder_index].transform(x)
            x_enc, x_dec = self.inner_model(x, encoder_index=encoder_index, decoder_index=decoder_index, **kwargs)
            if not self.data_transformer_list is None:
                x_dec = self.data_transformer_list[decoder_index].inverse_transform(x_dec)
            x_dec = self.decoder_outer_layer_list[decoder_index](x_dec)
            return x_enc, x_dec
        
        #if we only give it decoder_index, input must be already encoded, so just decode
        #and inverse transform
        #and return x_enc, x_dec
        elif encoder_index < 0:
            x_enc = x
            _, x_dec = self.inner_model(x_enc, encoder_index=encoder_index, decoder_index=decoder_index, **kwargs)
            if not self.data_transformer_list is None:
                x_dec = self.data_transformer_list[decoder_index].inverse_transform(x_dec)
            x_dec = self.decoder_outer_layer_list[decoder_index](x_dec)
            return x_enc, x_dec
        
        #if we only give it encoder_index, must want encoded for output
        #so encode and return x_enc
        elif decoder_index < 0:
            x = self.encoder_outer_layer_list[encoder_index](x)
            if not self.data_transformer_list is None:
                x = self.data_transformer_list[encoder_index].transform(x)
            x_enc = self.inner_model(x, encoder_index=encoder_index, decoder_index=decoder_index, **kwargs)
            return x_enc
    
    def freeze_inner_model(self, do_freeze=True):
        for param in self.inner_model.parameters():
            param.requires_grad = not do_freeze
    
    
    def set_dropout(self, new_dropout):
        self.dropout=new_dropout
        
    def prettystring(self):
        outstring=self.inner_model.prettystring()
        outstring+="_adapt"
        if self.linear_polynomial_order != 1:
            outstring+=".p%d"%(self.linear_polynomial_order)
        return outstring


    def save_checkpoint(self, filename, extra_dict=None):
        checkpoint={"state_dict": self.state_dict()}
        
        checkpoint['krakencoder_version']=get_version(include_date=True)
        
        checkpoint['input_size_list']=self.inputsize_list
        checkpoint['dropout']=self.dropout
        checkpoint['linear_polynomial_order']=self.linear_polynomial_order
        checkpoint['leaky_relu']=self.leaky_relu
        checkpoint['inner_model_extra_dict']=self.inner_model_extra_dict
        
        if extra_dict is not None:
            for k in extra_dict:
                checkpoint[k]=extra_dict[k]
        
        torch.save(checkpoint,filename)
        

    @staticmethod
    def load_checkpoint(filename, inner_model, data_transformer_list=None, inner_model_extra_dict={}, checkpoint_override=None, eval_mode=False):
        #warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization") #optional
        if torch.cuda.is_available():
            checkpoint=torch.load(filename, weights_only=False)
        else:
            checkpoint=torch.load(filename,map_location=torch.device('cpu'), weights_only=False)
        
        if checkpoint_override is not None:
            for k in checkpoint_override:
                checkpoint[k]=checkpoint_override[k]
        
        
        outer_weight_names=[k for k in checkpoint['state_dict'] if k.startswith('encoder_outer_layer_list') and k.endswith("weights")]
        default_size_list=[]
        for k in range(len(outer_weight_names)):
            poly_order=checkpoint['state_dict']['encoder_outer_layer_list.%d.0.weights'%(k)].shape[0]
            default_size_list+=[checkpoint['state_dict']['encoder_outer_layer_list.%d.0.weights'%(k)].shape[1]]
        
        
        checkpoint_default={'linear_polynomial_order':poly_order, 'leaky_relu':False, 'dropout':0, 'input_size_list':default_size_list}
        for k in checkpoint_default:
            if not k in checkpoint:
                checkpoint[k]=checkpoint_default[k]
        
        checkpoint['input_size_list']=default_size_list
        
        net=KrakenAdapter(inner_model, data_transformer_list=data_transformer_list, 
                          inputsize_list=checkpoint['input_size_list'], dropout=checkpoint['dropout'],
                          linear_polynomial_order=checkpoint['linear_polynomial_order'], leaky_relu=checkpoint['leaky_relu'],
                          inner_model_extra_dict=inner_model_extra_dict)
                        
        net.load_state_dict(checkpoint['state_dict'])
        
        if torch.cuda.is_available():
            net.to(torch.device("cuda"))
        
        extra_dict=checkpoint.copy()
        del extra_dict['state_dict']
        
        if eval_mode:
            net.eval()
        
        return net, extra_dict
    
        
class SimpleLinearTransform(nn.Module):
    def __init__(self, input_size, polynomial_order=1):
        super(SimpleLinearTransform, self).__init__()
        self.polynomial_order=polynomial_order
        self.weights = nn.Parameter(torch.Tensor(polynomial_order,input_size))
        self.biases = nn.Parameter(torch.Tensor(1,input_size))
        
        #initialize self.weights and self.biases
        #nn.init.xavier_uniform_(self.weights)
        nn.init.ones_(self.weights)
        nn.init.zeros_(self.biases)
    
    def forward(self, x):
        # Applying different linear transformations to each feature
        #output = x * self.weights + self.biases
        if self.polynomial_order == 0:
            return x
        output = self.biases
        for p in range(self.polynomial_order):
            output = output + (x**(p+1))*self.weights[p]
        return output
    
    def __repr__(self):
        return "SimpleLinearTransform(weights=%dx%d, polynomial_order=%d)"%(self.weights.shape[0],self.weights.shape[1],self.polynomial_order)
    