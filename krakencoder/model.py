import torch
import torch.nn as nn

#####################################

class Krakencoder(nn.Module):
    """
    A linked autoencoder model with multiple input types, each with its own encoder and decoder
    """
    def __init__(self,inputsize_list,latentsize=128,hiddenlayers=[],skip_relu=False, dropout=0, 
                 leakyrelu_negative_slope=0, relu_tanh_alternate=False, latent_activation=None, latent_normalize=False, 
                 linear_include_bias=True, no_bias_for_outer_layers=False, dropout_schedule_list=None,
                 dropout_final_layer=None, dropout_final_layer_list=None,
                 intergroup=False, intergroup_layers=[], intergroup_inputgroup_list=[],intergroup_transform_index_dict={}, intergroup_relu=True, intergroup_dropout=None):
        """
        Initialize the model with the specified hyperparameters
        
        Parameters:
        inputsize_list: list of int, number of features for each input type (Nfeat)
        latentsize: int (default=128), number of features in the latent space (Nlatent)
        hiddenlayers: list of int (default=[]), number of features in each hidden layer. []=no hidden layers, just single Nfeat->Nlatent
        
        skip_relu: bool (default=False), if True, do not apply ReLU activation after each layer
        leakyrelu_negative_slope: float (default=0), negative slope for LeakyReLU activation (if 0, just use ReLU)
        relu_tanh_alternate: bool (default=False), if True, alternate between ReLU and Tanh activations (As in Sarwar 2021)
        latent_activation: str (default=None), activation function to apply to the latent representation (None, 'tanh', 'sigmoid')
        latent_normalize: bool (default=False), if True, normalize the latent representation (L2 norm==1)
        linear_include_bias: bool (default=True), if True, include bias in all linear layers
        no_bias_for_outer_layers: bool (default=False), if True, do not include bias in the first and last linear layers
        
        dropout: float (default=0), dropout rate for all layers
        dropout_schedule_list: list of float (default=None), list of dropout rates to interpolate through over epochs 
            (eg [0.5,0] starts at 0.5 and ends at 0), if None, just use 'dropout' parameter for all epochs
            NOTE: setting 'dropout' over epochs is actually done in train.py. We just provide this list here to store in model params/prettystring
        dropout_final_layer: float (default=None), separate dropout rate for the final DECODER layer only (eg: for the final reconstruction). None=use 'dropout'
        dropout_final_layer_list: list of float (default=None), separate dropout rate for the final DECODER layer for each input type
        
        intergroup: bool (default=False), if True, create and apply additional intergroup transformations between groups of input types (eg SC->FC, FC->SC)
        intergroup_layers: list of int (default=[]), list of hidden layer sizes in each intergroup transformation layer. []=just single linear Nlatent->Nlatent
        intergroup_inputgroup_list: list of str (default=[]), list of group names for each input type (eg 'SC','FC')
        intergroup_transform_index_dict: dict (default={}), dictionary of intergroup transformation names to index (eg 'SC->FC':0, 'FC->SC':1)
        intergroup_relu: bool (default=True), if True, apply ReLU activation after each intergroup transformation layer
        intergroup_dropout: float (default=None), separate dropout rate for all intergroup transformation layers. None=use 'dropout'
        """
        super(Krakencoder, self).__init__()
        
        self.encoder_list=nn.ModuleList()
        self.decoder_list=nn.ModuleList()
        self.inputsize_list=inputsize_list
        self.latentsize=latentsize
        self.hiddenlayers=hiddenlayers
        self.skip_relu=skip_relu
        self.dropout=dropout
        self.dropout_schedule_list=dropout_schedule_list
        self.relu_tanh_alternate=relu_tanh_alternate
        self.leakyrelu_negative_slope=leakyrelu_negative_slope
        self.has_activation=False
        if latent_activation == 'none':
            latent_activation=None
        self.latent_activation=latent_activation
        self.latent_normalize=latent_normalize
        self.linear_include_bias=linear_include_bias
        self.no_bias_for_outer_layers=no_bias_for_outer_layers
        
        if dropout_schedule_list is not None:
            dropout=dropout_schedule_list[0]
            self.dropout=dropout_schedule_list[0]
        
        self.dropout_final_layer=dropout_final_layer
        self.dropout_final_layer_list=dropout_final_layer_list
            
        ############### intergroup 
        self.intergroup=intergroup
        self.intergroup_layers=intergroup_layers
        self.inputgroup_list=intergroup_inputgroup_list
        self.intergroup_transform_list=None
        self.intergroup_transform_index_dict=intergroup_transform_index_dict #sc->fc=0, fc->sc=1, etc...
        self.intergroup_relu=intergroup_relu
        if intergroup_dropout is None:
            intergroup_dropout=dropout
        self.intergroup_dropout=intergroup_dropout
        #make groups lowercase for case-insensitive matching
        self.inputgroup_list=[x.lower() for x in self.inputgroup_list]
        self.intergroup_transform_index_dict={k.lower():v for k,v in self.intergroup_transform_index_dict.items()}
        
        if intergroup:
            if len(self.inputgroup_list) != len(inputsize_list):
                raise ValueError("inputgroup_list must be same length as inputsize_list")

            self.intergroup_transform_list=nn.ModuleList()
            intergroup_name_list=[]
            intergroup_count=0
            for i in self.inputgroup_list:
                ilist=[]
                for j in self.inputgroup_list:
                    ijstr='%s->%s' % (i,j)
                    ilist+=[ijstr]
                    if i == j:
                        continue
                    if ijstr in self.intergroup_transform_index_dict:
                        continue
                    self.intergroup_transform_index_dict[ijstr]=intergroup_count
                    intergroup_count+=1
                intergroup_name_list+=[ilist]
            intergroup_count=len(self.intergroup_transform_index_dict.keys())
            
            do_bias=linear_include_bias
            for idx in range(intergroup_count):
                layer_list=[]
                prevlayersize=latentsize
                for ilayer,layersize in enumerate(intergroup_layers+[latentsize]):
                    layer_list+=[nn.Linear(prevlayersize,layersize, bias=do_bias)]
                    prevlayersize=layersize
                    if intergroup_dropout > 0:
                        layer_list+=[nn.Dropout(intergroup_dropout)]
                    if intergroup_relu:
                        if leakyrelu_negative_slope > 0:
                            layer_list+=[nn.LeakyReLU(leakyrelu_negative_slope)]
                        else:
                            layer_list+=[nn.ReLU()]
                if latent_normalize:
                    layer_list+=[Normalize(p=2)]
                self.intergroup_transform_list.append(nn.Sequential(*layer_list))
            print("intergroup_transform_index_dict",self.intergroup_transform_index_dict)
        ############### end intergroup 
        
        for i_input,inputsize in enumerate(inputsize_list):
            enc_size_list=[inputsize]+hiddenlayers+[latentsize]
            dec_size_list=enc_size_list[::-1]
            
            enc_layer_list=[]
            for i in range(len(enc_size_list)-1):
                do_bias=linear_include_bias
                if i == 0 and no_bias_for_outer_layers:
                    do_bias=False
                enc_layer_list+=[nn.Linear(enc_size_list[i],enc_size_list[i+1], bias=do_bias)]
                if dropout > 0: # and i < len(enc_size_list)-2:
                    #no dropout after we reach latent
                    enc_layer_list+=[nn.Dropout(dropout)]
                #no relu after we reach latent
                if not skip_relu and i < len(enc_size_list)-2:
                    self.has_activation=True
                    if leakyrelu_negative_slope > 0:
                        relu=nn.LeakyReLU(leakyrelu_negative_slope)
                    else:
                        relu=nn.ReLU()
                    
                    if relu_tanh_alternate:
                        #Sarwar-style alternating LeakyReLU+tanh
                        if i % 2 == 0:
                            enc_layer_list+=[relu]
                        else:
                            enc_layer_list+=[nn.Tanh()]
                    else:
                        enc_layer_list+=[relu]
                if latent_activation == "tanh":
                    enc_layer_list+=[nn.Tanh()]
                elif latent_activation == "sigmoid":
                    enc_layer_list+=[nn.Sigmoid()]
                    
            if latent_normalize:
                enc_layer_list+=[Normalize(p=2)]

                #enc_layer_list+=[nn.BatchNorm1d(latentsize)]
                
            dec_layer_list=[]
            for i in range(len(dec_size_list)-1):
                do_bias=linear_include_bias
                if i == len(dec_size_list)-2 and no_bias_for_outer_layers:
                    do_bias=False
                dec_layer_list+=[nn.Linear(dec_size_list[i],dec_size_list[i+1],bias=do_bias)]
                dropout_thislayer=dropout
                if i == len(dec_size_list)-2 and dropout_final_layer is not None:
                    #print("inputsize:",inputsize,"idec:",i,"dropout_final_layer:",dropout_final_layer)
                    dropout_thislayer=dropout_final_layer
                    if self.dropout_final_layer_list is not None:
                        dropout_thislayer=self.dropout_final_layer_list[i_input]
                if dropout > 0: # and i < len(dec_size_list)-2:
                    #no dropout after we reach output
                    dec_layer_list+=[nn.Dropout(dropout_thislayer)]
                #no relu after we reach output
                if not skip_relu and i < len(dec_size_list)-2:
                    self.has_activation=True
                    if leakyrelu_negative_slope > 0:
                        relu=nn.LeakyReLU(leakyrelu_negative_slope)
                    else:
                        relu=nn.ReLU()
                    if relu_tanh_alternate:
                        #Sarwar-style alternating LeakyReLU+tanh
                        if i % 2 == 0:
                            dec_layer_list+=[relu]
                        else:
                            dec_layer_list+=[nn.Tanh()]
                    else:
                        dec_layer_list+=[relu]
            
            self.encoder_list.append(nn.Sequential(*enc_layer_list))
            self.decoder_list.append(nn.Sequential(*dec_layer_list))
            
    def forward(self, x, encoder_index, decoder_index, transcoder_index_list=None):
        """
        Forward pass through the model
        If only encoder_index provided (decoder_index=-1), return latent result
        If both encoder and decoder (or just decoder) are provided, return latent, output
        
        Parameters:
        x: torch.Tensor, input data (Nsubj x Nfeat)
        encoder_index: int, index of the encoder to use. if -1, x is latent so just decode it
        decoder_index: int, index of the decoder to use. if -1, return latent only
        transcoder_index_list: list of int (optional), indices of transcoder layers to use. 
            If provided, it will encode->decode->encode->decode... for each index in the list
        """
        
        if encoder_index<0:
            x_enc = x
        else:
            x_enc = self.encoder_list[encoder_index](x)

        if decoder_index<0:
            return x_enc
        if transcoder_index_list is not None:
            for i in transcoder_index_list:
                x_dec=self.decoder_list[i](x_enc)
                x_enc=self.encoder_list[i](x_dec)
        
        ############### intergroup 
        if self.intergroup_transform_list is not None and encoder_index >= 0 and decoder_index >= 0:
            ijstr='%s->%s' % (self.inputgroup_list[encoder_index],self.inputgroup_list[decoder_index])
            if ijstr in self.intergroup_transform_index_dict:
                ijidx=self.intergroup_transform_index_dict[ijstr]
                #intercode then decode
                #don't save the intercode as x_env because we need that untouched for latentsim
                x_dec=self.decoder_list[decoder_index](self.intergroup_transform_list[ijidx](x_enc))
            else:
                x_dec = self.decoder_list[decoder_index](x_enc)
        else:
            x_dec = self.decoder_list[decoder_index](x_enc)
        ############### end intergroup 
                
        #x_dec = self.decoder_list[decoder_index](x_enc)
        return x_enc, x_dec
    
    def intergroup_transform_latent(self, x_enc, encoder_index, decoder_index, intergroup_transform_name=None):
        """
        Apply intergroup transformation to a latent vector.
         - Use intergroup_transform_name (eg 'SC->FC') to look up which transformation to apply
         - Cases where we just return x_enc:
            - if intergroup=False or intergroup_transform_list is empty
            - if encoder==decoder and (and no intergroup_transform_name provided)
        - If encoder!=decoder and intergroup_transform_name is None:
            - create intergroup_transform_name from encoder,decoder group names, and lookup the transformation
        
        Parameters:
        x_enc: torch.Tensor, latent vector (Nsubj x Nlatent)
        encoder_index: int, index of the encoder to use (to look up group name A for 'A->B' intergroup_transform_name)
        decoder_index: int, index of the decoder to use (to look up group name B for 'A->B' intergroup_transform_name)
        intergroup_transform_name: str (optional), name of the intergroup transformation to apply (eg 'SC->FC')
        
        Returns:
        x_enc: torch.Tensor, latent vector after intergroup transformation
        """
        if self.intergroup_transform_list is None:
            return x_enc
        
        if intergroup_transform_name is None and (encoder_index == decoder_index):
            return x_enc
        
        if intergroup_transform_name is None and encoder_index >=0 and decoder_index >= 0:
            intergroup_transform_name='%s->%s' % (self.inputgroup_list[encoder_index],self.inputgroup_list[decoder_index])
        
        if intergroup_transform_name in self.intergroup_transform_index_dict:
            ijidx=self.intergroup_transform_index_dict[intergroup_transform_name]
            x_enc=self.intergroup_transform_list[ijidx](x_enc)
        
        return x_enc
        
    def freeze_model(self, do_freeze=True):
        """Freeze or unfreeze all weights in this model"""
        for param in self.parameters():
            param.requires_grad = not do_freeze
        
    def set_dropout(self, new_dropout, new_intergroup_dropout=None, new_final_layer_dropout=None, new_final_layer_dropout_list=None):
        """
        Set dropout rate for all layers in this model. Useful for training schedules where dropout rate changes over epochs.
        Optional overrides for intergroup layers, final layer
        
        Parameters:
        new_dropout: float, new dropout rate for all layers.
        new_intergroup_dropout: float (optional), new dropout rate for intergroup layers only (if model created with intergroup=True)
        new_final_layer_dropout: float (optional), new dropout rate for final DECODER layer only (eg: for the final reconstruction)
        new_final_layer_dropout_list: list of floats (optional), a separate final DECODER dropout for each input type
        """
        if new_intergroup_dropout is None:
            new_intergroup_dropout=new_dropout
        self.dropout=new_dropout
        self.intergroup_dropout=new_intergroup_dropout
        
        if new_final_layer_dropout is not None:
            self.dropout_final_layer=new_final_layer_dropout
        
        if new_final_layer_dropout_list is not None:
            self.dropout_final_layer_list=new_final_layer_dropout_list
        
        for enc in self.encoder_list:
            for layer in enc:
                if isinstance(layer,nn.Dropout):
                    layer.p=new_dropout
                    
        for idec,dec in enumerate(self.decoder_list):
            for i,layer in enumerate(dec):
                dropout_thislayer=new_dropout
                if i==len(dec)-1 and self.dropout_final_layer is not None:
                    if self.dropout_final_layer is not None:
                        dropout_thislayer=self.dropout_final_layer
                    if self.dropout_final_layer_list is not None:
                        dropout_thislayer=self.dropout_final_layer_list[idec]
                if isinstance(layer,nn.Dropout):
                    layer.p=dropout_thislayer
                    
        ############### intergroup 
        if self.intergroup_transform_list is not None:
            for enc in self.intergroup_transform_list:
                for layer in enc:
                    if isinstance(layer,nn.Dropout):
                        layer.p=new_intergroup_dropout
        ############### end intergroup 
                    
    def prettystring(self):
        """
        Generate a filename-compatible string to describe the model architecture and hyperparameters
        """
        outstring="latent%d" % (self.latentsize)
        layerstring=""
        
        #group sequential same-size layers to avoid longer filenames
        curlayerstart=0
        if len(self.hiddenlayers) > 0:
            curlayersize=self.hiddenlayers[0]
            for i,l in enumerate(self.hiddenlayers+[-1]):
                if l != curlayersize:
                    if i==curlayerstart+1:
                        layerstring+="+layer%d.%d" % (curlayerstart+1,curlayersize)
                    else:
                        layerstring+="+layer%d-%d.%d" % (curlayerstart+1,i,curlayersize)
                    curlayerstart=i
                    curlayersize=l
            if layerstring.startswith("+"):
                layerstring="_"+layerstring[1:]
        
        if not layerstring:
            layerstring="_0layer"
        outstring+=layerstring
        
        actstring=""
        if self.has_activation:
            if self.leakyrelu_negative_slope>0:
                relustring="_leakyrelu%g" % (self.leakyrelu_negative_slope)
            else:
                relustring="_relu"

            if self.relu_tanh_alternate:
                actstring=relustring+"+tanh"
            elif self.leakyrelu_negative_slope>0:
                actstring=relustring
        
        if self.skip_relu:
            outstring+="_norelu"
        else:
            outstring+=actstring
        
        if not self.linear_include_bias:
            outstring+="_nobias"
        
        if self.linear_include_bias and self.no_bias_for_outer_layers:
            outstring+="_nobiasio"
        
        if self.latent_normalize:
            outstring+="_latentunit"
        
        if self.latent_activation is not None and self.latent_activation != 'none':
            outstring+="_latent."+self.latent_activation
        
        if (self.dropout_schedule_list is not None 
            and len(self.dropout_schedule_list)>1):
            if all([x==self.dropout_schedule_list[0] for x in self.dropout_schedule_list]):
                if self.dropout_schedule_list[0]>0:
                    outstring+="_drop%g" % (self.dropout_schedule_list[0])
            else:
                #outstring+="_drop%g-%g" % (self.dropout_init,self.dropout_final)
                outstring+="_drop"+"-".join(["%g" % (x) for x in self.dropout_schedule_list])
        elif self.dropout_schedule_list is not None and len(self.dropout_schedule_list)>0:
            if self.dropout_schedule_list[0]>0:
                outstring+="_drop%g" % (self.dropout_schedule_list[0])
        elif self.dropout>0:
            outstring+="_drop%g" % (self.dropout)
        if self.dropout_final_layer is not None:
            outstring+="x%g" % (self.dropout_final_layer)
            if self.dropout_final_layer_list is not None and not all([x==self.dropout_final_layer for x in self.dropout_final_layer_list]):
                outstring+="s"
        
        ############### intergroup 
        if self.intergroup:
            outstring+="_ig%d" % (len(self.intergroup_layers))
            if not self.intergroup_relu:
                outstring+=".norelu"
            if self.intergroup_dropout != self.dropout:
                outstring+=".d%g" % (self.intergroup_dropout)
        ############### end intergroup 
            
        return outstring
    
    def save_checkpoint(self, filename, extra_dict=None):
        """
        Save the model to a file, including all hyperparameters and model weights, as well as any additional information in extra_dict
        """
        checkpoint={"state_dict": self.state_dict()}
        
        checkpoint['input_size_list']=self.inputsize_list
        checkpoint['latentsize']=self.latentsize
        checkpoint['hiddenlayers']=self.hiddenlayers
        checkpoint['skip_relu']=self.skip_relu
        checkpoint['dropout']=self.dropout
        checkpoint['leakyrelu_negative_slope']=self.leakyrelu_negative_slope
        checkpoint['relu_tanh_alternate']=self.relu_tanh_alternate
        checkpoint['latent_activation']=self.latent_activation
        checkpoint['latent_normalize']=self.latent_normalize
        checkpoint['linear_include_bias']=self.linear_include_bias
        checkpoint['no_bias_for_outer_layers']=self.no_bias_for_outer_layers
        
        checkpoint['dropout_final_layer']=self.dropout_final_layer
        checkpoint['dropout_final_layer_list']=self.dropout_final_layer_list
        
        ############### intergroup 
        checkpoint['intergroup']=self.intergroup
        checkpoint['intergroup_layers']=self.intergroup_layers
        checkpoint['intergroup_inputgroup_list']=self.inputgroup_list
        checkpoint['intergroup_transform_index_dict']=self.intergroup_transform_index_dict
        checkpoint['intergroup_relu']=self.intergroup_relu
        checkpoint['intergroup_dropout']=self.intergroup_dropout
        ############### end intergroup
        
        if extra_dict is not None:
            for k in extra_dict:
                checkpoint[k]=extra_dict[k]
        
        torch.save(checkpoint,filename)
    
    @staticmethod
    def load_checkpoint(filename, checkpoint_override=None):
        """
        Static function to load a model from a file, including all hyperparameters and model weights.
        
        Parameters:
        filename: str, path to the checkpoint file (.pt)
        checkpoint_override: dict (optional), dictionary of hyperparameters to override in the checkpoint file
        
        Returns:
        net: Krakencoder model
        extra_dict: dictionary of additional information saved in the checkpoint file
        
        Example: 
        net, extra_dict = Krakencoder.load_checkpoint("model.pt")
        """
        #warnings.filterwarnings("ignore", category=UserWarning, message="CUDA initialization") #optional
        if torch.cuda.is_available():
            checkpoint=torch.load(filename)
        else:
            checkpoint=torch.load(filename,map_location=torch.device('cpu'))
        
        #put these defaults in since we added them after some models were saved
        if not 'linear_include_bias' in checkpoint:
            checkpoint['linear_include_bias']=True
        if not 'no_bias_for_outer_layers' in checkpoint:
            checkpoint['no_bias_for_outer_layers']=False
        
        if not 'dropout_final_layer' in checkpoint:
            checkpoint['dropout_final_layer']=None
        if not 'dropout_final_layer_list' in checkpoint:
            checkpoint['dropout_final_layer_list']=None
            
        if checkpoint_override is not None:
            for k in checkpoint_override:
                checkpoint[k]=checkpoint_override[k]
        
        """
        net=Krakencoder(checkpoint['input_size_list'],latentsize=checkpoint['latentsize'],hiddenlayers=checkpoint['hiddenlayers'],
            skip_relu=checkpoint['skip_relu'], dropout=checkpoint['dropout'], leakyrelu_negative_slope=checkpoint['leakyrelu_negative_slope'],
            relu_tanh_alternate=checkpoint['relu_tanh_alternate'],latent_activation=checkpoint['latent_activation'],latent_normalize=checkpoint['latent_normalize'],
            linear_include_bias=checkpoint['linear_include_bias'],no_bias_for_outer_layers=checkpoint['no_bias_for_outer_layers'])
        """
        
        ############### intergroup 
        if not 'intergroup' in checkpoint:
            checkpoint['intergroup']=False
        if not 'intergroup_layers' in checkpoint:
            checkpoint['intergroup_layers']=[]
        if not 'intergroup_inputgroup_list' in checkpoint:
            checkpoint['intergroup_inputgroup_list']=[]
        if not 'intergroup_transform_index_dict' in checkpoint:
            checkpoint['intergroup_transform_index_dict']={}
        if not 'intergroup_relu' in checkpoint:
            checkpoint['intergroup_relu']=True
        if not 'intergroup_dropout' in checkpoint:
            checkpoint['intergroup_dropout']=None
            
        if checkpoint_override is not None:
            for k in checkpoint_override:
                checkpoint[k]=checkpoint_override[k]
        
        net=Krakencoder(checkpoint['input_size_list'],latentsize=checkpoint['latentsize'],hiddenlayers=checkpoint['hiddenlayers'],
            skip_relu=checkpoint['skip_relu'], dropout=checkpoint['dropout'], leakyrelu_negative_slope=checkpoint['leakyrelu_negative_slope'],
            relu_tanh_alternate=checkpoint['relu_tanh_alternate'],latent_activation=checkpoint['latent_activation'],latent_normalize=checkpoint['latent_normalize'],
            linear_include_bias=checkpoint['linear_include_bias'],no_bias_for_outer_layers=checkpoint['no_bias_for_outer_layers'],
            dropout_final_layer=checkpoint['dropout_final_layer'],dropout_final_layer_list=checkpoint['dropout_final_layer_list'],
            intergroup=checkpoint['intergroup'], intergroup_layers=checkpoint['intergroup_layers'], 
            intergroup_inputgroup_list=checkpoint['intergroup_inputgroup_list'], 
            intergroup_transform_index_dict=checkpoint['intergroup_transform_index_dict'],intergroup_relu=checkpoint['intergroup_relu'],
            intergroup_dropout=checkpoint['intergroup_dropout'])
        ############### end intergroup 
        
        net.load_state_dict(checkpoint['state_dict'])
        
        if torch.cuda.is_available():
            net.to(torch.device("cuda"))
        
        extra_dict=checkpoint.copy()
        del extra_dict['state_dict']
        
        return net, extra_dict
    
class Normalize(nn.Module):
    """
    Normalize a tensor across a specified dimension
    """
    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return nn.functional.normalize(x,p=self.p,dim=1)
