import torch
import torch.nn as nn
import numpy as np

#####################################

class Krakencoder(nn.Module):
    def __init__(self,inputsize_list,latentsize=128,hiddenlayers=[],skip_relu=False, dropout=0, 
                 leakyrelu_negative_slope=0, relu_tanh_alternate=False, latent_activation=None, latent_normalize=False, 
                 linear_include_bias=True, no_bias_for_outer_layers=False):
        super(Krakencoder, self).__init__()
        
        self.encoder_list=nn.ModuleList()
        self.decoder_list=nn.ModuleList()
        self.inputsize_list=inputsize_list
        self.latentsize=latentsize
        self.hiddenlayers=hiddenlayers
        self.skip_relu=skip_relu
        self.dropout=dropout
        self.relu_tanh_alternate=relu_tanh_alternate
        self.leakyrelu_negative_slope=leakyrelu_negative_slope
        self.has_activation=False
        if latent_activation == 'none':
            latent_activation=None
        self.latent_activation=latent_activation
        self.latent_normalize=latent_normalize
        self.linear_include_bias=linear_include_bias
        self.no_bias_for_outer_layers=no_bias_for_outer_layers
        
        for inputsize in inputsize_list:
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
                if dropout > 0: # and i < len(dec_size_list)-2:
                    #no dropout after we reach output
                    dec_layer_list+=[nn.Dropout(dropout)]
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
            #enc_layer_list=[nn.Linear(enc_size_list[i],enc_size_list[i+1]) for i in range(len(enc_size_list)-1)]
            #dec_layer_list=[nn.Linear(dec_size_list[i],dec_size_list[i+1]) for i in range(len(dec_size_list)-1)]
            
            #[enc_layer_list.insert(i*2+1,nn.ReLU()) for i in range(len(enc_layer_list))]
            #[dec_layer_list.insert(i*2+1,nn.ReLU()) for i in range(len(dec_layer_list))]
            #no relu on the last encoding layer
            
            self.encoder_list.append(nn.Sequential(*enc_layer_list))
            self.decoder_list.append(nn.Sequential(*dec_layer_list))
            
    def forward(self, x, encoder_index, decoder_index, transcoder_index_list=None):
        #if only encoder_index provided, return latent result
        #if both encoder and decoder (or just decoder) are provided, return latent, output
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
    
        x_dec = self.decoder_list[decoder_index](x_enc)
        return x_enc, x_dec
    
    def prettystring(self):
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
        
        if self.dropout>0:
            outstring+="_drop%g" % (self.dropout)
        
        return outstring
    
    def save_checkpoint(self, filename, extra_dict=None):
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
        
        if extra_dict is not None:
            for k in extra_dict:
                checkpoint[k]=extra_dict[k]
        
        torch.save(checkpoint,filename)
    
    @staticmethod
    def load_checkpoint(filename, checkpoint_override=None):
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
        
        if checkpoint_override is not None:
            for k in checkpoint_override:
                checkpoint[k]=checkpoint_override[k]
        
        net=Krakencoder(checkpoint['input_size_list'],latentsize=checkpoint['latentsize'],hiddenlayers=checkpoint['hiddenlayers'],
            skip_relu=checkpoint['skip_relu'], dropout=checkpoint['dropout'], leakyrelu_negative_slope=checkpoint['leakyrelu_negative_slope'],
            relu_tanh_alternate=checkpoint['relu_tanh_alternate'],latent_activation=checkpoint['latent_activation'],latent_normalize=checkpoint['latent_normalize'],
            linear_include_bias=checkpoint['linear_include_bias'],no_bias_for_outer_layers=checkpoint['no_bias_for_outer_layers'])
        
        net.load_state_dict(checkpoint['state_dict'])
        
        extra_dict=checkpoint.copy()
        del extra_dict['state_dict']
        
        return net, extra_dict

class Normalize(nn.Module):

    def __init__(self, p=2):
        super(Normalize, self).__init__()
        self.p = p

    def forward(self, x):
        return nn.functional.normalize(x,p=self.p,dim=1)
