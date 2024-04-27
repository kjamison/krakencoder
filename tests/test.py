"""
Integration test to make sure model produces expected values in the installed environment
"""

import unittest

import torch
import numpy as np
from krakencoder.model import Krakencoder
from krakencoder.adaptermodel import KrakenAdapter
from krakencoder.utils import square2tri, numpyvar, torchfloat
from krakencoder.data import load_transformers_from_file

from scipy.io import loadmat
import os

class TestDummyEvaluationOutput(unittest.TestCase):
    def test_dummy_eval(self):
        """
        Test that the model produces the expected output for a dummy input on a very small test checkpoint containing two flavors of SC.
        In some cases, we've seen package version issues that cause models to produce different numerical outputs.
        
        In this test, we:
        1. Read in the SCsdstream and SCifod2act for 10 subjects
        2. Run that data through the pretrained model ENCODERS to generate a latent vector for each subject and flavor
        3. Average latent vectors across flavors to produce a "fusion" vector for each subject
        4. Compute average RMS between computed fusion vector and the data saved in the test output file
        5. Feed those fusion vectors through the model DECODERS to generate predicted connectomes
        6. Compute average RMS between predicted connectomes and the data saved in test output file
        7. All RMS values should be 0 or ~1e-8, depending on architecture. If they are < 1e-6, we pass the test
        """
        
        #test inputs and expected outputs are stored in the same directory as this script
        testdir=os.path.dirname(__file__)
        
        #checkpoint and transformation (just mean for cfeat)
        checkpoint_file=os.path.join(testdir,'krakenTEST_chkpt_SC_fs86_cfeat_4paths_latent128_20240425_ep002000.pt')
        ioxfm_file_list=[os.path.join(testdir,'krakenTEST_ioxfm_SC_fs86_cfeat_710train.npy')]
        
        #dummy input data and the expected outputs from a trusted run
        testfile_input=os.path.join(testdir,'dummydataSCTEST_inputs.mat')
        testfile_output=os.path.join(testdir,'dummydataSCTEST_20240425_ep002000_out.encoded.mat')
        testfile_output_predconn=os.path.join(testdir,'dummydataSCTEST_20240425_ep002000_in.encoded_out.all.mat')
        
        #load the checkpoint and the input transformers
        inner_net, checkpoint_info = Krakencoder.load_checkpoint(checkpoint_file, eval_mode=True)
        transformer_list, transformer_info_list = load_transformers_from_file(ioxfm_file_list, quiet=True)

        #create new model that wraps the inner kraken model and includes PCA transforms from raw data
        net=KrakenAdapter(inner_model=inner_net,
                          data_transformer_list=[transformer_list[conntype] if conntype in transformer_list else None for conntype in checkpoint_info['input_name_list']],
                          linear_polynomial_order=0,
                          eval_mode=True)
        
        #load the expected fusion outputs
        encoded_expectation=loadmat(testfile_output, simplify_cells=True)
        encoded_expectation=encoded_expectation['predicted_alltypes']['fusion']['encoded']

        #load the expected predicted connectomes
        predconn_expectation=loadmat(testfile_output_predconn,simplify_cells=True)
        predconn_expectation=predconn_expectation['predicted_alltypes']['encoded']

        #load dummy data and convert square matrices to stacked upper triangular
        conndata_squaremats=loadmat(testfile_input, simplify_cells=True)
        conndata={}
        for c in checkpoint_info['input_name_list']:
            conndata[c] = {'data': np.stack([square2tri(x) for x in conndata_squaremats[c]['data']])}
    
        #run dummy data through the ENCODERS
        encoded_data={}
        for encidx, c in enumerate(checkpoint_info['input_name_list']):
            encoded_data[c]=net(conndata[c]['data'],encoder_index=encidx, decoder_index=-1)

        #average the latent vectors across flavors to get "fusion" vectors
        encoded_fusion=torch.mean(torch.stack([encoded_data[c] for c in encoded_data]),axis=0)
        encoded_fusion=numpyvar(encoded_fusion)

        #compute the RMS between computed and expected outputs for each subject, then average across subjects
        #this should be ~1e-8
        rms_mean_fusion=np.mean(np.sqrt(np.mean((encoded_fusion-encoded_expectation)**2,axis=1)))

        print("rms_mean(fusion vs. expected):",rms_mean_fusion, "(should be <1e-6)")
        self.assertLess(rms_mean_fusion, 1e-6)
 
        #run the fusion vectors through the DECODERS
        predconn={}
        for decidx, c in enumerate(checkpoint_info['input_name_list']):
            _,predconn[c]=net(torchfloat(encoded_fusion), encoder_index=-1, decoder_index=decidx)
            predconn[c]=numpyvar(predconn[c])

        #compute the RMS for predicted connectomes and expected outputs
        rms_mean_predconn={}
        for c in predconn:
            #these should each be ~1e8
            rms_mean_predconn[c]=np.mean(np.sqrt(np.mean((predconn_expectation[c]-predconn[c])**2,axis=1)))
            print("rms_mean(%s predicted vs expected):" % (c),rms_mean_predconn[c], "(should be <1e-6)")
            self.assertLess(rms_mean_predconn[c], 1e-6)

if __name__ == '__main__':
    unittest.main()
