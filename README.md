# krakencoder
<img align="right" src="images/krakencoder_squid.png" alt="krakencoder cartoon" width=20%>

The Krakencoder is a joint connectome mapping tool that simultaneously, bidirectionally translates between structural and functional connectivity, and across different atlases and processing choices via a common latent representation.

<img src="images/krakencoder_overview.png" alt="krakencoder overview" width=60%>

### Citation
Keith W. Jamison, Zijin Gu, Qinxin Wang, Mert R. Sabuncu, Amy Kuceyeski, "Krakencoder: A unified brain connectome translation and fusion tool". bioRxiv [doi: XXXXXX](https://www.biorxiv.org/content/XXXXXXXX)

# Contents
1. [Code organization](#Code-organization)
2. [Examples](#examples)
3. [Pretrained connectivity types](#pretrained-connectivity-types)
4. [Requirements](#requirements)
5. [Downloads](#downloads)

# Code organization
### CLI-facing scripts:
* [`run_training.py`](run_training.py): Train a new model
* [`run_model.py`](run_model.py): Run a saved checkpoint on new data
* [`describe_checkpoint.py`](describe_checkpoint.py): Print information about a saved checkpoint

### Internal scripts:
* [`model.py`](krakencoder/model.py): Krakencoder model class definition
* [`train.py`](krakencoder/train.py): Training-related functions
* [`loss.py`](krakencoder/loss.py): Specifies different loss functions to be used during training and evaluation
* [`data.py`](krakencoder/data.py): Functions for loading and transforming input data
* [`plotfigures.py`](krakencoder/plotfigures.py): Functions for plotting loss curves and performance heatmaps
* [`utils.py`](krakencoder/utils.py): Miscellaneous utility functions

# Examples

## Generating predicted connectomes from new SC data, using pre-trained model:
```bash
python run_model.py --inputdata '[fs86_sdstream_volnorm]=mydata_fs86_sdstream_volnorm.mat' \
        '[fs86_ifod2act_volnorm]=mydata_fs86_ifod2act_volnorm.mat' \
        '[shen268_sdstream_volnorm]=mydata_shen268_sdstream_volnorm.mat' \
        '[shen268_ifod2act_volnorm]=mydata_shen268_ifod2act_volnorm.mat' \
        '[coco439_sdstream_volnorm]=mydata_coco439_sdstream_volnorm.mat' \
        '[coco439_ifod2act_volnorm]=mydata_coco439_ifod2act_volnorm.mat' \
    --adaptmode meanfit+meanshift \
    --checkpoint kraken_chkpt_SCFC_20240406_022034_ep002000.pt \
    --inputxform kraken_ioxfm_SCFC_coco439_993subj_pc256_25paths_710train_20220527.npy \
        kraken_ioxfm_SCFC_fs86_993subj_pc256_25paths_710train_20220527.npy \
        kraken_ioxfm_SCFC_shen268_993subj_pc256_25paths_710train_20220527.npy \
    --outputname all --output 'mydata_20240406_022034_ep002000_in.{input}.mat' \
    --fusion --fusioninclude fusion=all fusionSC=SC fusionFC=FC --onlyfusioninputs
```
* Each input file should have a 'data' field containing the [subjects x region x region] connectivity data for that input flavor.
* This will predict all 15 connectome flavors as outputs, based on whatever inputs are provided. The more input flavors you can provide, the better the predictions are.
* `--adaptmode meanfit+meanshift` uses a minimal approach for domain shift by linearly mapping the population mean of your input data to the population mean of the training data
* `--fusion` includes "fusion" predictions incorporating all inputs (or subsets, as below) into each predicted output.
* `--onlyfusioninputs` means the script will NOT output predictions for each individual input type, but only for fusion types
* `--fusioninclude fusion=all fusionSC=SC fusionFC=FC` produces "fusion", based on all inputs, and "fusion(SC|FC)" based on only SC or FC inputs
* Predicted outputs will be one file per input type flavor, for instance: `mydata_20240406_022034_ep002000_in.fusionSC.mat`

## Generating latent space representations from new data, using pre-trained model:
```bash
python run_model.py --inputdata '[fs86_sdstream_volnorm]=mydata_fs86_sdstream_volnorm.mat' \
        '[fs86_ifod2act_volnorm]=mydata_fs86_ifod2act_volnorm.mat' \
        '[shen268_sdstream_volnorm]=mydata_shen268_sdstream_volnorm.mat' \
        '[shen268_ifod2act_volnorm]=mydata_shen268_ifod2act_volnorm.mat' \
        '[coco439_sdstream_volnorm]=mydata_coco439_sdstream_volnorm.mat' \
        '[coco439_ifod2act_volnorm]=mydata_coco439_ifod2act_volnorm.mat' \
    --adaptmode meanfit+meanshift \
    --checkpoint kraken_chkpt_SCFC_20240406_022034_ep002000.pt \
    --inputxform kraken_ioxfm_SCFC_coco439_993subj_pc256_25paths_710train_20220527.npy \
        kraken_ioxfm_SCFC_fs86_993subj_pc256_25paths_710train_20220527.npy \
        kraken_ioxfm_SCFC_shen268_993subj_pc256_25paths_710train_20220527.npy \
    --fusion --outputname encoded --output mydata_20240406_022034_ep002000_out.{output}.mat
```
* Latent outputs will be in the file `mydata_20240406_022034_ep002000_out.encoded.mat`

## Reading predicted outputs:
```python
import numpy as np
from scipy.io import loadmat
from krakencoder.utils import tri2square

Mpred=loadmat('mydata_20240406_022034_ep002000_in.fusionSC.mat',simplify_cells=True)
#predicted outputs are stored in Mpred['predicted_alltypes'][inputtype][outputtype]
fusionSC_to_FCshen_triu=Mpred['predicted_alltypes']['fusionSC']['FCcov_shen268_hpf_FC'] 
#fusionSC_to_FCshen_triu is [Nsubj x 35778], where each row is a 1x(upper triangular) for a 268x268 matrix

#Now convert the [Nsubj x 35778] stacked upper triangular vectors to a list of [268x268] square matrices for each subject
#by default, the diagonal is filled with zeros. For FC, we might want to set those to 1 with tri2square(..., diagval=1)
nsubj=fusionSC_to_FCshen_triu.shape[0]
n=268
triu=np.triu_indices(n,k=1)
fusionSC_to_FCshen_list=[tri2square(fusionSC_to_FCshen_triu[i,:],tri_indices=triu, diagval=1) for i in range(nsubj)]

#or convert to an [Nsubj x region x region] 3D matrix:
fusionSC_to_FCshen_3D=np.stack(fusionSC_to_FCshen_list)

#or compute a single [region x region] mean across subjects:
fusionSC_to_FCshen_mean=np.mean(np.stack([tri2square(fusionSC_to_FCshen_triu[i,:],tri_indices=triu, diagval=1) for i in range(nsubj)]), axis=0)
```
# Pretrained connectivity types
The current pre-trained model has been trained on the following 15 connectivity flavors, including 3 FC and 2 SC estimates from each of 3 atlases:
* `FCcov_fs86_hpf_FC` `FCcov_fs86_hpfgsr_FC` `FCpcorr_fs86_hpf_FC` `fs86_ifod2act_volnorm` `fs86_sdstream_volnorm` 
* `FCcov_shen268_hpf_FC` `FCcov_shen268_hpfgsr_FC` `FCpcorr_shen268_hpf_FC` `shen268_ifod2act_volnorm` `shen268_sdstream_volnorm`
* `FCcov_coco439_hpf_FC` `FCcov_coco439_hpfgsr_FC` `FCpcorr_coco439_hpf_FC` `coco439_ifod2act_volnorm` `coco439_sdstream_volnorm`

### Functional Connectivity (FC) types
* `FCcov_<parc>_hpf_FC` Pearson correlation FC
* `FCcov_<parc>_hpfgsr_FC` Pearson correlation FC after global signal regression
* `FCpcorr_<parc>_hpf_FC` Regularized partial correlation FC
* Time series have been denoised using ICA+FIX, high-pass filter > 0.01 Hz, with nuisance regression using WM+CSF aCompCor and 24 motion parameters.

### Structural Connectivity (SC) types
* `<parc>_ifod2act_volnorm` Streamline counts from iFOD2+ACT (Probabilistic whole-brain tractography with anatomical constraint), with pairwise streamline counts normalized by region volumes
* `<parc>_sdstream_volnorm`  Streamline counts from SD_STREAM (Deterministic whole-brain tractography), with pairwise streamline counts normalized by region volumes
* Tractography was performed using MRtrix3, with whole-brain dynamic seeding, and 5 million streamlines per subject.

### Parcellations
* `FS86` or `FreeSurfer86`: 86-region FreeSurfer Desikan-Killiany (DKT) cortical atlas with "aseg" subcortical regions(ie: aparc+aseg.nii.gz) [Desikan 2006](https://pubmed.ncbi.nlm.nih.gov/16530430/), [Fischl 2002](https://pubmed.ncbi.nlm.nih.gov/11832223/)
    * This atlas includes the 68 cortical DKT regions + 18 subcortical (excluding brain-stem)
* `Shen268`: 268-region cortical+subcortical atlas from [Shen 2013](https://pubmed.ncbi.nlm.nih.gov/23747961/). This atlas is defined in MNI voxel space.
* `Coco439` or `CocoMMPSUIT439`: 439-region atlas combining parts of several atlases:
    * 358 cortical ROIs from the HCP multi-modal parcellation ([Glasser 2016](https://pubmed.ncbi.nlm.nih.gov/27437579/))
    * 12 subcortical ROIs from aseg, adjusted by FSL's FIRST tool ([Patenaude 2011](https://pubmed.ncbi.nlm.nih.gov/21352927/))
        * Hippocampus from HCP-MMP cortex is merged with aseg hippocampus
    * 30 thalamic nuclei from FreeSurfer7 [Iglesias 2018](https://pubmed.ncbi.nlm.nih.gov/30121337/) (50 nuclei merged down to 30 to remove the smallest nuclei, as with AAL3v1)
    * 12 subcortical nuclei from AAL3v1 [Rolls 2020](https://pubmed.ncbi.nlm.nih.gov/31521825/) (VTA L/R, SN_pc L/R, SN_pr L/R, Red_N L/R, LC L/R, Raphe D/M)
    * 27 SUIT cerebellar subregions [Diedrichsen 2009](https://pubmed.ncbi.nlm.nih.gov/19457380/) (10 left, 10 right, 7 vermis)

# Requirements
* python >= 3.8
* pytorch >= 1.10
* numpy >= 1.21.2
* scipy >= 1.7.2
* scikit_learn >= 0.23.2
* cycler >= 0.11.0
* matplotlib >= 3.5
* colorspacious >= 1.1.2
* ipython >= 7.31
* *See [`requirements.txt`](requirements.txt) and [`requirements_exact.txt`](requirements_exact.txt)*

# Downloads
* Data and other files associated with this model can found here: [https://osf.io/dfp92](https://osf.io/dfp92/?view_only=449aed6ae3e9471881be76cbb50480dc)
    * `kraken_ioxfm_SCFC_[fs86,shen268,coco439]_993subj_pc256_25paths_710train_20220527.npy`: precomputed PCA transformations for fs86, shen268, and coco439 atlases. Each file contains the PCA transformations for FC, FCgsr, FCpcorr, SCsdstream, and SCifod2act inputs for that atlas.
    * `kraken_chkpt_SCFC_20240406_022034_ep002000.pt`: pretrained model checkpoint