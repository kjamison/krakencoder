# krakencoder

## CLI-facing scripts:
* [`run_training.py`](run_training.py): Train a new model
* [`run_model.py`](run_model.py): Run a saved checkpoint on new data
* [`describe_checkpoint.py`](describe_checkpoint.py): Print information about a saved checkpoint

## Internal scripts:
* [`model.py`](krakencoder/model.py): Krakencoder model class definition
* [`train.py`](krakencoder/train.py): Training-related functions
* [`loss.py`](krakencoder/loss.py): Specifies different loss functions to be used during training and evaluation
* [`data.py`](krakencoder/data.py): Functions for loading and transforming input data
* [`plotfigures.py`](krakencoder/plotfigures.py): Functions for plotting loss curves and performance heatmaps
* [`utils.py`](krakencoder/utils.py): Miscellaneous utility functions

## Examples

### Generating latent space representations on new data, using pre-trained model:
```bash
python run_model.py --inputdata '[fs86_sdstream_volnorm]=mydata_fs86_sdstream_volnorm.mat' \
        '[fs86_ifod2act_volnorm]=mydata_fs86_ifod2act_volnorm.mat' \
        '[shen268_sdstream_volnorm]=mydata_shen268_sdstream_volnorm.mat' \
        '[shen268_ifod2act_volnorm]=mydata_shen268_ifod2act_volnorm.mat' \
        '[coco439_sdstream_volnorm]=mydata_coco439_sdstream_volnorm.mat' \
        '[coco439_ifod2act_volnorm]=mydata_coco439_ifod2act_volnorm.mat' \
    --adaptmode meanfit+meanshift \
    --checkpoint krak_chkpt_SCFC_20240406_022034_ep002000.pt \
    --inputxform krak_ioxfm_SCFC_coco439_993subj_pc256_25paths_710train_20220527.npy \
        krak_ioxfm_SCFC_fs86_993subj_pc256_25paths_710train_20220527.npy \
        krak_ioxfm_SCFC_shen268_993subj_pc256_25paths_710train_20220527.npy \
    --outputname encoded --output mydata_20240406_022034_ep002000_{output}.mat
```
* Each input file should have a 'data' field containing the [subjects x region x region] connectivity data for that input flavor.
* `--adaptmode meanfit+meanshift` uses a minimal approach for domain shift by linearly mapping the population mean of your input data to the population mean of the training data
* The more input flavors you can provide, the better the predictions are.
* Latent outputs will be in the file `mydata_20240406_022034_ep002000_encoded.mat`

### Generating predicted connectomes on new data, using pre-trained model:
```bash
python run_model.py --inputdata '[fs86_sdstream_volnorm]=mydata_fs86_sdstream_volnorm.mat' \
        '[fs86_ifod2act_volnorm]=mydata_fs86_ifod2act_volnorm.mat' \
        '[shen268_sdstream_volnorm]=mydata_shen268_sdstream_volnorm.mat' \
        '[shen268_ifod2act_volnorm]=mydata_shen268_ifod2act_volnorm.mat' \
        '[coco439_sdstream_volnorm]=mydata_coco439_sdstream_volnorm.mat' \
        '[coco439_ifod2act_volnorm]=mydata_coco439_ifod2act_volnorm.mat' \
    --adaptmode meanfit+meanshift \
    --checkpoint krak_chkpt_SCFC_20240406_022034_ep002000.pt \
    --inputxform krak_ioxfm_SCFC_coco439_993subj_pc256_25paths_710train_20220527.npy \
        krak_ioxfm_SCFC_fs86_993subj_pc256_25paths_710train_20220527.npy \
        krak_ioxfm_SCFC_shen268_993subj_pc256_25paths_710train_20220527.npy \
    --outputname all --output mydata_20240406_022034_ep002000_{output}.mat \
    --burst --burstinclude burst=all burstSC=SC burstFC=FC --burstnoself --burstnoatlas
```
* Each input file should have a 'data' field containing the [subjects x region x region] connectivity data for that input flavor.
* This will predict all 15 connectome flavors as outputs, based on whatever inputs are provided.
* This includes "fusion" predictions incorporating all inputs into each predicted output.
* Predicted outputs will be one file per output flavor, for instance: `mydata_20240406_022034_ep002000_FCcov_shen268_hpf_FC.mat`

### Reading predicted outputs:
```python
import numpy as np
from scipy.io import loadmat
from krakencoder.utils import tri2square

Mpred=loadmat('mydata_20240406_022034_ep002000_FCcov_shen268_hpf_FC.mat',simplify_cells=True)
#predicted outputs are stored in Mpred['predicted_alltypes'][inputtype][outputtype]
fusionSC_to_FCshen_triu=Mpred['predicted_alltypes']['burstSC']['FCcov_shen268_hpf_FC'] 
#fusionSC_to_FCshen_triu is [Nsubj x 35778], where each row is a 1x(upper triangular) for a 268x268 matrix

#Now convert the [Nsubj x 35778] stacked upper triangular vectors to a list of [268x268] square matrices for each subject
nsubj=fusionSC_to_FCshen_triu.shape[0]
n=268
triu=np.triu_indices(n,k=1)
fusionSC_to_FCshen_list=[tri2square(fusionSC_to_FCshen_triu[i,:],tri_indices=triu) for i in range(nsubj)]

#or convert to an [Nsubj x region x region] 3D matrix:
fusionSC_to_FCshen_3D=np.stack(fusionSC_to_FCshen_list)

#or compute a single [region x region] mean across subjects:
fusionSC_to_FCshen_mean=np.mean(np.stack([tri2square(fusionSC_to_FCshen_triu[i,:],tri_indices=triu) for i in range(nsubj)]), axis=0)
```

## Downloads
