# Cascade_Reconstruction
Tensorflow implementation of "A cascade reconstruction model with generalization ability evaluation for anomaly detection in videos" by Yuanhong Zhong, Xia Chen, Jinyang Jiang, Fan Ren.

## Data preparation
These datasets are from an official github of "Future Frame Prediction for Anomaly Detection - A New Baseline (CVPR 2018)".
* USCD [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]
* CUHK Avenue [[dataset](https://github.com/StevenLiuWen/ano_pred_cvpr2018)]

After geting the datasets, the optical flow magitude and orientation of each frame are calculated according to "C. Liu, Beyond pixels: Exploring new representations and applications for motion analysis, Doctoral Thesis. MIT. (2009).".

## Testing on saved models
There are the pretrained models of the papers, such as ped1, ped2 and avenue. Please manually download pretrained models from (https://pan.baidu.com/s/1NGuZ3cZO2EgrhxiIoFq_Rg) ncle, and move pretrains into /training_saver/'dataset name' folder.

