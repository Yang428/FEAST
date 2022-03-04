# LEAST Tracker

The main codes of the LEAST tracker will be available as soon as possible.

## Running Environments
* Pytorch 1.1.0, Python 3.6.12, Cuda 10.0, torchvision 0.3.0, cudatoolkit 10.0, Matlab R2016b.
* Ubuntu 16.04, NVIDIA GeForce GTX 1080Ti.

## Installation
The instrustions have been tested on an Ubuntu 16.04 system. In case of issues, we refer to these two links [1](https://github.com/alanlukezic/d3s) and [2](https://github.com/visionml/pytracking) for details.

#### Clone the GIT repository
```
git clone https://github.com/Yang428/LEAST.git.
```

#### Install dependent libraries
Run the installation script 'install.sh' to install all dependencies. We refer to [this link](https://github.com/visionml/pytracking/blob/master/INSTALL.md) for step-by-step instructions.
```
bash install.sh conda_install_path pytracking
```

#### Download the pre-trained networks
You can download the models from the [Baidu cloud link](https://pan.baidu.com/s/11kn8IyxN0AJ8D0C780FLUg), the extraction code is 'lxa4'. Then put the model files 'SegmNet.pth.tar and SegmNet_maskInitNet.pth.tar' to the subfolder 'pytracking/networks'.

## Testing the tracker
There are the [raw resullts](https://github.com/Yang428/LEAST/tree/master/resultsOnBenchmarks) on three datasets. 
1) Download the testing datasets Got-10k, TrackingNet and VOT2019 from the following Baidu cloud links.
* [Got-10k](https://pan.baidu.com/s/1TMCb2LMUigtchzrFaTw-EA), the extraction code is 'pcj6'.
* [TrackingNet](https://pan.baidu.com/s/1qc7FSjHXvAJkssJj-sRFKA), the extraction code is 'q00p'.
* [VOT2019](https://pan.baidu.com/s/1vf7l4sQMCxZY_fDsHkuwTA), the extraction code is '61kh'.
* Or you can download almost all tracking datasets from this web [link](https://blog.csdn.net/laizi_laizi/article/details/105447947#VisDrone_77).

2) Change the following paths to you own paths.
```
Network path: pytracking/parameters/segm/default_params.py  params.segm_net_path.
Results path: pytracking/evaluation/local.py  settings.network_path, settings.results_path, dataset_path.
```
3) Run the LEAST tracker on VOT19 sequences.
```
cd pytracking
python run_tracker.py segm default_params --dataset vot19 --sequence ants1 --debug 1
```
4) Run the LEAST tracker on Got10k and TrackingNet datasets.
```
cd pytracking
python run_experiment.py myexperiments got10k
python run_experiment.py myexperiments trackingnet
```

## Evaluation on VOT2019 using Matlab R2016b
We provide a [VOT Matlab toolkit](https://github.com/votchallenge/toolkit-legacy) integration for the LEAST tracker. There is the [tracker_LEAST.m](https://github.com/Yang428/LEAST/tree/master/pytracking/utils) Matlab file in the 'pytracking/utils', which can be connected with the toolkit. It uses the 'pytracking/vot_wrapper.py' script to integrate the tracker to the toolkit.

## Training the networks
The LEAST tracker is pre-trained for segmentation task only on the YouTube VOS dataset. Download the VOS training dataset (2018 version) and copy the files vos-list-train.txt and vos-list-val.txt from ltr/data_specs to the train directory of the VOS dataset.
1) Download the training dataset from [this link](https://youtube-vos.org/challenge/2018/).

2) Change the following paths to you own paths.
```
Workspace: ltr/admin/local.py  workspace_dir.
Dataset: ltr/admin/local.py  vos_dir.
```
3) Taining the segmentation network
```
cd ltr
python run_training.py segm segm_default
```
4) Taining the mask initialization network
```
cp ./LEAST/ ./LEAST_maskInit
cd ./LEAST_maskInit/ltr
move the file 
 (./actors/segm_actor_maskInitNet.py; ./data/segm_processing_maskInitNet.py; ./train_seetings/segm/segm_default_maskInitNet.py)
 to (./actors/segm_actor.py; ./data/segm_processing.py; ./train_seetings/segm/segm_default.py) respectively.
python run_training.py segm segm_default
```

## Acknowledgement
This a modified version of [D3S](https://github.com/alanlukezic/d3s) tracker which is based on the [pytracking](https://github.com/visionml/pytracking) framework. We would like to thank the author Martin Danelljan of pytracking and the author Alan Lukežič of D3S.
