# FEAST - Feature Representation and Aggregation Network for Segmentation Tracking

# Abstract
Visual tracking is a key component of computer vision and has a wide range of practical applications. Recently, the tracking-by-segmentation framework has been widely applied in visual tracking due to its astonishing performance on accuracy. It attempts to learn from the framework of video object segmentation to realize accurate tracking. Although segmentation-based trackers are effective for target scale estimation, the segmentation network makes the trackers have high requirements for the extracted target features due to the need for pixel-level segmentation. Therefore, in this article, we propose a novel feature representation and aggregation network and introduce it into the tracking-by-segmentation framework to extract and integrate rich features for accurate and robust segmentation tracking. To be specific, firstly, the proposed approach models three complementary feature representations, including contextual semantic, local position, and structural patch feature representations, through cross-attention, cross-correlation and dilated involution mechanisms respectively. Secondly, these features are fused by a simple feature aggregation network. Thirdly, the fusion features are fed into the segmentation network to obtain accurate target state estimation. In addition, to adapt the segmentation network to the appearance changes and partial occlusion, we introduce a template update strategy and a bounding box refinement module for robust segmentation and tracking. The extensive experimental results on twelve challenging tracking benchmarks show that the proposed tracker outperforms most of the state-of-the-art trackers and achieves very promising tracking performance on the OTB100 and VOT2018 benchmarks.

## Demo
![image](https://github.com/Yang428/FEAST/blob/master/resultsOnBenchmarks/Trans.gif)

## Running Environments
* Pytorch 1.1.0, Python 3.6.12, Cuda 9.0, torchvision 0.3.0, cudatoolkit 9.0, Matlab R2016b.
* Ubuntu 16.04, NVIDIA GeForce GTX 1080Ti, i7-6700K CPU @ 4.00GHz.


## Installation
The instructions have been tested on an Ubuntu 16.04 system. In case of issues, we refer to these two links [1](https://github.com/alanlukezic/d3s) and [2](https://github.com/visionml/pytracking) for details.

#### Clone the GIT repository
```
git clone https://github.com/Yang428/FEAST.git.
```

#### Install dependent libraries
Run the installation script 'install.sh' to install all dependencies. We refer to [this link](https://github.com/visionml/pytracking/blob/master/INSTALL.md) for step-by-step instructions.
```
bash install.sh conda_install_path pytracking
```

#### Or step by step install
```
conda create -n pytracking python=3.6
conda activate pytracking
conda install -y pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=9.0 -c pytorch
conda install -y matplotlib=2.2.2
conda install -y pandas
pip install opencv-python
pip install tensorboardX
conda install -y cython
pip install pycocotools
pip install jpeg4py 
sudo apt-get install libturbojpeg
```

#### Or copy my environment directly.

You can download the packed conda environment from the [Baidu cloud link](https://pan.baidu.com/s/1gMQOB2Zs1UPj6n8qzJc4Lg?pwd=qjl2), the extraction code is 'qjl2'.

#### Download the pre-trained networks
You can download the models from the [Baidu cloud link](https://pan.baidu.com/s/1mb2bP-4OAW2onoI-2wInmg?pwd=vl81), the extraction code is 'vl81'. Then put the model files 'SegmNet.pth.tar and IoUnet.pth.tar' to the subfolder 'pytracking/networks'.

## Testing the tracker
There are the [raw resullts](https://github.com/Yang428/FEAST/tree/master/resultsOnBenchmarks) on eleven datasets. 
1) Download the testing datasets Got-10k, TrackingNet, VOT2016, VOT2018, VOT2019, VOT2020, OTB100, UAV123, LaSOT, NFS and TCL128 from the following Baidu cloud links.
* [Got-10k](https://pan.baidu.com/s/1t_PvpIicHc0U9yR4upf-cA), the extraction code is '78hq'.
* [TrackingNet](https://pan.baidu.com/s/1BKtc4ndh_QrMiXF4fBB2sQ), the extraction code is '5pj8'.
* [VOT2016](https://pan.baidu.com/s/1iU88Aqq9mvv9V4ZwY4gUuw), the extraction code is '8f6w'.
* [VOT2018](https://pan.baidu.com/s/1ztAfNwahpDBDssnEYONDuw), the extraction code is 'jsgt'.
* [VOT2019](https://pan.baidu.com/s/1vf7l4sQMCxZY_fDsHkuwTA), the extraction code is '61kh'.
* [VOT2020](https://pan.baidu.com/s/16PFiEdnYQDIGh4ZDxeNB_w), the extraction code is 'kdag'.
* [OTB100](https://pan.baidu.com/s/1TC6BF9erhDCENGYElfS3sw), the extraction code is '9x8q'.
* [UAV123](https://pan.baidu.com/share/init?surl=OAUG8IrdqTRpGbK4Nv-bhA), the extraction code is 'vp4r'.
* [LaSOT](https://pan.baidu.com/s/1KBlrWGOFH9Fe85pCWN5ZkA&shfl=sharepset#list/path=%2F).
* [NFS](https://pan.baidu.com/share/init?surl=72r0r4y6UhAxzjc359yt6A), the extraction code is 'gc7u'.
* [TCL128](https://pan.baidu.com/share/init?surl=P4i63SrHhxMPShv7mWYZqg), the extraction code is '1h83'.
* Or you can download almost all tracking datasets from this web [link](https://blog.csdn.net/laizi_laizi/article/details/105447947#VisDrone_77).

2) Change the following paths to you own paths.
```
Network path: pytracking/parameters/feast/feast.py  params.segm_net_path.
Results path: pytracking/evaluation/local.py  settings.network_path, settings.results_path, dataset_path.
```
3) Run the FEAST tracker on Got10k, TrackingNet, OTB100, UAV123, LaSOT, NFS and TCL128 datasets.
```
cd pytracking
python run_experiment.py myexperiments got10k
python run_experiment.py myexperiments trackingnet
python run_experiment.py myexperiments otb
python run_experiment.py myexperiments uav
python run_experiment.py myexperiments lasot
python run_experiment.py myexperiments nfs
python run_experiment.py myexperiments tpl
```

## Evaluation on VOT16, VOT18 and VOT19 using Matlab R2016b
We provide a [VOT Matlab toolkit](https://github.com/votchallenge/toolkit-legacy) integration for the FEAST tracker. There is the [tracker_FEAST.m](https://github.com/Yang428/FEAST/tree/master/pytracking/utils) Matlab file in the 'pytracking/utils', which can be connected with the toolkit. It uses the 'pytracking/vot_wrapper.py' script to integrate the tracker to the toolkit.

## Evaluation on VOT2020 and VOT2021 using Python Toolkit
We provide a [VOT Python toolkit](https://github.com/votchallenge/toolkit) integration for the FEAST tracker. There is the [trackers.ini](https://github.com/Yang428/FEAST/tree/master/pytracking/utils) setting file in the 'pytracking/utils', which can be connected with the toolkit. It uses the 'pytracking/vot20_wrapper.py' script to integrate the tracker to the toolkit.
```
cd pytracking/workspace_vot2020
pip install git+https://github.com/votchallenge/vot-toolkit-python
vot initialize <vot2020> --workspace ./workspace_vot2020/
vot evaluate FEAST
vot analysis --workspace ./workspace_vot2020/FEAST
```

## Training the network
The FEAST network is trained only on the YouTube VOS dataset. Download the VOS training dataset (2018 version) and copy the files vos-list-train.txt and vos-list-val.txt from ltr/data_specs to the training directory of the VOS dataset. Download the bounding boxes from [this link](http://data.vicos.si/alanl/d3s/rectangles.zip) and copy them to the corresponding training sequence directories.
1) Download the training dataset from [this link](https://youtube-vos.org/challenge/2018/).

2) Change the following paths to you own paths.
```
Workspace: ltr/admin/local.py  workspace_dir.
Dataset: ltr/admin/local.py  vos_dir.
```
3) Taining the FEAST network
```
cd ltr
python run_training.py segm segm_default
```

## Acknowledgement
We would like to thank the author Martin Danelljan of [pytracking](https://github.com/visionml/pytracking) and the author Alan Lukežič of [D3S](https://github.com/alanlukezic/d3s).
