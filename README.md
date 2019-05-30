# Temporal Attentive Alignment for Video Domain Adaptation
This is the PyTorch implementation of our paper:

**Temporal Attentive Alignment for Video Domain Adaptation**  
[Min-Hung Chen](https://www.linkedin.com/in/chensteven), [Zsolt Kira](https://www.cc.gatech.edu/~zk15/), [Ghassan AlRegib](https://ghassanalregib.com/)  
CVPR Workshop (Learning from Unlabeled Videos), 2019  
[[arXiv](https://arxiv.org/abs/1905.10861)]

<p align="center">
<img src="webpage/Overview.png?raw=true" width="60%">
</p>

Although various image-based domain adaptation (DA) techniques have been proposed in recent years, domain shift in videos is still not well-explored. Most previous works only evaluate performance on small-scale datasets which are saturated. Therefore, we first propose a larger-scale dataset with larger domain discrepancy: UCF-HMDB_full. Second, we investigate different DA integration methods for videos, and show that simultaneously aligning and learning temporal dynamics achieves effective alignment even without sophisticated DA methods. Finally, we propose Temporal Attentive Adversarial Adaptation Network (TA3N), which explicitly attends to the temporal dynamics using domain discrepancy for more effective domain alignment, achieving state-of-the-art performance on three video DA datasets.

<p align="center">
<img src="webpage/SOTA_small.png?raw=true" width="49%">
<img src="webpage/SOTA_large.png?raw=true" width="50%">
</p>

---
## Contents
* [Requirements](#requirements)
* [Dataset Preparation](#dataset-preparation)
  * [Data Structure](#data-structure)
  * [File lists for training/validation](#file-lists-for-trainingvalidation)
* [Usage](#usage)
  * [Training](#training)
  * [Testing](#testing)
  * [Video Demo](#video-demo)
* [Options](#options)
  * [Domain Adaptation](#domain-adaptation)
  * [More options](#more-options)
* [Contact](#contact)

---
## Requirements
* support Python 3.6, PyTorch 0.4, CUDA 9.0, CUDNN 7.1.4
* install all the library with: `pip install -r requirements.txt`

---
## Dataset Preparation
### Data Structure
You need to extract frame-level features for each video to run the codes. To extract features, please check [`dataset_preparation/`](dataset_preparation/).

Folder Structure:
```
DATA_PATH/
  DATASET/
    list_DATASET_SUFFIX.txt
    RGB/
      CLASS_01/
        VIDEO_0001.mp4
        VIDEO_0002.mp4
        ...
      CLASS_02/
      ...

    RGB-Feature/
      VIDEO_0001/
        img_00001.t7
        img_00002.t7
        ...
      VIDEO_0002/
      ...
```
`RGB-Feature/` contains all the feature vectors for training/testing. `RGB/` contains all the raw videos.

There should be at least two `DATASET` folders: source training set  and validation set. If you want to do domain adaption, you need to have another `DATASET`: target training set.

The pre-trained feature representations will be released soon.
<!-- ([`Link`]()) -->

### File lists for training/validation
The file list `list_DATASET_SUFFIX.txt` is required for data feeding. Each line in the list contains the full path of the video folder, video frame number, and video class index. It looks like:
```
DATA_PATH/DATASET/RGB-Feature/VIDEO_0001/ 100 0
DATA_PATH/DATASET/RGB-Feature/VIDEO_0002/ 150 1
......
```
To generate the file list, please check [`dataset_preparation/`](dataset_preparation/).

---
## Usage
* training/validation: Run `./script_train_val.sh`
<!-- * demo video: Run `./script_demo_video.sh` -->

All the commonly used variables/parameters have comments in the end of the line. Please check [Options](#options).

#### Training
All the outputs will be under the directory `exp_path`.
* Outputs:
  * model weights: `checkpoint.pth.tar`, `model_best.pth.tar`
  * log files: `train.log`, `train_short.log`, `val.log`, `val_short.log`

#### Testing
You can choose one of model_weights for testing. All the outputs will be under the directory `exp_path`.

* Outputs:
  * score_data: used to check the model output (`scores_XXX.npz`)
  * confusion matrix: `confusion_matrix_XXX.png` and `confusion_matrix_XXX-topK.txt`

<!-- #### Video Demo
`demo_video.py` overlays the predicted categories and confidence values on one video. Please see "Results". -->

---
## Options
#### Domain Adaptation
<!-- In both `./script_train_val.sh` and `./script_demo_video.sh`, there are several options related to our Domain Adaptation approaches. -->
In `./script_train_val.sh`, there are several options related to our DA approaches.
* `use_target`: switch on/off the DA mode
  * `none`: not use target data (no DA)
  * `uSv`/`Sv`: use target data in a unsupervised/supervised way
* options for the DA approaches:
  * discrepancy-based: DAN, JAN
  * adversarial-based: RevGrad
  * Normalization-based: AdaBN
  * Ensemble-based: MCD

#### More options
For more details of all the arguments, please check [opts.py](opts.py).

#### Notes
The options in the scripts have comments with the following types:
* no comment: user can still change it, but NOT recommend (may need to change the code or have different experimental results)
* comments with choices (e.g. `true | false`): can only choose from choices
* comments as `depend on users`: totally depend on users (mostly related to data path)

---
## Citation
If you find this repository useful, please cite our paper:
```
@article{chen2019taaan,
title={Temporal Attentive Alignment for Video Domain Adaptation},
author={Chen, Min-Hung and Kira, Zsolt and AlRegib, Ghassan},
booktitle = {CVPR Workshop on Learning from Unlabeled Videos},
year={2019},
url={https://arxiv.org/abs/1905.10861}
}
```

---
### Acknowledgments
Some codes are borrowed from [TSN](https://github.com/yjxiong/temporal-segment-networks), [pytorch-tsn](https://github.com/yjxiong/tsn-pytorch), [TRN-pytorch](https://github.com/metalbubble/TRN-pytorch), and [Xlearn](https://github.com/thuml/Xlearn/tree/master/pytorch).

---
### Contact
[Min-Hung Chen](https://www.linkedin.com/in/chensteven) <br>
cmhungsteve AT gatech DOT edu
