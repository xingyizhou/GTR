# GTR model zoo

## Introduction

This file documents a collection of models reported in our paper.
Our experiments are trained on a DGX machine with 8 32G V100 GPUs.
Most of our models use 4 GPUs.

#### How to Read the Tables

The "Name" column contains a link to the config file. 
To train a model, run 

```
python train_net.py --num-gpus 4 --config-file /path/to/config/name.yaml
``` 

To evaluate a model with a trained/ pretrained model, run 

```
python train_net.py --config-file /path/to/config/name.yaml --eval-only MODEL.WEIGHTS /path/to/weight.pth
``` 

## MOT

#### Validation set


|         Name           |   MOTA     |  IDF1  |  HOTA  | DetA  | AssA  | Download |
|------------------------|------------|--------|--------|-------|-------|----------|
|[GTR_MOT_FPN](../configs/GTR_MOT_FPN.yaml) | 71.3      |  75.9  | 63.0   | 60.4  | 66.2 | [model](https://drive.google.com/file/d/1fY4DrOHR9lxzU9XuLZy6_ae2srXZBu2h/view?usp=sharing) |
|[GTR_MOT_FPN (local)](../configs/GTR_MOT_FPN.yaml)| 71.1      |  74.2  | 62.1   | 60.2  | 64.4 | same as above |

#### Test set

|         Name           |   MOTA     |  IDF1  |  HOTA  | DetA  | AssA  | Download |
|------------------------|------------|--------|--------|-------|-------|----------|
|[GTR_MOTFull_FPN](../configs/GTR_MOT_FPN.yaml) | 75.3      |  71.5  | 59.1   | 61.6  | 57.0 | [model](https://drive.google.com/file/d/1fn74PGscvecxTQRXlWqtXgp3VZI3gYT_/view?usp=sharing) |

#### Note

- The validation set follows the half-half training set split from [CenterTrack](https://github.com/xingyizhou/CenterTrack).
- All models are finetuned from a detection-only model trained on Crowdhuman ([config](../configs/CH_FPN_1x.yaml), [model](https://drive.google.com/file/d/15wFGs2HyepezQKvyLb0BpJ_5PMMxStd0/view?usp=sharing)). Download or train the model and place it as `GTR_ROOT/models/CH_FPN_1x.pth` before training. Training the detection-only models takes ~12 hours on 4 GPUs.
- Training GTR takes ~3 hours on 4 V100 GPUs (32G memory).
- `GTR_MOT_FPN` is our model with a temporal-window size of 32. It needs more than 12G GPU memory in testing. To change the temporal-window size, append `INPUT.VIDEO.TEST_LEN 16` to the command.
- `GTR_MOT_FPN (local)` is our local tracker baseline, which applies [FairMOT](https://github.com/ifzhang/FairMOT) to our detections and features. To run it, append `VIDEO_TEST.LOCAL_TRACK True` to the command.

## TAO

|         Name          |   validation mAP |  Test mAP | Download |
|-----------------------|------------------|-----------|----------|
|[GTR_TAO_DR2101](../configs/GTR_TAO_DR2101.yaml) | 22.5  | 20.1 | [model](https://drive.google.com/file/d/1TqkLpFZvOMY5HTTaAWz25RxtLHdzQ-CD/view?usp=sharing) |

#### Note

- The model is evaluated on TAO keyframes only, which are sampled in ~1 frame-per-second.
- Our model is trained on LVIS+COCO only. The TAO training set is not used anywhere.
- Our model is finetuned on a detection-only CenterNet2 model trained on LVIS+COCO ([config](./configs/C2_LVISCOCO_DR2101_4x.yaml), [model](https://drive.google.com/file/d/1WCrfbyNhMryB4ryV5piLG3NLgU3pUvcz/view?usp=sharing)). Download or train the model and place it as `GTR_ROOT/models/C2_LVISCOCO_DR2101_4x.pth` before training. Training the detection-only models takes ~3 days on 8 GPUs.
- Training GTR takes ~13 hours on 4 V100 GPUs (32G memory).