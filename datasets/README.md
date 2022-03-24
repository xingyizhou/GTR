# Prepare datasets for GTR

Our MOT models are trained on [MOT17](https://motchallenge.net/data/MOT17/) and [CrowdHuman](https://www.crowdhuman.org/download.html) and are evaluated on MOT17.
Our TAO models are trained on [LVIS](https://www.lvisdataset.org/) and [COCO](https://cocodataset.org/) images), and evaluated on [TAO](http://taodataset.org/#). 

Before starting processing, please download the datasets from the official websites and place or sim-link them under `$Detic_ROOT/datasets/`. 

```
$Detic_ROOT/datasets/
    lvis/
    coco/
    mot/
    crowdhuman/
    tao/
```

Please follow the following instruction to pre-process individual datasets.

`metadata/` is our preprocessed meta-data (included in the repo). See the below [section](#Metadata) for details.
Please follow the following instruction to pre-process individual datasets.

### MOT

First, download and place them in the following way

```
mot/
    MOT17/
        train/
            MOT17-02-FRCNN/
            ...
        test/
            MOT17-01-FRCNN/
            ...
```

Then create the half-half train/ validation split and convert the annotation format

```
python tools/convert_mot2coco.py
```
This creates `datasets/mot/MOT17/annotations/train_half_conf0.json` and 
`datasets/mot/MOT17/annotations/val_half_conf0.json`.
Note that these files are different from [CenterTrack](https://github.com/xingyizhou/CenterTrack/blob/master/readme/DATA.md#mot-2017) as CenterTrack [filters](https://github.com/xingyizhou/CenterTrack/blob/master/src/tools/convert_mot_to_coco.py#L97) annotations with a visibility threshold of 0.25.

To generate the annotation files for the train/test split, change the `SPLITS` in `tools/convert_mot2coco.py` to `SPLITS = ['train', 'test']` and run 
`python tools/convert_mot2coco.py` again.

### Crowdhuman

Download the data and place them as the following:

```
crowdhuman/
    CrowdHuman_train/
        Images/
    CrowdHuman_val/
        Images/
    annotation_train.odgt
    annotation_val.odgt
```

Convert the annotation format by 

```
python tools/convert_crowdhuman_amodal.py
```

This creates `datasets/crowdhuman/annotations/train_amodal.json` and 
`datasets/crowdhuman/annotations/train_amodal.json`.

### COCO and LVIS

Download COCO and LVIS data place them in the following way:

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
```

Next, prepare the merged annotation file using 

~~~
python tools/merge_lvis_coco.py
~~~

This creates `datasets/lvis/lvis_v1_train+coco_box.json`

### TAO

Download the data following the official [instructions](https://github.com/TAO-Dataset/tao/blob/master/docs/download.md) and place them as 

```
tao/
    frames/
        val/
            ArgoVerse/
            AVA/
            BDD/
            Charades/
            HACS/
            LaSOT/
            TFCC100M/
        train/
            ArgoVerse/
            ...
        test/
            ArgoVerse/
            ...
    annotations/
        train.json
        validation.json
        test_without_annotations.json
```

Our model only uses the annotated frames ("keyframe"). To make the data management easier, we first copy the keyframes to a new folder

```
python tools/move_tao_keyframes.py --gt datasets/tao/annotations/validation.json --img_dir datasets/tao/frames --img_dir datasets/tao/keyframes
```

This creates `tao/keyframes/`

The TAO annotations are originally based on LVIS v0.5. We update them to LVIS v1 for validation.

```
python tools/create_tao_v1.py datasets/tao/annotations/validation.json
```

This creates `datasets/tao/annotations/validation_v1.json`.

For TAO test set, we'll convert the LVIS v1 labels back to v0.5 for the server-based test set evaluation.
