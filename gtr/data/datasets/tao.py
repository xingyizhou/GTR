from collections import defaultdict
import logging
import os

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.data.datasets.lvis_v0_5_categories import LVIS_CATEGORIES
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES as LVIS_V1_CATEGORIES
from detectron2.data.datasets.lvis import get_lvis_instances_meta, register_lvis_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

logger = logging.getLogger(__name__)

__all__ = ["load_tao_json", "register_tao_instances", "get_tao_instances_meta"]


def register_tao_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in TAO's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_tao_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="tao", **metadata
    )

def register_tao_v1_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in TAO's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_tao_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="lvis", **metadata
    )

def load_tao_json(json_file, image_root, dataset_name=None):
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    # if 'v1' in dataset_name:
    meta = get_lvis_instances_meta('lvis_v1')
    if 'coco' in dataset_name:
        meta = _get_builtin_metadata('coco')

    id_map = None
    cat_ids = sorted(lvis_api.get_cat_ids())
    id_map = {v: i for i, v in enumerate(cat_ids)}
    meta['thing_dataset_id_to_contiguous_id'] = id_map
    if not (min(cat_ids) == 1 and max(cat_ids) == len(cat_ids)):
        logger.warning(
            """
            Category ids in annotations are not in [1, #categories]! We'll apply a mapping for you.
            """
        )
        logger.warning(id_map)
    MetadataCatalog.get(dataset_name).set(**meta)


    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file
    )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in the LVIS v0.5 format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    all_frames = defaultdict(list)
    for (img_dict, anno_dict_list) in imgs_anns:
        record = {}
        if "file_name" in img_dict:
            file_name = img_dict["file_name"]
            if img_dict["file_name"].startswith("COCO"):
                file_name = file_name[-16:]
        else:
            # e.g., http://images.cocodataset.org/train2017/000000391895.jpg
            file_name = img_dict["coco_url"][30:]
        
        record["file_name"] = os.path.join(image_root, file_name)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["not_exhaustive_category_ids"] = img_dict.get("not_exhaustive_category_ids", [])
        record["neg_category_ids"] = img_dict.get("neg_category_ids", [])
        # NOTE: modified by zhouxy to convert to 0-based
        record["neg_category_ids"] = [x - 1 for x in record["neg_category_ids"]]
        image_id = record["image_id"] = img_dict["id"]
        record['video_id'] = img_dict['video_id']
        record['frame_index'] = img_dict['frame_index']
        all_frames[record['video_id']].append(img_dict['frame_index'])
        objs = []
        for anno in anno_dict_list:
            # Check that the image_id in this annotation is the same as
            # the image_id we're looking at.
            # This fails only when the data parsing logic or the annotation file is buggy.
            assert anno["image_id"] == image_id
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            if id_map:
                obj["category_id"] = id_map[anno["category_id"]]
            # obj["category_id"] = anno["category_id"] - 1  # Convert 1-indexed to 0-indexed
            if 'track_id' in anno:
                obj['track_id'] = anno['track_id']
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    for video_id in all_frames:
        all_frames[video_id] = sorted(all_frames[video_id])
    for record in dataset_dicts:
        record['frame_id'] = \
            all_frames[record['video_id']].index(record['frame_index']) + 1 # frame_id starts from 1
        del record['frame_index']
    del all_frames

    return dataset_dicts


def get_tao_instances_meta():
    assert len(LVIS_CATEGORIES) == 1230
    cat_ids = [k["id"] for k in LVIS_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta

def get_tao_v1_instances_meta():
    assert len(LVIS_V1_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_V1_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_V1_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


_PREDEFINED_SPLITS_TAO = {
    "tao_train": ("tao/keyframes/", "tao/annotations/train.json"),
    "tao_val": ("tao/keyframes/", "tao/annotations/validation.json"),
    "tao_test": ("tao/frames/", "tao/annotations/test_without_annotations.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_TAO.items():
    register_tao_instances(
        key,
        get_tao_instances_meta(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )


_PREDEFINED_SPLITS_TAOV1 = {
    "tao_train_v1": ("tao/keyframes/", "tao/annotations/train_v1.json"),
    "tao_val_v1": ("tao/keyframes/", "tao/annotations/validation_v1.json"),
    "tao_val_v1_mini": ("tao/keyframes/", "tao/annotations/validation_v1_mini.json"),
}

for key, (image_root, json_file) in _PREDEFINED_SPLITS_TAOV1.items():
    register_tao_v1_instances(
        key,
        get_lvis_instances_meta("lvis_v1"),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
