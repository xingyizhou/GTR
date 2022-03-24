# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Xingyi Zhou: handle neg_category_ids
import logging
import os

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets.lvis_v1_categories import LVIS_CATEGORIES

logger = logging.getLogger(__name__)

def register_lvis_v1_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in LVIS's json annotation format for instance detection and segmentation.

    Args:
        name (str): a name that identifies the dataset, e.g. "lvis_v0.5_train".
        metadata (dict): extra metadata associated with this dataset. It can be an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str): directory which contains all the images.
    """
    DatasetCatalog.register(name, lambda: load_lvis_v1_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root, evaluator_type="lvis", **metadata
    )


def load_lvis_v1_json(json_file, image_root, dataset_name=None):
    from lvis import LVIS

    json_file = PathManager.get_local_path(json_file)

    timer = Timer()
    lvis_api = LVIS(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    if dataset_name is not None:
        meta = get_lvis_v1_instances_meta()
        MetadataCatalog.get(dataset_name).set(**meta)

    img_ids = sorted(lvis_api.imgs.keys())
    imgs = lvis_api.load_imgs(img_ids)
    anns = [lvis_api.img_ann_map[img_id] for img_id in img_ids]

    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), "Annotation ids in '{}' are not unique".format(
        json_file
    )

    imgs_anns = list(zip(imgs, anns))

    logger.info("Loaded {} images in the LVIS v1 format from {}".format(len(imgs_anns), json_file))

    dataset_dicts = []
    inst_count = 0
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
        # NOTE: modified by Xingyi to convert to 0-based
        record["neg_category_ids"] = [x - 1 for x in record["neg_category_ids"]]
        image_id = record["image_id"] = img_dict["id"]

        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            if anno.get('iscrowd', 0) > 0:
                continue
            obj = {"bbox": anno["bbox"], "bbox_mode": BoxMode.XYWH_ABS}
            obj["category_id"] = anno["category_id"] - 1  # Convert 1-indexed to 0-indexed
            inst_count = inst_count + 1
            obj['instance_id'] = inst_count
            obj['score'] = anno['score'] if 'score' in anno else 1.
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    print('inst_count', dataset_name, inst_count)
    return dataset_dicts


def get_lvis_v1_instances_meta():
    assert len(LVIS_CATEGORIES) == 1203
    cat_ids = [k["id"] for k in LVIS_CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(LVIS_CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["synonyms"][0] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta

_PREDEFINED_SPLITS_LVIS_V1 = {
    "lvis_v1_train+coco_box": ("coco/", "lvis/lvis_v1_train+coco_box.json"),
    "lvis_v1_train+coco_mask": ("coco/", "lvis/lvis_v1_train+coco_mask.json"),

}


for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVIS_V1.items():
    register_lvis_v1_instances(
        key,
        get_lvis_v1_instances_meta(),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )
