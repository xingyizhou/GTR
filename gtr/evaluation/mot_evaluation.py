import itertools
import json
import numpy as np
import os
from collections import defaultdict
from multiprocessing import freeze_support
import pycocotools.mask as mask_util
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from fvcore.common.file_io import PathManager
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from ..tracking.naive_tracker import track
from ..tracking import trackeval

def eval_track(out_dir, year):
    freeze_support()

    default_eval_config = trackeval.Evaluator.get_default_eval_config()
    default_eval_config['DISPLAY_LESS_PROGRESS'] = True
    
    default_dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()

    default_metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity']}
    config = {
        **default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs
    config['GT_FOLDER'] = 'datasets/mot/MOT{}/'.format(year)
    config['SPLIT_TO_EVAL'] = 'half_val'
    config['TRACKERS_FOLDER'] = out_dir
    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}
    print('config', config)
    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric())
    evaluator.evaluate(dataset_list, metrics_list)


def save_cocojson_as_mottxt(out_dir, videos, video2images, per_image_preds):
    if os.path.exists(out_dir):
        print('removing', out_dir)
        os.system('rm -rf {}'.format(out_dir))
    os.makedirs(out_dir)
    for video in videos:
        video_id = video['id']
        file_name = video['file_name']
        out_path = out_dir + '/{}.txt'.format(file_name)
        f = open(out_path, 'w')
        images = video2images[video_id]
        tracks = defaultdict(list)
        for image_info in images:
            result = per_image_preds[image_info['id']]
            frame_id = image_info['frame_id']
            for item in result:
                if not ('track_id' in item):
                    assert 0, 'No track ID!!'
                tracking_id = item['track_id']
                bbox = item['bbox']
                bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
                tracks[tracking_id].append([frame_id] + bbox)
        rename_track_id = 0
        for track_id in sorted(tracks):
            rename_track_id += 1
            for t in tracks[track_id]:
                f.write('{},{},{:.2f},{:.2f},{:.2f},{:.2f},-1,-1,-1,-1\n'.format(
                    t[0], rename_track_id, t[1], t[2], t[3], t[4])) # 
        f.close()


def track_and_eval_mot(out_dir, data, preds, dataset_name):
    videos = sorted(data['videos'], key=lambda x: x['id'])
    images = sorted(data['images'], key=lambda x: x['id'])
    video2images = defaultdict(list)
    for image in images:
        video2images[image['video_id']].append(image)
    for video in video2images:
        video2images[video] = sorted(
            video2images[video], key=lambda x: x['frame_id'])
    per_image_preds = defaultdict(list)
    for x in preds:
        if x['score'] > 0.4:
            per_image_preds[x['image_id']].append(x)
    has_track_id = len(preds) > 0 and 'track_id' in preds[0]
    del preds
    year = '20' if '20' in dataset_name else '17'
    split = 'trainval' if year == '17' else 'train'
    mot_out_dir = out_dir + '/moteval/{}/pred/data/'.format(split)
    if not has_track_id:
        print('Runing naive tracker')
        mot_out_dir = out_dir + '/moteval/{}/naive/data/'.format(split)
        for video in videos:
            images = video2images[video['id']]
            print('Runing tracking ...', video['file_name'], len(images))
            preds = [per_image_preds[x['id']] for x in images]
            preds = track(preds)
    save_cocojson_as_mottxt(
        mot_out_dir, videos, video2images, per_image_preds)
    eval_track(out_dir + '/moteval', year)


def custom_instances_to_coco_json(instances, img_id):
    """
    Add track_id
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    has_track_id = instances.has("track_ids")
    if has_track_id:
        track_ids = instances.track_ids

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        if has_track_id:
            result['track_id'] = int(track_ids[k].item())
        results.append(result)
    
    return results

class MOTEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, *, use_fast_impl=True):
        super().__init__(dataset_name, cfg, distributed, output_dir=output_dir, use_fast_impl=use_fast_impl)
        self.dataset_name = dataset_name


    def process(self, inputs, outputs):
        """
        custom_instances_to_coco_json
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = custom_instances_to_coco_json(
                    instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)


    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        assert img_ids is None
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)

        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        file_path = os.path.join(self._output_dir, "coco_instances_results.json")
        self._logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "w") as f:
            f.write(json.dumps(coco_results))
            f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

        track_and_eval_mot(
            self._output_dir, self._coco_api.dataset, coco_results,
            dataset_name=self.dataset_name)
