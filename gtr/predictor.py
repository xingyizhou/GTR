import cv2
import numpy as np
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer, random_colors
from detectron2.utils.video_visualizer import _create_text_labels
from detectron2.utils.visualizer import ColorMode, Visualizer


class TrackingVisualizer(VideoVisualizer):
    def __init__(self, metadata, instance_mode=ColorMode.IMAGE):
        """
        """
        self.metadata = metadata
        self._old_instances = []
        assert instance_mode in [
            ColorMode.IMAGE,
            ColorMode.IMAGE_BW,
        ], "Other mode not supported yet."
        self._instance_mode = instance_mode
        self._assigned_colors = {}
        self._max_num_instances = 10000
        self._num_colors = 74
        self._color_pool = random_colors(self._num_colors, rgb=True, maximum=1)
        self.color_idx = 0


    def draw_instance_predictions(self, frame, predictions):
        """
        """
        frame_visualizer = Visualizer(frame, self.metadata)
        num_instances = len(predictions)
        if num_instances == 0:
            return frame_visualizer.output

        boxes = predictions.pred_boxes.tensor.numpy() if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.numpy() if predictions.has("pred_classes") else None
        predictions.track_ids = predictions.track_ids % self._max_num_instances
        track_ids = predictions.track_ids.numpy() 
        if predictions.has("pred_masks"):
            masks = predictions.pred_masks
        else:
            masks = None

        colors = self._assign_colors_by_track_id(predictions)
        labels = _create_text_labels(
            classes, scores, self.metadata.get("thing_classes", None))
        labels = ['({}){}'.format(x, y[:y.rfind(' ')]) \
            for x, y in zip(track_ids, labels)]

        frame_visualizer.overlay_instances(
            boxes=None if masks is not None else boxes,  # boxes are a bit distracting
            masks=None if masks is None else masks,
            labels=labels,
            assigned_colors=colors,
            alpha=0.5,
        )

        return frame_visualizer.output

    def _assign_colors_by_track_id(self, instances):
        '''
        Allow duplicated colors
        '''
        colors = []
        for id_tensor in instances.track_ids:
            id = id_tensor.item()
            if id in self._assigned_colors:
                colors.append(self._color_pool[self._assigned_colors[id]])
            else:
                self.color_idx = (self.color_idx + 1) % self._num_colors
                color = self._color_pool[self.color_idx]
                self._assigned_colors[id] = self.color_idx
                colors.append(color)
        # print('self._assigned_colors', self._assigned_colors)
        return colors

class GTRPredictor(DefaultPredictor):
    @torch.no_grad()
    def __call__(self, original_frames):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
        """
        if self.input_format == "RGB":
            original_frames = \
                [x[:, :, ::-1] for x in original_frames]
        height, width = original_frames[0].shape[:2]
        frames = [self.aug.get_transform(x).apply_image(x) \
            for x in original_frames]
        frames = [torch.as_tensor(x.astype("float32").transpose(2, 0, 1))\
            for x in frames]
        inputs = [{"image": x, "height": height, "width": width, "video_id": 0} \
            for x in frames]
        predictions = self.model(inputs)
        return predictions


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.video_predictor = GTRPredictor(cfg)


    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break


    def _process_predictions(self, tracker_visualizer, frame, predictions):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        predictions = predictions["instances"].to(self.cpu_device)
        vis_frame = tracker_visualizer.draw_instance_predictions(
            frame, predictions)
        vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
        return vis_frame


    def run_on_video(self, video):
        """
        """
        tracker_visualizer = TrackingVisualizer(self.metadata, self.instance_mode)
        frames = [x for x in self._frame_from_video(video)]
        outputs = self.video_predictor(frames)
        for frame, instances in zip(frames, outputs):
            yield self._process_predictions(tracker_visualizer, frame, instances)


    def run_on_images(self, frames):
        """
        """
        tracker_visualizer = TrackingVisualizer(self.metadata, self.instance_mode)
        outputs = self.video_predictor(frames)
        for frame, instances in zip(frames, outputs):
            yield self._process_predictions(tracker_visualizer, frame, instances)