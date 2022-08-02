import sys
import tempfile
import numpy as np
import cv2
import imageio

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from cog import BasePredictor, Path, Input

# GTR libraries
sys.path.insert(0, "third_party/CenterNet2/")
from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
from gtr.predictor import GTRPredictor, TrackingVisualizer


class Predictor(BasePredictor):
    def setup(self):
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_gtr_config(cfg)
        cfg.merge_from_file("configs/GTR_TAO_DR2101.yaml")
        cfg.MODEL.WEIGHTS = "models/GTR_TAO_DR2101.pth"
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model

        metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.predictor = GTRPredictor(cfg)
        self.tracker_visualizer = TrackingVisualizer(metadata)

    def predict(
        self,
        video: Path = Path(
            description="Input video.",
        ),
    ) -> Path:
        # Load images from video
        video = cv2.VideoCapture(str(video))
        fps = video.get(cv2.CAP_PROP_FPS)

        frames = [x for x in frame_from_video(video)]
        video.release()

        # Run model
        outputs = self.predictor(frames)

        out_frames = []
        for frame, instances in zip(frames, outputs):
            out_frame = process_predictions(self.tracker_visualizer, frame, instances)
            out_frames.append(out_frame)

        output_path = Path(tempfile.mkdtemp()) / "output.mp4"
        imageio.mimwrite(str(output_path), [x[..., ::-1] for x in out_frames], fps=fps)

        return output_path


def frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def process_predictions(tracker_visualizer, frame, predictions):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    predictions = predictions["instances"].to("cpu")
    vis_frame = tracker_visualizer.draw_instance_predictions(frame, predictions)
    vis_frame = cv2.cvtColor(vis_frame.get_image(), cv2.COLOR_RGB2BGR)
    return vis_frame
