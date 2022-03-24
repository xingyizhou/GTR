# Copyright (c) Facebook, Inc. and its affiliates.
# Modified from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/meta_arch/rcnn.py
# The original file is under Apache-2.0 License

# Modified by Xingyi Zhou: support not_clamp_box

from typing import Dict, List, Optional, Tuple
import torch
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.structures import Instances

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.structures import Instances, ROIMasks

def custom_detector_postprocess(
    results: Instances, output_height: int, output_width: int, 
    mask_threshold: float = 0.5, not_clamp_box=False,
):
    """
    allow not clamp box for MOT datasets
    """
    # Change to 'if is_tracing' after PT1.7
    if isinstance(output_height, torch.Tensor):
        # Converts integer tensors to float temporaries to ensure true
        # division is performed when computing scale_x and scale_y.
        output_width_tmp = output_width.float()
        output_height_tmp = output_height.float()
        new_size = torch.stack([output_height, output_width])
    else:
        new_size = (output_height, output_width)
        output_width_tmp = output_width
        output_height_tmp = output_height

    scale_x, scale_y = (
        output_width_tmp / results.image_size[1],
        output_height_tmp / results.image_size[0],
    )
    results = Instances(new_size, **results.get_fields())

    if results.has("pred_boxes"):
        output_boxes = results.pred_boxes
    elif results.has("proposal_boxes"):
        output_boxes = results.proposal_boxes
    else:
        output_boxes = None
    assert output_boxes is not None, "Predictions must contain boxes!"

    output_boxes.scale(scale_x, scale_y)
    if not not_clamp_box:
        output_boxes.clip(results.image_size) # TODO (Xingyi): note modified

    results = results[output_boxes.nonempty()]

    if results.has("pred_masks"):
        if isinstance(results.pred_masks, ROIMasks):
            roi_masks = results.pred_masks
        else:
            # pred_masks is a tensor of shape (N, 1, M, M)
            roi_masks = ROIMasks(results.pred_masks[:, 0, :, :])
        results.pred_masks = roi_masks.to_bitmasks(
            results.pred_boxes, output_height, output_width, mask_threshold
        ).tensor  # TODO return ROIMasks/BitMask object in the future

    if results.has("pred_keypoints"):
        results.pred_keypoints[:, :, 0] *= scale_x
        results.pred_keypoints[:, :, 1] *= scale_y

    return results


@META_ARCH_REGISTRY.register()
class CustomRCNN(GeneralizedRCNN):
    '''
    Allow not clip box for MOT datasets
    '''
    @configurable
    def __init__(
        self, **kwargs):
        """
        add not_clamp_box
        """
        not_clamp_box = kwargs.pop('not_clamp_box', False)
        super().__init__(**kwargs)
        self.not_clamp_box = not_clamp_box

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['not_clamp_box'] = cfg.INPUT.NOT_CLAMP_BOX
        return ret

    def inference(
        self,
        batched_inputs: Tuple[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Allow not clamp box for MOT datasets
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(
                features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CustomRCNN._postprocess(
                results, batched_inputs, images.image_sizes,
                not_clamp_box=self.not_clamp_box)
        else:
            return results

    @staticmethod
    def _postprocess(instances, batched_inputs: Tuple[Dict[str, torch.Tensor]], 
        image_sizes, not_clamp_box=False):
        """
        Allow not clip box for MOT datasets
        """
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = custom_detector_postprocess(
                results_per_image, height, width, not_clamp_box=not_clamp_box)
            processed_results.append({"instances": r})
        return processed_results