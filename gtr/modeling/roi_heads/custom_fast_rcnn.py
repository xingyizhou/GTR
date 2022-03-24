import math
from typing import List, Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.structures import Boxes, Instances
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.fast_rcnn import _log_classification_stats
from centernet.modeling.roi_heads.fed_loss import load_class_freq, get_fed_loss_inds

__all__ = ["CustomFastRCNNOutputLayers"]

class CustomFastRCNNOutputLayers(FastRCNNOutputLayers):
    def __init__(
        self, 
        cfg, 
        input_shape: ShapeSpec,
        **kwargs
    ):
        '''
        Support not_clamp_box
        '''
        super().__init__(cfg, input_shape, **kwargs)
        self.use_sigmoid_ce = cfg.MODEL.ROI_BOX_HEAD.USE_SIGMOID_CE
        self.not_clamp_box = cfg.INPUT.NOT_CLAMP_BOX
        self.mult_proposal_score = cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE

        if self.use_sigmoid_ce:
            prior_prob = cfg.MODEL.ROI_BOX_HEAD.PRIOR_PROB
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            nn.init.constant_(self.cls_score.bias, bias_value)

        self.use_fed_loss = cfg.MODEL.ROI_BOX_HEAD.USE_FED_LOSS
        if self.use_fed_loss:
            self.fed_loss_num_cat = cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_NUM_CAT
            self.register_buffer(
                'freq_weight', 
                load_class_freq(
                    cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH, 
                    cfg.MODEL.ROI_BOX_HEAD.FED_LOSS_FREQ_WEIGHT,
                )
            )
    
    def losses(self, predictions, proposals):
        """
        enable advanced loss
        """
        scores, proposal_deltas = predictions
        gt_classes = (
            cat([p.gt_classes for p in proposals], dim=0) if len(proposals) else torch.empty(0)
        )
        _log_classification_stats(scores, gt_classes)

        if len(proposals):
            proposal_boxes = cat([p.proposal_boxes.tensor for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            gt_boxes = cat(
                [(p.gt_boxes if p.has("gt_boxes") else p.proposal_boxes).tensor for p in proposals],
                dim=0,
            )
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)

        if self.use_sigmoid_ce:
            loss_cls = self.sigmoid_cross_entropy_loss(scores, gt_classes)
        else:
            loss_cls = self.softmax_cross_entropy_loss(scores, gt_classes)
        return {
            "loss_cls": loss_cls, 
            "loss_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes)
        }


    def sigmoid_cross_entropy_loss(self, pred_class_logits, gt_classes):
        if pred_class_logits.numel() == 0:
            return pred_class_logits.new_zeros([1])[0] # This is more robust than .sum() * 0.

        B = pred_class_logits.shape[0]
        C = pred_class_logits.shape[1] - 1

        target = pred_class_logits.new_zeros(B, C + 1)
        target[range(len(gt_classes)), gt_classes] = 1 # B x (C + 1)
        target = target[:, :C] # B x C

        weight = 1
        if self.use_fed_loss and (self.freq_weight is not None): # fedloss
            appeared = get_fed_loss_inds(
                gt_classes, 
                num_sample_cats=self.fed_loss_num_cat,
                C=C,
                weight=self.freq_weight)
            appeared_mask = appeared.new_zeros(C + 1)
            appeared_mask[appeared] = 1 # C + 1
            appeared_mask = appeared_mask[:C]
            fed_w = appeared_mask.view(1, C).expand(B, C)
            weight = weight * fed_w.float()

        cls_loss = F.binary_cross_entropy_with_logits(
            pred_class_logits[:, :-1], target, reduction='none') # B x C
        loss =  torch.sum(cls_loss * weight) / B  
        return loss


    def predict_probs(self, predictions, proposals):
        """
        support sigmoid
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.use_sigmoid_ce:
            probs = scores.sigmoid()
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)


    def inference(self, predictions, proposals):
        """
        add tracking outputs and not clamp box
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        if self.mult_proposal_score:
            proposal_scores = [p.get('objectness_logits') for p in proposals]
            scores = [(s * ps[:, None]) ** 0.5 \
                for s, ps in zip(scores, proposal_scores)]
        image_shapes = [x.image_size for x in proposals]
        track_ids = [x.track_id if x.has('track_id') else None \
            for x in proposals]
        reid_features = [x.reid_features if x.has('reid_features') else None \
            for x in proposals]
        return custom_fast_rcnn_inference(
            boxes,
            scores,
            track_ids,
            reid_features,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.not_clamp_box,
        )

def custom_fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    track_ids: List[torch.Tensor],
    reid_features: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    not_clamp_box: bool,
):
    """
    allow not clamp box
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, track_ids_image, reid_features_image, \
                image_shape, score_thresh, nms_thresh, topk_per_image, not_clamp_box,
        )
        for scores_per_image, boxes_per_image, track_ids_image, reid_features_image, image_shape \
            in zip(scores, boxes, track_ids, reid_features, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
    boxes,
    scores,
    track_ids,
    reid_features,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    not_clamp_box: bool,
):
    """
    not clip box and
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    if not not_clamp_box:
        boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    if track_ids is not None:
        result.track_ids = track_ids[filter_inds[:, 0]]
    if reid_features is not None:
        result.reid_features = reid_features[filter_inds[:, 0]]
    return result, filter_inds[:, 0]
