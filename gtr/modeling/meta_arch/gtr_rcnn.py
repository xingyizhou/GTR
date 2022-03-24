import cv2
import torch
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.structures import Boxes, pairwise_iou, Instances

from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from .custom_rcnn import CustomRCNN
from ..roi_heads.custom_fast_rcnn import custom_fast_rcnn_inference

@META_ARCH_REGISTRY.register()
class GTRRCNN(CustomRCNN):
    @configurable
    def __init__(self, **kwargs):
        """
        """
        self.test_len = kwargs.pop('test_len')
        self.overlap_thresh = kwargs.pop('overlap_thresh')
        self.min_track_len = kwargs.pop('min_track_len')
        self.max_center_dist = kwargs.pop('max_center_dist')
        self.decay_time = kwargs.pop('decay_time')
        self.asso_thresh = kwargs.pop('asso_thresh')
        self.with_iou = kwargs.pop('with_iou')
        self.local_track = kwargs.pop('local_track')
        self.local_no_iou = kwargs.pop('local_no_iou')
        self.local_iou_only = kwargs.pop('local_iou_only')
        self.not_mult_thresh = kwargs.pop('not_mult_thresh')
        super().__init__(**kwargs)


    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        ret['test_len'] = cfg.INPUT.VIDEO.TEST_LEN
        ret['overlap_thresh'] = cfg.VIDEO_TEST.OVERLAP_THRESH     
        ret['asso_thresh'] = cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        ret['min_track_len'] = cfg.VIDEO_TEST.MIN_TRACK_LEN
        ret['max_center_dist'] = cfg.VIDEO_TEST.MAX_CENTER_DIST
        ret['decay_time'] = cfg.VIDEO_TEST.DECAY_TIME
        ret['with_iou'] = cfg.VIDEO_TEST.WITH_IOU
        ret['local_track'] = cfg.VIDEO_TEST.LOCAL_TRACK
        ret['local_no_iou'] = cfg.VIDEO_TEST.LOCAL_NO_IOU
        ret['local_iou_only'] = cfg.VIDEO_TEST.LOCAL_IOU_ONLY
        ret['not_mult_thresh'] = cfg.VIDEO_TEST.NOT_MULT_THRESH
        return ret


    def forward(self, batched_inputs):
        """
        All batched images are from the same video
        During testing, the current implementation requires all frames 
            in a video are loaded.
        TODO (Xingyi): one-the-fly testing
        """
        if not self.training:
            if self.local_track:
                return self.local_tracker_inference(batched_inputs)
            else:
                return self.sliding_inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances)
        _, detector_losses = self.roi_heads(
            images, features, proposals, gt_instances)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses


    def sliding_inference(self, batched_inputs):
        video_len = len(batched_inputs)
        instances = []
        id_count = 0
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False)
            instances.extend([x for x in instances_wo_id])

            if frame_id == 0: # first frame
                instances[0].track_ids = torch.arange(
                    1, len(instances[0]) + 1,
                    device=instances[0].reid_features.device)
                id_count = len(instances[0]) + 1
            else:
                win_st = max(0, frame_id + 1 - self.test_len)
                win_ed = frame_id + 1
                instances[win_st: win_ed], id_count = self.run_global_tracker(
                    batched_inputs[win_st: win_ed],
                    instances[win_st: win_ed],
                    k=min(self.test_len - 1, frame_id),
                    id_count=id_count) # n_k x N
            if frame_id - self.test_len >= 0:
                instances[frame_id - self.test_len].remove(
                    'reid_features')

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances


    def run_global_tracker(self, batched_inputs, instances, k, id_count):
        n_t = [len(x) for x in instances]
        N, T = sum(n_t), len(n_t)

        reid_features = torch.cat(
                [x.reid_features for x in instances], dim=0)[None]
        asso_output, pred_boxes, _, _ = self.roi_heads._forward_transformer(
            instances, reid_features, k) # [n_k x N], N x 4

        asso_output = asso_output[-1].split(n_t, dim=1) # T x [n_k x n_t]
        asso_output = self.roi_heads._activate_asso(asso_output) # T x [n_k x n_t]
        asso_output = torch.cat(asso_output, dim=1) # n_k x N

        n_k = len(instances[k])
        Np = N - n_k
        ids = torch.cat(
            [x.track_ids for t, x in enumerate(instances) if t != k],
            dim=0).view(Np) # Np
        k_inds = [x for x in range(sum(n_t[:k]), sum(n_t[:k + 1]))]
        nonk_inds = [i for i in range(N) if not i in k_inds]
        asso_nonk = asso_output[:, nonk_inds] # n_k x Np
        k_boxes = pred_boxes[k_inds] # n_k x 4
        nonk_boxes = pred_boxes[nonk_inds] # Np x 4
        
        if self.roi_heads.delay_cls:
            # filter based on classification score similarity
            cls_scores = torch.cat(
                [x.cls_scores for x in instances], dim=0)[:, :-1] # N x (C + 1)
            cls_scores_k = cls_scores[k_inds] # n_k x (C + 1)
            cls_scores_nonk = cls_scores[nonk_inds] # Np x (C + 1)
            cls_similarity = torch.mm(
                cls_scores_k, cls_scores_nonk.permute(1, 0)) # n_k x Np
            asso_nonk[cls_similarity < 0.01] = 0

        unique_ids = torch.unique(ids) # M
        M = len(unique_ids) # number of existing tracks
        id_inds = (unique_ids[None, :] == ids[:, None]).float() # Np x M

        # (n_k x Np) x (Np x M) --> n_k x M
        if self.decay_time > 0:
            # (n_k x Np) x (Np x M) --> n_k x M
            dts = torch.cat([x.reid_features.new_full((len(x),), T - t - 2) \
                for t, x in enumerate(instances) if t != k], dim=0) # Np
            asso_nonk = asso_nonk * (self.decay_time ** dts[None, :])

        traj_score = torch.mm(asso_nonk, id_inds) # n_k x M
        if id_inds.numel() > 0:
            last_inds = (id_inds * torch.arange(
                Np, device=id_inds.device)[:, None]).max(dim=0)[1] # M
            last_boxes = nonk_boxes[last_inds] # M x 4
            last_ious = pairwise_iou(
                Boxes(k_boxes), Boxes(last_boxes)) # n_k x M
        else:
            last_ious = traj_score.new_zeros(traj_score.shape)
        
        if self.with_iou:
            traj_score = torch.max(traj_score, last_ious)
        
        if self.max_center_dist > 0.: # filter out too far-away trjactories
            # traj_score n_k x M
            k_boxes = pred_boxes[k_inds] # n_k x 4
            nonk_boxes = pred_boxes[nonk_inds] # Np x 4
            k_ct = (k_boxes[:, :2] + k_boxes[:, 2:]) / 2
            k_s = ((k_boxes[:, 2:] - k_boxes[:, :2]) ** 2).sum(dim=1) # n_k
            nonk_ct = (nonk_boxes[:, :2] + nonk_boxes[:, 2:]) / 2
            dist = ((k_ct[:, None] - nonk_ct[None, :]) ** 2).sum(dim=2) # n_k x Np
            norm_dist = dist / (k_s[:, None] + 1e-8) # n_k x Np
            # id_inds # Np x M
            valid = norm_dist < self.max_center_dist # n_k x Np
            valid_assn = torch.mm(
                valid.float(), id_inds).clamp_(max=1.).long().bool() # n_k x M
            traj_score[~valid_assn] = 0 # n_k x M

        match_i, match_j = linear_sum_assignment((- traj_score).cpu()) #
        track_ids = ids.new_full((n_k,), -1)
        for i, j in zip(match_i, match_j):
            thresh = self.overlap_thresh * id_inds[:, j].sum() \
                if not (self.not_mult_thresh) else self.overlap_thresh
            if traj_score[i, j] > thresh:
                track_ids[i] = unique_ids[j]

        for i in range(n_k):
            if track_ids[i] < 0:
                id_count = id_count + 1
                track_ids[i] = id_count
        instances[k].track_ids = track_ids

        assert len(track_ids) == len(torch.unique(track_ids)), track_ids
        return instances, id_count


    def _remove_short_track(self, instances):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        num_insts_track = id_inds.sum(dim=1) # M
        remove_track_id = num_insts_track < self.min_track_len # M
        unique_ids[remove_track_id] = -1
        ids = unique_ids[torch.where(id_inds.permute(1, 0))[1]]
        ids = ids.split([len(x) for x in instances])
        for k in range(len(instances)):
            instances[k] = instances[k][ids[k] >= 0]
        return instances


    def _delay_cls(self, instances, video_id):
        ids = torch.cat([x.track_ids for x in instances], dim=0) # N
        unique_ids = ids.unique() # M
        M = len(unique_ids) # #existing tracks
        id_inds = (unique_ids[:, None] == ids[None, :]).float() # M x N
        # update scores
        cls_scores = torch.cat(
            [x.cls_scores for x in instances], dim=0) # N x (C + 1)
        traj_scores = torch.mm(id_inds, cls_scores) / \
            (id_inds.sum(dim=1)[:, None] + 1e-8) # M x (C + 1)
        _, traj_inds = torch.where(id_inds.permute(1, 0)) # N
        cls_scores = traj_scores[traj_inds] # N x (C + 1)

        n_t = [len(x) for x in instances]
        boxes = [x.pred_boxes.tensor for x in instances]
        track_ids = ids.split(n_t, dim=0)
        cls_scores = cls_scores.split(n_t, dim=0)
        instances, _ = custom_fast_rcnn_inference(
            boxes, cls_scores, track_ids, [None for _ in n_t],
            [x.image_size for x in instances],
            self.roi_heads.box_predictor[-1].test_score_thresh,
            self.roi_heads.box_predictor[-1].test_nms_thresh,
            self.roi_heads.box_predictor[-1].test_topk_per_image,
            self.not_clamp_box,
        )
        for inst in instances:
            inst.track_ids = inst.track_ids + inst.pred_classes * 10000 + \
                video_id * 100000000
        return instances

    def local_tracker_inference(self, batched_inputs):
        from ...tracking.local_tracker.fairmot import FairMOT
        local_tracker = FairMOT(
            no_iou=self.local_no_iou,
            iou_only=self.local_iou_only)

        video_len = len(batched_inputs)
        instances = []
        ret_instances = []
        for frame_id in range(video_len):
            instances_wo_id = self.inference(
                batched_inputs[frame_id: frame_id + 1], 
                do_postprocess=False)
            instances.extend([x for x in instances_wo_id])
            inst = instances[frame_id]
            dets = torch.cat([
                inst.pred_boxes.tensor, 
                inst.scores[:, None]], dim=1).cpu()
            id_feature = inst.reid_features.cpu()
            tracks = local_tracker.update(dets, id_feature)
            track_inds = [x.ind for x in tracks]
            ret_inst = inst[track_inds]
            track_ids = [x.track_id for x in tracks]
            ret_inst.track_ids = ret_inst.pred_classes.new_tensor(track_ids)
            ret_instances.append(ret_inst)
        instances = ret_instances

        if self.min_track_len > 0:
            instances = self._remove_short_track(instances)
        if self.roi_heads.delay_cls:
            instances = self._delay_cls(
                instances, video_id=batched_inputs[0]['video_id'])
        instances = CustomRCNN._postprocess(
                instances, batched_inputs, [
                    (0, 0) for _ in range(len(batched_inputs))],
                not_clamp_box=self.not_clamp_box)
        return instances