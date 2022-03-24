import copy
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from detectron2.modeling.poolers import ROIPooler
from .custom_fast_rcnn import CustomFastRCNNOutputLayers, custom_fast_rcnn_inference
from .association_head import ATTWeightHead, FCHead
from .transformer import Transformer

@ROI_HEADS_REGISTRY.register()
class GTRROIHeads(CascadeROIHeads):
    @configurable
    def __init__(self, **kwargs):
        '''
        TODO (Xingyi): refactor cfg
        '''
        cfg = kwargs.pop('cfg', None)
        input_shape = kwargs.pop('input_shape', None)
        super().__init__(**kwargs)
        if cfg is None:
            return

        self.no_box_head = cfg.MODEL.ROI_HEADS.NO_BOX_HEAD
        if self.no_box_head:
            del self.box_predictor
            del self.box_pooler
            del self.box_head

        assert not cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT
        self.asso_on = cfg.MODEL.ASSO_ON
        if self.asso_on:
            self._init_asso_head(cfg, input_shape)


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret['cfg'] = cfg
        ret['input_shape'] = input_shape
        return ret


    def _init_asso_head(self, cfg, input_shape):
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        self.asso_in_features = in_features
        self.feature_dim = cfg.MODEL.ASSO_HEAD.FC_DIM
        self.num_fc = cfg.MODEL.ASSO_HEAD.NUM_FC
        self.asso_thresh_train = cfg.MODEL.ASSO_HEAD.ASSO_THRESH
        self.asso_thresh_test = cfg.MODEL.ASSO_HEAD.ASSO_THRESH_TEST
        self.asso_weight = cfg.MODEL.ASSO_HEAD.ASSO_WEIGHT
        self.neg_unmatched = cfg.MODEL.ASSO_HEAD.NEG_UNMATCHED
        self.with_temp_emb = cfg.MODEL.ASSO_HEAD.WITH_TEMP_EMB
        self.no_pos_emb = cfg.MODEL.ASSO_HEAD.NO_POS_EMB
        
        self.asso_thresh_test = self.asso_thresh_test \
            if self.asso_thresh_test > 0 else self.asso_thresh_train

        num_encoder_layers = cfg.MODEL.ASSO_HEAD.NUM_ENCODER_LAYERS
        num_decoder_layers = cfg.MODEL.ASSO_HEAD.NUM_DECODER_LAYERS
        num_heads = cfg.MODEL.ASSO_HEAD.NUM_HEADS
        dropout = cfg.MODEL.ASSO_HEAD.DROPOUT
        norm = cfg.MODEL.ASSO_HEAD.NORM
        num_weight_layers = cfg.MODEL.ASSO_HEAD.NUM_WEIGHT_LAYERS
        no_decoder_self_att = cfg.MODEL.ASSO_HEAD.NO_DECODER_SELF_ATT
        no_encoder_self_att = cfg.MODEL.ASSO_HEAD.NO_ENCODER_SELF_ATT

        asso_in_channels = input_shape[in_features[0]].channels
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.asso_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.asso_head = FCHead(
            input_shape=ShapeSpec(
                channels=asso_in_channels, 
                height=pooler_resolution, width=pooler_resolution),
            fc_dim=self.feature_dim,
            num_fc=self.num_fc,
        )

        self.asso_predictor = ATTWeightHead(
            self.feature_dim, num_layers=num_weight_layers, dropout=dropout)

        self.transformer = Transformer(
            d_model=self.feature_dim, 
            nhead=num_heads, 
            num_encoder_layers=num_encoder_layers, 
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=self.feature_dim, 
            dropout=dropout, 
            return_intermediate_dec=True,
            norm=norm,
            no_decoder_self_att=no_decoder_self_att,
            no_encoder_self_att=no_encoder_self_att
        )
        
        if not self.no_pos_emb:
            self.learn_pos_emb_num = 16
            self.pos_emb = nn.Embedding(
                self.learn_pos_emb_num * 4, self.feature_dim // 4)
            if self.with_temp_emb:
                self.learn_temp_emb_num = 16
                self.temp_emb = nn.Embedding(
                    self.learn_temp_emb_num, self.feature_dim)


    def _forward_asso(self, features, instances, targets=None):
        """
        """
        if not self.asso_on:
            return {} if self.training else instances
        asso_thresh = self.asso_thresh_train if self.training \
            else self.asso_thresh_test
        fg_inds = [
            x.objectness_logits > asso_thresh for x in instances]
        proposals = [x[inds] for (x, inds) in zip(instances, fg_inds)]
        features = [features[f] for f in self.asso_in_features]
        proposal_boxes = [x.proposal_boxes for x in proposals] # 
        pool_features = self.asso_pooler(features, proposal_boxes)
        reid_features = self.asso_head(pool_features)
        reid_features = reid_features.view(
            1, -1, self.feature_dim) # 1 x N x F
        n_t = [len(x) for x in proposals]
        if not self.training: # delay transformer
            instances = [inst[inds] for inst, inds in zip(instances, fg_inds)]
            features = reid_features.view(-1, self.feature_dim).split(n_t, dim=0)
            for inst, feat in zip(instances, features):
                inst.reid_features = feat
            return instances
        else:
            asso_outputs, pred_box, pred_time, query_inds = \
                self._forward_transformer(proposals, reid_features)
            assert len(proposals) == len(targets)
            target_box, target_time = self._get_boxes_time(targets) # G x 4
            if sum(len(x) for x in targets) == 0 or \
                max(x.gt_instance_ids.max().item() for x in targets if len(x) > 0) == 0:
                asso_loss = features[0].new_zeros((1,), dtype=torch.float32)[0]
                return {'loss_asso': asso_loss}
            target_inst_id = torch.cat(
                [x.gt_instance_ids for x in targets if len(x) > 0])
            asso_gt, match_cues = self._get_asso_gt(
                pred_box, pred_time, target_box, target_time, 
                target_inst_id, n_t) # K x N, 
            asso_loss = 0
            for x in asso_outputs:
                asso_loss += self.detr_asso_loss(x, asso_gt, match_cues, n_t)
            return {'loss_asso': self.asso_weight * asso_loss}


    def _forward_transformer(self, proposals, reid_features, query_frame=None):
        T = len(proposals)
        n_t = [len(x) for x in proposals]
        pred_box, pred_time = self._get_boxes_time(proposals) # N x 4
        N = sum(n_t)
        D = self.feature_dim
        if self.no_pos_emb:
            pos_emb = None
        else:
            pos_emb = self._box_pe(pred_box) # N x F
            if self.with_temp_emb:
                temp_emb = self._temp_pe(pred_time.clone().float() / T)
                pos_emb = (pos_emb + temp_emb) / 2.
            pos_emb = pos_emb.view(1, N, D)

        query = None
        query_inds = None
        M = N
        if query_frame is not None:
            c = query_frame
            query_inds = [x for x in range(sum(n_t[:c]), sum(n_t[:c + 1]))]
            M = len(query_inds)

        feats, memory = self.transformer(
            reid_features, pos_embed=pos_emb, query_embed=query,
            query_inds=query_inds)
        # feats: L x [1 x M x F], memory: 1 x N x F
        asso_outputs = [self.asso_predictor(x, memory).view(M, N) \
            for x in feats] # L x [M x N]
        return asso_outputs, pred_box, pred_time, query_inds

    
    def _activate_asso(self, asso_output):
        asso_active = []
        for asso in asso_output:
            # asso: M x n_t
            asso = torch.cat(
                [asso, asso.new_zeros((asso.shape[0], 1))], dim=1).softmax(
                    dim=1)[:, :-1]
            asso_active.append(asso)
        return asso_active
    

    def _get_asso_gt(self, pred_box, pred_time, \
        target_box, target_time, target_inst_id, n_t):
        '''
        Inputs:
            pred_box: N x 4
            pred_time: N
            targer_box: G x 4
            targer_time: G
            target_inst_id: G
            K: len(unique(target_inst_id))
        Return:
            ret: K x N or K x T
            match_cues: K x 3 or N
        '''
        ious = pairwise_iou(Boxes(pred_box), Boxes(target_box)) # N x G
        ious[pred_time[:, None] != target_time[None, :]] = -1.
        inst_ids = torch.unique(target_inst_id[target_inst_id > 0])
        K, N = len(inst_ids), len(pred_box)
        match_cues = pred_box.new_full((N,), -1, dtype=torch.long)

        T = len(n_t)

        ret = pred_box.new_zeros((K, T), dtype=torch.long)
        ious_per_frame = ious.split(n_t, dim=0) # T x [n_t x G]
        for k, inst_id in enumerate(inst_ids):
            target_inds = target_inst_id == inst_id # G
            base_ind = 0
            for t in range(T):
                iou_t = ious_per_frame[t][:, target_inds] # n_t x gk
                if iou_t.numel() == 0:
                    ret[k, t] = n_t[t]
                else:
                    val, inds = iou_t.max(dim=0) # n_t x gk --> gk
                    ind = inds[val > 0.0]
                    assert (len(ind) <= 1), '{} {}'.format(
                        target_inst_id, n_t)
                    if len(ind) == 1:
                        obj_ind = ind[0].item()
                        ret[k, t] = obj_ind
                        match_cues[base_ind + obj_ind] = k
                    else:
                        ret[k, t] = n_t[t]
                base_ind += n_t[t]

        return ret, match_cues


    def detr_asso_loss(self, asso_pred, asso_gt, match_cues, n_t):
        '''
        Inputs:
            asso_pred: M x N
            asso_gt: K x N or K x T
            n_t: T (list of int)
        Return:
            float
        '''
        src_inds, target_inds = self._match(
            asso_pred, asso_gt, match_cues, n_t)

        loss = 0
        num_objs = 0
        zero = asso_pred.new_zeros((asso_pred.shape[0], 1)) # M x 1
        asso_pred_image = asso_pred.split(n_t, dim=1) # T x [M x n_t]
        for t in range(len(n_t)):
            asso_pred_with_bg = torch.cat(
                [asso_pred_image[t], zero], dim=1) # M x (n_t + 1)
            if self.neg_unmatched:
                asso_gt_t = asso_gt.new_full(
                    (asso_pred.shape[0],), n_t[t]) # M
                asso_gt_t[src_inds] = asso_gt[target_inds, t] # M
            else:
                asso_pred_with_bg = asso_pred_with_bg[src_inds] # K x (n_t + 1)
                asso_gt_t = asso_gt[target_inds, t] # K
            num_objs += (asso_gt_t != n_t[t]).float().sum()
            loss += F.cross_entropy(
                asso_pred_with_bg, asso_gt_t, reduction='none')
        return loss.sum() / (num_objs + 1e-4)


    @torch.no_grad()
    def _match(self, asso_pred, asso_gt, match_cues, n_t):
        '''
        Inputs:
            asso_pred: M x N
            asso_gt: K x N or K x T
            match_cues: K x 3 or N
        Return:
            indices: 
        '''
        src_inds = torch.where(match_cues >= 0)[0]
        target_inds = match_cues[src_inds]
        return (src_inds, target_inds)


    def _get_boxes_time(self, instances):
        boxes, times = [], []
        for t, p in enumerate(instances):
            h, w = p._image_size
            if p.has('proposal_boxes'):
                p_boxes = p.proposal_boxes.tensor.clone()
            elif p.has('pred_boxes'):
                p_boxes = p.pred_boxes.tensor.clone()
            else:
                p_boxes = p.gt_boxes.tensor.clone()
            p_boxes[:, [0, 2]] /= w
            p_boxes[:, [1, 3]] /= h
            boxes.append(p_boxes) # ni x 4
            times.append(p_boxes.new_full(
                (p_boxes.shape[0],), t, dtype=torch.long))
        boxes = torch.cat(boxes, dim=0) # N x 4
        times = torch.cat(times, dim=0) # N
        return boxes.detach(), times.detach()


    def _box_pe(self, boxes):
        '''
        '''
        N = boxes.shape[0]
        boxes = boxes.view(N, 4)
        xywh = torch.cat([
            (boxes[:, 2:] + boxes[:, :2]) / 2, 
            (boxes[:, 2:] - boxes[:, :2])], dim=1)
        xywh = xywh * self.learn_pos_emb_num
        l = xywh.clamp(min=0, max=self.learn_pos_emb_num - 1).long() # N x 4
        r = (l + 1).clamp(min=0, max=self.learn_pos_emb_num - 1).long() # N x 4
        lw = (xywh - l.float()) # N x 4
        rw = 1. - lw
        f = self.pos_emb.weight.shape[1]
        pos_emb_table = self.pos_emb.weight.view(
            self.learn_pos_emb_num, 4, f) # T x 4 x (F // 4)
        pos_le = pos_emb_table.gather(0, l[:, :, None].expand(N, 4, f)) # N x 4 x f 
        pos_re = pos_emb_table.gather(0, r[:, :, None].expand(N, 4, f)) # N x 4 x f
        pos_emb = lw[:, :, None] * pos_re + rw[:, :, None] * pos_le
        return pos_emb.view(N, 4 * f)


    def _temp_pe(self, temps):
        '''
        '''
        N = temps.shape[0]
        temps = temps * self.learn_temp_emb_num
        l = temps.clamp(min=0, max=self.learn_temp_emb_num - 1).long() # N x 4
        r = (l + 1).clamp(min=0, max=self.learn_temp_emb_num - 1).long() # N x 4
        lw = (temps - l.float()) # N
        rw = 1. - lw
        le = self.temp_emb.weight[l] # T x F --> N x F
        re = self.temp_emb.weight[r] # N x F
        temp_emb = lw[:, None] * re + rw[:, None] * le
        return temp_emb.view(N, self.feature_dim)


    @classmethod
    def _init_box_head(self, cfg, input_shape):
        self.mult_proposal_score = cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE
        self.not_clamp_box = cfg.INPUT.NOT_CLAMP_BOX
        self.delay_cls = cfg.MODEL.ROI_BOX_HEAD.DELAY_CLS
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], cascade_bbox_reg_weights):
            box_predictors.append(
                CustomFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        ret['box_predictors'] = box_predictors
        return ret


    def _forward_box(self, features, proposals, targets=None):
        """
        Add mult proposal scores at testing
        """
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [
                    p.get('scores') for p in proposals]
            else:
                proposal_scores = [
                    p.get('objectness_logits') for p in proposals]
        has_track_id = proposals[0].has('track_ids')
        has_reid_features = proposals[0].has('reid_features')
        if has_track_id or has_reid_features or self.delay_cls:
            ori_proposals = copy.deepcopy(proposals)
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]


            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            if self.delay_cls:
                pred_instances = ori_proposals
                for score, box, inst in zip(scores, boxes, pred_instances):
                    inst.cls_scores = score
                    inst.pred_boxes = Boxes(box)
                    inst.remove('proposal_boxes')
                return pred_instances

            if has_track_id:
                track_ids = [x.track_ids for x in ori_proposals]
                has_track_score = ori_proposals[0].has('track_score') 
                if has_track_score:
                    M = ori_proposals[0].track_score.shape[1]
                    C = scores[0].shape[1] - 1
                    pred_probs_video = ori_proposals[0].track_score.new_zeros(
                        (M, C + 1))
                    norm_score = pred_probs_video.new_zeros((M, 1))
                    for score, p in zip(scores, ori_proposals):
                        # p.track_score: n_t x M
                        # score: n_t x (C + 1)
                        pred_probs_video += (p.track_score[:, :, None] * \
                            score[:, None, :]).sum(dim=0) # M x (C + 1)
                        norm_score += p.track_score.sum(dim=0)[:, None] # M x 1
                    pred_probs_video = pred_probs_video / (norm_score + 1e-8) # M x (C + 1)
                    new_scores = []
                    for score, p in zip(scores, ori_proposals):
                        # p.track_score: n_t x M
                        new_score = (p.track_score[:, :, None] * \
                            pred_probs_video[None, :, :]) ** 0.5 # n_t x M x (C + 1)
                        if new_score.numel() > 0:
                            new_score, ids = new_score.max(dim=1) # n_t x (C + 1)
                        else:
                            new_score = new_score.new_zeros(
                                (new_score.shape[0], C + 1))
                        new_scores.append(new_score) # n_t x (C + 1)
                    scores = new_scores
            else:
                track_ids = [None for _ in proposals]
            if has_reid_features: # TODO (Xingyi): reimplement
                reid_features = [x.reid_features for x in ori_proposals]
            else:
                reid_features = [None for _ in proposals]
            pred_instances, _ = custom_fast_rcnn_inference(
                boxes,
                scores,
                track_ids,
                reid_features,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
                self.not_clamp_box,
            )
            return pred_instances

    def forward(self, images, features, proposals, targets=None):
        """
        enable reid head
        enable association
        """
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            if self.no_box_head:
                losses = {}
            else:
                losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_asso(features, proposals, targets))
            return proposals, losses
        else:
            # During testing, forward association head first to filter out
            #   background proposals and get reid features for delay classification
            proposals = self._forward_asso(features, proposals)
            if self.no_box_head:
                pred_instances = proposals
                for p in pred_instances:
                    p.pred_boxes = p.proposal_boxes
                    p.scores = p.objectness_logits
                    p.pred_classes = torch.zeros(
                        (len(p),), dtype=torch.long, device=p.pred_boxes.device)
                    p.remove('proposal_boxes')
                    p.remove('objectness_logits')
            else:
                pred_instances = self._forward_box(features, proposals)
            if not self.delay_cls:
                pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}
