import torch

def _freeze_except_roi_heads_id(model):
    for v in model.parameters():
        v.requires_grad = False
    try:
        for child in model.module.roi_heads.children():
            if child._get_name() == 'Sequential':
                continue
            print('unfreezing', child._get_name())
            for v in child.parameters():
                v.requires_grad = True
    except:
        for child in model.roi_heads.children():
            print('unfreezing', child._get_name())
            for v in child.parameters():
                v.requires_grad = True
    return model

def _freeze_except_roi_heads(model):
    for v in model.parameters():
        v.requires_grad = False
    try:
        for child in model.module.roi_heads.children():
            print('unfreezing', child._get_name())
            for v in child.parameters():
                v.requires_grad = True
    except:
        for child in model.roi_heads.children():
            print('unfreezing', child._get_name())
            for v in child.parameters():
                v.requires_grad = True
    return model

def _freeze_roi_heads(model):
    try:
        for child in model.module.roi_heads.children():
            for v in child.parameters():
                v.requires_grad = False
    except:
        for child in model.roi_heads.children():
            for v in child.parameters():
                v.requires_grad = False
    return model


def _freeze_backbonebottomup(model):
    try:
        for child in model.module.backbone.bottom_up.children():
            print('Freezing', child)
            for v in child.parameters():
                v.requires_grad = False
    except:
        for child in model.backbone.bottom_up.children():
            for v in child.parameters():
                v.requires_grad = False
    return model


def _freeze_backbone(model):
    try:
        for child in model.module.backbone.children():
            for v in child.parameters():
                v.requires_grad = False
    except:
        for child in model.backbone.children():
            for v in child.parameters():
                v.requires_grad = False
    return model

def _freeze_except_cascade_cls(model):
    for v in model.parameters():
        v.requires_grad = False

    for child in model.module.roi_heads.box_predictor.children():
        for v in child.cls_score.parameters():
            v.requires_grad = True

    return model

def _freeze_cls(model):
    for child in model.module.roi_heads.box_predictor.children():
        for v in child.cls_score.parameters():
            v.requires_grad = False

    return model

def _freeze_except_cascade_cls_centernet(model):
    for v in model.parameters():
        v.requires_grad = False

    for child in model.module.roi_heads.box_predictor.children():
        for v in child.cls_score.parameters():
            v.requires_grad = True
    
    try:
        print('unfreezing cls_logits')
        for v in model.module.proposal_generator.centernet_head.cls_logits.parameters():
            v.requires_grad = True
    except:
        print('unfreezing agn_gm')
        for v in model.module.proposal_generator.centernet_head.agn_hm.parameters():
            v.requires_grad = True
    return model

def _freeze_except_cascade_rpn_cls_reg(model):
    for v in model.parameters():
        v.requires_grad = False

    for child in model.module.roi_heads.box_predictor.children():
        for v in child.cls_score.parameters():
            v.requires_grad = True
        for v in child.bbox_pred.parameters():
            v.requires_grad = True

    print('unfreezing cls_logits')
    for v in model.module.proposal_generator.rpn_head.objectness_logits.parameters():
        v.requires_grad = True
    for v in model.module.proposal_generator.rpn_head.anchor_deltas.parameters():
        v.requires_grad = True
    return model

def _freeze_except_cascade_cls_reg(model):
    for v in model.parameters():
        v.requires_grad = False

    for child in model.module.roi_heads.box_predictor.children():
        for v in child.cls_score.parameters():
            v.requires_grad = True
        for v in child.bbox_pred.parameters():
            v.requires_grad = True

    return model

def check_if_freeze_model(model, cfg):
    if cfg.MODEL.FREEZE_TYPE == 'ExceptROIheads':
        print('Freezing  except ROI heads!')
        model = _freeze_except_roi_heads(model)
    elif cfg.MODEL.FREEZE_TYPE == 'ExceptROIheadsID':
        print('Freezing  except ROI heads ID!')
        model = _freeze_except_roi_heads_id(model)
    elif cfg.MODEL.FREEZE_TYPE == 'ExceptClassifier':
        print('Freezing  except cascade classification layers!')
        model = _freeze_except_cascade_cls(model)
    elif cfg.MODEL.FREEZE_TYPE == 'ExceptClassifierProposal':
        print('Freezing  except cascade classification layers!')
        model = _freeze_except_cascade_cls_centernet(model)
    elif cfg.MODEL.FREEZE_TYPE == 'ExceptClassifierRegression':
        print('Freezing except cascade classification/ reg/ RPN layers!')
        model = _freeze_except_cascade_cls_reg(model)
    elif cfg.MODEL.FREEZE_TYPE == 'ExceptClassifierProposalRegression':
        print('Freezing except cascade classification/ reg/ RPN layers!')
        model = _freeze_except_cascade_rpn_cls_reg(model)
    elif cfg.MODEL.FREEZE_TYPE == 'Classifier':
        print('Freezing Classifier')
        model = _freeze_cls(model)
    elif cfg.MODEL.FREEZE_TYPE == 'ROIheads':
        print('Freezing ROIheads')
        model = _freeze_roi_heads(model)
    elif cfg.MODEL.FREEZE_TYPE == 'BackboneBottomup':
        print('Freezing BackboneBottomup')
        model = _freeze_backbonebottomup(model)
    elif cfg.MODEL.FREEZE_TYPE == 'Backbone':
        print('Freezing Backbone')
        model = _freeze_backbone(model)
    else:
        assert cfg.MODEL.FREEZE_TYPE == '', cfg.MODEL.FREEZE_TYPE
    return model
