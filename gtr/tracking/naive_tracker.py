from collections import defaultdict
import json
import argparse
import torch
import cv2
import numpy as np
import os
from scipy.optimize import linear_sum_assignment

def pairwise_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).prod(dim=1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).prod(dim=1)

    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 0.99:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)

def associate(boxes, pre_boxes, pre_ids, iou_thresh):
    if len(pre_boxes) == 0 or len(boxes) == 0:
        return torch.zeros(len(boxes), dtype=torch.long) - 1
    ious = pairwise_iou(boxes, pre_boxes) # n x m
    # best_ious, inds = ious.max(dim=1) # n
    # ids = pre_ids[inds] # n
    # ids[best_ious < iou_thresh] = -1 # n
    # match = greedy_assignment(1. - ious)
    # match = linear_assignment(1. - ious)
    match_i, match_j = linear_sum_assignment(1. - ious)
    ids = torch.full((len(boxes),), -1, dtype=torch.long)
    for i, j in zip(match_i, match_j):
        if ious[i, j] > iou_thresh:
            ids[i] = pre_ids[j]
    return ids


def dict2tensors(frame):
    if len(frame) == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
    else:
        boxes = torch.tensor([x['bbox'] for x in frame], dtype=torch.float32)
        boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:]
    return boxes

def track(dets, iou_thresh=0.01):
    '''
    dets: list of lists of detection dict
    '''
    # initialize the first frame
    pre_boxes = dict2tensors(dets[0])
    count = 0
    for x in dets[0]:
        count = count + 1
        x['track_id'] = count
    pre_ids = torch.tensor([x['track_id'] for x in dets[0]], dtype=torch.long)
    for frame in dets[1:]:
        n = len(frame)
        if n == 0:
            # TODO
            continue
        boxes = dict2tensors(frame)
        ids = associate(
            boxes, pre_boxes, pre_ids, 
            iou_thresh=iou_thresh)
        for k, x in enumerate(frame):
            if ids[k] == -1:
                count = count + 1
                x['track_id'] = count 
            else:
                x['track_id'] = ids[k].item()
        pre_boxes = boxes
        pre_ids = torch.tensor([x['track_id'] for x in frame], dtype=torch.long)
    return dets