# This file is modified from FairMOT https://github.com/ifzhang/FairMOT/blob/master/src/lib/tracker/multitracker.py
# The original code is under MIT license

import cv2
import torch.nn.functional as F

import numpy as np
from numba import jit
from collections import deque

import numpy as np

from .fairmot_utils import KalmanFilter, BaseTrack, TrackState
from . import fairmot_matching as matching

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()
    def __init__(self, tlwh, score, temp_feat, buffer_size=30, ind=-1,
        is_activated=False):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = is_activated

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9
        self.ind = ind

    def update_features(self, feat):
        if feat is None:
            self.curr_feat = None
            return
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)


    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
                multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(
            self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        #self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )

        self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.ind = new_track.ind

    def update(self, new_track, frame_id, update_feature=True):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh))
        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        if update_feature:
            self.update_features(new_track.curr_feat)

        self.ind = new_track.ind

    @property
    # @jit(nopython=True)
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    # @jit(nopython=True)
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    # @jit(nopython=True)
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    # @jit(nopython=True)
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class FairMOT(object):
    def __init__(self, 
        thresh=0, 
        iou_thresh=0.5,
        confirm_thresh=0.7,
        reid_thresh=0.4,
        no_iou=False,
        iou_only=False):
        self.thresh = thresh
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        self.frame_id = 0

        self.det_thresh = thresh
        self.buffer_size = 30
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.iou_thresh=iou_thresh
        self.confirm_thresh=confirm_thresh
        self.reid_thresh=reid_thresh
        self.no_iou = no_iou
        self.iou_only = iou_only
        

    def update(self, dets, id_feature):
        '''
        Inputs:
            dets: N x 5: (l, t, r, b, s)
            id_feature: N x D
        '''
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        dets = dets.numpy()
        if self.iou_only:
            id_feature = [None for _ in range(len(dets))]
        else:
            id_feature = F.normalize(id_feature, dim=1)
            id_feature = id_feature.numpy()

        if len(dets) > 0:
            '''Detections'''
            detections = [
                STrack(STrack.tlbr_to_tlwh(
                    tlbrs[:4]), tlbrs[4], f, 30, ind=i,)
                    for i, (tlbrs, f) in enumerate(zip(dets[:, :5], id_feature))]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with embedding'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        if self.iou_only:
            matches = np.zeros((0, 2), dtype=np.int32)
            u_track = [i for i in range(len(strack_pool))]
            u_detection = [i for i in range(len(detections))]
        else:
            dists = matching.embedding_distance(strack_pool, detections)
            dists = matching.fuse_motion(
                self.kalman_filter, dists, strack_pool, detections)
            matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=self.reid_thresh)

            for itracked, idet in matches:
                track = strack_pool[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(detections[idet], self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

        if not self.no_iou:
            ''' Step 3: Second association, with IOU'''
            detections = [detections[i] for i in u_detection]
            r_tracked_stracks = [strack_pool[i] for i in u_track \
                if strack_pool[i].state == TrackState.Tracked]
            dists = matching.iou_distance(r_tracked_stracks, detections)
            matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=self.iou_thresh)

            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections[idet]
                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)
                
        for it in u_track:
            try:
                track = r_tracked_stracks[it]
            except:
                track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        if not self.no_iou:
            dists = matching.iou_distance(unconfirmed, detections)
        else:
            dists = matching.embedding_distance(unconfirmed, detections)
            dists = matching.fuse_motion(
                self.kalman_filter, dists, unconfirmed, detections)

        matches, u_unconfirmed, u_detection = matching.linear_assignment(
            dists, thresh=self.confirm_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks \
            if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(
            self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(
            self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks \
            if track.is_activated]
        return output_stracks

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb