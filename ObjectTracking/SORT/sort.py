from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from utils import KalmanFilter

class BoxTracker(object):
    count = 0
    def __init__(self, bbox):
        self.id = BoxTracker.count
        BoxTracker += 1

        # x = [u, v, s, r, u', v', s']
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.F = np.array([[1,0,0,0,1,0,0],
                           [0,1,0,0,0,1,0],
                           [0,0,1,0,0,0,1],
                           [0,0,0,1,0,0,0],
                           [0,0,0,0,1,0,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,1]])
        self.H = np.array([[1,0,0,0,0,0,0],
                           [0,1,0,0,0,0,0],
                           [0,0,1,0,0,0,0],
                           [0,0,0,1,0,0,0]])
        self.P *= 10.
        self.P[4:,4:] *= 100. # 초기 속도 예측에 대한 높은 불확실성을 반영
        self.Q[4:,4:] *= 0.01
        self.Q[-1,-1] *= 0.01
        self.R[2:,2:] *= 10.

class Sort(object):
    def __init__(self, t_lost=1, t_probation=3, iou_threshold=0.3):
        self.t_lost = t_lost
        self.t_probation = t_probation
        self.iou_threshold = iou_threshold

        self.trackers: List[BoxTracker] = []

    def update(self, dets):
        def notValidPred(pred):
            return False

        preds = []
        error_preds = []
        for i, trk in enumerate(self.trackers):
            pred_bbox = trk.predict()

            if notValidPred(pred_bbox):
                error_preds.append(i)
            else:
                preds.append(pred_bbox)
        preds = np.array(preds)

        for i in reversed(error_preds):
            self.trackers.pop(i)
        
        matches, unmatched_dets, unmatched_trks = self.match_dets_with_trks(dets, preds)

        for m in matches:
            self.trackers[m[1]].update(dets[m[0]])

        for i in unmatched_dets:
            trk = BoxTracker(dets[i])
            self.trackers.append(trk)

    def match_dets_with_trks(self, dets, preds):
        def cal_iou(bb_a, bb_b):
            bb_a = np.expand_dims(bb_a, 0)
            bb_b = np.expand_dims(bb_b, 1)

            xmin_batch = np.maximum(bb_a[..., 0], bb_b[..., 0])
            ymin_batch = np.maximum(bb_a[..., 1], bb_b[..., 1])
            xmax_batch = np.minimum(bb_a[..., 2], bb_b[..., 2])
            ymax_batch = np.minimum(bb_a[..., 3], bb_b[..., 3])

            w_batch = np.maximum(0., xmax_batch - xmin_batch)
            h_batch = np.maximum(0., ymax_batch - ymin_batch)
            overlap = w_batch * h_batch
            union = (bb_a[..., 2] - bb_a[..., 0]) * (bb_a[..., 3] - bb_a[..., 1]) + \
                    (bb_b[..., 2] - bb_b[..., 0]) * (bb_b[..., 3] - bb_b[..., 1]) - overlap
            
            iou_matrix = overlap / union

            return iou_matrix
        
        iou_matrix = cal_iou(dets, preds)

        matches = []
        unmatched_dets = []
        unmatched_trks = []
        if min(iou_matrix.shape) > 1: # 매칭의 가능성이 있다면
            matched_indices = linear_sum_assignment(-iou_matrix)
            
            for det_i, pred_i in matched_indices:
                if(iou_matrix[det_i][pred_i] < self.iou_threshold):
                    unmatched_dets.append(det_i)
                    unmatched_trks.append(pred_i)
                else:
                    matches.append([det_i, pred_i])

            for i in range(len(dets)):
                if i not in matched_indices[:, 0]:
                    unmatched_dets.append(i)
            
            for i in range(len(preds)):
                if i not in matched_indices[:, 1]:
                    unmatched_trks.append(i)
        else:
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(preds)))
        
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)
        
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

if __name__ == "__main__":

    motracker = Sort()

    while True:
        detections = get_dets()

        trackers = motracker.track(detections)