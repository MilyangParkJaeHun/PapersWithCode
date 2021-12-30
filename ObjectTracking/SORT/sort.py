from typing import List

import numpy as np

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

class Sort(object):
    def __init__(self, t_lost=1, t_probation=3, iou_threshold=0.3):
        self.t_lost = t_lost
        self.t_probation = t_probation
        self.iou_threshold = iou_threshold

        self.trackers: List[BoxTracker] = []