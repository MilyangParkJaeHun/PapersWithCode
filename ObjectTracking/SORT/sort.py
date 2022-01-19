import os
import glob
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from scipy.optimize import linear_sum_assignment


from utils import KalmanFilter

from typing import List

np.random.seed(0)

def x_to_bbox(x):
    """ 칼만 필터 상태 벡터 값을 bounding box 값으로 변환

    Args:
        x (u, v, s, r, u', v', s'): 칼만 필터 상태 벡터

    Returns:
        bbox: xmin, ymin, xmax, ymax 로 이루어진 bounding box 값
    """
    u, v, s, r = x[:4]
    w = np.sqrt(s * r)
    h = s / w
    
    xmin = u - w/2.
    ymin = v - h/2.
    xmax = u + w/2.
    ymax = v + h/2.
    return np.array([xmin, ymin, xmax, ymax])

def bbox_to_z(bbox):
    """ bounding box 값을 측정 벡터 값으로 변환

    Args:
        bbox (xmin, ymin, xmax, ymax): bounding box 정보

    Returns:
        [u, v, s, r]: x 중심 좌표, y 중심 좌표, 넓이, 비율 정보
    """
    xmin, ymin, xmax, ymax, _ = bbox
    w = xmax - xmin
    h = ymax - ymin
    u = xmin + w/2.
    v = ymin + h/2.
    s = w * h
    r = w / float(h)
    return np.array([u, v, s, r]).reshape((4, 1))

class BoxTracker(object): # 칼만 필터 기반 단일 객체 추적기
    count = 0 # 단일 객체 추적기의 id 를 할당해주기 위한 변수
    def __init__(self, bbox):
        """ 단일 객체 추적기 생성자

        Args:
            bbox (xmin, ymin, xmax, ymax): 객체 추적을 시작할 bbox 좌표값
        """
        self.id = BoxTracker.count
        BoxTracker.count += 1

        # x = [u, v, s, r, u', v', s']
        self.kf = KalmanFilter(dim_x=7, dim_z=4) # 객체 위치 예측을 수행할 칼만필터 instance 생성
        self.kf.F = np.array([[1,0,0,0,1,0,0],
                           [0,1,0,0,0,1,0],
                           [0,0,1,0,0,0,1],
                           [0,0,0,1,0,0,0],
                           [0,0,0,0,1,0,0],
                           [0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],
                           [0,1,0,0,0,0,0],
                           [0,0,1,0,0,0,0],
                           [0,0,0,1,0,0,0]])
        self.kf.P *= 10. # 초기 상태의 예측 불확실성을 고려
        self.kf.P[4:,4:] *= 100. # 초기 속도 예측에 대한 높은 불확실성을 반영
        self.kf.Q[4:,4:] *= 0.01 # 속도에 대해 완만한 추정값을 얻기 위해 작게 조정
        self.kf.Q[-1,-1] *= 0.01 # 넓이의 속도는 높이와 너비의 속도가 곱해져 반영되므로 0.01 을 다시 반영
        self.kf.R[2:,2:] *= 10. # 너비와 비율이 측정값의 영향을 덜 받고 완만하게 추정하도록 R 값을 크게 반영

        self.kf.x[:4] = bbox_to_z(bbox) # 첫 bbox 를 측정 벡터로 변환하여 칼만필터 상태 벡터 초기화에 사용

        self.age = 0 # 추적을 유지해온 횟수
        self.hits = 0 # 매칭에 성공한 횟수
        self.time_since_update = 0 # 마지막 매칭에 성공한 이후부터 지나온 횟수

    def predict(self):
        """ 추적하던 객체의 다음 위치 예측

        Returns:
            bbox: 추적하던 객체의 다음 시점의 bbox 예측값
        """
        if((self.kf.x[6]+self.kf.x[2])<=0): # 예측될 bbox 의 넓이가 0 보다 작다면 0 으로 설정
            self.kf.x[6] *= 0.0
        self.kf.predict() # 예측 수행
        next_bbox = x_to_bbox(self.kf.x) 

        self.age += 1
        if self.time_since_update > 0:
            self.hits = 0
        self.time_since_update += 1

        return next_bbox

    def update(self, bbox):
        """ 측정값을 사용하여 칼만 필터 업데이트

        Args:
            bbox (xmin, ymin, xmax, ymax): 객체 검출기로부터의 bbox 측정값
        """
        self.kf.update(bbox_to_z(bbox)) # 업데이트 수행
        self.time_since_update = 0
        self.hits += 1

    def get_state(self):
        """ 추적하는 객체의 bbox 값 출력

        Returns:
            bbox: xmin, ymin, xmax, ymax
        """
        return x_to_bbox(self.kf.x)
    
    def get_scale(self):
        """ 추적하는 객체의 bbox 넓이 출력

        Returns:
            float: 추적하는 객체의 bbox 넓이
        """
        return self.kf.x[2]

class Sort(object): # 다중 객체 추적을 수행하는 Class
    def __init__(self, t_lost=1, t_probation=3, iou_threshold=0.3):
        """ 다중 객체 추적기 생성자

        Args:
            t_lost (int, optional): 최대 추적 실패 용인 횟수, t_lost 동안 추적에 실패하면 추적을 중단.
            t_probation (int, optional): 초기 속도 관찰을 위한 period, 추적기가 처음 생성되면 t_probation 동안 
                                        추적만 유지하고 추적 결과는 출력하지 않음 
            iou_threshold (float, optional): 매칭에 필요한 최소 iou score, 객체 검출 박스와 추적 예측 박스의
                                            iou score 가 iou_threshold 를 넘지 못하면 매칭 실패로 판단
        """
        self.t_lost = t_lost
        self.t_probation = t_probation
        self.iou_threshold = iou_threshold

        self.trackers: List[BoxTracker] = [] # 단일 객체 추적기 리스트

        self.frame_count = 0 # 처리한 프레임 수

    def update(self, dets):
        """ 객체 검출 결과를 기반으로 새로운 추적 결과 도출

        Args:
            dets ([xmin, ymin, xmax, ymax]): 객체 검출 결과 리스트

        Returns:
            [xmin, ymin, xmax, ymax, id]: 추적 결과 리스트
        """
        self.frame_count += 1

        def notValidPred(scale, bbox):
            """ 비정상적 예측 결과 필터링

            Args:
                scale (float): 예측된 bbox 의 넓이
                bbox (xmin, ymin, xmax, ymax): 예측된 bbox 정보

            Returns:
                bool: 비정상적이면 True, 정상적이면 False
            """
            return scale <= 0 or np.any(np.isnan(bbox))

        preds = [] # 예측 결과가 저장되는 리스트
        error_preds = [] # 비정상 예측 결과를 도출한 추적기의 index 리스트
        for i, trk in enumerate(self.trackers): # 모든 단일 추적기의 다음 위치 예측
            pred_bbox = trk.predict() # 다음 위치 예측
            s = trk.get_scale()

            if notValidPred(s, pred_bbox): # 비정상 결과 필터링
                error_preds.append(i)
            else:
                preds.append(pred_bbox)
        preds = np.array(preds).reshape(-1,4)

        for i in reversed(error_preds): # 비정상적 결과를 도출한 추적기 제거
            self.trackers.pop(i)
        
        if len(dets) == 0 and len(preds) == 0: # 객체 검출과 예측 결과가 존재하지 않는다면 매칭 실패
            return np.array([])
        elif len(dets) == 0: # 객체 검출이 안됐다면 현재 존재하는 모든 추적기는 매칭 실패
            matches, unmatched_dets = [], []
            unmatched_trks = list(range(len(self.trackers)))
        elif len(preds) == 0: # 정상적인 예측 결과가 존재하지 않는다면 모든 객체 검출 bbox 는 매칭 실패
            matches, unmatched_trks = [], []
            unmatched_dets = list(range(len(dets)))
        else: # 객체 검출 bbox 가 존재하고 예측 결과가 존재한다면 최적 매칭 계산 시도
            matches, unmatched_dets, unmatched_trks = self.match_dets_with_trks(dets, preds) # 검출 결과와 추적 예측 결과 최적 매칭

        for m in matches: # 매칭에 성공한 검출 결과를 기반으로 추적기 업데이트
            self.trackers[m[1]].update(dets[m[0]])

        for i in unmatched_dets: # 매칭에 실패한 검출 결과를 추적하는 새로운 추적기 생성
            trk = BoxTracker(dets[i])
            self.trackers.append(trk)

        trk_res = []
        for i, trk in reversed(list(enumerate(self.trackers))):
            bbox = trk.get_state()
            if trk.hits >= self.t_probation or self.frame_count <= self.t_probation: # 추적을 t_probation 동안 유지한 경우만 결과 사용
                trk_res.append([*bbox, trk.id + 1])
            if trk.time_since_update > self.t_lost: # t_lost 동안 추적에 실패하면 추적 중단
                self.trackers.pop(i)
        return np.array(trk_res, dtype=object)

    def match_dets_with_trks(self, dets, preds):
        """ 객체 검출과 예측값의 최적 매칭을 

        Args:
            dets ([xmin, ymin, xmax, ymax]): 객체 검출 bbox
            preds ([xmin, ymin, xmax, ymax]): 객체의 다음 위치 예측 bbox
        """
        def cal_iou(bb_a, bb_b): # iou score 계산
            """ bbox batch A 와 bbox batch B 사이의 iou score matrix 계산

            Args:
                bb_a ([bbox]): 하나 이상의 bbox 를 포함하는 리스트
                bb_b ([bbox]): 하나 이상의 bbox 를 포함하는 리스트

            Returns:
                2DMatrix: A 와 B 간의 iou score 값으로 이루어진 2차원 매트릭스
            """
            # input(dets, preds) 의 순서에 맞춰 ouput 이 출력되도록 확장 차원 순서 조정
            bb_b = np.expand_dims(bb_b, 0)
            bb_a = np.expand_dims(bb_a, 1)

            xmin_batch = np.maximum(bb_a[..., 0], bb_b[..., 0])
            ymin_batch = np.maximum(bb_a[..., 1], bb_b[..., 1])
            xmax_batch = np.minimum(bb_a[..., 2], bb_b[..., 2])
            ymax_batch = np.minimum(bb_a[..., 3], bb_b[..., 3])

            w_batch = np.maximum(0., xmax_batch - xmin_batch)
            h_batch = np.maximum(0., ymax_batch - ymin_batch)
            overlap = w_batch * h_batch
            union = (bb_a[..., 2] - bb_a[..., 0]) * (bb_a[..., 3] - bb_a[..., 1]) + \
                    (bb_b[..., 2] - bb_b[..., 0]) * (bb_b[..., 3] - bb_b[..., 1]) - overlap
            
            iou_matrix = overlap / union # iou score 계산

            return iou_matrix

        iou_matrix = cal_iou(dets, preds) # dets 와 preds 의 모든 조합에 대한 iou score 계산

        matches = []
        unmatched_dets = []
        unmatched_trks = []
        if min(iou_matrix.shape) > 0: # 매칭의 가능성이 있다면 iou score 기반으로 최적 매칭 서치
            det_indices, pred_indices = linear_sum_assignment(-iou_matrix) # 헝가리안 알고리즘(이분 그래프에서 최소 가중치 매칭을 찾는 알고리즘이므로 최대 가중치 매칭을 위해 iou_matrix 의 부호를 반전)
            matched_indices = np.array(list(zip(det_indices, pred_indices))) # ((det_index, pre_index)) 꼴로 매칭 결과 변환
            
            for det_i, pred_i in matched_indices:
                if(iou_matrix[det_i][pred_i] < self.iou_threshold): # 매칭된 결과의 iou score 가 iou threshold 보다 낮다면 매칭 실패로 간주
                    unmatched_dets.append(det_i)
                    unmatched_trks.append(pred_i)
                else:
                    matches.append([det_i, pred_i]) # iou threshold 보다 크다면 매칭 성공으로 간주

            for i in range(len(dets)): # 매칭에 실패한 객체 검출 결과의 인덱스 저장
                if i not in matched_indices[:, 0]:
                    unmatched_dets.append(i)

            for i in range(len(preds)): # 매칭에 실패한 추적기의 인덱스 저장
                if i not in matched_indices[:, 1]:
                    unmatched_trks.append(i)
            
        else: # 모든 매칭에 실패했으므로 실패한 객체 검출 결과와 추적기의 인덱스 저장
            unmatched_dets = list(range(len(dets)))
            unmatched_trks = list(range(len(preds)))
        
        if len(matches) == 0:
            matches = np.empty((0,2), dtype=int)
        else:
            matches = np.array(matches)
        
        return matches, np.array(unmatched_dets), np.array(unmatched_trks)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    parser.add_argument("--seq_path", help="Path to detections.", type=str, default='data')
    parser.add_argument("--phase", help="Subdirectory in seq_path.", type=str, default='train')
    parser.add_argument("--max_age", 
                        help="Maximum number of frames to keep alive a track without associated detections.", 
                        type=int, default=1)
    parser.add_argument("--min_hits", 
                        help="Minimum number of associated detections before track is initialised.", 
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3) #used only for display
    if(display):
        if not os.path.exists('mot_benchmark'):
            print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
            exit()
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, aspect='equal')

    if not os.path.exists('output'):
        os.makedirs('output')

    pattern = os.path.join(args.seq_path, phase, '*', 'det', 'det.txt') 
    for seq_dets_fn in glob.glob(pattern):
        BoxTracker.count = 0

        motracker = Sort(t_lost=args.max_age,
                        t_probation=args.min_hits,
                        iou_threshold=args.iou_threshold) # 다중 객체 추적기 instance 생성 (motracker : Multiple Object Tracker)
        
        seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
        seq = seq_dets_fn[pattern.find('*'):].split(os.path.sep)[0]

        with open(os.path.join('output', '%s.txt'%(seq)),'w') as out_file:
            print("Processing %s."%(seq))
            for frame in range(int(seq_dets[:,0].max())):
                frame += 1 #detection and frame numbers begin at 1
                dets = seq_dets[seq_dets[:, 0]==frame, 2:7]
                dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]
                total_frames += 1

                if(display):
                    fn = os.path.join('mot_benchmark', phase, seq, 'img1', '%06d.jpg'%(frame))
                    im =io.imread(fn)
                    ax1.imshow(im)
                    plt.title(seq + ' Tracked Targets')
                
                start_time = time.time()
                trackers = motracker.update(dets) # 객체 검출 결과를 입력으로 추적 결과 도출
                cycle_time = time.time() - start_time
                total_time += cycle_time

                for d in trackers:
                    print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
                    if(display):
                        d = d.astype(np.int32)
                        ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
                
                if(display):
                    fig.canvas.flush_events()
                    plt.draw()
                    ax1.cla()
                    
    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (total_time, total_frames, total_frames / total_time))

    if(display):
        print("Note: to get real runtime results run without the option: --display")
