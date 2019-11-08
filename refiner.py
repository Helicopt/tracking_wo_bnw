import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
# from scipy.optimize import linear_sum_assignment
# from scipy.spatial.distance import cdist
from torch.autograd import Variable

import cv2

from senseTk.common import *
from tracktor.utils import bbox_overlaps, bbox_transform_inv, clip_boxes

Th1 = 0.5
Th2 = 0.5

class Refiner(object):

    cl = 1

    def __init__(self, detector):
        self.obj_detect = detector

    def reset(self, seq, seq_det, prev_dir):
        seq_det = seq_det.split('_')[-1]
        if seq_det=='DPM17':
            seq_det = 'DPM'
        elif seq_det=='SDP17':
            seq_det = 'SDP'
        elif seq_det=='FRCNN17':
            seq_det = 'FRCNN'
        else:
            seq_det = ''
        f = seq._seq_name+'-'+seq_det+'.txt'
        self.fr = 0
        self.r = TrackSet(os.path.join(prev_dir, f))
        self.res = {}
        self.prev = []
        self.threshold1 = Th1
        self.threshold2 = Th2

    @staticmethod
    def _get_dets(x, scale):
        pos = []
        for i in x:
            d = torch.FloatTensor([i.x1-1, i.y1-1, i.w + i.x1-2, i.h + i.y1-2]).view(1, -1) * scale
            pos.append(d)
        return torch.cat(pos, 0)


    def step(self, blob):
        self.fr += 1

        l = []
        pre = []
        mapping = {}
        cur = []
        drop = 0
        append = 0
        # refine all results
        self.obj_detect.load_image(blob['data'][0], blob['im_info'][0])

        if len(self.r[self.fr]):
            dt = self.r[self.fr]
            dets = dets = self._get_dets(dt, blob['im_info'][0][2])
            _, scores, bbox_pred, rois = self.obj_detect.test_rois(dets)
            boxes = bbox_transform_inv(rois, bbox_pred)
            boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).cpu().data
            bbox_pred = boxes[:, self.cl*4:(self.cl+1)*4]
            scores = scores[:, self.cl]
            for i in range(len(dt)):
                sc = float(scores[i])
                x1, y1, x2, y2 = bbox_pred[i] / blob['im_info'][0][2]
                if sc > self.threshold1:
                    k = Det(float(x1+1), float(y1+1), float(x2 - x1+1), float(y2 - y1+1), fr = dt[i].fr)
                    k.conf = sc
                    k.uid = dt[i].uid
                    # print('%s ==> %s'%(str(dt[i]), str(k)))
                    l.append(k)
                    cur.append(k)
                    mapping[k.uid] = 1
                else:
                    drop += 1

        for j in self.prev:
            if j.uid not in mapping:
                pre.append(j)
        if len(pre):
            dets = dets = self._get_dets(pre, blob['im_info'][0][2])
            _, scores, bbox_pred, rois = self.obj_detect.test_rois(dets)
            boxes = bbox_transform_inv(rois, bbox_pred)
            boxes = clip_boxes(Variable(boxes), blob['im_info'][0][:2]).cpu().data
            bbox_pred = boxes[:, self.cl*4:(self.cl+1)*4]
            scores = scores[:, self.cl]
            for i in range(len(pre)):
                sc = float(scores[i])
                x1, y1, x2, y2 = bbox_pred[i] / blob['im_info'][0][2]
                if sc > self.threshold2:
                    k = Det(float(x1+1), float(y1+1), float(x2 - x1+1), float(y2 - y1+1), fr = self.fr)
                    flag = True
                    for j in cur:
                        if j.iou(k)>0.6:
                            flag = False
                            break
                    if not flag: continue
                    k.conf = sc
                    k.uid = pre[i].uid
                    l.append(k)
                    mapping[k.uid] = 1
                    append += 1
        self.prev = l

        if self.fr%10==1:
            print('dealing frame %03d'%self.fr)
            print('drop %d append %d alter %d'%(drop, append, append - drop))


        # refresh all results
        for t in l:
            if t.uid not in self.res:
                self.res[t.uid] = {}
            self.res[t.uid][t.fr - 1] = np.array([t.x1 - 1, t.y1 - 1, t.x2 - 2, t.y2 - 2, t.conf])

    def get_results(self):
        return self.res