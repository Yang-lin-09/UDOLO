# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Helper functions and class to calculate Average Precisions for 3D object detection.
"""
import os
import sys
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from eval_det import eval_det_cls, eval_det_multiprocessing
from eval_det import get_iou_obb
from nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls
from box_util import get_3d_box
from utils.integrate_utils import _pc_bbox3_filter_bev, _pc_bbox3_filter_batch

import _pickle as cPickle


def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def flip_axis_to_camera_torch(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    '''
    pc2 = pc.detach()
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def flip_axis_to_depth_torch(pc):
    pc2 = pc.detach()
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2


def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def parse_predictions(end_points, config_dict, inputs, frame):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}
        inputs:
        frame:

    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    pred_center = end_points['center']  # B,num_proposal,3
    pred_heading_class = torch.argmax(end_points['heading_scores'], -1)  # B,num_proposal
    pred_heading_residual = torch.gather(end_points['heading_residuals'], 2,
                                         pred_heading_class.unsqueeze(-1))  # B,num_proposal,1
    pred_heading_residual.squeeze_(2)
    pred_size_class = torch.argmax(end_points['size_scores'], -1)  # B,num_proposal
    pred_size_residual = torch.gather(end_points['size_residuals'], 2,
                                      pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1,
                                                                                         3))  # B,num_proposal,1,3
    pred_size_residual.squeeze_(2)
    pred_sem_cls = torch.argmax(end_points['sem_cls_scores'], -1)  # B,num_proposal
    sem_cls_probs = softmax(end_points['sem_cls_scores'].detach().cpu().numpy())  # B,num_proposal,10
    pred_sem_cls_prob = np.max(sem_cls_probs, -1)  # B,num_proposal

    num_proposal = pred_center.shape[1]
    # Since we operate in upright_depth coord for points, while util functions
    # assume upright_camera coord.
    bsize = pred_center.shape[0]
    pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3))
    pred_center_upright_camera = flip_axis_to_camera(pred_center.detach().cpu().numpy())

    for i in range(bsize):
        for j in range(num_proposal):
            heading_angle = config_dict['dataset_config'].class2angle( \
                pred_heading_class[i, j].detach().cpu().numpy(), pred_heading_residual[i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size( \
                int(pred_size_class[i, j].detach().cpu().numpy()), pred_size_residual[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
            pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    K = pred_center.shape[1]  # K==num_proposal
    nonempty_box_mask = np.ones((bsize, K))

    if config_dict['remove_empty_box']:
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'][:, :, 0:3]  # B,N,3
        for i in range(bsize):
            box3d = flip_axis_to_depth(pred_corners_3d_upright_camera[i])
            pc_in_box = _pc_bbox3_filter_bev(torch.from_numpy(box3d).cuda().float(), batch_pc[i])
            pc_in_box = pc_in_box.sum(dim=1)
            nonvalid = pc_in_box < 5
            nonempty_box_mask[i, nonvalid.data.cpu().numpy()] = 0

    obj_logits_raw = end_points['objectness_scores'].detach()
    obj_logits = torch.nn.Softmax(dim=-1)(obj_logits_raw)[:, :, 1]
    if config_dict['feedback']:
        integrate = end_points['integrate']
        pred = torch.Tensor(
            [_[[3, 7, 4, 0, 2, 6, 5, 1]] for _ in pred_corners_3d_upright_camera[0]]).cuda()
        ids = integrate.matching(pred, mode='corners')

        # late integrate
        match_tra_box3d, match_pre_ids, scores, sem_cls_probs_track = integrate.get_match_box3d('corners')
        if match_tra_box3d is not None:
            scores = scores * 0.99
            match_tra_box3d = torch.stack(
                [_[[3, 7, 4, 0, 2, 6, 5, 1]] for _ in match_tra_box3d]).unsqueeze(0).cpu().numpy()
            pred_corners_3d_upright_camera = np.concatenate([pred_corners_3d_upright_camera, match_tra_box3d], axis=1)
            ids = torch.cat([ids, match_pre_ids])

            obj_logits = torch.cat([obj_logits, scores.unsqueeze(0)], dim=1)

            track_num = match_tra_box3d.shape[1]
            nonempty_track = np.ones(track_num)[np.newaxis, ...]
            nonempty_box_mask = np.concatenate([nonempty_box_mask, nonempty_track], axis=1)

            pred_sem_cls_track = torch.argmax(sem_cls_probs_track, -1)  # B,num_proposal
            pred_sem_cls = torch.cat([pred_sem_cls, pred_sem_cls_track.unsqueeze(0)], dim=1)
            sem_cls_probs = np.concatenate([sem_cls_probs, sem_cls_probs_track.unsqueeze(0).data.cpu().numpy()], axis=1)

            K += track_num

        ids = ids.unsqueeze(0).cpu().numpy()

    obj_prob = obj_logits.cpu().numpy()

    if not config_dict['use_3d_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_2d_with_prob = np.zeros((K, 5))
            for j in range(K):
                boxes_2d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 2] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_2d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_2d_with_prob[j, 4] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
        # ---------- NMS input: pred_with_prob in (B,K,7) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 7))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                 config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------
    elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
        # ---------- NMS input: pred_with_prob in (B,K,8) -----------
        pred_mask = np.zeros((bsize, K))
        for i in range(bsize):
            boxes_3d_with_prob = np.zeros((K, 8))
            for j in range(K):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = obj_prob[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls[i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 1
        end_points['pred_mask'] = pred_mask
        # ---------- NMS output: pred_mask in (B,K) -----------

    batch_pred_map_cls = []  # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    for i in range(bsize):
        scene_name = inputs['scene_name']
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(config_dict['dataset_config'].num_class):
                if config_dict['feedback']:
                    # print(1, sem_cls_probs[i, j, ii], obj_prob[i, j], config_dict['conf_thresh'], obj_prob[i, j] > config_dict['conf_thresh'], pred_mask[i, j] == 1, pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh'])
                    cur_list += [
                        (ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j], ids[i, j]) \
                        for j in range(K) if pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
                else:
                    # print(2, sem_cls_probs[i, j, ii], obj_prob[i, j], config_dict['conf_thresh'], obj_prob[i, j] > config_dict['conf_thresh'], pred_mask[i, j] == 1, pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh'])
                    cur_list += [(ii, pred_corners_3d_upright_camera[i, j], sem_cls_probs[i, j, ii] * obj_prob[i, j], 0) \
                                 for j in range(K) if pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']]
            batch_pred_map_cls.append(cur_list)
        else:
            if config_dict['feedback']:
                # print(3, obj_prob[i, j], config_dict['conf_thresh'], obj_prob[i, j] > config_dict['conf_thresh'], pred_mask[i, j] == 1, pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh'])
                batch_pred_map_cls.append(
                    [(pred_sem_cls[i, j].item(), pred_corners_3d_upright_camera[i, j], obj_prob[i, j], ids[i, j]) \
                     for j in range(K) if pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])
            else:
                # print(4, obj_prob[i, j], config_dict['conf_thresh'], obj_prob[i, j] > config_dict['conf_thresh'], pred_mask[i, j] == 1, pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh'])
                batch_pred_map_cls.append(
                    [(pred_sem_cls[i, j].item(), pred_corners_3d_upright_camera[i, j], obj_prob[i, j], 0, scene_name[i]) \
                     for j in range(K) if pred_mask[i, j] == 1 and obj_prob[i, j] > config_dict['conf_thresh']])
    end_points['batch_pred_map_cls'] = batch_pred_map_cls
    
    if config_dict['feedback']:
        # TODO support only one class and per_class_proposal
        if len(batch_pred_map_cls[0]) == 0:
            pred = torch.rand(1, 8, 3).cuda()
            scores = torch.Tensor(1).zero_().cuda()
            ids = torch.Tensor(1).zero_().cuda()
            sem_scores = torch.Tensor(1, config_dict['dataset_config'].num_class).zero_().cuda()
        else:
            pred = torch.stack(
                [torch.Tensor(pred_corners_3d_upright_camera[0, j])[[3, 7, 4, 0, 2, 6, 5, 1]] for j in range(K) if
                 pred_mask[0, j] == 1 and obj_prob[0, j] > config_dict['conf_thresh']]).cuda()
            ids = torch.cat([torch.Tensor([ids[0, j]]) for j in range(K) if
                             pred_mask[0, j] == 1 and obj_prob[0, j] > config_dict['conf_thresh']]).cuda().long()
            scores = torch.cat([torch.Tensor([obj_prob[0, j]]) for j in range(K) if
                                pred_mask[0, j] == 1 and obj_prob[0, j] > config_dict['conf_thresh']]).cuda()
            sem_scores = torch.stack([torch.Tensor(sem_cls_probs[0, j]) for j in range(K) if
                                      pred_mask[0, j] == 1 and obj_prob[0, j] > config_dict['conf_thresh']]).cuda()
        mask = scores > 0.1
        if mask.sum() != 0:
            pred = pred[mask]
            scores = scores[mask]
            ids = ids[mask]
            sem_scores = sem_scores[mask]

        integrate.late(pred, scores, ids, sem_scores, mode='corners')

    return batch_pred_map_cls


def parse_groundtruths(end_points, config_dict):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}

    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    center_label = end_points['center_label']
    heading_class_label = end_points['heading_class_label']
    heading_residual_label = end_points['heading_residual_label']
    size_class_label = end_points['size_class_label']
    size_residual_label = end_points['size_residual_label']
    box_label_mask = end_points['box_label_mask']
    sem_cls_label = end_points['sem_cls_label']
    bsize = center_label.shape[0]

    K2 = center_label.shape[1]  # K2==MAX_NUM_OBJ
    gt_corners_3d_upright_camera = np.zeros((bsize, K2, 8, 3))
    gt_center_upright_camera = flip_axis_to_camera(center_label[:, :, 0:3].detach().cpu().numpy())
    for i in range(bsize):
        for j in range(K2):
            if box_label_mask[i, j] == 0: continue
            heading_angle = config_dict['dataset_config'].class2angle(heading_class_label[i, j].detach().cpu().numpy(),
                                                                      heading_residual_label[
                                                                          i, j].detach().cpu().numpy())
            box_size = config_dict['dataset_config'].class2size(int(size_class_label[i, j].detach().cpu().numpy()),
                                                                size_residual_label[i, j].detach().cpu().numpy())
            corners_3d_upright_camera = get_3d_box(box_size, heading_angle, gt_center_upright_camera[i, j, :])
            gt_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

    batch_gt_map_cls = []
    for i in range(bsize):
        batch_gt_map_cls.append(
            [(sem_cls_label[i, j].item(), gt_corners_3d_upright_camera[i, j]) for j in
             range(gt_corners_3d_upright_camera.shape[1]) if box_label_mask[i, j] == 1])
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls


class APCalculator(object):
    ''' Calculating Average Precision '''

    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()

    def load(self, gt_map_cls, pre_map_cls):
        self.gt_map_cls = gt_map_cls
        self.pred_map_cls = pre_map_cls

    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """

        bsize = len(batch_pred_map_cls)
        assert (bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i]
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i]
            self.scan_cnt += 1

    def compute_metrics(self, DUMP_DIR=None, name='front', not_eval=False):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        if DUMP_DIR is not None:
            predict_dir = os.path.join(DUMP_DIR, name + '_predict.pkl')
            gt_dir = os.path.join(DUMP_DIR, 'gt.pkl')
            with open(predict_dir, 'wb') as fid:
                cPickle.dump(self.pred_map_cls, fid)
            with open(gt_dir, 'wb') as fid:
                cPickle.dump(self.gt_map_cls, fid)
            if not_eval:
                return

        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh,
                                                 get_iou_func=get_iou_obb)
        ret_dict = {}
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision' % (clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall' % (clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall' % (clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {}  # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {}  # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0
