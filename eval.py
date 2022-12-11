# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Evaluation routine for 3D object detection with SUN RGB-D and ScanNet.
"""

import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import open3d as o3d

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from box_util import get_3d_box, corners2xyzrylwh
from kalman_utils import AB3DMOT
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='sunrgbd', help='Dataset name. sunrgbd or scannet. [default: sunrgbd]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=100000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=32, help='Point Number [default: 256]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--vote_factor', type=int, default=1, help='Number of votes generated from each seed [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps',
                    help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresholds', default='0.25,0.5', help='A list of AP IoU thresholds [default: 0.25,0.5]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use SUN RGB-D V2 box labels.')
parser.add_argument('--use_3d_nms', action='store_true', help='Use 3D NMS instead of 2D NMS.')
parser.add_argument('--use_cls_nms', action='store_true', help='Use per class NMS.')
parser.add_argument('--use_old_type_nms', action='store_true', help='Use old type of NMS, IoBox2Area.')
parser.add_argument('--per_class_proposal', action='store_true', help='Duplicate each proposal num_class times.')
parser.add_argument('--nms_iou', type=float, default=0.25, help='NMS IoU threshold. [default: 0.25]')
parser.add_argument('--conf_thresh', type=float, default=0.5,
                    help='Filter out predictions with obj prob less than it. [default: 0.05]')
parser.add_argument('--faster_eval', action='store_true',
                    help='Faster evaluation by skippling empty bounding box removal.')
parser.add_argument('--shuffle_dataset', action='store_true', help='Shuffle the dataset (random order).')

parser.add_argument('--feedback', action='store_true', help='')
parser.add_argument('--aug_score_based', action='store_true', help='Use score in sample 3d boxes.')
parser.add_argument('--use_kalman', action='store_true', help='If save result.')
parser.add_argument('--voxel_size', type=tuple, default=(0.04, 0.04), help='voxel size for voxel map')
parser.add_argument('--map_area', type=tuple, default=(200, 200), help='voxel map area')
parser.add_argument('--detect_thresh', type=int, default=3, help='threshold for number of detection')
parser.add_argument('--score_decay', type=float, default=1.0, help='score decay')
parser.add_argument('--sem_score_decay', type=float, default=0.0, help='rescore decay')
parser.add_argument('--boxes_decay', type=float, default=0.0, help='boxes decay')
parser.add_argument('--valid_score', type=float, default=0.1,
                    help='valid score for backend box which condition frontend')
parser.add_argument('--num_point_thresh', type=int, default=10, help='number of point')
parser.add_argument('--num_proposal_per_box', type=int, default=1, help='number of point')
parser.add_argument('--iou_thresh', type=float, default=0.5, help='iou_thresh')
parser.add_argument('--top_n', type=int, default=64, help='number of proposal')
parser.add_argument('--top_n_pred', type=int, default=32, help='number of prediction proposal')
parser.add_argument('--view_coord', type=str, default='camera', help='view coord')
parser.add_argument('--world_coord', type=str, default='depth', help='world coord')

parser.add_argument('--vis_disable', action = 'store_true')

FLAGS = parser.parse_args()

if FLAGS.use_cls_nms:
    assert (FLAGS.use_3d_nms)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DUMP_DIR = FLAGS.dump_dir
CHECKPOINT_PATH = FLAGS.checkpoint_path
assert (CHECKPOINT_PATH is not None)
FLAGS.DUMP_DIR = DUMP_DIR
AP_IOU_THRESHOLDS = [float(x) for x in FLAGS.ap_iou_thresholds.split(',')]

# Prepare DUMP_DIR
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DUMP_FOUT = open(os.path.join(DUMP_DIR, 'log_eval.txt'), 'w')
DUMP_FOUT.write(str(FLAGS) + '\n')

VIS_DISABLE = FLAGS.vis_disable


def log_string(out_str):
    DUMP_FOUT.write(out_str + '\n')
    DUMP_FOUT.flush()
    print(out_str)


# Init datasets and dataloaders
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


if FLAGS.dataset == "scannet":
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetSVDetectionDataset
    from model_util_scannet import ScannetSVDatasetConfig

    DATASET_CONFIG = ScannetSVDatasetConfig()
    TEST_DATASET = ScannetSVDetectionDataset('val', num_points=NUM_POINT,
                                             augment=False,
                                             use_color=FLAGS.use_color, use_height=(not FLAGS.no_height), fix_seed=True)
else:
    print('Unknown dataset %s. Exiting...' % (FLAGS.dataset))
    exit(-1)
print(len(TEST_DATASET))
TEST_DATALOADER = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE,
                             shuffle=FLAGS.shuffle_dataset, num_workers=4, worker_init_fn=my_worker_init_fn)

# Init the model and optimzier
MODEL = importlib.import_module(FLAGS.model)  # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color) * 3 + int(not FLAGS.no_height) * 1

net = MODEL.VoteNet(num_class=DATASET_CONFIG.num_class,
                    num_heading_bin=DATASET_CONFIG.num_heading_bin,
                    num_size_cluster=DATASET_CONFIG.num_size_cluster,
                    mean_size_arr=DATASET_CONFIG.mean_size_arr,
                    input_feature_dim=num_input_channel,
                    FLAGS=FLAGS)
net.to(device)
criterion = MODEL.get_loss

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Load checkpoint if there is any
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    log_string("Loaded checkpoint %s (epoch: %d)" % (CHECKPOINT_PATH, epoch))

# Used for AP calculation
CONFIG_DICT = {'remove_empty_box': (not FLAGS.faster_eval), 'use_3d_nms': FLAGS.use_3d_nms, 'nms_iou': FLAGS.nms_iou,
               'use_old_type_nms': FLAGS.use_old_type_nms, 'cls_nms': FLAGS.use_cls_nms,
               'per_class_proposal': FLAGS.per_class_proposal,
               'conf_thresh': FLAGS.conf_thresh, 'dataset_config': DATASET_CONFIG,
               'feedback': FLAGS.feedback, }


# ------------------------------------------------------------------------- GLOBAL CONFIG END

def is_inbbox(point, bbox):
    '''
            7 -------- 4                 z
           /|         /|                /
          6 -------- 5 .               /
          | |        | |              |--------> x
          . 3 -------- 0              |
          |/         |/               |
          2 -------- 1                y
    '''
    
    i = bbox[3] - bbox[2]
    j = bbox[1] - bbox[2]
    k = bbox[6] - bbox[2]
    v = point - bbox[2]
    
    if 0 < np.dot(v, i) and np.dot(v, i) < np.dot(i, i):
        if 0 < np.dot(v, j) and np.dot(v, j) < np.dot(j, j):
            if 0 < np.dot(v, k) and np.dot(v, k) < np.dot(k, k):
                return True
            return False
        return False
    return False
    
    
def get_index(points, bboxes):
    
    '''
    Reference: https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
    '''
    
    point_class = {}
    
    for i in range(points.shape[0]):
        for j in range(len(bboxes)):
            if is_inbbox(points[i], bboxes[j][1]):
                if j in point_class.keys():
                    point_class[j].append(i)
                else:
                    point_class[j] = [i, ]
    
    return point_class

def get_maching_num(prev, now):
    
    if prev == None:
        return 0, []
    
    num = 0 
    id_list = []
    for i in range(len(now[0])):
        for j in range(len(prev[0])):
            if now[0][i][3] == prev[0][j][3] and now[0][i][3] != 0 and now[0][i][3] not in id_list:
                num += 1
                id_list.append(now[0][i][3])
    return num, id_list

def direct_icp(frame_pre, frame, imu = None, imu_weight = None):
    '''
    Inputs:
        frame_pre, frame: [nc, 3, n]
            nc is the # clusters, n is the # points
        imu: [4, 4]
            rotation matrix
        imu_weight:
            the weight of imu information
    Outputs:

    '''
    
    cluster_num = len(frame_pre)
    # # virtual points for each cluster
    virtual_point_num = 10
    
    b = np.zeros((virtual_point_num * cluster_num, 1), dtype=np.float32)
    A = np.zeros((virtual_point_num * cluster_num, 16), dtype=np.float32)
    x = np.zeros((3, virtual_point_num), dtype=np.float32)
    

    for i in range(cluster_num):
        cluster_pre = frame_pre[i]
        cluster = frame[i]
        
        num_points_pre = frame_pre[i].shape[1]
        num_points = frame[i].shape[1]
        
        cluster_bar = np.sum(cluster, 1) / num_points
        
        x_min_pre, y_min_pre, z_min_pre = np.min(frame_pre[i], 1)
        x_max_pre, y_max_pre, z_max_pre = np.max(frame_pre[i], 1)
        
        for j in range(virtual_point_num):
            x[0, j] = np.random.uniform(x_min_pre, x_max_pre)
            x[1, j] = np.random.uniform(y_min_pre, y_max_pre)
            x[2, j] = np.random.uniform(z_min_pre, z_max_pre)
            
            b[(i - 1) * virtual_point_num + j, 0] += np.linalg.norm(x[:, j].reshape(-1, 1) - cluster_pre) ** 2 / num_points_pre
            b[(i - 1) * virtual_point_num + j, 0] -= np.linalg.norm(x[:, j].reshape(-1, 1) - cluster) ** 2 / num_points
            
            b[(i - 1) * virtual_point_num + j, 0] -= np.linalg.norm(x[: j]) ** 2
            
            # 3 * 9
            L = np.kron(x[: j], np.eye(3))
            
            A[(i - 1) * virtual_point_num + j, :] = np.concatenate((-2 * np.dot(cluster_bar, L), -2 * cluster_bar, 2 * x[:, j], 1))

    pose_rel = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(b)
    
    return pose_rel

def evaluate_one_epoch():
    stat_dict = {}
    ap_calculator_list = [APCalculator(iou_thresh, DATASET_CONFIG.class2type) \
                          for iou_thresh in AP_IOU_THRESHOLDS]

    net.eval()  # set model to eval mode (for bn and dp)
    
    if not VIS_DISABLE:
        vis = o3d.visualization.Visualizer()
        pcd = o3d.geometry.PointCloud()
        vis.create_window()
        
        vis_ctrl = vis.get_view_control()

    
    initialized = False
    
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
         [4, 5], [5, 6], [6, 7], [4, 7],
         [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))]
    
    pred = 0
    pred_matching = 0
    gt = 0
    
    prev = None
    
    for batch_idx, batch_data_label in enumerate(tqdm(TEST_DATALOADER)):
        # Forward pass
        loss, end_points = net(batch_data_label, DATASET_CONFIG)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_pred_map_cls = parse_predictions(end_points, CONFIG_DICT,
                                                                           batch_data_label,
                                                                           batch_idx)
        

        batch_gt_map_cls = parse_groundtruths(end_points, CONFIG_DICT)

        point = batch_data_label['point_clouds_camera'].cpu().numpy()[0]
        
        pred += len(batch_pred_map_cls[0])
        gt += len(batch_gt_map_cls[0])
        match_num, match_list = get_maching_num(prev, batch_pred_map_cls)
        pred_matching += match_num
        
        print('pred: {}/{}, gt: {}/{}, pred_matching: {}/{}'.format(pred, pred / (batch_idx + 1), gt, gt / (batch_idx + 1), pred_matching, pred_matching / (batch_idx + 1)))
        
        point_class = get_index(point, batch_pred_map_cls[0])
        
        if match_num >= 4:
            
            frame_pre = []
            frame = []
    
            for i in range(len(match_list)):
                bbox_id = match_list[i]
                
                for j in range(len(batch_pred_map_cls[0])):
                    if bbox_id == batch_pred_map_cls[0][i][3]:
                        frame.append(point[point_class[j]].T)

                for j in range(len(prev[0])):
                    if bbox_id == prev[0][i][3]:
                        frame_pre.append(point[point_class[j]].T)
            
            pose_rel = direct_icp(frame, frame_pre)
        
        prev = batch_pred_map_cls
        # only for batch size = 1
        # dets = {}
        # dets['dets'] = np.array([np.concatenate(corners2xyzrylwh(batch_pred_map_cls[0][i][1])) for i in range(len(batch_pred_map_cls[0]))])
        
        if not VIS_DISABLE:
            pcd.points = o3d.utility.Vector3dVector(point)
            
            vis.clear_geometries()
            
            vis.add_geometry(pcd)
            
            line_sets = [o3d.geometry.LineSet() for _ in range(len(batch_pred_map_cls[0]))]
            for i in range(len(batch_pred_map_cls[0])):
                
                line_sets[i].points = o3d.utility.Vector3dVector(batch_pred_map_cls[0][i][1])
                line_sets[i].lines = o3d.utility.Vector2iVector(lines)
                line_sets[i].colors = o3d.utility.Vector3dVector(colors)
                vis.add_geometry(line_sets[i])
            
            vis_ctrl.set_front([0, 0, 1])
            vis_ctrl.set_lookat([0, 0, 1])
            vis_ctrl.set_up([0, -1, 0])
            
            vis.poll_events()
            vis.update_renderer()
                    
            time.sleep(0.5)
        
        
        for ap_calculator in ap_calculator_list:
            ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

    # Evaluate average precision
    for i, ap_calculator in enumerate(ap_calculator_list):
        print('-' * 10, 'iou_thresh: %f' % (AP_IOU_THRESHOLDS[i]), '-' * 10)
        metrics_dict = ap_calculator.compute_metrics(DUMP_DIR, 'oriented_result')
        for key in metrics_dict:
            log_string('eval %s: %f' % (key, metrics_dict[key]))

    mean_loss = stat_dict['loss'] / float(batch_idx + 1)
    return mean_loss


def eval():
    log_string(str(datetime.now()))
    # Reset numpy seed.
    # REF: https://github.com/pytorch/pytorch/issues/5059
    np.random.seed()
    loss = evaluate_one_epoch()


if __name__ == '__main__':
    eval()
