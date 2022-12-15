import numpy as np
import cv2
import os
import pickle
import open3d as o3d
import time

DATA_PATH = '/home/ylin/Code/UDOLO/scannet/scene0011_00_raw/scene0011_00_unaligned_bbox.npy'
INTRI_PATH = '/home/ylin/Code/UDOLO/scannet/scans/scene0011_00/intrinsic/intrinsic_depth.txt'
POSE_DIR = '/home/ylin/Code/UDOLO/scannet/scans/scene0011_00/pose'
DEPTH_DIR = '/home/ylin/Code/UDOLO/scannet/scans/scene0011_00/depth'
POINT_PATH = '/home/ylin/Code/UDOLO/scannet/scene0011_00_raw/scene0011_00_vert.npy'

'''
    Note:
    ply vertex, depth image: not alighed
    coordiantes system: opencv
    scannet pose: camera to world, not extri
    pose file: listdir out of order
'''

def generate_depth(screen_to_camera):
    
    full_name_list = os.listdir(DEPTH_DIR)
    full_point_cloud = []
    for i in range(len(full_name_list)):
        
        d = cv2.imread(os.path.join(DEPTH_DIR, '{}.png'.format(i)), -1).astype(np.float32)
        
        d /= 1000
        h, w = d.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        xx = xx * d
        yy = yy * d
        pc = np.stack([xx, yy, d], axis=2)
        pc = pc.reshape(-1, 3)
        pc = np.dot(screen_to_camera, pc.T).T
        pos_z = np.nonzero(pc[:, 2] > 0)[0]
        point_cloud = pc[pos_z]
        full_point_cloud.append(point_cloud)
    
    return full_point_cloud

def generate_pose():
    full_name_list = os.listdir(POSE_DIR)
    full_pose = []
    
    for i in range(len(full_name_list)):
        pose = np.genfromtxt(os.path.join(POSE_DIR, full_name_list[i]))
        full_pose.append(pose)
    
    return full_pose

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
    
    i = bbox[2] - bbox[6]
    j = bbox[5] - bbox[6]
    k = bbox[7] - bbox[6]
    v = point - bbox[6]
    
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
        for j in range(bboxes.shape[0]):
            if is_inbbox(points[i], bboxes[j]):
                if j in point_class.keys():
                    point_class[j].append(i)
                else:
                    point_class[j] = [i, ]
    
    return point_class

def get_3d_bbox(bbox):
    
    '''
            3 -------- 0              z  y
           /|         /|              | /
          7 -------- 4 .              |/
          | |        | |              |--------> x
          . 2 -------- 1              
          |/         |/               
          6 -------- 5                
    '''
    
    l, h, w = bbox[:, 3], bbox[:, 4], bbox[:, 5]
    center = bbox[:, :3]
    
    # print(center[0], l[0], h[0], w[0])
    x_corners = np.array([l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2])
    z_corners = np.array([w / 2, w / 2, w / 2, w / 2, -w / 2, -w / 2, -w / 2, -w / 2])
    y_corners = np.array([h / 2, -h / 2, -h / 2, h / 2, h / 2, -h / 2, -h / 2, h / 2])

    # n, 3, 8
    bbox_corners = np.vstack((x_corners, y_corners, z_corners)).T.reshape(-1, 3, 8)
    
    # n, 3, 8
    center = np.repeat(center, 8, 1).reshape(-1, 3, 8)

    # n, 8, 3
    bbox_corners = (center + bbox_corners).transpose((0, 2, 1))
    # print(bbox_corners[0])
    return bbox_corners

def get_camera_bbox(world_bbox):
    
    full_name_list = os.listdir(POSE_DIR)
    full_camera_bbox = []
    
    world_bbox = get_3d_bbox(world_bbox)

    for i in range(len(full_name_list)):
        
        pose = np.linalg.inv(np.genfromtxt(os.path.join(POSE_DIR, '{}.txt'.format(i))))
        r = pose[:3, :3]
        t = pose[:3, 3]
        
        # print(pose, r, t, sep = '\n')
        
        camera_bbox = (np.dot(world_bbox.reshape(-1, 3), r.T) + t).reshape(-1, 8, 3)
        
        # print(camera_bbox[0])
        # print(np.dot(r, world_bbox[0].T).T)
        full_camera_bbox.append(camera_bbox)
    
    return full_camera_bbox
    

def get_camera_edge():
    pass

def generate_camera_bbox():
    pass

def get_pose_err(gt_pose_pre, gt_pose, pose_rel):
    
    '''
    Inputs:
        gt_pose_pre, gt_pose: 4 * 4
        pose_rel: 3 * 4
    '''
    
    rel_poses = np.matmul(np.linalg.inv(gt_pose_pre), gt_pose).astype(np.float32)
    r_err = np.linalg.norm((rel_poses[:3, :3] - pose_rel[:3, :3]), 'fro')
    t_err = np.linalg.norm((rel_poses[:3, 3] - pose_rel[:, 3]))

    print('error:', r_err, t_err)
    return r_err, t_err

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
    
    b = np.zeros((virtual_point_num * cluster_num, 1), dtype=np.float64)
    A = np.zeros((virtual_point_num * cluster_num, 16), dtype=np.float64)
    x = np.zeros((3, virtual_point_num), dtype=np.float64)
    
    for i in range(cluster_num):
        cluster_pre = frame_pre[i]
        cluster = frame[i]
        
        
        num_points_pre = frame_pre[i].shape[1]
        num_points = frame[i].shape[1]
        
        
        cluster_bar = np.sum(cluster, 1) / num_points
        
        
        x_min_pre, y_min_pre, z_min_pre = np.min(frame_pre[i], 1)
        x_max_pre, y_max_pre, z_max_pre = np.max(frame_pre[i], 1)
        
        for j in range(virtual_point_num):
            x[0, j] = (x_max_pre - x_min_pre) * np.random.rand() + x_min_pre
            x[1, j] = (y_max_pre - y_min_pre) * np.random.rand() + y_min_pre
            x[2, j] = (z_max_pre - z_min_pre) * np.random.rand() + z_min_pre
            for k in range(num_points_pre):
                b[i * virtual_point_num + j, 0] += np.linalg.norm(x[:, j] - cluster_pre[:, k]) ** 2 / num_points_pre
            
            for k in range(num_points):
                b[i * virtual_point_num + j, 0] -= np.linalg.norm(cluster[:, k]) ** 2 / num_points
                      
            b[i * virtual_point_num + j, 0] -= np.linalg.norm(x[:, j]) ** 2
            
            # 3 * 9
            L = np.kron(x[:, j], np.eye(3))
            
            A[i * virtual_point_num + j, :] = np.concatenate((-2 * np.dot(cluster_bar, L), -2 * cluster_bar, 2 * x[:, j], [1, ]))
    
    cluster_matrix = np.eye(cluster_num)
    for i in range(cluster_num):
        cluster_matrix[i, i] = frame_pre[i].shape[1]
    W = np.kron(cluster_matrix, np.eye(virtual_point_num))
    
    pose_rel = np.linalg.inv(A.T.dot(W).dot(A)).dot(A.T).dot(b)
    pose_rel = pose_rel[:12, :].reshape(3, -1)[:, :4]
    
    return pose_rel

if __name__ == '__main__':
    world_bbox_ = np.load(DATA_PATH)
    world_bbox = get_3d_bbox(world_bbox_)
    intr = np.genfromtxt(INTRI_PATH)[:3, :3]
    screen_to_camera = np.linalg.inv(intr)
    
    """ world_point = np.load(POINT_PATH)[:, 0:3].astype(np.float32)
    pose = np.linalg.inv(np.genfromtxt(os.path.join(POSE_DIR, '0.txt')))
    r = pose[:3, :3]
    t = pose[:3, 3]
    world_point_1 = np.dot(world_point, r.T) + t
    
    pose = np.linalg.inv(np.genfromtxt(os.path.join(POSE_DIR, '1.txt')))
    r = pose[:3, :3]
    t = pose[:3, 3]
    world_point_2 = np.dot(world_point, r.T) + t """
    
    
    """ points = generate_depth(screen_to_camera)
    pose = generate_pose()
    bbox = get_camera_bbox(world_bbox_)
    # print(bbox[0].shape)
    
    print(len(points))
    print(len(bbox))
    print(len(pose))
    
    mydict = {'points': points, 'bbox': bbox, 'pose': pose}
    output = open('scene0011_00.pkl', 'wb')
    pickle.dump(mydict, output)
    output.close() """

    
    scene11 = pickle.load(open('scene0011_00.pkl', 'rb'))
    points = scene11['points']
    pose = scene11['pose']
    bbox = scene11['bbox']
    # points = [world_point_1, world_point_2]
    
    print('loaded!')
    
    """ vis = o3d.visualization.Visualizer()
    
    pcd = o3d.geometry.PointCloud()
    pcd2 = o3d.geometry.PointCloud()
    vis.create_window()
    
    vis_ctrl = vis.get_view_control()
    opt = vis.get_render_option()
    
    
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
        [4, 5], [5, 6], [6, 7], [4, 7],
        [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [[1, 0, 0] for _ in range(len(lines))] """
    
    matching_num = 0
    prev_points = None
    prev_keys = None
    r_err_list = []
    t_err_list = []
        
    for i in range(0, len(points)):
        points_class = get_index(points[i], bbox[i])
        keys = list(points_class.keys())
        print(keys)
        
        if prev_points != None:
            for j in prev_keys:
                if j in keys:
                    matching_num += 1
        
        if matching_num >= 4:
            frame = []
            frame_pre = []
            for j in prev_keys:
                if j in keys:
                    frame.append(points[i][points_class[j]].T)
                    frame_pre.append(points[i - 1][prev_points[j]].T)

            pose_rel = direct_icp(frame_pre, frame)
            r_err, t_err = get_pose_err(pose[i - 1], pose[i], pose_rel)
            
            r_err_list.append(r_err)
            t_err_list.append(t_err)
            
            print('mean err:', sum(r_err_list) / len(r_err_list), sum(t_err_list) / len(t_err_list))
        prev_points = points_class
        prev_keys = keys
        matching_num = 0
    
        """ # pcd.points = o3d.utility.Vector3dVector(world_point)
        pcd2.points = o3d.utility.Vector3dVector(points[i])
        
        # pcd.paint_uniform_color([0, 1, 0])
        # pcd2.paint_uniform_color([0, 0, 1])
        
        vis.clear_geometries()
        
        # vis.add_geometry(pcd)
        vis.add_geometry(pcd2)
        
        line_sets = [o3d.geometry.LineSet() for _ in range(bbox[i].shape[0])]
        for j in range(bbox[i].shape[0]):
            line_sets[j].points = o3d.utility.Vector3dVector(bbox[i][j])
            line_sets[j].lines = o3d.utility.Vector2iVector(lines)
            line_sets[j].colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(line_sets[j])
        
        #vis_ctrl.set_front([0, 1, 0])
        #vis_ctrl.set_lookat(look_at)
        #vis_ctrl.set_up([0, 0, 1])
        opt.show_coordinate_frame = True
        
        vis.poll_events()
        vis.update_renderer()
                
        time.sleep(0.5) """