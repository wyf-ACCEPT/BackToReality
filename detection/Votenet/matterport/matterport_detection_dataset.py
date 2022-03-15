# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import pc_util
import matterport_utils
from model_util_matterport import rotate_aligned_boxes

from model_util_matterport import MatterportDatasetConfig_md40
DC = MatterportDatasetConfig_md40()
MAX_NUM_OBJ = 256#64
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

class MatterportDetectionDataset(Dataset):
       
    def __init__(self, split_set='train', data_path="matterport_train_detection_data_md40", num_points=20000, use_color=False, use_height=False, augment=False, center_jitter=0):

        self.data_path = os.path.join(BASE_DIR, data_path)
        self.center_jitter = center_jitter
        all_scan_names = list(set([os.path.basename(x)[0:18] if x.startswith('scene_aug') \
            else os.path.basename(x)[0:12] for x in os.listdir(self.data_path)]))

        if split_set == 'all':            
            self.scan_names = all_scan_names
        elif split_set in ['train', 'val', 'test']:
            split_filenames = os.path.join(ROOT_DIR, 'matterport/meta_data',
                'matterport3d_{}.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [sname for sname in self.scan_names \
                if sname in all_scan_names]
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        elif split_set == 'train_aug':
            split_filenames = os.path.join(ROOT_DIR, 'matterport/meta_data',
                'matterport3d_train.txt'.format(split_set))
            with open(split_filenames, 'r') as f:
                self.scan_names = f.read().splitlines()   
            # remove unavailiable scans
            self.scan_names = [sname for sname in all_scan_names \
                if sname in self.scan_names or 'aug' in sname]
            num_scans = len(self.scan_names)
            print('kept {} scans out of {}'.format(len(self.scan_names), num_scans))
            num_scans = len(self.scan_names)
        else:
            print('illegal split name')
            return
        
        self.num_points = num_points
        self.use_color = use_color        
        self.use_height = use_height
        self.augment = augment
        self.augment_random_params = np.random.rand(self.__len__(), 2)
        if self.center_jitter != 0 and 'obj' not in self.data_path:
            #self.delta = (np.random.rand(self.__len__(), MAX_NUM_OBJ, 3) - 0.5) * self.center_jitter
            #np.save('annotation_error.npy', self.delta)
            #sys.exit(0)
            self.delta = np.load('matterport/annotation_error.npy')
        elif self.center_jitter != 0 and 'obj' in self.data_path:
            self.delta = (np.random.rand(self.__len__(), MAX_NUM_OBJ, 3) - 0.5) * self.center_jitter
        else:
            self.delta = np.zeros((self.__len__(), MAX_NUM_OBJ, 3))
        # np.array([(np.random.rand()-0.5)*self.center_jitter, (np.random.rand()-0.5)*self.center_jitter, (np.random.rand()-0.5)*self.center_jitter]) * size_gts
       
    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            angle_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            angle_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            point_votes: (N,3) with votes XYZ
            point_votes_mask: (N,) with 0/1 with 1 indicating the point is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            pcl_color: unused
        """
        
        scan_name = self.scan_names[idx]    
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name)+'_vert.npy')
        instance_labels = np.load(os.path.join(self.data_path, scan_name)+'_ins_label.npy')
        semantic_labels = np.load(os.path.join(self.data_path, scan_name)+'_sem_label.npy')
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name)+'_bbox.npy')

        if not self.use_color:
            point_cloud = mesh_vertices[:,0:3] # do not use color for now
            pcl_color = mesh_vertices[:,3:6]
        else:
            point_cloud = mesh_vertices[:,0:6] 
            point_cloud[:,3:] = (point_cloud[:,3:]-MEAN_COLOR_RGB)/256.0
        
        if self.use_height:
            floor_height = np.percentile(point_cloud[:,2],0.99)
            height = point_cloud[:,2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) 
            
        # ------------------------------- LABELS ------------------------------        
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ))    
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        size_gts = np.zeros((MAX_NUM_OBJ, 3))
        
        point_cloud, choices = pc_util.random_sampling(point_cloud,
            self.num_points, return_choices=True)        
        instance_labels = instance_labels[choices]
        semantic_labels = semantic_labels[choices]
        
        pcl_color = pcl_color[choices]
        
        target_bboxes_mask[0:instance_bboxes.shape[0]] = 1
        target_bboxes[0:instance_bboxes.shape[0],:] = instance_bboxes[:,0:6]
        
        # ------------------------------- DATA AUGMENTATION ------------------------------        
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:,0] = -1 * point_cloud[:,0]
                instance_bboxes[:,0] = -1 * instance_bboxes[:,0]
                instance_bboxes[:,6] = np.pi - instance_bboxes[:,6]
                target_bboxes[:,0] = -1 * target_bboxes[:,0]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random()*np.pi/3) - np.pi/6 # -30 ~ +30 degree
            rot_mat = matterport_utils.rotz(rot_angle)

            point_cloud[:,0:3] = np.dot(point_cloud[:,0:3], np.transpose(rot_mat))
            instance_bboxes[:,0:3] = np.dot(instance_bboxes[:,0:3], np.transpose(rot_mat))
            instance_bboxes[:,6] -= rot_angle
            target_bboxes[:,0:3] = np.dot(target_bboxes[:,0:3], np.transpose(rot_mat))

        # compute votes *AFTER* augmentation
        # generate votes
        # Note: since there's no map between bbox instance labels and
        # pc instance_labels (it had been filtered 
        # in the data preparation step) we'll compute the instance bbox
        # from the points sharing the same instance label.
        
        point_votes = np.zeros([self.num_points, 3])
        point_votes_mask = np.zeros(self.num_points)
        for i_instance in np.unique(instance_labels):            
            # find all points belong to that instance
            ind = np.where(instance_labels == i_instance)[0]
            # find the semantic label            
            if semantic_labels[ind[0]] in DC.nyu40ids:
                x = point_cloud[ind,:3]
                center = 0.5*(x.min(0) + x.max(0))
                point_votes[ind, :] = center - x
                point_votes_mask[ind] = 1.0
        
        for i in range(instance_bboxes.shape[0]):
            instance_bbox = instance_bboxes[i]
            angle_class, angle_residual = DC.angle2class(instance_bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
        
        point_votes = np.tile(point_votes, (1, 3)) # make 3 votes identical 
        
        class_ind = [np.where(DC.nyu40ids == x)[0][0] for x in instance_bboxes[:,-1]]
        # NOTE: set size class as semantic class. Consider use size2class.
        size_classes[0:instance_bboxes.shape[0]] = class_ind
        size_residuals[0:instance_bboxes.shape[0], :] = \
            target_bboxes[0:instance_bboxes.shape[0], 3:6] - DC.mean_size_arr[class_ind,:]
        size_gts[0:instance_bboxes.shape[0], :] = target_bboxes[0:instance_bboxes.shape[0], 3:6]
        
        # Using center jitter to approaximate the error of human labeling
        gt_centers = target_bboxes[:,0:3]
        if self.center_jitter != 0:
            # gt_centers += np.array([np.random.rand()*0.1-0.05, np.random.rand()*0.1-0.05, np.random.rand()*0.1-0.05]) * size_gts
            # gt_centers += np.array([(np.random.rand()-0.5)*self.center_jitter, (np.random.rand()-0.5)*self.center_jitter, (np.random.rand()-0.5)*self.center_jitter]) * size_gts
            gt_centers += size_gts * self.delta[idx]
            
        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = gt_centers.astype(np.float32)
        ret_dict['center_jitter'] = (size_gts * self.delta[idx]).astype(np.float32)
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))                                
        target_bboxes_semcls[0:instance_bboxes.shape[0]] = \
            [DC.nyu40id2class[x] for x in instance_bboxes[:,-1][0:instance_bboxes.shape[0]]]                
        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        if instance_bboxes.shape[0] != 0:
            ret_dict['cloud_label'] = np.eye(13)[ret_dict['sem_cls_label'][0:instance_bboxes.shape[0]]].max(axis=0)
        else:
            ret_dict['cloud_label'] = np.zeros(13)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)
        ret_dict['vote_label'] = point_votes.astype(np.float32)
        ret_dict['vote_label_mask'] = point_votes_mask.astype(np.int64)
        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['pcl_color'] = pcl_color
        return ret_dict


############# Visualizaion ########

def viz_votes(pc, point_votes, point_votes_mask, name=''):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask==1)
    pc_obj = pc[inds,0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds,0:3]    
    pc_util.write_ply(pc_obj, 'pc_obj{}.ply'.format(name))
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1{}.ply'.format(name))
    
def viz_obb(pc, label, mask, angle_classes, angle_residuals,
    size_classes, size_residuals, name=''):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i,0:3]
        heading_angle = 0 # hard code to 0
        box_size = DC.mean_size_arr[size_classes[i], :] + size_residuals[i, :]
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)        
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, 'gt_obbs{}.ply'.format(name))
    pc_util.write_ply(label[mask==1,:], 'gt_centroids{}.ply'.format(name))

    
if __name__=='__main__':
    dset = MatterportDetectionDataset(use_height=True, num_points=40000)
    
    count_sum = np.zeros((13, 13))
    for d in dset:
        cl = d['cloud_label']
        idx_list = []
        for i in range(13):
            if cl[i] == 1.0:
                idx_list.append(i)
        for i in idx_list:
            for j in idx_list:
                count_sum[i][j] += 1
    np.save('context_vectors.npy', count_sum)
    
    context_vectors = np.load('context_vectors.npy')
    cl_rate = 25
    # list of tuple (scene_idx, score)
    bathtub = []
    bench = []
    desk = []
    dresser = []
    bathtub_vector = context_vectors[0]
    bathtub_vector[0] = 0
    bench_vector = context_vectors[2]
    bench_vector[2] = 0
    desk_vector = context_vectors[5]
    desk_vector[5] = 0
    dresser_vector = context_vectors[7]
    dresser_vector[7] = 0
    for d in dset:
        cl = d['cloud_label']
        scan_name = dset.scan_names[d['scan_idx']]
        bathtub.append((scan_name, np.dot(cl, bathtub_vector) - cl_rate * sum(cl)))
        bench.append((scan_name, np.dot(cl, bench_vector) - cl_rate * sum(cl)))
        desk.append((scan_name, np.dot(cl, desk_vector) - cl_rate * sum(cl)))
        dresser.append((scan_name, np.dot(cl, dresser_vector) - cl_rate * sum(cl)))
    bathtub = sorted(bathtub, key=lambda x: x[1], reverse=True)[:70]
    bench = sorted(bench, key=lambda x: x[1], reverse=True)[:15]
    desk = sorted(desk, key=lambda x: x[1], reverse=True)[:20]
    dresser = sorted(dresser, key=lambda x: x[1], reverse=True)[:50]
    bathtub_scan = [x[0] for x in bathtub]
    bench_scan = [x[0] for x in bench]
    desk_scan = [x[0] for x in desk]
    dresser_scan = [x[0] for x in dresser]
    print([bathtub_scan, bench_scan, desk_scan, dresser_scan])
    np.save('scans_toadd_scarce.npy', [bathtub_scan, bench_scan, desk_scan, dresser_scan])
    
