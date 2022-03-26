import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
from .losses import smoothl1_loss, l1_loss, SigmoidFocalClassificationLoss
from .ap_helper import flip_axis_to_camera
from utils.box_util import get_3d_box
from torch.autograd import Variable, Function


def compute_points_obj_cls_loss_hard_topk(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
    gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
    gt_size = end_points['size_gts'][:, :, 0:3]  # B, K2, 3
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]
    K2 = gt_center.shape[1]

    point_instance_label = end_points['point_instance_label']  # B, num_points
    object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
    object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
    delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, K2, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
    euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
    topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + \
                (box_label_mask[:, :, None] - 1)  # BxK2xtopk
    topk_inds = topk_inds.long()  # BxK2xtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, K2xtopk
    batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous()

    objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    objectness_label[objectness_label_mask < 0] = 0

    total_num_points = B * K
    end_points[f'points_hard_topk{topk}_pos_ratio'] = \
        torch.sum(objectness_label.float()) / float(total_num_points)
    end_points[f'points_hard_topk{topk}_neg_ratio'] = 1 - end_points[f'points_hard_topk{topk}_pos_ratio']

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
    objectness_loss = cls_loss_src.sum() / B

    # Compute recall upper bound
    padding_array = torch.arange(0, B).to(point_instance_label.device) * 10000
    padding_array = padding_array.unsqueeze(1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = torch.unique(point_instance_label).shape[0] - 1
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1)
    num_query_bboxes = torch.unique(pos_points_instance_label).shape[0] - 1
    if num_gt_bboxes > 0:
        end_points[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes

    return objectness_loss


def compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers):
    """ Compute objectness loss for the proposals.
    """

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal

    objectness_loss_sum = 0.0
    for prefix in prefixes:
        # Associate proposal and GT objects
        seed_inds = end_points['seed_inds'].long()  # B,num_seed in [0,num_points-1]
        gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
        query_points_sample_inds = end_points['query_points_sample_inds'].long()

        B = seed_inds.shape[0]
        K = query_points_sample_inds.shape[1]
        K2 = gt_center.shape[1]

        seed_obj_gt = torch.gather(end_points['point_obj_mask'], 1, seed_inds)  # B,num_seed
        query_points_obj_gt = torch.gather(seed_obj_gt, 1, query_points_sample_inds)  # B, query_points

        point_instance_label = end_points['point_instance_label']  # B, num_points
        seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
        query_points_instance_label = torch.gather(seed_instance_label, 1, query_points_sample_inds)  # B,query_points

        objectness_mask = torch.ones((B, K)).cuda()

        # Set assignment
        object_assignment = query_points_instance_label  # (B,K) with values in 0,1,...,K2-1
        object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox

        end_points[f'{prefix}objectness_label'] = query_points_obj_gt
        end_points[f'{prefix}objectness_mask'] = objectness_mask
        end_points[f'{prefix}object_assignment'] = object_assignment
        total_num_proposal = query_points_obj_gt.shape[0] * query_points_obj_gt.shape[1]
        end_points[f'{prefix}pos_ratio'] = \
            torch.sum(query_points_obj_gt.float().cuda()) / float(total_num_proposal)
        end_points[f'{prefix}neg_ratio'] = \
            torch.sum(objectness_mask.float()) / float(total_num_proposal) - end_points[f'{prefix}pos_ratio']

        # Compute objectness loss
        objectness_scores = end_points[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = objectness_mask.float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        cls_loss_src = criterion(objectness_scores.transpose(2, 1).contiguous().view(B, K, 1),
                                 query_points_obj_gt.unsqueeze(-1),
                                 weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B

        end_points[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss

    return objectness_loss_sum, end_points


def compute_box_and_sem_cls_loss(end_points, config, num_decoder_layers,
                                 center_loss_type='smoothl1', center_delta=1.0,
                                 size_loss_type='smoothl1', size_delta=1.0,
                                 heading_loss_type='smoothl1', heading_delta=1.0):
    """ Compute 3D bounding box and semantic classification loss.
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = end_points[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]
        # Compute center loss
        pred_center = end_points[f'{prefix}center']
        gt_center = end_points['center_label'][:, :, 0:3]

        if center_loss_type == 'smoothl1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute heading loss
        heading_class_label = torch.gather(end_points['heading_class_label'], 1,
                                           object_assignment)  # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        heading_class_loss = criterion_heading_class(end_points[f'{prefix}heading_scores'].transpose(2, 1),
                                                     heading_class_label)  # (B,K)
        heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        heading_residual_label = torch.gather(end_points['heading_residual_label'], 1,
                                              object_assignment)  # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1],
                                                       num_heading_bin).zero_()
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                       1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        heading_residual_normalized_error = torch.sum(
            end_points[f'{prefix}heading_residuals_normalized'] * heading_label_one_hot,
            -1) - heading_residual_normalized_label

        if heading_loss_type == 'smoothl1':
            heading_residual_normalized_loss = heading_delta * smoothl1_loss(heading_residual_normalized_error,
                                                                             delta=heading_delta)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            heading_residual_normalized_loss = l1_loss(heading_residual_normalized_error)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute size loss
        size_class_label = torch.gather(end_points['size_class_label'], 1,
                                        object_assignment)  # select (B,K) from (B,K2)
        criterion_size_class = nn.CrossEntropyLoss(reduction='none')
        size_class_loss = criterion_size_class(end_points[f'{prefix}size_scores'].transpose(2, 1),
                                               size_class_label)  # (B,K)
        size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        size_residual_label = torch.gather(
            end_points['size_residual_label'], 1,
            object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)

        size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
        size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1),
                                    1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
        size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
        predicted_size_residual_normalized = torch.sum(
            end_points[f'{prefix}size_residuals_normalized'] * size_label_one_hot_tiled,
            2)  # (B,K,3)

        mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
            0)  # (1,1,num_size_cluster,3)
        mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
        size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)

        size_residual_normalized_error = predicted_size_residual_normalized - size_residual_label_normalized

        if size_loss_type == 'smoothl1':
            size_residual_normalized_loss = size_delta * smoothl1_loss(size_residual_normalized_error,
                                                                       delta=size_delta)  # (B,K,3) -> (B,K)
            size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                    torch.sum(objectness_label) + 1e-6)
        elif size_loss_type == 'l1':
            size_residual_normalized_loss = l1_loss(size_residual_normalized_error)  # (B,K,3) -> (B,K)
            size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                    torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # 3.4 Semantic cls loss
        sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(end_points[f'{prefix}sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        end_points[f'{prefix}center_loss'] = center_loss
        end_points[f'{prefix}heading_cls_loss'] = heading_class_loss
        end_points[f'{prefix}heading_reg_loss'] = heading_residual_normalized_loss
        end_points[f'{prefix}size_cls_loss'] = size_class_loss
        end_points[f'{prefix}size_reg_loss'] = size_residual_normalized_loss
        box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + 0.1 * size_class_loss + size_residual_normalized_loss
        end_points[f'{prefix}box_loss'] = box_loss
        end_points[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss
    # WithSem Center Prediction
    #criterion_center_cls = nn.CrossEntropyLoss(reduction='none')
    #box_label_mask = end_points['box_label_mask']
    #center_cls_loss = criterion_center_cls(end_points['center_sem_scores'], end_points['sem_cls_label'])  # (B,K2)
    #center_cls_loss = torch.sum(center_cls_loss * box_label_mask) / (torch.sum(box_label_mask) + 1e-6)
    #sem_cls_loss_sum += center_cls_loss
    return box_loss_sum, sem_cls_loss_sum, end_points


def get_loss(end_points, config, num_decoder_layers,
             query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk=5,
             center_loss_type='smoothl1', center_delta=1.0,
             size_loss_type='smoothl1', size_delta=1.0,
             heading_loss_type='smoothl1', heading_delta=1.0):
    """ Loss functions
    """
    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss = compute_points_obj_cls_loss_hard_topk(end_points, query_points_obj_topk)

        end_points['query_points_generation_loss'] = query_points_generation_loss
    else:
        query_points_generation_loss = 0.0

    # Obj loss
    objectness_loss_sum, end_points = \
        compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers)

    end_points['sum_heads_objectness_loss'] = objectness_loss_sum

    # Box loss and sem cls loss
    box_loss_sum, sem_cls_loss_sum, end_points = compute_box_and_sem_cls_loss(
        end_points, config, num_decoder_layers,
        center_loss_type, center_delta=center_delta,
        size_loss_type=size_loss_type, size_delta=size_delta,
        heading_loss_type=heading_loss_type, heading_delta=heading_delta)
    end_points['sum_heads_box_loss'] = box_loss_sum
    end_points['sum_heads_sem_cls_loss'] = sem_cls_loss_sum

    # means average proposal with prediction loss
    loss = query_points_generator_loss_coef * query_points_generation_loss + \
           1.0 / (num_decoder_layers + 1) * (
                   obj_loss_coef * objectness_loss_sum + box_loss_coef * box_loss_sum + sem_cls_loss_coef * sem_cls_loss_sum)
    loss *= 10

    end_points['loss'] = loss
    return loss, end_points



####################### Loss for Domain Adaptation #######################


def compute_points_obj_cls_loss_hard_topk_weak(end_points, topk):
    box_label_mask = end_points['box_label_mask']
    seed_inds = end_points['seed_inds'].long()  # B, K
    seed_xyz = end_points['seed_xyz']  # B, K, 3
    seeds_obj_cls_logits = end_points['seeds_obj_cls_logits']  # B, 1, K
    gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
    #gt_size = end_points['size_gts'][:, :, 0:3]  # B, K2, 3
    B = gt_center.shape[0]
    K = seed_xyz.shape[1]
    K2 = gt_center.shape[1]

    #point_instance_label = end_points['point_instance_label']  # B, num_points
    #object_assignment = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    #object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox
    #object_assignment_one_hot = torch.zeros((B, K, K2)).to(seed_xyz.device)
    #object_assignment_one_hot.scatter_(2, object_assignment.unsqueeze(-1), 1)  # (B, K, K2)
    delta_xyz = seed_xyz.unsqueeze(2) - gt_center.unsqueeze(1)  # (B, K, K2, 3)
    #delta_xyz = delta_xyz / (gt_size.unsqueeze(1) + 1e-6)  # (B, K, K2, 3)
    new_dist = torch.sum(delta_xyz ** 2, dim=-1)
    euclidean_dist1 = torch.sqrt(new_dist + 1e-6)  # BxKxK2
    #euclidean_dist1 = euclidean_dist1 * object_assignment_one_hot + 100 * (1 - object_assignment_one_hot)  # BxKxK2
    euclidean_dist1 = euclidean_dist1.transpose(1, 2).contiguous()  # BxK2xK
    topk_inds = torch.topk(euclidean_dist1, topk, largest=False)[1] * box_label_mask[:, :, None] + \
                (box_label_mask[:, :, None] - 1)  # BxK2xtopk
    topk_inds = topk_inds.long()  # BxK2xtopk
    topk_inds = topk_inds.view(B, -1).contiguous()  # B, K2xtopk
    batch_inds = torch.arange(B).unsqueeze(1).repeat(1, K2 * topk).to(seed_xyz.device)
    batch_topk_inds = torch.stack([batch_inds, topk_inds], -1).view(-1, 2).contiguous()

    objectness_label = torch.zeros((B, K + 1), dtype=torch.long).to(seed_xyz.device)
    objectness_label[batch_topk_inds[:, 0], batch_topk_inds[:, 1]] = 1
    objectness_label = objectness_label[:, :K]
    #objectness_label_mask = torch.gather(point_instance_label, 1, seed_inds)  # B, num_seed
    #objectness_label[objectness_label_mask < 0] = 0

    total_num_points = B * K
    end_points[f'points_hard_topk{topk}_pos_ratio'] = \
        torch.sum(objectness_label.float()) / float(total_num_points)
    end_points[f'points_hard_topk{topk}_neg_ratio'] = 1 - end_points[f'points_hard_topk{topk}_pos_ratio']

    # Compute objectness loss
    criterion = SigmoidFocalClassificationLoss()
    cls_weights = (objectness_label >= 0).float()
    cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
    cls_weights /= torch.clamp(cls_normalizer, min=1.0)
    cls_loss_src = criterion(seeds_obj_cls_logits.view(B, K, 1), objectness_label.unsqueeze(-1), weights=cls_weights)
    objectness_loss = cls_loss_src.sum() / B

    # Compute recall upper bound
    '''
    padding_array = torch.arange(0, B).to(point_instance_label.device) * 10000
    padding_array = padding_array.unsqueeze(1)  # B,1
    point_instance_label_mask = (point_instance_label < 0)  # B,num_points
    point_instance_label = point_instance_label + padding_array  # B,num_points
    point_instance_label[point_instance_label_mask] = -1
    num_gt_bboxes = torch.unique(point_instance_label).shape[0] - 1
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    pos_points_instance_label = seed_instance_label * objectness_label + (objectness_label - 1)
    num_query_bboxes = torch.unique(pos_points_instance_label).shape[0] - 1
    if num_gt_bboxes > 0:
        end_points[f'points_hard_topk{topk}_upper_recall_ratio'] = num_query_bboxes / num_gt_bboxes
    '''

    return objectness_loss

def nn_distance(pc1, pc2, l1smooth=False, delta=1.0, l1=False):
    """
    Input:
        pc1: (B,N,C) torch tensor
        pc2: (B,M,C) torch tensor
        l1smooth: bool, whether to use l1smooth loss
        delta: scalar, the delta used in l1smooth loss
    Output:
        dist1: (B,N) torch float32 tensor
        idx1: (B,N) torch int64 tensor
        dist2: (B,M) torch float32 tensor
        idx2: (B,M) torch int64 tensor
    """
    N = pc1.shape[1]
    M = pc2.shape[1]
    pc1_expand_tile = pc1.unsqueeze(2).repeat(1,1,M,1)
    pc2_expand_tile = pc2.unsqueeze(1).repeat(1,N,1,1)
    pc_diff = pc1_expand_tile - pc2_expand_tile
    
    if l1smooth:
        pc_dist = torch.sum(huber_loss(pc_diff, delta), dim=-1) # (B,N,M)
    elif l1:
        pc_dist = torch.sum(torch.abs(pc_diff), dim=-1) # (B,N,M)
    else:
        pc_dist = torch.sum(pc_diff**2, dim=-1) # (B,N,M)
    dist1, idx1 = torch.min(pc_dist, dim=2) # (B,N)
    dist2, idx2 = torch.min(pc_dist, dim=1) # (B,M)
    return dist1, idx1, dist2, idx2

def compute_objectness_loss_based_on_query_points_weak(end_points, num_decoder_layers):
    """ Compute objectness loss for the proposals.
    """

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal

    objectness_loss_sum = 0.0
    for prefix in prefixes:
        # Associate proposal and GT objects
        seed_inds = end_points['seed_inds'].long()  # B,num_seed in [0,num_points-1]
        gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
        query_points_sample_inds = end_points['query_points_sample_inds'].long()
        aggregated_vote_xyz = end_points['query_points_xyz']

        B = seed_inds.shape[0]
        K = query_points_sample_inds.shape[1]
        K2 = gt_center.shape[1]
        dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

        euclidean_dist1 = torch.sqrt(dist1+1e-6)
        objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
        #objectness_mask = torch.zeros((B,K)).cuda()
        objectness_label[euclidean_dist1<0.3] = 1
        #objectness_mask[euclidean_dist1<0.3] = 1
        #objectness_mask[euclidean_dist1>0.6] = 1

        #seed_obj_gt = torch.gather(end_points['point_obj_mask'], 1, seed_inds)  # B,num_seed
        #query_points_obj_gt = torch.gather(seed_obj_gt, 1, query_points_sample_inds)  # B, query_points

        #point_instance_label = end_points['point_instance_label']  # B, num_points
        #seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
        #query_points_instance_label = torch.gather(seed_instance_label, 1, query_points_sample_inds)  # B,query_points

        objectness_mask = torch.ones((B, K)).cuda()
        object_assignment = ind1.cuda()

        # Set assignment
        #object_assignment = query_points_instance_label  # (B,K) with values in 0,1,...,K2-1
        #object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox

        end_points[f'{prefix}objectness_label'] = objectness_label
        end_points[f'{prefix}objectness_mask'] = objectness_mask
        end_points[f'{prefix}object_assignment'] = object_assignment

        # Compute objectness loss
        objectness_scores = end_points[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = objectness_mask.float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        cls_loss_src = criterion(objectness_scores.transpose(2, 1).contiguous().view(B, K, 1),
                                 objectness_label.unsqueeze(-1),
                                 weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B

        end_points[f'{prefix}objectness_loss'] = objectness_loss
        objectness_loss_sum += objectness_loss

    return objectness_loss_sum, end_points

def compute_center_and_sem_cls_loss(end_points, config, num_decoder_layers,
                                 center_loss_type='smoothl1', center_delta=1.0,
                                 size_loss_type='smoothl1', size_delta=1.0,
                                 heading_loss_type='smoothl1', heading_delta=1.0):
    """ Compute 3D bounding box and semantic classification loss.
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = end_points[f'{prefix}object_assignment']
        batch_size = object_assignment.shape[0]
        # Compute center loss
        pred_center = end_points[f'{prefix}center']
        gt_center = end_points['center_label'][:, :, 0:3]

        size_class_label = torch.gather(end_points['size_class_label'], 1,
                                        object_assignment)  # select (B,K) from (B,K2)
        center_margin = torch.from_numpy(0.05 * mean_size_arr[size_class_label.cpu(), :]).cuda()   # (B,K,3)

        if center_loss_type == 'smoothl1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss -= center_margin
            center_loss[center_loss < 0] = 0
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss -= center_margin
            center_loss[center_loss < 0] = 0
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute size loss
        #size_class_label = torch.gather(end_points['size_class_label'], 1,
        #                                object_assignment)  # select (B,K) from (B,K2)
        criterion_size_class = nn.CrossEntropyLoss(reduction='none')
        size_class_loss = criterion_size_class(end_points[f'{prefix}size_scores'].transpose(2, 1),
                                               size_class_label)  # (B,K)
        size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        # 3.4 Semantic cls loss
        sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(end_points[f'{prefix}sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        end_points[f'{prefix}center_loss'] = center_loss
        end_points[f'{prefix}size_cls_loss'] = size_class_loss
        box_loss = center_loss + 0.1 * size_class_loss
        end_points[f'{prefix}box_loss'] = box_loss
        end_points[f'{prefix}sem_cls_loss'] = sem_cls_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss
    # WithSem Center Prediction
    #criterion_center_cls = nn.CrossEntropyLoss(reduction='none')
    #box_label_mask = end_points['box_label_mask']
    #center_cls_loss = criterion_center_cls(end_points['center_sem_scores'], end_points['sem_cls_label'])  # (B,K2)
    #center_cls_loss = torch.sum(center_cls_loss * box_label_mask) / (torch.sum(box_label_mask) + 1e-6)
    #sem_cls_loss_sum += center_cls_loss
    return box_loss_sum, sem_cls_loss_sum, end_points

def get_loss_weak(end_points, config, num_decoder_layers,
             query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk=5,
             center_loss_type='smoothl1', center_delta=1.0,
             size_loss_type='smoothl1', size_delta=1.0,
             heading_loss_type='smoothl1', heading_delta=1.0):
    """ Loss functions
    """
    if 'seeds_obj_cls_logits' in end_points.keys():
        query_points_generation_loss_ = compute_points_obj_cls_loss_hard_topk(end_points, query_points_obj_topk)
        query_points_generation_loss__ = compute_points_obj_cls_loss_hard_topk_weak(end_points, query_points_obj_topk)
        query_points_generation_loss = 0.000 * query_points_generation_loss_ + query_points_generation_loss__

        end_points['query_points_generation_loss'] = query_points_generation_loss
    else:
        query_points_generation_loss = 0.0

    # Obj loss
    objectness_loss_sum_, end_points = \
        compute_objectness_loss_based_on_query_points(end_points, num_decoder_layers)
    objectness_loss_sum__, end_points = \
        compute_objectness_loss_based_on_query_points_weak(end_points, num_decoder_layers)
    objectness_loss_sum = 0.000 * objectness_loss_sum_ + objectness_loss_sum__
    

    end_points['sum_heads_objectness_loss'] = objectness_loss_sum

    # Box loss and sem cls loss
    box_loss_sum_, sem_cls_loss_sum_, end_points = compute_box_and_sem_cls_loss(
        end_points, config, num_decoder_layers,
        center_loss_type, center_delta=center_delta,
        size_loss_type=size_loss_type, size_delta=size_delta,
        heading_loss_type=heading_loss_type, heading_delta=heading_delta)
    box_loss_sum__, sem_cls_loss_sum__, end_points = compute_center_and_sem_cls_loss(
        end_points, config, num_decoder_layers,
        center_loss_type, center_delta=center_delta,
        size_loss_type=size_loss_type, size_delta=size_delta,
        heading_loss_type=heading_loss_type, heading_delta=heading_delta)
    box_loss_sum, sem_cls_loss_sum = 0.000 * box_loss_sum_ + box_loss_sum__, 0.000 * sem_cls_loss_sum_ + sem_cls_loss_sum__
    end_points['sum_heads_box_loss'] = box_loss_sum
    end_points['sum_heads_sem_cls_loss'] = sem_cls_loss_sum

    # means average proposal with prediction loss
    loss = query_points_generator_loss_coef * query_points_generation_loss + \
           1.0 / (num_decoder_layers + 1) * (
                   obj_loss_coef * objectness_loss_sum + box_loss_coef * box_loss_sum + sem_cls_loss_coef * sem_cls_loss_sum)
    loss *= 10

    end_points['loss'] = loss
    return loss, end_points


class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True, sigmoid=False, reduce=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1) * 1.0)
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.sigmoid = sigmoid
        self.reduce = reduce
    def forward(self, inputs, targets, global_weight=None):
        N = inputs.size(0)
        # print(N)
        C = inputs.size(1)
        if self.sigmoid:
            P = F.sigmoid(inputs)
            #F.softmax(inputs)
            if targets == 0:
                probs = 1 - P#(P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
            if targets == 1:
                probs = P  # (P * class_mask).sum(1).view(-1, 1)
                log_p = probs.log()
                batch_loss = - (torch.pow((1 - probs), self.gamma)) * log_p
        else:
            #inputs = F.sigmoid(inputs)
            P = F.softmax(inputs, dim=-1)
            class_mask = inputs.data.new(N, C).fill_(0)
            class_mask = Variable(class_mask)
            ids = targets.view(-1, 1)
            class_mask.scatter_(1, ids.data, 1.)

            if inputs.is_cuda and not self.alpha.is_cuda:
                self.alpha = self.alpha.cuda()
            alpha = self.alpha[ids.data.view(-1)]

            probs = (P * class_mask).sum(1).view(-1, 1)

            log_p = probs.log()
            # print('probs size= {}'.format(probs.size()))
            # print(probs)

            batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
            # print('-----bacth_loss------')
            # print(batch_loss)

        if not self.reduce:
            return batch_loss
        if self.size_average:
            if global_weight is not None:
                global_weight = global_weight.view(-1, 1)
                batch_loss = batch_loss * global_weight
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


def get_loss_DA(end_points_S, end_points_T, config, num_decoder_layers,
             query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk=5,
             center_loss_type='smoothl1', center_delta=1.0,
             size_loss_type='smoothl1', size_delta=1.0,
             heading_loss_type='smoothl1', heading_delta=1.0):
    """ Loss functions
    """
    loss = 0.5 * get_loss(end_points_S, config, num_decoder_layers, query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk, center_loss_type, center_delta, size_loss_type, size_delta, heading_loss_type, heading_delta)[0] + get_loss_weak(end_points_T, config, num_decoder_layers, query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk, center_loss_type, center_delta, size_loss_type, size_delta, heading_loss_type, heading_delta)[0]

    FL_global = FocalLoss(class_num=2, gamma=3)
    #CST_loss = torch.nn.MSELoss(size_average=True)

    # Source domain global
    global_d_pred_S = end_points_S['global_d_pred']
    domain_S = Variable(torch.zeros(global_d_pred_S.size(0)).long().cuda())
    source_dloss = 1.0 * FL_global(global_d_pred_S, domain_S)
    
    # Target domain global
    global_d_pred_T = end_points_T['global_d_pred']
    domain_T = Variable(torch.ones(global_d_pred_T.size(0)).long().cuda())
    target_dloss = 1.0 * FL_global(global_d_pred_T, domain_T)
    
    
    # S/T local
    prefixes = ['last_']
    for prefix in prefixes:
        local_d_pred_S = end_points_S[f'{prefix}local_d_pred'].transpose(1,2).contiguous().squeeze(-1)
        source_dloss += 1.0 * torch.mean(local_d_pred_S**2 * end_points_S[f'{prefix}objectness_label'])

        local_d_pred_T = end_points_T[f'{prefix}local_d_pred'].transpose(1,2).contiguous().squeeze(-1)
        target_dloss += 1.0 * torch.mean((1-local_d_pred_T)**2 * end_points_T[f'{prefix}objectness_label'])
    

    DA_loss = source_dloss + target_dloss
    loss += 10 * DA_loss

    return loss, end_points_S, end_points_T


def compute_jitter_loss(end_points):
    # center_jitter: B 64 3
    # jitter_pred: B 3 64
    jitter_loss = ((end_points['center_jitter']-end_points['jitter_pred'].transpose(1,2).contiguous())**2).mean()
    end_points['jitter_loss'] = jitter_loss
    return jitter_loss


def get_loss_DA_jitter(end_points_S, end_points_T, epoch, config, num_decoder_layers,
             query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk=5,
             center_loss_type='smoothl1', center_delta=1.0,
             size_loss_type='smoothl1', size_delta=1.0,
             heading_loss_type='smoothl1', heading_delta=1.0):
    """ Loss functions
    """
    if epoch > -1:
        end_points_S['center_label'] -= min(epoch/120.0, 1.0) * end_points_S['center_jitter']
        end_points_T['center_label'] -= min(epoch/120.0, 1.0) * end_points_T['jitter_pred'].transpose(1,2) * end_points_T['box_label_mask'].unsqueeze(-1)
        end_points_T['center_label'] = end_points_T['center_label'].detach()

    # Jitter loss
    jitter_loss_S = compute_jitter_loss(end_points_S)


    loss = 0.5 * get_loss(end_points_S, config, num_decoder_layers, query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk, center_loss_type, center_delta, size_loss_type, size_delta, heading_loss_type, heading_delta)[0] + get_loss_weak(end_points_T, config, num_decoder_layers, query_points_generator_loss_coef, obj_loss_coef, box_loss_coef, sem_cls_loss_coef,
             query_points_obj_topk, center_loss_type, center_delta, size_loss_type, size_delta, heading_loss_type, heading_delta)[0]

    FL_global = FocalLoss(class_num=2, gamma=3)
    #CST_loss = torch.nn.MSELoss(size_average=True)

    # Source domain global
    global_d_pred_S = end_points_S['global_d_pred']
    domain_S = Variable(torch.zeros(global_d_pred_S.size(0)).long().cuda())
    source_dloss = 1.0 * FL_global(global_d_pred_S, domain_S)
    
    # Target domain global
    global_d_pred_T = end_points_T['global_d_pred']
    domain_T = Variable(torch.ones(global_d_pred_T.size(0)).long().cuda())
    target_dloss = 1.0 * FL_global(global_d_pred_T, domain_T)
    
    
    # S/T local
    prefixes = ['last_']
    for prefix in prefixes:
        local_d_pred_S = end_points_S[f'{prefix}local_d_pred'].transpose(1,2).contiguous().squeeze(-1)
        source_dloss += 1.0 * torch.mean(local_d_pred_S**2 * end_points_S[f'{prefix}objectness_label'])

        local_d_pred_T = end_points_T[f'{prefix}local_d_pred'].transpose(1,2).contiguous().squeeze(-1)
        target_dloss += 1.0 * torch.mean((1-local_d_pred_T)**2 * end_points_T[f'{prefix}objectness_label'])
    

    DA_loss = source_dloss + target_dloss + jitter_loss_S * 0.5
    loss += 10 * DA_loss

    return loss, end_points_S, end_points_T


####################### Loss for Self Training #######################


def get_pseudo_labels(end_points, pred_center, pred_sem_cls, pred_objectness, pred_heading_scores,
                      pred_heading_residuals, pred_size_scores, pred_size_residuals, config_dict):
    MAX_NUM_OBJ = 64
    batch_size, num_proposal = pred_center.shape[:2]
    label_mask = torch.zeros((batch_size, MAX_NUM_OBJ), dtype=torch.long).cuda()

    # obj score threshold
    pred_objectness = torch.sigmoid(pred_objectness)
    # the second element is positive score
    pos_obj = pred_objectness[:, :, 0]
    neg_obj = 1 - pos_obj
    objectness_mask = pos_obj > config_dict['obj_threshold']
    neg_objectness_mask = neg_obj > 0.9  # deprecated

    # cls score threshold
    pred_sem_cls = nn.Softmax(dim=2)(pred_sem_cls)
    max_cls, argmax_cls = torch.max(pred_sem_cls, dim=2)
    cls_mask = max_cls > config_dict['cls_threshold']

    supervised_mask = end_points['supervised_mask']
    unsupervised_inds = torch.nonzero(1 - supervised_mask).squeeze(1).long()

    final_mask = torch.logical_and(cls_mask, objectness_mask)

    # we only keep MAX_NUM_OBJ predictions
    # however, after filtering the number can still exceed this
    # so we keep the ones with larger pos_obj * max_cls
    inds = torch.argsort(pos_obj * max_cls * final_mask, dim=1, descending=True)

    inds = inds[:, :MAX_NUM_OBJ].long()
    final_mask_sorted = torch.gather(final_mask, dim=1, index=inds)
    end_points['pseudo_gt_ratio'] = torch.sum(final_mask_sorted).float() / final_mask_sorted.view(-1).shape[0]

    neg_objectness_mask = torch.gather(neg_objectness_mask, dim=1, index=inds)

    max_size, argmax_size = torch.max(pred_size_scores, dim=2)
    size_inds = argmax_size.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3)
    max_heading, argmax_heading = torch.max(pred_heading_scores, dim=2)
    heading_inds = argmax_heading.unsqueeze(-1)

    # now only one class residuals
    pred_heading_residuals = torch.gather(pred_heading_residuals, dim=2, index=heading_inds).squeeze(2)
    pred_size_residuals = torch.gather(pred_size_residuals, dim=2, index=size_inds).squeeze(2)

    if config_dict['use_lhs']:
        pred_center_ = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        pred_heading_class_ = torch.gather(argmax_heading, dim=1, index=inds)
        pred_heading_residual_ = torch.gather(pred_heading_residuals, dim=1, index=inds)
        pred_size_class_ = torch.gather(argmax_size, dim=1, index=inds)
        pred_size_residual_ = torch.gather(pred_size_residuals, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
        num_proposal = pred_center_.shape[1]
        bsize = pred_center_.shape[0]
        pred_box_parameters = np.zeros((bsize, num_proposal, 7), dtype=np.float32)
        pred_box_parameters[:, :, 0:3] = pred_center_.detach().cpu().numpy()
        pred_corners_3d_upright_camera = np.zeros((bsize, num_proposal, 8, 3), dtype=np.float32)
        pred_center_upright_camera = flip_axis_to_camera(pred_center_.detach().cpu().numpy())
        for i in range(bsize):
            for j in range(num_proposal):
                heading_angle = config_dict['dataset_config'].class2angle( \
                    pred_heading_class_[i, j].detach().cpu().numpy(),
                    pred_heading_residual_[i, j].detach().cpu().numpy())
                box_size = config_dict['dataset_config'].class2size( \
                    int(pred_size_class_[i, j].detach().cpu().numpy()),
                    pred_size_residual_[i, j].detach().cpu().numpy())
                pred_box_parameters[i, j, 3:6] = box_size
                pred_box_parameters[i, j, 6] = heading_angle
                corners_3d_upright_camera = get_3d_box(box_size, heading_angle, pred_center_upright_camera[i, j, :])
                pred_corners_3d_upright_camera[i, j] = corners_3d_upright_camera

        # pred_corners_3d_upright_camera, _ = predictions2corners3d(end_points, config_dict)
        pred_mask = np.ones((batch_size, MAX_NUM_OBJ))
        nonempty_box_mask = np.ones((batch_size, MAX_NUM_OBJ))
        pos_obj_numpy = torch.gather(pos_obj, dim=1, index=inds).detach().cpu().numpy()
        pred_sem_cls_numpy = torch.gather(argmax_cls, dim=1, index=inds).detach().cpu().numpy()
        #iou_numpy = torch.gather(iou_pred, dim=1, index=inds).detach().cpu().numpy()
        for i in range(batch_size):
            boxes_3d_with_prob = np.zeros((MAX_NUM_OBJ, 8))
            for j in range(MAX_NUM_OBJ):
                boxes_3d_with_prob[j, 0] = np.min(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 1] = np.min(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 2] = np.min(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 3] = np.max(pred_corners_3d_upright_camera[i, j, :, 0])
                boxes_3d_with_prob[j, 4] = np.max(pred_corners_3d_upright_camera[i, j, :, 1])
                boxes_3d_with_prob[j, 5] = np.max(pred_corners_3d_upright_camera[i, j, :, 2])
                boxes_3d_with_prob[j, 6] = pos_obj_numpy[i, j] #* iou_numpy[i, j]
                boxes_3d_with_prob[j, 7] = pred_sem_cls_numpy[
                    i, j]  # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[i, :] == 1)[0]

            # here we do not consider orientation, in accordance to test time nms
            pick = lhs_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_mask[i, :] == 1, :],
                                         config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert (len(pick) > 0)
            pred_mask[i, nonempty_box_inds[pick]] = 0
        # end_points['pred_mask'] = pred_mask
        final_mask_sorted[torch.from_numpy(pred_mask).bool().cuda()] = 0

    
    label_mask[final_mask_sorted] = 1
    heading_label = torch.gather(argmax_heading, dim=1, index=inds)
    heading_residual_label = torch.gather(pred_heading_residuals.squeeze(-1), dim=1, index=inds)
    size_label = torch.gather(argmax_size, dim=1, index=inds)
    size_residual_label = torch.gather(pred_size_residuals, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
    sem_cls_label = torch.gather(argmax_cls, dim=1, index=inds)
    center_label = torch.gather(pred_center, dim=1, index=inds.unsqueeze(-1).expand(-1, -1, 3))
    center_label[(1 - label_mask).unsqueeze(-1).expand(-1, -1, 3).bool()] = -1000
    #iou_label = torch.gather(iou_pred, dim=1, index=inds)

    return label_mask, center_label, sem_cls_label, heading_label, heading_residual_label, size_label, size_residual_label


def compute_objectness_loss_based_on_query_points_pseudo(end_points, num_decoder_layers):
    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal

    '''
    # Associate proposal and GT objects
    seed_inds = end_points['seed_inds'].long()  # B,num_seed in [0,num_points-1]
    gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
    query_points_sample_inds = end_points['query_points_sample_inds'].long()

    B = seed_inds.shape[0]
    K = query_points_sample_inds.shape[1]
    K2 = gt_center.shape[1]

    seed_obj_gt = torch.gather(end_points['unlabeled_point_obj_mask'], 1, seed_inds)  # B,num_seed
    query_points_obj_gt = torch.gather(seed_obj_gt, 1, query_points_sample_inds)  # B, query_points

    point_instance_label = end_points['unlabeled_point_instance_label']  # B, num_points
    seed_instance_label = torch.gather(point_instance_label, 1, seed_inds)  # B,num_seed
    query_points_instance_label = torch.gather(seed_instance_label, 1, query_points_sample_inds)  # B,query_points

    objectness_mask = torch.ones((B, K)).cuda()

    # Set assignment
    object_assignment = query_points_instance_label  # (B,K) with values in 0,1,...,K2-1
    object_assignment[object_assignment < 0] = K2 - 1  # set background points to the last gt bbox

    end_points['unlabeled_objectness_label'] = query_points_obj_gt
    end_points['unlabeled_objectness_mask'] = objectness_mask
    end_points['unlabeled_object_assignment'] = object_assignment
    '''
    # Associate proposal and GT objects
    seed_inds = end_points['seed_inds'].long()  # B,num_seed in [0,num_points-1]
    gt_center = end_points['center_label'][:, :, 0:3]  # B, K2, 3
    query_points_sample_inds = end_points['query_points_sample_inds'].long()
    aggregated_vote_xyz = end_points['query_points_xyz']

    B = seed_inds.shape[0]
    K = query_points_sample_inds.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_label[euclidean_dist1<0.3] = 1

    objectness_mask = torch.ones((B, K)).cuda()
    object_assignment = ind1.cuda()

    end_points['unlabeled_objectness_label'] = objectness_label
    end_points['unlabeled_objectness_mask'] = objectness_mask
    end_points['unlabeled_object_assignment'] = object_assignment

    objectness_loss_sum = 0.0
    for prefix in prefixes:
        # Compute objectness loss
        objectness_scores = end_points[f'{prefix}objectness_scores']
        criterion = SigmoidFocalClassificationLoss()
        cls_weights = objectness_mask.float()
        cls_normalizer = cls_weights.sum(dim=1, keepdim=True).float()
        cls_weights /= torch.clamp(cls_normalizer, min=1.0)
        cls_loss_src = criterion(objectness_scores.transpose(2, 1).contiguous().view(B, K, 1),
                                 objectness_label.unsqueeze(-1),
                                 weights=cls_weights)
        objectness_loss = cls_loss_src.sum() / B

        objectness_loss_sum += objectness_loss
    return objectness_loss_sum, end_points


def compute_box_and_sem_cls_loss_pseudo(
        end_points, config, num_decoder_layers, center_loss_type, center_delta,
        size_loss_type, size_delta, heading_loss_type, heading_delta):
    '''loss
    '''
    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    mean_size_arr = config.mean_size_arr

    if num_decoder_layers > 0:
        prefixes = ['proposal_'] + ['last_'] + [f'{i}head_' for i in range(num_decoder_layers - 1)]
    else:
        prefixes = ['proposal_']  # only proposal
    box_loss_sum = 0.0
    sem_cls_loss_sum = 0.0
    for prefix in prefixes:
        object_assignment = end_points['unlabeled_object_assignment']
        batch_size = object_assignment.shape[0]

        # Compute center loss
        pred_center = end_points[f'{prefix}center']
        gt_center = end_points['unlabeled_center_label'][:, :, 0:3]

        if center_loss_type == 'smoothl1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = smoothl1_loss(assigned_gt_center - pred_center, delta=center_delta)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        elif center_loss_type == 'l1':
            objectness_label = end_points[f'{prefix}objectness_label'].float()
            object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
            assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
            center_loss = l1_loss(assigned_gt_center - pred_center)  # (B,K)
            center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute heading loss
        heading_class_label = torch.gather(end_points['unlabeled_heading_class_label'], 1,
                                           object_assignment)  # select (B,K) from (B,K2)
        criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
        heading_class_loss = criterion_heading_class(end_points[f'{prefix}heading_scores'].transpose(2, 1),
                                                     heading_class_label)  # (B,K)
        heading_class_loss = torch.sum(heading_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        heading_residual_label = torch.gather(end_points['unlabeled_heading_residual_label'], 1,
                                              object_assignment)  # select (B,K) from (B,K2)
        heading_residual_normalized_label = heading_residual_label / (np.pi / num_heading_bin)

        # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
        heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1],
                                                       num_heading_bin).zero_()
        heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1),
                                       1)  # src==1 so it's *one-hot* (B,K,num_heading_bin)
        heading_residual_normalized_error = torch.sum(
            end_points[f'{prefix}heading_residuals_normalized'] * heading_label_one_hot,
            -1) - heading_residual_normalized_label

        if heading_loss_type == 'smoothl1':
            heading_residual_normalized_loss = heading_delta * smoothl1_loss(heading_residual_normalized_error,
                                                                             delta=heading_delta)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        elif heading_loss_type == 'l1':
            heading_residual_normalized_loss = l1_loss(heading_residual_normalized_error)  # (B,K)
            heading_residual_normalized_loss = torch.sum(
                heading_residual_normalized_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # Compute size loss
        size_class_label = torch.gather(end_points['unlabeled_size_class_label'], 1,
                                        object_assignment)  # select (B,K) from (B,K2)
        criterion_size_class = nn.CrossEntropyLoss(reduction='none')
        size_class_loss = criterion_size_class(end_points[f'{prefix}size_scores'].transpose(2, 1),
                                               size_class_label)  # (B,K)
        size_class_loss = torch.sum(size_class_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        size_residual_label = torch.gather(
            end_points['unlabeled_size_residual_label'], 1,
            object_assignment.unsqueeze(-1).repeat(1, 1, 3))  # select (B,K,3) from (B,K2,3)

        size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
        size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1),
                                    1)  # src==1 so it's *one-hot* (B,K,num_size_cluster)
        size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1, 1, 1, 3)  # (B,K,num_size_cluster,3)
        predicted_size_residual_normalized = torch.sum(
            end_points[f'{prefix}size_residuals_normalized'] * size_label_one_hot_tiled,
            2)  # (B,K,3)

        mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(
            0)  # (1,1,num_size_cluster,3)
        mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2)  # (B,K,3)
        size_residual_label_normalized = size_residual_label / mean_size_label  # (B,K,3)

        size_residual_normalized_error = predicted_size_residual_normalized - size_residual_label_normalized

        if size_loss_type == 'smoothl1':
            size_residual_normalized_loss = size_delta * smoothl1_loss(size_residual_normalized_error,
                                                                       delta=size_delta)  # (B,K,3) -> (B,K)
            size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                    torch.sum(objectness_label) + 1e-6)
        elif size_loss_type == 'l1':
            size_residual_normalized_loss = l1_loss(size_residual_normalized_error)  # (B,K,3) -> (B,K)
            size_residual_normalized_loss = torch.sum(size_residual_normalized_loss * objectness_label.unsqueeze(2)) / (
                    torch.sum(objectness_label) + 1e-6)
        else:
            raise NotImplementedError

        # 3.4 Semantic cls loss
        sem_cls_label = torch.gather(end_points['unlabeled_sem_cls_label'], 1, object_assignment)  # select (B,K) from (B,K2)
        criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
        sem_cls_loss = criterion_sem_cls(end_points[f'{prefix}sem_cls_scores'].transpose(2, 1), sem_cls_label)  # (B,K)
        sem_cls_loss = torch.sum(sem_cls_loss * objectness_label) / (torch.sum(objectness_label) + 1e-6)

        box_loss = center_loss + 0.1 * heading_class_loss + heading_residual_normalized_loss + 0.1 * size_class_loss + size_residual_normalized_loss

        box_loss_sum += box_loss
        sem_cls_loss_sum += sem_cls_loss
    return box_loss_sum, sem_cls_loss_sum, end_points


def get_pseudo_detection_loss(end_points, config, config_dict, num_decoder_layers, box_loss_coef, sem_cls_loss_coef,
            center_loss_type, center_delta, size_loss_type, size_delta, heading_loss_type, heading_delta):
    """ Loss functions
    """
    # Obj loss
    objectness_loss_sum, end_points = \
        compute_objectness_loss_based_on_query_points_pseudo(end_points, num_decoder_layers)

    end_points['sum_heads_objectness_loss'] = objectness_loss_sum

    # Box loss and sem cls loss
    box_loss_sum, sem_cls_loss_sum, end_points = compute_box_and_sem_cls_loss_pseudo(
        end_points, config, num_decoder_layers,
        center_loss_type, center_delta=center_delta,
        size_loss_type=size_loss_type, size_delta=size_delta,
        heading_loss_type=heading_loss_type, heading_delta=heading_delta)
    end_points['sum_heads_box_loss'] = box_loss_sum
    end_points['sum_heads_sem_cls_loss'] = sem_cls_loss_sum

    # means average proposal with prediction loss
    loss = 1.0 / (num_decoder_layers + 1) * (box_loss_coef * box_loss_sum + sem_cls_loss_coef * sem_cls_loss_sum)
    loss *= 10
    end_points['unlabeled_detection_loss'] = loss

    return loss, end_points


def get_loss_pseudo(end_points, end_points_teacher, config, config_dict,
             num_decoder_layers, box_loss_coef, sem_cls_loss_coef,
             center_loss_type='smoothl1', center_delta=1.0,
             size_loss_type='smoothl1', size_delta=1.0,
             heading_loss_type='smoothl1', heading_delta=1.0):
    """ Loss functions
    """
    prefix = "4head_"

    labeled_num = torch.nonzero(end_points['supervised_mask']).squeeze(1).shape[0]
    pred_center = end_points_teacher[f'{prefix}center'][labeled_num:]
    pred_sem_cls = end_points_teacher[f'{prefix}sem_cls_scores'][labeled_num:]
    pred_objectness = end_points_teacher[f'{prefix}objectness_scores'][labeled_num:]
    pred_heading_scores = end_points_teacher[f'{prefix}heading_scores'][labeled_num:]
    pred_heading_residuals = end_points_teacher[f'{prefix}heading_residuals'][labeled_num:]
    pred_size_scores = end_points_teacher[f'{prefix}size_scores'][labeled_num:]
    pred_size_residuals = end_points_teacher[f'{prefix}size_residuals'][labeled_num:]

    # generate pseudo labels
    label_mask, center_label, sem_cls_label, \
    heading_label, heading_residual_label, \
    size_label, size_residual_label = \
        get_pseudo_labels(end_points, pred_center, pred_sem_cls, pred_objectness, pred_heading_scores,
                          pred_heading_residuals, pred_size_scores, pred_size_residuals, config_dict)
    
    end_points['unlabeled_center_label'] = center_label
    end_points['unlabeled_box_label_mask'] = label_mask
    end_points['unlabeled_sem_cls_label'] = sem_cls_label
    end_points['unlabeled_heading_class_label'] = heading_label
    end_points['unlabeled_heading_residual_label'] = heading_residual_label
    end_points['unlabeled_size_class_label'] = size_label
    end_points['unlabeled_size_residual_label'] = size_residual_label

    consistency_loss, end_points = get_pseudo_detection_loss(end_points, config, config_dict, num_decoder_layers,
             box_loss_coef, sem_cls_loss_coef, center_loss_type, center_delta, size_loss_type, size_delta, heading_loss_type, heading_delta)

    return consistency_loss, end_points
