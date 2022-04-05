# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from nn_distance import nn_distance, huber_loss

FAR_THRESHOLD = 0.6
NEAR_THRESHOLD = 0.3
GT_VOTE_FACTOR = 3 # number of GT votes per point
OBJECTNESS_CLS_WEIGHTS = [0.2,0.8] # put larger weights on positive objectness

def compute_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3
    seed_inds = end_points['seed_inds'].long() # B,num_seed in [0,num_points-1]

    # Get groundtruth votes for the seed points
    # vote_label_mask: Use gather to select B,num_seed from B,num_point
    #   non-object point has no GT vote mask = 0, object point has mask = 1
    # vote_label: Use gather to select B,num_seed,9 from B,num_point,9
    #   with inds in shape B,num_seed,9 and 9 = GT_VOTE_FACTOR * 3
    seed_gt_votes_mask = torch.gather(end_points['vote_label_mask'], 1, seed_inds)
    seed_inds_expand = seed_inds.view(batch_size,num_seed,1).repeat(1,1,3*GT_VOTE_FACTOR)
    seed_gt_votes = torch.gather(end_points['vote_label'], 1, seed_inds_expand)
    seed_gt_votes += end_points['seed_xyz'].repeat(1,1,3)

    # Compute the min of min of distance
    vote_xyz_reshape = vote_xyz.view(batch_size*num_seed, -1, 3) # from B,num_seed*vote_factor,3 to B*num_seed,vote_factor,3
    seed_gt_votes_reshape = seed_gt_votes.view(batch_size*num_seed, GT_VOTE_FACTOR, 3) # from B,num_seed,3*GT_VOTE_FACTOR to B*num_seed,GT_VOTE_FACTOR,3
    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz_reshape, seed_gt_votes_reshape, l1=True)
    votes_dist, _ = torch.min(dist2, dim=1) # (B*num_seed,vote_factor) to (B*num_seed,)
    votes_dist = votes_dist.view(batch_size, num_seed)
    vote_loss = torch.sum(votes_dist*seed_gt_votes_mask.float())/(torch.sum(seed_gt_votes_mask.float())+1e-6)
    return vote_loss

def compute_weak_vote_loss(end_points):
    """ Compute vote loss: Match predicted votes to GT votes.

    Args:
        end_points: dict (read-only)
    
    Returns:
        vote_loss: scalar Tensor
            
    Overall idea:
        If the seed point belongs to an object (votes_label_mask == 1),
        then we require it to vote for the object center.

        Each seed point may vote for multiple translations v1,v2,v3
        A seed point may also be in the boxes of multiple objects:
        o1,o2,o3 with corresponding GT votes c1,c2,c3

        Then the loss for this seed point is:
            min(d(v_i,c_j)) for i=1,2,3 and j=1,2,3
    """

    # Load ground truth votes and assign them to seed points
    batch_size = end_points['seed_xyz'].shape[0]
    num_seed = end_points['seed_xyz'].shape[1] # B,num_seed,3
    vote_xyz = end_points['vote_xyz'] # B,num_seed*vote_factor,3

    gt_center = end_points['center_label'][:,:,0:3] # B,K2,3

    # A predicted vote to no where is not penalized as long as there is a good vote near the GT vote.
    dist1, _, dist2, _ = nn_distance(vote_xyz, gt_center, l1=True) # dist1: B,num_seed*vote_factor, dist2: B,K2
    dist1 = dist1.view(batch_size, num_seed, -1) # dist1: B,num_seed,vote_factor
    votes_dist, _ = torch.min(dist1, dim=2) # (B,num_seed,vote_factor) to (B,num_seed,)
    box_label_mask = end_points['box_label_mask'] # B,K2
    sem_cls_label = end_points['sem_cls_label'] # B,K2
    object_weight = torch.ones_like(sem_cls_label).cuda()
    #object_weight[(sem_cls_label == 4) + (sem_cls_label == 6) + (sem_cls_label == 11)] = 10
    vote_loss = torch.mean(votes_dist) + torch.sum(dist2*object_weight*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    
    return vote_loss

def compute_objectness_loss(end_points):
    """ Compute objectness loss for the proposals.

    Args:
        end_points: dict (read-only)

    Returns:
        objectness_loss: scalar Tensor
        objectness_label: (batch_size, num_seed) Tensor with value 0 or 1
        objectness_mask: (batch_size, num_seed) Tensor with value 0 or 1
        object_assignment: (batch_size, num_seed) Tensor with long int
            within [0,num_gt_object-1]
    """ 
    # Associate proposal and GT objects by point-to-point distances
    aggregated_vote_xyz = end_points['aggregated_vote_xyz']
    # aggregated_vote_xyz = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    B = gt_center.shape[0]
    K = aggregated_vote_xyz.shape[1]
    K2 = gt_center.shape[1]
    dist1, ind1, dist2, _ = nn_distance(aggregated_vote_xyz, gt_center) # dist1: BxK, dist2: BxK2

    # Generate objectness label and mask
    # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
    # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
    euclidean_dist1 = torch.sqrt(dist1+1e-6)
    objectness_label = torch.zeros((B,K), dtype=torch.long).cuda()
    objectness_mask = torch.zeros((B,K)).cuda()
    objectness_label[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1<NEAR_THRESHOLD] = 1
    objectness_mask[euclidean_dist1>FAR_THRESHOLD] = 1

    # Compute objectness loss
    objectness_scores = end_points['objectness_scores']
    criterion = nn.CrossEntropyLoss(torch.Tensor(OBJECTNESS_CLS_WEIGHTS).cuda(), reduction='none')
    objectness_loss = criterion(objectness_scores.transpose(2,1), objectness_label)
    objectness_loss = torch.sum(objectness_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    # Set assignment
    object_assignment = ind1 # (B,K) with values in 0,1,...,K2-1

    return objectness_loss, objectness_label, objectness_mask, object_assignment

def compute_box_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2

    # Compute heading loss
    heading_class_label = torch.gather(end_points['heading_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_heading_class = nn.CrossEntropyLoss(reduction='none')
    heading_class_loss = criterion_heading_class(end_points['heading_scores'].transpose(2,1), heading_class_label) # (B,K)
    heading_class_loss = torch.sum(heading_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    heading_residual_label = torch.gather(end_points['heading_residual_label'], 1, object_assignment) # select (B,K) from (B,K2)
    heading_residual_normalized_label = heading_residual_label / (np.pi/num_heading_bin)

    # Ref: https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/3
    heading_label_one_hot = torch.cuda.FloatTensor(batch_size, heading_class_label.shape[1], num_heading_bin).zero_()
    heading_label_one_hot.scatter_(2, heading_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_heading_bin)
    heading_residual_normalized_loss = huber_loss(torch.sum(end_points['heading_residuals_normalized']*heading_label_one_hot, -1) - heading_residual_normalized_label, delta=1.0) # (B,K)
    heading_residual_normalized_loss = torch.sum(heading_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    size_residual_label = torch.gather(end_points['size_residual_label'], 1, object_assignment.unsqueeze(-1).repeat(1,1,3)) # select (B,K,3) from (B,K2,3)
    size_label_one_hot = torch.cuda.FloatTensor(batch_size, size_class_label.shape[1], num_size_cluster).zero_()
    size_label_one_hot.scatter_(2, size_class_label.unsqueeze(-1), 1) # src==1 so it's *one-hot* (B,K,num_size_cluster)
    size_label_one_hot_tiled = size_label_one_hot.unsqueeze(-1).repeat(1,1,1,3) # (B,K,num_size_cluster,3)
    predicted_size_residual_normalized = torch.sum(end_points['size_residuals_normalized']*size_label_one_hot_tiled, 2) # (B,K,3)

    mean_size_arr_expanded = torch.from_numpy(mean_size_arr.astype(np.float32)).cuda().unsqueeze(0).unsqueeze(0) # (1,1,num_size_cluster,3) 
    mean_size_label = torch.sum(size_label_one_hot_tiled * mean_size_arr_expanded, 2) # (B,K,3)
    size_residual_label_normalized = size_residual_label / mean_size_label # (B,K,3)
    size_residual_normalized_loss = torch.mean(huber_loss(predicted_size_residual_normalized - size_residual_label_normalized, delta=1.0), -1) # (B,K,3) -> (B,K)
    size_residual_normalized_loss = torch.sum(size_residual_normalized_loss*objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, heading_class_loss, heading_residual_normalized_loss, size_class_loss, size_residual_normalized_loss, sem_cls_loss


def smoothl1_loss(error, delta=1.0):
    """Smooth L1 loss.
    x = error = pred - gt or dist(pred,gt)
    0.5 * |x|^2                 if |x|<=d
    |x| - 0.5 * d               if |x|>d
    """
    diff = torch.abs(error)
    loss = torch.where(diff < delta, 0.5 * diff * diff / delta, diff - 0.5 * delta)
    return loss


def compute_center_and_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    object_assignment = end_points['object_assignment']
    batch_size = object_assignment.shape[0]

    # Compute center loss
    
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:,:,0:3]
    dist1, ind1, dist2, _ = nn_distance(pred_center, gt_center) # dist1: BxK, dist2: BxK2
    box_label_mask = end_points['box_label_mask']
    objectness_label = end_points['objectness_label'].float()
    centroid_reg_loss1 = \
        torch.sum(dist1*objectness_label)/(torch.sum(objectness_label)+1e-6)
    centroid_reg_loss2 = \
        torch.sum(dist2*box_label_mask)/(torch.sum(box_label_mask)+1e-6)
    center_loss = centroid_reg_loss1 + centroid_reg_loss2
    '''
    pred_center = end_points['center']
    gt_center = end_points['center_label'][:, :, 0:3]
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment)  # select (B,K) from (B,K2)
    center_margin = torch.from_numpy(0.05 * mean_size_arr[size_class_label.cpu(), :]).cuda()   # (B,K,3)

    objectness_label = end_points['objectness_label'].float()
    object_assignment_expand = object_assignment.unsqueeze(2).repeat(1, 1, 3)
    assigned_gt_center = torch.gather(gt_center, 1, object_assignment_expand)  # (B, K, 3) from (B, K2, 3)
    center_loss = smoothl1_loss(assigned_gt_center - pred_center)  # (B,K)
    center_loss -= center_margin
    center_loss[center_loss < 0] = 0
    center_loss = torch.sum(center_loss * objectness_label.unsqueeze(2)) / (torch.sum(objectness_label) + 1e-6)
    '''
    
    # Compute size loss
    size_class_label = torch.gather(end_points['size_class_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_size_class = nn.CrossEntropyLoss(reduction='none')
    size_class_loss = criterion_size_class(end_points['size_scores'].transpose(2,1), size_class_label) # (B,K)
    size_class_loss = torch.sum(size_class_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    # 3.4 Semantic cls loss
    sem_cls_label = torch.gather(end_points['sem_cls_label'], 1, object_assignment) # select (B,K) from (B,K2)
    criterion_sem_cls = nn.CrossEntropyLoss(reduction='none')
    sem_cls_loss = criterion_sem_cls(end_points['sem_cls_scores'].transpose(2,1), sem_cls_label) # (B,K)
    sem_cls_loss = torch.sum(sem_cls_loss * objectness_label)/(torch.sum(objectness_label)+1e-6)

    return center_loss, size_class_loss, sem_cls_loss

def compute_sem_cls_loss(end_points, config):
    """ Compute 3D bounding box and semantic classification loss.

    Args:
        end_points: dict (read-only)

    Returns:
        center_loss
        heading_cls_loss
        heading_reg_loss
        size_cls_loss
        size_reg_loss
        sem_cls_loss
    """

    num_heading_bin = config.num_heading_bin
    num_size_cluster = config.num_size_cluster
    num_class = config.num_class
    mean_size_arr = config.mean_size_arr

    cloud_label = end_points['cloud_label'] # Bxnum_class
    batch_size = cloud_label.shape[0]

    # 3.4 Semantic cls loss
    cloud_pred = end_points['sem_cls_scores'].transpose(2,1) # Bxnum_classxK
    cloud_pred_gap = torch.mean(cloud_pred, dim=2) # Bxnum_class
    BCEWL = nn.BCEWithLogitsLoss()
    sem_cls_loss = BCEWL(cloud_pred_gap.float(), cloud_label.float())
    return sem_cls_loss

def get_loss(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, heading_cls_loss, heading_reg_loss, size_cls_loss, size_reg_loss, sem_cls_loss = \
        compute_box_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['heading_cls_loss'] = heading_cls_loss
    end_points['heading_reg_loss'] = heading_reg_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['size_reg_loss'] = size_reg_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    box_loss = center_loss + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss + size_reg_loss
    end_points['box_loss'] = box_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points


def get_loss_weak(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss = compute_weak_vote_loss(end_points)
    end_points['vote_loss'] = vote_loss

    # Obj loss
    objectness_loss, objectness_label, objectness_mask, object_assignment = \
        compute_objectness_loss(end_points)
    end_points['objectness_loss'] = objectness_loss
    end_points['objectness_label'] = objectness_label
    end_points['objectness_mask'] = objectness_mask
    end_points['object_assignment'] = object_assignment
    total_num_proposal = objectness_label.shape[0]*objectness_label.shape[1]
    end_points['pos_ratio'] = \
        torch.sum(objectness_label.float().cuda())/float(total_num_proposal)
    end_points['neg_ratio'] = \
        torch.sum(objectness_mask.float())/float(total_num_proposal) - end_points['pos_ratio']

    # Box loss and sem cls loss
    center_loss, size_cls_loss, sem_cls_loss = compute_center_and_sem_cls_loss(end_points, config)
    end_points['center_loss'] = center_loss
    end_points['size_cls_loss'] = size_cls_loss
    end_points['sem_cls_loss'] = sem_cls_loss
    
    box_loss = center_loss + 0.1*size_cls_loss
    sem_cls_loss = sem_cls_loss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss
    loss *= 10
    end_points['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label.long()).float()*objectness_mask)/(torch.sum(objectness_mask)+1e-6)
    end_points['obj_acc'] = obj_acc

    return loss, end_points


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5),
                                   putting more focus on hard, misclassi?ed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True,sigmoid=False,reduce=True):
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
            # print(class_mask)


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

def get_loss_DA(end_points_S, end_points_T, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, global_d_pred, vote_xyz, local_d_pred,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    source_coefficient = 0.1

    # Vote loss
    vote_loss_S = compute_weak_vote_loss(end_points_S)
    vote_loss_T = compute_weak_vote_loss(end_points_T)
    vote_loss = source_coefficient*vote_loss_S + vote_loss_T
    end_points_S['vote_loss'] = vote_loss_S
    end_points_T['vote_loss'] = vote_loss_T

    # Obj loss
    objectness_loss_S, objectness_label_S, objectness_mask_S, object_assignment = \
        compute_objectness_loss(end_points_S)
    end_points_S['objectness_loss'] = objectness_loss_S
    end_points_S['objectness_label'] = objectness_label_S
    end_points_S['objectness_mask'] = objectness_mask_S
    end_points_S['object_assignment'] = object_assignment
    total_num_proposal = objectness_label_S.shape[0]*objectness_label_S.shape[1]
    end_points_S['pos_ratio'] = \
        torch.sum(objectness_label_S.float().cuda())/float(total_num_proposal)
    end_points_S['neg_ratio'] = \
        torch.sum(objectness_mask_S.float())/float(total_num_proposal) - end_points_S['pos_ratio']
    
    objectness_loss_T, objectness_label_T, objectness_mask_T, object_assignment = \
        compute_objectness_loss(end_points_T)
    end_points_T['objectness_loss'] = objectness_loss_T
    end_points_T['objectness_label'] = objectness_label_T
    end_points_T['objectness_mask'] = objectness_mask_T
    end_points_T['object_assignment'] = object_assignment
    total_num_proposal = objectness_label_T.shape[0]*objectness_label_T.shape[1]
    end_points_T['pos_ratio'] = \
        torch.sum(objectness_label_T.float().cuda())/float(total_num_proposal)
    end_points_T['neg_ratio'] = \
        torch.sum(objectness_mask_T.float())/float(total_num_proposal) - end_points_T['pos_ratio']
    
    objectness_loss = source_coefficient*objectness_loss_S + objectness_loss_T

    # Box loss and sem cls loss
    center_loss_S, heading_cls_loss, heading_reg_loss, size_cls_loss_S, size_reg_loss, sem_cls_loss_S = \
        compute_box_and_sem_cls_loss(end_points_S, config)
    end_points_S['center_loss'] = center_loss_S
    end_points_S['heading_cls_loss'] = heading_cls_loss
    end_points_S['heading_reg_loss'] = heading_reg_loss
    end_points_S['size_cls_loss'] = size_cls_loss_S
    end_points_S['size_reg_loss'] = size_reg_loss
    end_points_S['sem_cls_loss'] = sem_cls_loss_S
    box_loss_S = center_loss_S + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss_S + size_reg_loss
    end_points_S['box_loss'] = box_loss_S
    
    center_loss_T, size_cls_loss_T, sem_cls_loss_T = compute_center_and_sem_cls_loss(end_points_T, config)
    end_points_T['center_loss'] = center_loss_T
    end_points_T['size_cls_loss'] = size_cls_loss_T
    end_points_T['sem_cls_loss'] = sem_cls_loss_T
    box_loss_T = center_loss_T + 0.1*size_cls_loss_T

    box_loss = 10*source_coefficient*box_loss_S + box_loss_T
    sem_cls_loss = source_coefficient*sem_cls_loss_S + sem_cls_loss_T

    ## Domain Align Loss
    FL_global = FocalLoss(class_num=2, gamma=3)
    #FL_vote = FocalLoss(class_num=2, gamma=3)

    da_coefficient = 0.5
    
    # Source domain
    global_d_pred_S = end_points_S['global_d_pred']
    local_d_pred_S = end_points_S['local_d_pred'].transpose(1,2).contiguous()
    domain_S = Variable(torch.zeros(global_d_pred_S.size(0)).long().cuda())
    #object_weight_local_S = F.softmax(end_points_S['objectness_scores'], dim=-1)[:,:,1:]
    object_weight_local_S = end_points_S['objectness_label'].unsqueeze(-1)
    source_dloss = da_coefficient * torch.mean(local_d_pred_S**2 * object_weight_local_S) + da_coefficient * FL_global(global_d_pred_S, domain_S)
    
    # Target domain
    global_d_pred_T = end_points_T['global_d_pred']
    local_d_pred_T = end_points_T['local_d_pred'].transpose(1,2).contiguous()
    domain_T = Variable(torch.ones(global_d_pred_T.size(0)).long().cuda())
    #object_weight_local_T = F.softmax(end_points_T['objectness_scores'], dim=-1)[:,:,1:]
    object_weight_local_T = end_points_T['objectness_label'].unsqueeze(-1)
    target_dloss = da_coefficient * torch.mean((1-local_d_pred_T)**2 * object_weight_local_T) + da_coefficient * FL_global(global_d_pred_T, domain_T)

    DA_loss = source_dloss + target_dloss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + DA_loss
    loss *= 10
    end_points_S['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points_S['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label_S.long()).float()*objectness_mask_S)/(torch.sum(objectness_mask_S)+1e-6)
    end_points_S['obj_acc'] = obj_acc

    return loss, end_points_S, end_points_T


def compute_jitter_loss(end_points):
    # center_jitter: B 64 3
    # jitter_pred: B 3 64
    jitter_loss = ((end_points['center_jitter']-end_points['jitter_pred'].transpose(1,2).contiguous())**2).mean()
    end_points['jitter_loss'] = jitter_loss
    return jitter_loss


def get_loss_DA_jitter(end_points_S, end_points_T, epoch, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, global_d_pred, vote_xyz, local_d_pred,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """
    if epoch > -1:
        end_points_S['center_label'] -= min(epoch/60.0, 1.0) * end_points_S['center_jitter']
        end_points_T['center_label'] -= min(epoch/60.0, 1.0) * end_points_T['jitter_pred'].transpose(1,2) * end_points_T['box_label_mask'].unsqueeze(-1)
        end_points_T['center_label'] = end_points_T['center_label'].detach()

    source_coefficient = 0.5

    # Jitter loss
    jitter_loss_S = compute_jitter_loss(end_points_S)
    end_points_S['jitter_loss'] = jitter_loss_S

    # Vote loss
    vote_loss_S = compute_weak_vote_loss(end_points_S)
    vote_loss_T = compute_weak_vote_loss(end_points_T)
    vote_loss = source_coefficient*vote_loss_S + vote_loss_T
    end_points_S['vote_loss'] = vote_loss_S
    end_points_T['vote_loss'] = vote_loss_T

    # Obj loss
    objectness_loss_S, objectness_label_S, objectness_mask_S, object_assignment = \
        compute_objectness_loss(end_points_S)
    end_points_S['objectness_loss'] = objectness_loss_S
    end_points_S['objectness_label'] = objectness_label_S
    end_points_S['objectness_mask'] = objectness_mask_S
    end_points_S['object_assignment'] = object_assignment
    total_num_proposal = objectness_label_S.shape[0]*objectness_label_S.shape[1]
    end_points_S['pos_ratio'] = \
        torch.sum(objectness_label_S.float().cuda())/float(total_num_proposal)
    end_points_S['neg_ratio'] = \
        torch.sum(objectness_mask_S.float())/float(total_num_proposal) - end_points_S['pos_ratio']
    
    objectness_loss_T, objectness_label_T, objectness_mask_T, object_assignment = \
        compute_objectness_loss(end_points_T)
    end_points_T['objectness_loss'] = objectness_loss_T
    end_points_T['objectness_label'] = objectness_label_T
    end_points_T['objectness_mask'] = objectness_mask_T
    end_points_T['object_assignment'] = object_assignment
    total_num_proposal = objectness_label_T.shape[0]*objectness_label_T.shape[1]
    end_points_T['pos_ratio'] = \
        torch.sum(objectness_label_T.float().cuda())/float(total_num_proposal)
    end_points_T['neg_ratio'] = \
        torch.sum(objectness_mask_T.float())/float(total_num_proposal) - end_points_T['pos_ratio']
    
    objectness_loss = source_coefficient*objectness_loss_S + objectness_loss_T

    # Box loss and sem cls loss
    center_loss_S, heading_cls_loss, heading_reg_loss, size_cls_loss_S, size_reg_loss, sem_cls_loss_S = \
        compute_box_and_sem_cls_loss(end_points_S, config)
    end_points_S['center_loss'] = center_loss_S
    end_points_S['heading_cls_loss'] = heading_cls_loss
    end_points_S['heading_reg_loss'] = heading_reg_loss
    end_points_S['size_cls_loss'] = size_cls_loss_S
    end_points_S['size_reg_loss'] = size_reg_loss
    end_points_S['sem_cls_loss'] = sem_cls_loss_S
    box_loss_S = center_loss_S + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss_S + size_reg_loss
    end_points_S['box_loss'] = box_loss_S
    
    center_loss_T, size_cls_loss_T, sem_cls_loss_T = compute_center_and_sem_cls_loss(end_points_T, config)
    end_points_T['center_loss'] = center_loss_T
    end_points_T['size_cls_loss'] = size_cls_loss_T
    end_points_T['sem_cls_loss'] = sem_cls_loss_T
    box_loss_T = center_loss_T + 0.1*size_cls_loss_T

    box_loss = source_coefficient*box_loss_S + box_loss_T
    sem_cls_loss = source_coefficient*sem_cls_loss_S + sem_cls_loss_T

    ## Domain Align Loss
    FL_global = FocalLoss(class_num=2, gamma=3)
    #FL_vote = FocalLoss(class_num=2, gamma=3)

    da_coefficient = 0.5
    
    # Source domain
    global_d_pred_S = end_points_S['global_d_pred']
    local_d_pred_S = end_points_S['local_d_pred'].transpose(1,2).contiguous()
    jitter_d_pred_S = end_points_S['jitter_d_pred'].transpose(1,2).contiguous()
    domain_S = Variable(torch.zeros(global_d_pred_S.size(0)).long().cuda())
    #object_weight_local_S = F.softmax(end_points_S['objectness_scores'], dim=-1)[:,:,1:]
    jitter_weight_S = end_points_S['box_label_mask'].unsqueeze(-1)
    object_weight_local_S = end_points_S['objectness_label'].unsqueeze(-1)
    source_dloss = da_coefficient * torch.mean(local_d_pred_S**2 * object_weight_local_S) + da_coefficient * FL_global(global_d_pred_S, domain_S)# + da_coefficient * torch.mean(jitter_d_pred_S**2 * jitter_weight_S)
    
    # Target domain
    global_d_pred_T = end_points_T['global_d_pred']
    local_d_pred_T = end_points_T['local_d_pred'].transpose(1,2).contiguous()
    jitter_d_pred_T = end_points_T['jitter_d_pred'].transpose(1,2).contiguous()
    domain_T = Variable(torch.ones(global_d_pred_T.size(0)).long().cuda())
    #object_weight_local_T = F.softmax(end_points_T['objectness_scores'], dim=-1)[:,:,1:]
    jitter_weight_T = end_points_T['box_label_mask'].unsqueeze(-1)
    object_weight_local_T = end_points_T['objectness_label'].unsqueeze(-1)
    target_dloss = da_coefficient * torch.mean((1-local_d_pred_T)**2 * object_weight_local_T) + da_coefficient * FL_global(global_d_pred_T, domain_T)# + da_coefficient * torch.mean((1-jitter_d_pred_T)**2 * jitter_weight_T)

    DA_loss = source_dloss + target_dloss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + DA_loss + source_coefficient*jitter_loss_S
    loss *= 10
    end_points_S['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points_S['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label_S.long()).float()*objectness_mask_S)/(torch.sum(objectness_mask_S)+1e-6)
    end_points_S['obj_acc'] = obj_acc

    return loss, end_points_S, end_points_T


def get_loss_DA_separate(end_points_S, end_points_T, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, global_d_pred, vote_xyz, local_d_pred,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss_S = compute_vote_loss(end_points_S)
    vote_loss_T = compute_weak_vote_loss(end_points_T)
    vote_loss = vote_loss_S + vote_loss_T
    end_points_S['vote_loss'] = vote_loss_S
    end_points_T['vote_loss'] = vote_loss_T

    # Obj loss
    objectness_loss_S, objectness_label_S, objectness_mask_S, object_assignment = \
        compute_objectness_loss(end_points_S)
    end_points_S['objectness_loss'] = objectness_loss_S
    end_points_S['objectness_label'] = objectness_label_S
    end_points_S['objectness_mask'] = objectness_mask_S
    end_points_S['object_assignment'] = object_assignment
    total_num_proposal = objectness_label_S.shape[0]*objectness_label_S.shape[1]
    end_points_S['pos_ratio'] = \
        torch.sum(objectness_label_S.float().cuda())/float(total_num_proposal)
    end_points_S['neg_ratio'] = \
        torch.sum(objectness_mask_S.float())/float(total_num_proposal) - end_points_S['pos_ratio']
    
    objectness_loss_T, objectness_label_T, objectness_mask_T, object_assignment = \
        compute_objectness_loss(end_points_T)
    end_points_T['objectness_loss'] = objectness_loss_T
    end_points_T['objectness_label'] = objectness_label_T
    end_points_T['objectness_mask'] = objectness_mask_T
    end_points_T['object_assignment'] = object_assignment
    total_num_proposal = objectness_label_T.shape[0]*objectness_label_T.shape[1]
    end_points_T['pos_ratio'] = \
        torch.sum(objectness_label_T.float().cuda())/float(total_num_proposal)
    end_points_T['neg_ratio'] = \
        torch.sum(objectness_mask_T.float())/float(total_num_proposal) - end_points_T['pos_ratio']
    
    objectness_loss = objectness_loss_S + objectness_loss_T

    # Box loss and sem cls loss
    center_loss_S, heading_cls_loss, heading_reg_loss, size_cls_loss_S, size_reg_loss, sem_cls_loss_S = \
        compute_box_and_sem_cls_loss(end_points_S, config)
    end_points_S['center_loss'] = center_loss_S
    end_points_S['heading_cls_loss'] = heading_cls_loss
    end_points_S['heading_reg_loss'] = heading_reg_loss
    end_points_S['size_cls_loss'] = size_cls_loss_S
    end_points_S['size_reg_loss'] = size_reg_loss
    end_points_S['sem_cls_loss'] = sem_cls_loss_S
    box_loss = center_loss_S + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss_S + size_reg_loss
    end_points_S['box_loss'] = box_loss
    
    center_loss_T, size_cls_loss_T, sem_cls_loss_T = compute_center_and_sem_cls_loss(end_points_T, config)
    end_points_T['center_loss'] = center_loss_T
    end_points_T['size_cls_loss'] = size_cls_loss_T
    end_points_T['sem_cls_loss'] = sem_cls_loss_T
    
    box_loss += center_loss_T + 0.1*size_cls_loss_T
    sem_cls_loss = sem_cls_loss_S + sem_cls_loss_T

    # Source domain
    local_d_pred_S = end_points_S['local_d_pred'].transpose(1,2).contiguous()
    object_weight_S = F.softmax(end_points_S['objectness_scores'], dim=-1)[:,:,1:]
    source_dloss = 1.0 * torch.mean(local_d_pred_S ** 2 * object_weight_S)

    # Target domain
    local_d_pred_T = end_points_T['local_d_pred'].transpose(1,2).contiguous()
    object_weight_T = F.softmax(end_points_T['objectness_scores'], dim=-1)[:,:,1:]
    target_dloss = 1.0 * torch.mean((1-local_d_pred_T) ** 2 * object_weight_T)

    DA_loss = source_dloss + target_dloss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + DA_loss
    loss *= 10
    end_points_S['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points_S['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label_S.long()).float()*objectness_mask_S)/(torch.sum(objectness_mask_S)+1e-6)
    end_points_S['obj_acc'] = obj_acc

    return loss, end_points_S, end_points_T


def get_loss_cam(end_points, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, vote_xyz,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Final loss function
    pred_cam = end_points['cam'] # Bxnum_classx256
    pred_cam_gap = torch.mean(pred_cam, dim=2) # Bxnum_class
    cloud_label = end_points['cloud_label'] # Bxnum_class
    
    BCEWL = nn.BCEWithLogitsLoss()
    loss = BCEWL(pred_cam_gap.float(), cloud_label.float())
    end_points['loss'] = loss

    return loss, end_points


def get_loss_DA_cam(end_points_S, end_points_T, config):
    """ Loss functions

    Args:
        end_points: dict
            {   
                seed_xyz, seed_inds, global_d_pred, vote_xyz, local_d_pred,
                center,
                heading_scores, heading_residuals_normalized,
                size_scores, size_residuals_normalized,
                sem_cls_scores, #seed_logits,#
                center_label,
                heading_class_label, heading_residual_label,
                size_class_label, size_residual_label,
                sem_cls_label,
                box_label_mask,
                vote_label, vote_label_mask
            }
        config: dataset config instance
    Returns:
        loss: pytorch scalar tensor
        end_points: dict
    """

    # Vote loss
    vote_loss_S = compute_vote_loss(end_points_S)
    vote_loss = vote_loss_S
    end_points_S['vote_loss'] = vote_loss_S

    # Obj loss
    objectness_loss_S, objectness_label_S, objectness_mask_S, object_assignment = \
        compute_objectness_loss(end_points_S)
    end_points_S['objectness_loss'] = objectness_loss_S
    end_points_S['objectness_label'] = objectness_label_S
    end_points_S['objectness_mask'] = objectness_mask_S
    end_points_S['object_assignment'] = object_assignment
    total_num_proposal = objectness_label_S.shape[0]*objectness_label_S.shape[1]
    end_points_S['pos_ratio'] = \
        torch.sum(objectness_label_S.float().cuda())/float(total_num_proposal)
    end_points_S['neg_ratio'] = \
        torch.sum(objectness_mask_S.float())/float(total_num_proposal) - end_points_S['pos_ratio']
    objectness_loss = objectness_loss_S

    # Box loss and sem cls loss
    center_loss_S, heading_cls_loss, heading_reg_loss, size_cls_loss_S, size_reg_loss, sem_cls_loss_S = \
        compute_box_and_sem_cls_loss(end_points_S, config)
    end_points_S['center_loss'] = center_loss_S
    end_points_S['heading_cls_loss'] = heading_cls_loss
    end_points_S['heading_reg_loss'] = heading_reg_loss
    end_points_S['size_cls_loss'] = size_cls_loss_S
    end_points_S['size_reg_loss'] = size_reg_loss
    end_points_S['sem_cls_loss'] = sem_cls_loss_S
    box_loss = center_loss_S + 0.1*heading_cls_loss + heading_reg_loss + 0.1*size_cls_loss_S + size_reg_loss
    end_points_S['box_loss'] = box_loss
    
    sem_cls_loss_T = compute_sem_cls_loss(end_points_T, config)
    end_points_T['sem_cls_loss'] = sem_cls_loss_T
    
    sem_cls_loss = sem_cls_loss_S + 2*sem_cls_loss_T

    ## Domain Align Loss
    FL_global = FocalLoss(class_num=2, gamma=5)
    FL_vote = FocalLoss(class_num=2, gamma=3)

    # Source domain
    global_d_pred_S = end_points_S['global_d_pred']
    vote_feature_d_pred_S = end_points_S['vote_feature_d_pred']
    local_d_pred_S = end_points_S['local_d_pred'].transpose(1,2).contiguous()
    domain_S = Variable(torch.zeros(global_d_pred_S.size(0)).long().cuda())
    object_weight_local_S = F.softmax(end_points_S['objectness_scores'], dim=-1)[:,:,1:]
    source_dloss = 0.5 * torch.mean(local_d_pred_S ** 2 * object_weight_local_S) + 0.5 * FL_global(global_d_pred_S, domain_S) + 0.5 * FL_vote(vote_feature_d_pred_S, domain_S)

    # Target domain
    global_d_pred_T = end_points_T['global_d_pred']
    vote_feature_d_pred_T = end_points_T['vote_feature_d_pred']
    local_d_pred_T = end_points_T['local_d_pred'].transpose(1,2).contiguous()
    domain_T = Variable(torch.ones(global_d_pred_T.size(0)).long().cuda())
    object_weight_local_T = F.softmax(end_points_T['objectness_scores'], dim=-1)[:,:,1:]
    target_dloss = 0.5 * torch.mean((1-local_d_pred_T) ** 2 * object_weight_local_T) + 0.5 * FL_global(global_d_pred_T, domain_T) + 0.5 * FL_vote(vote_feature_d_pred_T, domain_T)

    DA_loss = source_dloss + target_dloss

    # Final loss function
    loss = vote_loss + 0.5*objectness_loss + box_loss + 0.1*sem_cls_loss + DA_loss
    loss *= 10
    end_points_S['loss'] = loss

    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(end_points_S['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==objectness_label_S.long()).float()*objectness_mask_S)/(torch.sum(objectness_mask_S)+1e-6)
    end_points_S['obj_acc'] = obj_acc

    return loss, end_points_S, end_points_T
