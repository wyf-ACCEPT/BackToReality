# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.nn.functional as F
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
from pointnet2_utils import furthest_point_sample, gather_operation
from pointnet2_modules import PointnetSAModuleCenters
from backbone_module import Pointnet2Backbone, Pointnet2Backbone_jitter
from voting_module import VotingModule
from proposal_module import ProposalModule
from dump_helper import dump_results
from loss_helper import get_loss_DA, get_loss_weak, get_loss_DA_jitter


class GradReverse(Function):

    @staticmethod
    def forward(self, x):
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        #pdb.set_trace()
        return (grad_output * -1.0)


def grad_reverse(x):
    return GradReverse.apply(x)


class VoteNet_DA(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)

        # global domain prediction
        self.global_netD1 = nn.Sequential(
            nn.Conv1d(256,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )
        self.global_netD2 = nn.Linear(128, 2)
        '''
        self.global_netD = nn.Sequential(
            nn.Conv1d(256,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,2,1)
            )
        '''

        # local domain prediction
        self.local_netD = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1,1)
            )

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
        # predict center jitter from center features
        # do not forget to use the class information!!!!            
    
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features
        
        end_points = self.pnet(xyz, features, end_points)

        ## domain prediction
        # global
        global_d_pred = self.global_netD1(grad_reverse(end_points['seed_features'])) # Bx128x1024
        global_d_pred = torch.mean(global_d_pred, dim=2) # Bx128
        global_d_pred = self.global_netD2(global_d_pred) # Bx2
        '''
        global_d_pred = self.global_netD(grad_reverse(end_points['seed_features'])) # Bx2x1024
        global_d_pred = torch.mean(global_d_pred, dim=2) # Bx2
        '''
        end_points['global_d_pred'] = global_d_pred

        # local
        local_d_pred = self.local_netD(grad_reverse(end_points['aggregated_vote_features'])) # Bx1x256
        local_d_pred = torch.sigmoid(local_d_pred)
        end_points['local_d_pred'] = local_d_pred

        return end_points


class VoteNet_DA_jitter(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone_jitter(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)

        # global domain prediction
        self.global_netD1 = nn.Sequential(
            nn.Conv1d(256,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )
        self.global_netD2 = nn.Linear(128, 2)
        '''
        self.global_netD = nn.Sequential(
            nn.Conv1d(256,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,2,1)
            )
        '''

        # local domain prediction
        self.local_netD = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1,1)
            )

        # jitter prediction
        self.jitter_netD = nn.Sequential(
            nn.Conv1d(150,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1,1)
            )

        self.jitter_net = nn.Sequential(
            nn.Conv1d(150,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,3,1)
            )

    def forward(self, inputs, center_xyz=None, center_cls=None):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], center_xyz, center_cls, end_points)
        # predict center jitter from center features
        # do not forget to use the class information!!!!
        if center_xyz is not None:
            end_points['jitter_pred'] = self.jitter_net(end_points['center_features']) # B, 3, 64
    
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features
        
        end_points = self.pnet(xyz, features, end_points)

        ## domain prediction
        # global
        global_d_pred = self.global_netD1(grad_reverse(end_points['seed_features'])) # Bx128x1024
        global_d_pred = torch.mean(global_d_pred, dim=2) # Bx128
        global_d_pred = self.global_netD2(global_d_pred) # Bx2
        end_points['global_d_pred'] = global_d_pred

        # local
        local_d_pred = self.local_netD(grad_reverse(end_points['aggregated_vote_features'])) # Bx1x256
        local_d_pred = torch.sigmoid(local_d_pred)
        end_points['local_d_pred'] = local_d_pred

        
        # jitter
        if center_xyz is not None:
            jitter_d_pred = self.jitter_netD(grad_reverse(end_points['center_features'])) # Bx1x64
            jitter_d_pred = torch.sigmoid(jitter_d_pred)
            end_points['jitter_d_pred'] = jitter_d_pred


        return end_points


class VoteNet_DA_jitter2(nn.Module):
    r"""
        A deep neural network for 3D object detection with end-to-end optimizable hough voting.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        vote_factor: (default: 1)
            Number of votes generated from each seed point.
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
        input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='vote_fps'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling=sampling

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and detection
        self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
            mean_size_arr, num_proposal, sampling)

        # global domain prediction
        self.global_netD1 = nn.Sequential(
            nn.Conv1d(256,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU()
            )
        self.global_netD2 = nn.Linear(128, 2)
        '''
        self.global_netD = nn.Sequential(
            nn.Conv1d(256,256,1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,2,1)
            )
        '''

        # local domain prediction
        self.local_netD = nn.Sequential(
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,128,1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128,1,1)
            )

        # jitter prediction
        self.ctjt_head = PointnetSAModuleCenters(
                npoint=64,
                radius=0.8,
                nsample=16,
                mlp=[128, 128],
                use_xyz=True,
                normalize_xyz=False
            )

        self.jitter_net = nn.Sequential(
            nn.Conv1d(150,64,1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64,3,1)
            )

    def forward(self, inputs, center_xyz=None, center_cls=None):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """
        end_points = {}
        batch_size = inputs['point_clouds'].shape[0]

        end_points = self.backbone_net(inputs['point_clouds'], end_points)
    
        # --------- HOUGH VOTING ---------
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        end_points['vote_xyz'] = xyz
        end_points['vote_features'] = features
        
        end_points = self.pnet(xyz, features, end_points)

        if center_xyz is not None:
            center_features = self.ctjt_head(end_points['aggregated_vote_xyz'], end_points['aggregated_vote_features'].detach(), center_xyz)
            end_points['center_features'] = torch.cat([center_features, torch.eye(22)[center_cls].transpose(1,2).cuda()], dim=1) # B, 128+22, 64
        # predict center jitter from center features
        # do not forget to use the class information!!!!
        if center_xyz is not None:
            end_points['jitter_pred'] = self.jitter_net(end_points['center_features']) # B, 3, 64

        ## domain prediction
        # global
        global_d_pred = self.global_netD1(grad_reverse(end_points['seed_features'])) # Bx128x1024
        global_d_pred = torch.mean(global_d_pred, dim=2) # Bx128
        global_d_pred = self.global_netD2(global_d_pred) # Bx2
        '''
        global_d_pred = self.global_netD(grad_reverse(end_points['seed_features'])) # Bx2x1024
        global_d_pred = torch.mean(global_d_pred, dim=2) # Bx2
        '''
        end_points['global_d_pred'] = global_d_pred

        # local
        local_d_pred = self.local_netD(grad_reverse(end_points['aggregated_vote_features'])) # Bx1x256
        local_d_pred = torch.sigmoid(local_d_pred)
        end_points['local_d_pred'] = local_d_pred

        return end_points


if __name__=='__main__':
    sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
    from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
    from loss_helper import get_loss

    # Define model
    model = VoteNet(10,12,10,np.random.random((10,3))).cuda()
    
    try:
        # Define dataset
        TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

        # Model forward pass
        sample = TRAIN_DATASET[5]
        inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
    except:
        print('Dataset has not been prepared. Use a random sample.')
        inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

    end_points = model(inputs)
    for key in end_points:
        print(key, end_points[key])

    try:
        # Compute loss
        for key in sample:
            end_points[key] = torch.from_numpy(sample[key]).unsqueeze(0).cuda()
        loss, end_points = get_loss(end_points, DC)
        print('loss', loss)
        end_points['point_clouds'] = inputs['point_clouds']
        end_points['pred_mask'] = np.ones((1,128))
        dump_results(end_points, 'tmp', DC)
    except:
        print('Dataset has not been prepared. Skip loss and dump.')

