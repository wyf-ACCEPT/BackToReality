# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Training routine for 3D object detection with SUN RGB-D or ScanNet.

Sample usage:
python train.py --dataset sunrgbd --log_dir log_sunrgbd

To use Tensorboard:
At server:
    python -m tensorboard.main --logdir=<log_dir_name> --port=6006
At local machine:
    ssh -L 1237:localhost:6006 <server_name>
Then go to local browser and type:
    localhost:1237
"""

import os
import sys
import time
import numpy as np
from datetime import datetime
import argparse
import importlib
from itertools import cycle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
from ap_helper import APCalculator, parse_predictions, parse_groundtruths

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='votenet_DA', help='Model file name [default: votenet]')
parser.add_argument('--dataset', default='scannet', help='Dataset name. sunrgbd or scannet. [default: scannet]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
parser.add_argument('--log_dir', default='log_debug', help='Dump dir to save model checkpoint [default: log_scannet_DA]')
parser.add_argument('--dump_dir', default=None, help='Dump dir to save sample outputs [default: None]')
parser.add_argument('--num_point', type=int, default=40000, help='Point Number [default: 40000]')
parser.add_argument('--num_target', type=int, default=256, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--ap_iou_thresh', type=float, default=0.25, help='AP IoU threshold [default: 0.25]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=4, help='Batch Size during training [default: 4]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
parser.add_argument('--use_sunrgbd_v2', action='store_true', help='Use V2 box labels for SUN RGB-D dataset')
parser.add_argument('--overwrite', action='store_true', help='Overwrite existing log and dump folders.')
parser.add_argument('--dump_results', action='store_true', help='Dump results.')
parser.add_argument('--center_jitter', type=float, default=0.1, help='magnitude of perturbation at the center [default: 0.1 (means 10%% jitter of the object size)].')
parser.add_argument('--dataset_without_mesh', action='store_true', help='Not use the mesh information.')
FLAGS = parser.parse_args()

# ------------------------------------------------------------------------- SET RANDOM SEED (if needed)
def set_random_seed(seed_value):
    import random
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True

SEED_VALUE = 42
set_random_seed(SEED_VALUE)

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))
LOG_DIR = FLAGS.log_dir
DEFAULT_DUMP_DIR = os.path.join(BASE_DIR, os.path.basename(LOG_DIR))
DUMP_DIR = FLAGS.dump_dir if FLAGS.dump_dir is not None else DEFAULT_DUMP_DIR
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'train_BR.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH
FLAGS.DUMP_DIR = DUMP_DIR

# Prepare LOG_DIR and DUMP_DIR
if os.path.exists(LOG_DIR) and FLAGS.overwrite:
    print('Log folder %s already exists. Are you sure to overwrite? (Y/N)'%(LOG_DIR))
    c = input()
    if c == 'n' or c == 'N':
        print('Exiting..')
        exit()
    elif c == 'y' or c == 'Y':
        print('Overwrite the files in the log and dump folers...')
        os.system('rm -r %s %s'%(LOG_DIR, DUMP_DIR))

if not os.path.exists(LOG_DIR):
    os.mkdir(LOG_DIR)

LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train_%02d.txt'  % (FLAGS.center_jitter*100)), 'a')
LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

# Init datasets and dataloaders 
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# Create Dataset and Dataloader
if FLAGS.dataset == 'matterport':
    sys.path.append(os.path.join(ROOT_DIR, 'matterport'))
    from matterport_detection_dataset import MatterportDetectionDataset, MAX_NUM_OBJ
    from model_util_matterport import MatterportDatasetConfig_md40
    DATASET_WITHOUT_MESH = FLAGS.dataset_without_mesh
    if DATASET_WITHOUT_MESH:
        data_path = 'matterport_train_detection_data_md40_obj_aug'
    else:
        data_path = 'matterport_train_detection_data_md40_obj_mesh_aug'
        
    DATASET_CONFIG = MatterportDatasetConfig_md40()
    TRAIN_DATASET_S = MatterportDetectionDataset('train_aug', data_path=data_path, num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TEST_DATASET_S = MatterportDetectionDataset('val', data_path=data_path, num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TRAIN_DATASET_T = MatterportDetectionDataset('train', 'matterport_train_detection_data_md40', num_points=NUM_POINT,
        augment=True, center_jitter=FLAGS.center_jitter,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TEST_DATASET_T = MatterportDetectionDataset('val', 'matterport_train_detection_data_md40', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
elif FLAGS.dataset == 'scannet':
    sys.path.append(os.path.join(ROOT_DIR, 'scannet'))
    from scannet_detection_dataset import ScannetDetectionDataset, MAX_NUM_OBJ
    from model_util_scannet import ScannetDatasetConfig_md40
    DATASET_WITHOUT_MESH = FLAGS.dataset_without_mesh
    if DATASET_WITHOUT_MESH:
        data_path = 'scannet_train_detection_data_md40_obj_aug'
    else:
        data_path = 'scannet_train_detection_data_md40_obj_mesh_aug'
        
    DATASET_CONFIG = ScannetDatasetConfig_md40()
    TRAIN_DATASET_S = ScannetDetectionDataset('train_aug', data_path=data_path, num_points=NUM_POINT,
        augment=True,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TEST_DATASET_S = ScannetDetectionDataset('val', data_path=data_path, num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TRAIN_DATASET_T = ScannetDetectionDataset('train', 'scannet_train_detection_data_md40', num_points=NUM_POINT,
        augment=True, center_jitter=FLAGS.center_jitter,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
    TEST_DATASET_T = ScannetDetectionDataset('val', 'scannet_train_detection_data_md40', num_points=NUM_POINT,
        augment=False,
        use_color=FLAGS.use_color, use_height=(not FLAGS.no_height))
else:
    print('Unknown dataset %s. Exiting...'%(FLAGS.dataset))
    exit(-1)
# print(len(TRAIN_DATASET_S), len(TEST_DATASET_S))
# print(len(TRAIN_DATASET_T), len(TEST_DATASET_T))
TRAIN_DATALOADER_S = DataLoader(TRAIN_DATASET_S, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER_S = DataLoader(TEST_DATASET_S, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, worker_init_fn=my_worker_init_fn)
TRAIN_DATALOADER_T = DataLoader(TRAIN_DATASET_T, batch_size=BATCH_SIZE,
    shuffle=True, num_workers=0, worker_init_fn=my_worker_init_fn)
TEST_DATALOADER_T = DataLoader(TEST_DATASET_T, batch_size=BATCH_SIZE,
    shuffle=False, num_workers=0, worker_init_fn=my_worker_init_fn)
print(len(TRAIN_DATALOADER_S), len(TEST_DATALOADER_S))
print(len(TRAIN_DATALOADER_T), len(TEST_DATALOADER_T))

# Init the model and optimizer
MODEL = importlib.import_module(FLAGS.model) # import network module
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_input_channel = int(FLAGS.use_color)*3 + int(not FLAGS.no_height)*1

if FLAGS.model == 'boxnet':
    print('Unknown network!')
    exit(-1)
else:
    Detector = MODEL.VoteNet_DA

net = Detector(num_class=DATASET_CONFIG.num_class,
               num_heading_bin=DATASET_CONFIG.num_heading_bin,
               num_size_cluster=DATASET_CONFIG.num_size_cluster,
               mean_size_arr=DATASET_CONFIG.mean_size_arr,
               num_proposal=FLAGS.num_target,
               input_feature_dim=num_input_channel,
               vote_factor=FLAGS.vote_factor,
               sampling=FLAGS.cluster_sampling)

if torch.cuda.device_count() > 1:
    log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
net.to(device)
criterion = MODEL.get_loss_DA

# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] - 10
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)

def get_current_lr(epoch):
    lr = BASE_LEARNING_RATE
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Used for AP calculation
CONFIG_DICT = {'remove_empty_box':False, 'use_3d_nms':True,
    'nms_iou':0.25, 'use_old_type_nms':False, 'cls_nms':True,
    'per_class_proposal': False, 'conf_thresh':0.05,
    'dataset_config':DATASET_CONFIG}

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    net.train() # set model to training mode
    for batch_idx, (batch_data_label_S, batch_data_label_T) in enumerate(zip(TRAIN_DATALOADER_S, cycle(TRAIN_DATALOADER_T))):
        for key in batch_data_label_S:
            batch_data_label_S[key] = batch_data_label_S[key].to(device)
        for key in batch_data_label_T:
            batch_data_label_T[key] = batch_data_label_T[key].to(device)

        # Forward pass
        optimizer.zero_grad()
        inputs_S = {'point_clouds': batch_data_label_S['point_clouds']}
        inputs_T = {'point_clouds': batch_data_label_T['point_clouds']}
        end_points_S = net(inputs_S)
        end_points_T = net(inputs_T)
        
        # Compute loss and gradients, update parameters.
        for key in batch_data_label_S:
            assert(key not in end_points_S)
            end_points_S[key] = batch_data_label_S[key]
        for key in batch_data_label_T:
            assert(key not in end_points_T)
            end_points_T[key] = batch_data_label_T[key]
        loss, end_points_S, _ = criterion(end_points_S, end_points_T, DATASET_CONFIG)
        loss.backward()
        optimizer.step()
        print(loss)
        
        # Accumulate statistics and print out
        for key in end_points_S:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points_S[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            for key in sorted(stat_dict.keys()):
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0

def evaluate_one_epoch(visual_final=False):
    stat_dict = {} # collect statistics
    ap_calculator_S = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
        class2type_map=DATASET_CONFIG.class2type)
    ap_calculator_T = APCalculator(ap_iou_thresh=FLAGS.ap_iou_thresh,
        class2type_map=DATASET_CONFIG.class2type)
    net.eval() # set model to eval mode (for bn and dp)
    for batch_idx, (batch_data_label_S, batch_data_label_T) in enumerate(zip(TEST_DATALOADER_S, TEST_DATALOADER_T)):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label_S:
            batch_data_label_S[key] = batch_data_label_S[key].to(device)
        for key in batch_data_label_T:
            batch_data_label_T[key] = batch_data_label_T[key].to(device)
        
        # Forward pass
        inputs_S = {'point_clouds': batch_data_label_S['point_clouds']}
        inputs_T = {'point_clouds': batch_data_label_T['point_clouds']}
        with torch.no_grad():
            end_points_S = net(inputs_S)
            end_points_T = net(inputs_T)

        # Compute loss
        for key in batch_data_label_S:
            assert(key not in end_points_S)
            end_points_S[key] = batch_data_label_S[key]
        for key in batch_data_label_T:
            assert(key not in end_points_T)
            end_points_T[key] = batch_data_label_T[key]
        loss, end_points_S, end_points_T = criterion(end_points_S, end_points_T, DATASET_CONFIG)

        if visual_final:
            CONFIG_DICT['conf_thresh'] = 0.7
        
        batch_pred_map_cls_S = parse_predictions(end_points_S, CONFIG_DICT) 
        batch_gt_map_cls_S = parse_groundtruths(end_points_S, CONFIG_DICT) 
        ap_calculator_S.step(batch_pred_map_cls_S, batch_gt_map_cls_S)
        
        batch_pred_map_cls_T = parse_predictions(end_points_T, CONFIG_DICT) 
        batch_gt_map_cls_T = parse_groundtruths(end_points_T, CONFIG_DICT) 
        
        if visual_final:
            if batch_idx == 20:
                print('Stop here!')
            from utils.pc_util import write_bbox, write_ply, write_ply_rgb
            point_cloud = end_points_T['point_clouds'][:, :, :3].detach().cpu().numpy()
            pcl_color = end_points_T['pcl_color'].detach().cpu().numpy()
            visual_path = os.path.join(BASE_DIR, 'visual')
            
            for i in range(point_cloud.shape[0]):
                all_3d_box_pred = np.concatenate([batch_pred_map_cls_T[i][k][1].reshape(1,8,3) for k in range(len(batch_pred_map_cls_T[i]))], axis=0)
                center_axis_pred = all_3d_box_pred.mean(1)
                center_flip_pred = np.zeros(center_axis_pred.shape)
                center_flip_pred[:,0], center_flip_pred[:,1], center_flip_pred[:,2] = center_axis_pred[:,0], center_axis_pred[:,2], -center_axis_pred[:,1]
                size_flip_pred = (all_3d_box_pred.max(1) - all_3d_box_pred.min(1))[:, [0,2,1]]
                bbox_pred = np.concatenate((center_flip_pred, size_flip_pred), axis=1)
                write_bbox(bbox_pred, os.path.join(visual_path, '%02d_bbox_VSSDA.ply' % (i+batch_idx*BATCH_SIZE)))
            
        ap_calculator_T.step(batch_pred_map_cls_T, batch_gt_map_cls_T)


    # Evaluate average precision
    print("================= Source =================")
    metrics_dict_S = ap_calculator_S.compute_metrics()
    for key in metrics_dict_S:
        log_string('eval %s: %f'%(key, metrics_dict_S[key]))
    print("================= Target =================")
    metrics_dict_T = ap_calculator_T.compute_metrics()
    for key in metrics_dict_T:
        log_string('eval %s: %f'%(key, metrics_dict_T[key]))
        if 'mAP' in key:
            open(os.path.join(LOG_DIR, 'Eval_mAP.txt'), 'a').write('eval %s: %f%%\n'%(key, metrics_dict_T[key] * 100))


def train(start_epoch):
    global EPOCH_CNT 
    open(os.path.join(LOG_DIR, 'Eval_mAP.txt'), 'a').write('\nStart at %s.\n' % (time.asctime()))
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        np.random.seed()
        
        train_one_epoch()
        if EPOCH_CNT == 0 or EPOCH_CNT % 10 == 9: # Eval every 10 epochs
            open(os.path.join(LOG_DIR, 'Eval_mAP.txt'), 'a').write('(Batch %s) ' % EPOCH_CNT)
            evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1-10, # after training one epoch, the start_epoch should be epoch+1
                     'optimizer_state_dict': optimizer.state_dict()
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(LOG_DIR, 'train_BR.tar'))

if __name__=='__main__':
    train(start_epoch)
