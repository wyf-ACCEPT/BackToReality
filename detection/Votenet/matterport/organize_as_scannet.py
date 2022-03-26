import os
import sys


with open('scenes_train.txt', 'r') as f:
    strings = f.readlines()
    train_split = open('for_scannet/matterport3d_train.txt', 'w')
    for house_id, s in enumerate(strings):
        house_dir = './v1/scans/{}/region_segmentations/'.format(s[:-1])
        house_files = os.listdir(house_dir)
        regions_num = len(house_files) // 4
        for region_id in range(regions_num):
            region_fold = 'scene{:04d}_{:02d}'.format(house_id, region_id)
            os.system('mkdir for_scannet/scans/{}'.format(region_fold))
            os.system('cp {}region{}.fsegs.json ./for_scannet/scans/{}/'.format(house_dir, region_id, region_fold))
            os.system('cp {}region{}.ply ./for_scannet/scans/{}/'.format(house_dir, region_id, region_fold))
            os.system('cp {}region{}.semseg.json ./for_scannet/scans/{}/'.format(house_dir, region_id, region_fold))
            os.system('cp {}region{}.vsegs.json ./for_scannet/scans/{}/'.format(house_dir, region_id, region_fold))
            train_split.write('{}\n'.format(region_fold))
    train_split.close()

with open('scenes_val.txt', 'r') as f:
    strings = f.readlines()
    train_split = open('for_scannet/matterport3d_val.txt', 'w')
    for house_id, s in enumerate(strings):
        house_id += 61
        house_dir = './v1/scans/{}/region_segmentations/'.format(s[:-1])
        house_files = os.listdir(house_dir)
        regions_num = len(house_files) // 4
        for region_id in range(regions_num):
            region_fold = 'scene{:04d}_{:02d}'.format(house_id, region_id)
            os.system('mkdir for_scannet/scans/{}'.format(region_fold))
            os.system('cp {}region{}.fsegs.json ./for_scannet/scans/{}/'.format(house_dir, region_id, region_fold))
            os.system('cp {}region{}.ply ./for_scannet/scans/{}/'.format(house_dir, region_id, region_fold))
            os.system('cp {}region{}.semseg.json ./for_scannet/scans/{}/'.format(house_dir, region_id, region_fold))
            os.system('cp {}region{}.vsegs.json ./for_scannet/scans/{}/'.format(house_dir, region_id, region_fold))
            train_split.write('{}\n'.format(region_fold))
    train_split.close()

'''
with open('scenes_test.txt', 'r') as f:
    strings = f.readlines()
    train_split = open('for_scannet/matterport3d_test.txt', 'w')
    for house_id, s in enumerate(strings):
        house_id += 72
        house_dir = './v1/scans/{}/region_segmentations/'.format(s[:-1])
        house_files = os.listdir(house_dir)
        regions_num = len(house_files) // 4
        for region_id in range(regions_num):
            region_fold = 'scene{:04d}_{:02d}'.format(house_id, region_id)
            os.system('mkdir for_scannet/scans_test/{}'.format(region_fold))
            os.system('cp {}region{}.fsegs.json ./for_scannet/scans_test/{}/'.format(house_dir, region_id, region_fold))
            os.system('cp {}region{}.ply ./for_scannet/scans_test/{}/'.format(house_dir, region_id, region_fold))
            os.system('cp {}region{}.semseg.json ./for_scannet/scans_test/{}/'.format(house_dir, region_id, region_fold))
            os.system('cp {}region{}.vsegs.json ./for_scannet/scans_test/{}/'.format(house_dir, region_id, region_fold))
            train_split.write('{}\n'.format(region_fold))
    train_split.close()
'''
