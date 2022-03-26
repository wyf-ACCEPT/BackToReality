import os
import sys
import json
import csv
import numpy as np
from plyfile import PlyData
#import matterport_utils


def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs.copy()
    return object_id_to_segs, label_to_segs


def get_id_to_label(filename):
    assert os.path.isfile(filename)
    id_to_label = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1
            label = data['segGroups'][i]['label']
            id_to_label[object_id] = label
    return id_to_label


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def create_color_palette():
    return [
       (255, 255, 255),
       (152, 223, 138),
       (31, 119, 180),
       (255, 187, 120),
       (188, 189, 34),
       (140, 86, 75),
       (255, 152, 150),
       (214, 39, 40),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (214, 39, 40),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52),
       (152, 223, 138),
       (31, 119, 180),
       (255, 187, 120),
       (188, 189, 34),
       (140, 86, 75),
       (255, 152, 150),
       (214, 39, 40),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (214, 39, 40),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14),
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),
       (112, 128, 144),
       (96, 207, 209), 
       (227, 119, 194),
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),
       (100, 85, 144),
       (197, 176, 213),
       (148, 103, 189),
       (196, 156, 148),
       (23, 190, 207),
       (178, 76, 76),  
       (247, 182, 210),
       (66, 188, 102), 
       (219, 219, 141),
       (140, 57, 197), 
       (202, 185, 52)
    ]


def visualize(pred_file, mesh_file, output_file):
    if not output_file.endswith('.ply'):
        util.print_error('output file must be a .ply file')
    colors = create_color_palette()
    num_colors = len(colors)
    ids = np.load(pred_file)
    with open(mesh_file, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        if num_verts != len(ids):
            print("num error!")
            sys.exit(0)
        # *_vh_clean_2.ply has colors already
        for i in range(num_verts):
            if ids[i] >= num_colors:
                ids[i] %= 30
            color = colors[ids[i]]
            plydata['vertex']['red'][i] = color[0]
            plydata['vertex']['green'][i] = color[1]
            plydata['vertex']['blue'][i] = color[2]
    plydata.write(output_file)

