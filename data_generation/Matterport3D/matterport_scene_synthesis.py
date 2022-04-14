import open3d as o3d
import numpy as np
import cv2
from segment_tools import export_random
import os
import copy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from ins_ply import read_aggregation, read_segmentation, create_color_palette, get_id_to_label
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import warnings
warnings.filterwarnings("ignore")



def get_MER(points):
    points = copy.deepcopy(points)
    xys = points[:, 0:2] * 1000
    xys = xys.astype('int')
    (x_center, y_center), (y_size, x_size), angle = cv2.minAreaRect(xys)
    x_center /= 1e3; y_center /= 1e3; y_size /= 1e3; x_size /= 1e3
    angle = (90 - angle) / 180 * np.pi
    return (x_center, y_center), (y_size, x_size), angle


# MER: ((x, y), (long, short), theta)
def get_solid_MER(points):
    points = copy.deepcopy(points)
    xys = points[:, 0:2]
    xys *= 1000
    xys = xys.astype('int')
    rect = cv2.minAreaRect(xys)
    k_means = KMeans(n_clusters=2)
    k_means.fit(xys)
    cluster_label = k_means.predict(xys)
    choose0 = (sum(cluster_label == 0) < sum(cluster_label == 1))
    if choose0:
        xys_part = xys[cluster_label == 0]
        xys_other = xys[cluster_label == 1]
    else:
        xys_part = xys[cluster_label == 1]
        xys_other = xys[cluster_label == 0]
    rect_part = cv2.minAreaRect(xys_part)
    Is_solid = (rect_part[1][0] * rect_part[1][1] * 2.5 > rect[1][0] * rect[1][1])
    if Is_solid:
        pass
    else:
        rect = cv2.minAreaRect(xys_other)
    
    if rect[1][1] > rect[1][0]:
        l_s = (rect[1][1]/1000, rect[1][0]/1000)
    else:
        l_s = (rect[1][0]/1000, rect[1][1]/1000)
    if rect[1][0] >= rect[1][1]:
        theta = -rect[2]
        if theta == 0:
            theta = 180
    else:
        theta = -rect[2] + 90
    return ((rect[0][0] / 1000, rect[0][1] / 1000), l_s, theta)


def find_nearest_object(ls_ratio, info_dict, object_name, require_support=False):
    min_dis = 100
    min_code = ""
    for key, value in info_dict.items():
        if value[0][0][1] == 0:
            continue
        if abs(value[0][0][0] / value[0][0][1] - ls_ratio) < min_dis:
            if require_support == True and value[2] == False:
                continue
            min_dis = abs(value[0][0][0] / value[0][0][1] - ls_ratio)
            min_code = key
    if min_code == "" and require_support == True:
        return find_nearest_object(ls_ratio, info_dict, object_name)
    txt = object_name + "_" + min_code + ".txt"
    return txt, info_dict[min_code]
        

def generate_initial_random_positions(mesh_file, agg_file, seg_file, modelnet40_path, scan_name):
    xyz_obj_dict = export_random(mesh_file, agg_file, seg_file, scan_name)
    obj_prop = np.load('CONFIG/object40_property.npy', allow_pickle=True, encoding='bytes').item()
    modelnet40_names = np.loadtxt('CONFIG/modelnet40_shape_names.txt', dtype='object')
    supporter_list = ["tv_stand", "desk", "bed", "bookshelf", "table", "night_stand"]
    # object_id: [(x,y,z), (sx,sy,sz), object_txt, Is_supporter, theta, support_MER/None, height/None]
    # theta is the orientation of the object (anticlockwise)
    # supporter has theta, support_MER and height
    # object with plane but not supporter has theta, None, None
    # object without plane has None, None, None
    # MER: ((x, y), (long, short), theta)
    positions = {}
    for key, value in xyz_obj_dict.items():
        obj_name = modelnet40_names[value[2] - 1]
        txts = os.listdir(os.path.join(modelnet40_path, obj_name))
        txts.remove("this_class_info.npy")
        this_class_info = np.load(os.path.join(modelnet40_path, obj_name,
         "this_class_info.npy"), allow_pickle=True).item()

        # choose xy or yx
        if np.random.rand() > 0.5:
            _, _, _, dxavg, dyavg, dzavg = obj_prop[value[2]][0:6]
        else:
            dxavg, dyavg, dzavg, _, _, _ = obj_prop[value[2]][0:6]
        dxmin, dymin, dzmin = 0.8 * dxavg, 0.8 * dyavg, 0.8 * dzavg
        dxmax, dymax, dzmax = 1.3 * dxavg, 1.3 * dyavg, 1.3 * dzavg
        dx = dxmin + np.random.rand() * (dxmax - dxmin)
        dy = dymin + np.random.rand() * (dymax - dymin)
        dz = dzmin + np.random.rand() * (dzmax - dzmin)
        x, y, z = value[0]
        ls_ratio = max(dx, dy) / min(dx, dy)
        if obj_name in supporter_list:
            txt, obj_info = find_nearest_object(ls_ratio, this_class_info, obj_name, require_support=True)
        else:
            txt, obj_info = find_nearest_object(ls_ratio, this_class_info, obj_name)
        obj_txt = os.path.join(modelnet40_path, obj_name, txt)
        pc_txt = np.loadtxt(obj_txt, delimiter=',')
        pc_txt[:, [1, 2]] = pc_txt[:, [2, 1]]
        pc_txt[:, [4, 5]] = pc_txt[:, [4, 5]]
        ddx, ddy, ddz = max(pc_txt[:, 0]) - min(pc_txt[:, 0]), max(pc_txt[:, 1]) - min(pc_txt[:, 1]), max(pc_txt[:, 2]) - min(pc_txt[:, 2])
        scale = (dx * dx * dz / ddx / ddy / ddz)**(1 / 3)
        # special category: door, curtain, ...
        # only constrain the height
        if obj_name in ["curtain", "door", "sofa", "desk"]:
            scale = dz / ddz
        # special category: keyboard
        # only constrain the horizontal property
        if obj_name in ["keyboard"]:
            scale = (dx * dy / ddx / ddy)**(1 / 2)
        theta = np.random.rand() * 360
        if obj_name in supporter_list:
            MER = ((x, y), (scale * max(ddx, ddy), scale * min(ddx, ddy)), (theta + obj_info[0][1]) % 180)
            height = z + scale * obj_info[1]
            positions[key] = [(x, y, z), (scale, scale, scale), obj_txt, True, theta, MER, height]
        else:
            positions[key] = [(x, y, z), (scale, scale, scale), obj_txt, False, theta, None, None]

    '''
    # no floor/wall points
    wall_points = np.array([])
    floor_points = np.array([])
    '''
    
    # get floor/wall points
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    scene_vertices = np.array(mesh.vertices)
    map2nyu40 = np.load('CONFIG/map2nyu40.npy', allow_pickle=True).item()
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)
    for label, segs in label_to_segs.items():
        label_id = map2nyu40[label]
        for seg in segs:
            try:
                verts = seg_to_verts[seg]
            except:
                continue
            label_ids[verts] = label_id
    wall_points = scene_vertices[label_ids == 1]
    #floor_points = scene_vertices[label_ids == 2]
    floor_points = scene_vertices[abs(scene_vertices[:, 2]) < 0.05]
    
    return positions, (floor_points, wall_points)


def point_in_MER(x, y, MER):
    dx = x - MER[0][0]
    dy = abs(MER[0][1] - y)
    dd = (dx ** 2 + dy ** 2)** 0.5
    cosf = dx / dd
    f = np.arccos(cosf) / np.pi * 180
    if MER[2] >= 90:
        theta = f - MER[2] + 90
        dx_align = abs(dd * np.cos(theta / 180 * np.pi))
        dy_align = abs(dd * np.sin(theta / 180 * np.pi))
        if dx_align < MER[1][1] / 2 and dy_align < MER[1][0] / 2:
            return True
    else:
        theta = f - MER[2]
        dx_align = abs(dd * np.cos(theta / 180 * np.pi))
        dy_align = abs(dd * np.sin(theta / 180 * np.pi))
        if dx_align < MER[1][0] / 2 and dy_align < MER[1][1] / 2:
            return True
    return False


def generate_gravity_aware_positions(positions, floor_points):
    new_positions = positions.copy()
    if len(floor_points) == 0:
        ground_z = 0
    else:
        ground_z = np.mean(floor_points[:, 2])
    supporter_MER = {}  # id: MER
    # Stage1: not be supported, things on the ground or dangling (lamp/sink) 
    # Stage2: supported objects
    # stage1_id: [stage2_id, ...]
    stage_map = {}
    for key, value in positions.items():
        obj_name = value[2].split('/')[-2]
        if value[3] == True:
            supporter_MER[key] = value[5]
            stage_map[key] = []
            x, y, z = value[0]
            _, _, sz = value[1]
            pc_txt = np.loadtxt(value[2], delimiter=',')
            pc_txt[:, [1, 2]] = pc_txt[:, [2, 1]]
            pc_txt[:, [4, 5]] = pc_txt[:, [4, 5]]
            new_z = ground_z - sz * min(pc_txt[:, 2])
            new_positions[key][0] = (x, y, new_z)
            new_positions[key][6] = value[6] + (new_z - z)
    for key, value in positions.items():
        if value[3] == False:
            min_center_dis2 = 100
            x, y = value[0][0], value[0][1]
            choosed_supporter = -1
            for supporter_id, MER in supporter_MER.items():
                if point_in_MER(x, y, MER) and (x - MER[0][0])** 2 + (y - MER[0][1])** 2 < min_center_dis2:
                    choosed_supporter = supporter_id
                    min_center_dis2 = (x - MER[0][0])** 2 + (y - MER[0][1])** 2
            # some object will never be supported
            # so we need to correct the wrong choice if needed
            obj_name = value[2].split('/')[-2]
            if obj_name not in ["monitor", "plant", "lamp", "sink", "cup", "keyboard", "bottle", "laptop"]:
                choosed_supporter = -1
            
            if choosed_supporter == -1:
                stage_map[key] = []
            else:
                stage_map[choosed_supporter].append(key)

            x, y, z = value[0]
            _, _, sz = value[1]
            pc_txt = np.loadtxt(value[2], delimiter=',')
            pc_txt[:, [1, 2]] = pc_txt[:, [2, 1]]
            pc_txt[:, [4, 5]] = pc_txt[:, [4, 5]]
            # special category: sink, lamp
            if choosed_supporter == -1:
                if (obj_name == "lamp" and z > 1.2) or obj_name == "sink":
                    new_z = z
                else:
                    new_z = ground_z - sz * min(pc_txt[:, 2])
            else:
                new_z = new_positions[choosed_supporter][6] - sz * min(pc_txt[:, 2])
            new_positions[key][0] = (x, y, new_z)
    return new_positions, stage_map


def anticlock_rotate_matrix(theta):
    # anticlockwise means y-->x
    theta *= -1
    return np.array([[np.cos(np.pi / 180 * theta), np.sin(np.pi / 180 * theta)],
     [-np.sin(np.pi / 180 * theta), np.cos(np.pi / 180 * theta)]])


def position_to_xyz(position, Is_density=False, ratio=None):
    # If consider density, the total number of points of a object is 10000*ratio
    obj_xyz = np.loadtxt(position[2], delimiter=',')[:, 0:3]
    obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
    if Is_density:
        ds_k = int(1 // ratio)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_xyz)
        down_pcd = pcd.uniform_down_sample(every_k_points=ds_k)
        obj_xyz = np.array(down_pcd.points)
        #print(position[2].split('/')[-2] + ": ", obj_xyz.shape[0], ", ", ds_k)
    # scale
    obj_xyz[:, 0] *= position[1][0]
    obj_xyz[:, 1] *= position[1][1]
    obj_xyz[:, 2] *= position[1][2]
    # rotate
    theta = position[4]
    obj_xyz[:, 0:2] = np.matmul(obj_xyz[:, 0:2], anticlock_rotate_matrix(theta))
    # translate
    obj_xyz[:, 0] += position[0][0]
    obj_xyz[:, 1] += position[0][1]
    obj_xyz[:, 2] += position[0][2]
    return obj_xyz


def Is_collide(xyz1, xyz2, threshold):
    D = pairwise_distances(xyz1, xyz2, metric='euclidean')
    return D.min() < threshold


def generate_collision_aware_positions(positions, stage_map, floor_points, threshold=0.05):
    obj_xyzs = {} # id: xyz
    obj_dxy = {}  # id: [dx, dy]
    if len(floor_points) > 0:
        xmin, xmax = floor_points[:, 0].min(), floor_points[:, 0].max()
        ymin, ymax = floor_points[:, 1].min(), floor_points[:, 1].max()
        stage1_center = [(xmin + xmax) / 2, (ymin + ymax) / 2]
    else:
        stage1_center = [0, 0]
    stage1_distance = {}  # stage1_id: dis_to_center
    for key, value in positions.items():
        if key in stage_map.keys():
            stage1_distance[key] = ((value[0][0] - stage1_center[0])** 2 + (value[0][1] - stage1_center[1])** 2)** 0.5
        # to boost the collision detection, we need to downsample the xyz
        xyz = position_to_xyz(value)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        down_pcd = pcd.uniform_down_sample(every_k_points=5)
        down_xyz = np.array(down_pcd.points)
        obj_xyzs[key] = down_xyz
        obj_dxy[key] = [0, 0]
    near_to_far = sorted(stage1_distance.keys(), key=lambda x: stage1_distance[x])
    # only consider nearer (to center) objects
    for i, obj_id in enumerate(near_to_far[1:]):
        consider_obj_ids = near_to_far[:i + 1]
        new_x, new_y, _ = positions[obj_id][0]
        move_vector = [0, 0]
        Is_collide_bool = False
        for consider_id in consider_obj_ids:
            con_x, con_y, _ = positions[consider_id][0]
            try:
                move_vector[0] += 1 / (new_x - con_x)
            except:
                move_vector[0] += 10
            try:
                move_vector[1] += 1 / (new_y - con_y)
            except:
                move_vector[1] += 10
            if Is_collide_bool:
                continue
            if Is_collide(obj_xyzs[obj_id], obj_xyzs[consider_id], threshold=threshold):
                Is_collide_bool = True
        scale = 0.1 / ((move_vector[0])** 2 + (move_vector[1])** 2)** 0.5
        move_vector[0] *= scale
        move_vector[1] *= scale
        while Is_collide_bool:
            obj_xyzs[obj_id][:, 0] += move_vector[0]
            obj_xyzs[obj_id][:, 1] += move_vector[1]
            obj_dxy[obj_id][0] += move_vector[0]
            obj_dxy[obj_id][1] += move_vector[1]
            Is_collide_bool = False
            for consider_id in consider_obj_ids:
                if Is_collide_bool:
                    break
                if Is_collide(obj_xyzs[obj_id], obj_xyzs[consider_id], threshold=threshold):
                    Is_collide_bool = True
    # stage2 objects
    for stage1_id, stage2_ids in stage_map.items():
        if len(stage2_ids) == 0:
            continue
        for stage2_id in stage2_ids:
            obj_xyzs[stage2_id][:, 0] += obj_dxy[stage1_id][0]
            obj_xyzs[stage2_id][:, 1] += obj_dxy[stage1_id][1]
            obj_dxy[stage2_id][0] += obj_dxy[stage1_id][0]
            obj_dxy[stage2_id][1] += obj_dxy[stage1_id][1]
        if len(stage2_ids) == 1:
            continue
        stage2_center = [positions[stage1_id][0][0], positions[stage1_id][0][1]]
        stage2_distance = {}  # stage2_id: dis_to_center
        for stage2_id in stage2_ids:
            value = positions[stage2_id]
            stage2_distance[stage2_id] = ((value[0][0] - stage2_center[0])** 2 + (value[0][1] - stage2_center[1])** 2)** 0.5
        far_to_near = sorted(stage2_distance.keys(), key=lambda x: stage2_distance[x], reverse=True)
        for i, obj_id in enumerate(far_to_near[1:]):
            consider_obj_ids = far_to_near[:i + 1]
            new_x, new_y, _ = positions[obj_id][0]
            # to avoid falling down, we need to correct the direction toward the center
            max_moving_len = 0
            move_vector = [0, 0]
            Is_collide_bool = False
            for consider_id in consider_obj_ids:
                con_x, con_y, _ = positions[consider_id][0]
                moving_len = (1 / (new_x - con_x)** 2 + 1 / (new_y - con_y)** 2)** 0.5
                if moving_len > max_moving_len:
                    max_moving_len = moving_len
                move_vector[0] += 1 / (new_x - con_x)
                move_vector[1] += 1 / (new_y - con_y)
                if Is_collide_bool:
                    continue
                if Is_collide(obj_xyzs[obj_id], obj_xyzs[consider_id], threshold=threshold):
                    Is_collide_bool = True
            center_move_vector = [stage2_center[0] - new_x, stage2_center[1] - new_y]
            scale_center = max_moving_len / ((center_move_vector[0])** 2 + (center_move_vector[1])** 2)** 0.5
            move_vector[0] += scale_center * center_move_vector[0]
            move_vector[1] += scale_center * center_move_vector[1]
            scale = 0.05 / ((move_vector[0])** 2 + (move_vector[1])** 2)** 0.5
            move_vector[0] *= scale
            move_vector[1] *= scale
            while Is_collide_bool:
                obj_xyzs[obj_id][:, 0] += move_vector[0]
                obj_xyzs[obj_id][:, 1] += move_vector[1]
                obj_dxy[obj_id][0] += move_vector[0]
                obj_dxy[obj_id][1] += move_vector[1]
                Is_collide_bool = False
                for consider_id in consider_obj_ids:
                    if Is_collide_bool:
                        break
                    if Is_collide(obj_xyzs[obj_id], obj_xyzs[consider_id], threshold=threshold):
                        Is_collide_bool = True
    new_positions = positions.copy()
    for key, value in new_positions.items():
        dx, dy = obj_dxy[key]
        value[0] = (value[0][0] + dx, value[0][1] + dy, value[0][2])
    return new_positions
    

def positions_to_pcd(positions, fw_points, Is_floor=True, Is_wall=False, Is_density=False, Is_HPR=False):
    if len(positions) == 0:
        return None
    xyzrgb_list = []
    colors = create_color_palette()
    # floor
    if Is_floor:
        floor_points = fw_points[0]
        ground_z = np.mean(floor_points[:, 2])
        floor_points[:, 2] = ground_z
        floor_color = colors[0]
        floor_rgb = np.array([list(floor_color)] * floor_points.shape[0])
        floor_xyzrgb = np.concatenate([floor_points, floor_rgb], axis=1)
        xyzrgb_list += list(floor_xyzrgb)
    # wall
    if Is_wall:
        wall_points = fw_points[1]
        wall_color = colors[0]
        wall_rgb = np.array([list(wall_color)] * wall_points.shape[0])
        wall_xyzrgb = np.concatenate([wall_points, wall_rgb], axis=1)
        xyzrgb_list += list(wall_xyzrgb)
    # density
    if Is_density:
        S_dict = {}
        for key, value in positions.items():
            obj_xyz = np.loadtxt(value[2], delimiter=',')[:, 0:3]
            obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
            ddx, ddy, ddz = max(obj_xyz[:, 0]) - min(obj_xyz[:, 0]), max(obj_xyz[:, 1]) - min(obj_xyz[:, 1]), max(obj_xyz[:, 2]) - min(obj_xyz[:, 2])
            dx, dy, dz = ddx * value[1][0], ddy * value[1][1], ddz * value[1][2]
            S_larger = dx * dy * dz / min(dx, dy, dz)
            S_dict[key] = S_larger
        S_max = max(S_dict.values())
    # position to points
    for key, value in positions.items():
        color = colors[key]
        if Is_density:
            obj_xyz = position_to_xyz(value, Is_density=True, ratio=S_dict[key]/S_max)
        else:
            obj_xyz = position_to_xyz(value)
        obj_rgb = np.array([list(color)] * obj_xyz.shape[0])
        obj_xyzrgb = np.concatenate([obj_xyz, obj_rgb], axis=1)
        xyzrgb_list += list(obj_xyzrgb)
    # points to pcd
    xyzrgb = np.array(xyzrgb_list)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyzrgb[:, 0:3])
    pcd.colors = o3d.utility.Vector3dVector(xyzrgb[:, 3:6] / 255)
    # HPR
    if Is_HPR:
        xs, ys = xyzrgb[:, 0], xyzrgb[:, 1]
        print("x in (", xs.min(), ",", xs.max(), "), y in (", ys.min(), ",", ys.max(), ")")
        _, pt_map = pcd.hidden_point_removal([0, 0, 2.5], 100)
        pcd = pcd.select_by_index(pt_map)
    return pcd


MDN_dict = {'airplane': 1, 'bathtub': 2, 'bed': 3, 'bench': 4, 'bookshelf': 5, 'bottle': 6, 'bowl': 7, 'car': 8, 'chair': 9, 'cone': 10, 'cup': 11, 'curtain': 12, 'desk': 13, 'door': 14, 'dresser': 15, 'flower_pot': 16, 'glass_box': 17, 'guitar': 18, 'keyboard': 19, 'lamp': 20, 'laptop': 21, 'mantel': 22, 'monitor': 23, 'night_stand': 24, 'person': 25, 'piano': 26, 'plant': 27, 'radio': 28, 'range_hood': 29, 'sink': 30, 'sofa': 31, 'stairs': 32, 'stool': 33, 'table': 34, 'tent': 35, 'toilet': 36, 'tv_stand': 37, 'vase': 38, 'wardrobe': 39, 'xbox': 40}

def export_for_md40(scan_name, Is_density=True, Is_HPR=False, Is_floor=False):
    positions = np.load('./augment_random_positions_matterport/'+scan_name+'.npy', allow_pickle=True).item()
    xyz_oid_list = []
    if Is_floor:
        floor_points = positions['floor_points']
        #floor_points[:, 2] = np.mean(floor_points[:, 2])
    positions.pop('floor_points')
    # density
    if Is_density:
        S_dict = {}
        for key, value in positions.items():
            obj_xyz = np.loadtxt(value[2], delimiter=',')[:, 0:3]
            obj_xyz[:, [1, 2]] = obj_xyz[:, [2, 1]]
            ddx, ddy, ddz = max(obj_xyz[:, 0]) - min(obj_xyz[:, 0]), max(obj_xyz[:, 1]) - min(obj_xyz[:, 1]), max(obj_xyz[:, 2]) - min(obj_xyz[:, 2])
            dx, dy, dz = ddx * value[1][0], ddy * value[1][1], ddz * value[1][2]
            S_larger = dx * dy * dz / min(dx, dy, dz)
            S_dict[key] = S_larger
        S_max = max(S_dict.values())
    # position to points
    instance_bboxes = np.zeros((len(positions),8))
    #label_to_nyuid = np.load('map2nyu40.npy', allow_pickle=True).item()
    label_to_modelnet40id = np.load("CONFIG/map2modelnet.npy", allow_pickle=True).item()
    obj_prop = np.load('CONFIG/object40_property.npy', allow_pickle=True, encoding='bytes').item()
    for key, value in positions.items():
        oid = key
        if Is_density:
            obj_xyz = position_to_xyz(value, Is_density=True, ratio=S_dict[key]/S_max)
        else:
            obj_xyz = position_to_xyz(value)
        obj_id = np.array([[oid]] * obj_xyz.shape[0])
        obj_xyzoid = np.concatenate([obj_xyz, obj_id], axis=1)
        xyz_oid_list += list(obj_xyzoid)
    if Is_floor:
        floor_xyzoid = np.concatenate([floor_points, np.array([[0]] * floor_points.shape[0])], axis=1)
        xyz_oid_list += list(floor_xyzoid)
    xyz_oid = np.array(xyz_oid_list)
    if Is_HPR:
        xyz = xyz_oid[:,:3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        xs, ys = xyz[:, 0], xyz[:, 1]
        delta_x, delta_y = (xs.max() - xs.min()) / 3, (ys.max() - ys.min()) / 3
        camera1 = [xs.min() + delta_x, ys.min() + delta_y, 2]
        camera2 = [xs.min() + 2 * delta_x, ys.min() + delta_y, 2]
        camera3 = [xs.min() + delta_x, ys.min() + 2 * delta_y, 2]
        camera4 = [xs.min() + 2 * delta_x, ys.min() + 2 * delta_y, 2]
        _, pt_map1 = pcd.hidden_point_removal(camera1, 100)
        _, pt_map2 = pcd.hidden_point_removal(camera2, 100)
        _, pt_map3 = pcd.hidden_point_removal(camera3, 100)
        _, pt_map4 = pcd.hidden_point_removal(camera4, 100)
        pt_map = np.unique(pt_map1 + pt_map2 + pt_map3 + pt_map4)
        xyz_oid = xyz_oid[pt_map]
    count_i = 0
    #np.save('xxw.npy', xyz_oid[:,:3])
    #sys.exit(0)
    oid_to_modelnet40id = {}
    for oid, value in positions.items():
        obj_xyz = xyz_oid[xyz_oid[:,3] == oid][:,:3]
        if len(obj_xyz) == 0:
            continue
        zmin = np.min(obj_xyz[:,2])
        zmax = np.max(obj_xyz[:,2])
        (x_center, y_center), (x_size, y_size), angle = get_MER(obj_xyz)
        obj_name = value[2].split('/')[-2]
        sem_label = MDN_dict[obj_name]
        oid_to_modelnet40id[oid] = sem_label
        #if sem_label in [4,7,6,5,33,14,3,32,10,36]:
        #    dxavg_xy, dyavg_xy, dzavg_xy, dxavg_yx, dyavg_yx, dzavg_yx = obj_prop[label_to_modelnet40id[id_to_label[oid]]][0:6]
        #    if (xmax-xmin) < min(dxavg_xy, dxavg_yx)/2 and (ymax-ymin) < min(dyavg_xy, dyavg_yx)/2 and (zmax-zmin) < min(dzavg_xy, dzavg_yx)/2:
        #        sem_label = -1
        #        xyz_oid[xyz_oid[:,3] == oid][:,3] = -1
        bbox = np.array([x_center, y_center, (zmin+zmax)/2, x_size, y_size, zmax-zmin, angle, sem_label])
        instance_bboxes[count_i,:] = bbox
        count_i += 1
    mesh_vertices = (xyz_oid.copy())[:,:3]
    instance_ids = xyz_oid[:, 3]
    semantic_ids = np.zeros_like(instance_ids)
    for i in range(len(semantic_ids)):
        if instance_ids[i] == 0:
            semantic_ids[i] = 0
        else:
            semantic_ids[i] = oid_to_modelnet40id[instance_ids[i]]
    return mesh_vertices, semantic_ids, instance_ids, instance_bboxes


if __name__ == "__main__":
    MATTERPORT_DIR = "/path/to/matterport/for_scannet/scans"
    scan_names = os.listdir(MATTERPORT_DIR)
    for scan_name in scan_names:
        print(scan_name)
        scan_folder = "/path/to/matterport/for_scannet/scans/" + scan_name + "/"
        mesh_file = os.path.join(scan_folder, 'region{}.ply'.format(int(scan_name[-2:])))
        agg_file = os.path.join(scan_folder, 'region{}.semseg.json'.format(int(scan_name[-2:])))
        seg_file = os.path.join(scan_folder, 'region{}.vsegs.json'.format(int(scan_name[-2:])))
        modelnet40_path = "./modelnet40_normal_resampled"
        try:
            positions, fw_points = generate_initial_random_positions(mesh_file, agg_file, seg_file, modelnet40_path, scan_name)
        except:
            print("Failed!")
            continue
        positions, stage_map = generate_gravity_aware_positions(positions, fw_points[0])
        try:
            positions = generate_collision_aware_positions(positions, stage_map, fw_points[0])
        except:
            try:
                positions = generate_collision_aware_positions(positions, stage_map, fw_points[0])
            except:
                continue
        positions['floor_points'] = fw_points[0]
        np.save('/path/to/BackToReality/data_generation/Matterport3D/augment_random_positions_matterport/' + scan_name + '.npy', positions)
    
    # augmentation
    sys.path.append(os.path.join(ROOT_DIR, 'detection' ,'votenet', 'matterport'))
    from matterport_detection_dataset import MatterportDetectionDataset
    dset = MatterportDetectionDataset(use_height=True, num_points=40000)
    obj_prop = np.load('CONFIG/object40_property.npy', allow_pickle=True, encoding='bytes').item()
    ls = np.load('CONFIG/scans_toadd_scarce.npy', allow_pickle=True)
    for d in dset:
        cl = d['cloud_label']
        scan_name = dset.scan_names[d['scan_idx']]
        print(scan_name)
        scan_folder = "/path/to/matterport/for_scannet/scans/" + scan_name + "/"
        mesh_file = os.path.join(scan_folder, 'region{}.ply'.format(int(scan_name[-2:])))
        agg_file = os.path.join(scan_folder, 'region{}.semseg.json'.format(int(scan_name[-2:])))
        seg_file = os.path.join(scan_folder, 'region{}.vsegs.json'.format(int(scan_name[-2:])))
        modelnet40_path = "./modelnet40_normal_resampled"
        add_bathtub = (scan_name in ls[0])
        add_bench = (scan_name in ls[1])
        add_desk = (scan_name in ls[2])
        add_dresser = (scan_name in ls[3])
        try:
            positions, fw_points = generate_initial_random_positions(mesh_file, agg_file, seg_file, modelnet40_path)
        except:
            print("Failed!")
            continue
        xyzs = np.array([position[0] for position in positions.values()])
        if len(xyzs) == 0:
            continue
        xmin, xmax, ymin, ymax, zmin, zmax = xyzs[:, 0].min(), xyzs[:, 0].max(), xyzs[:, 1].min(), xyzs[:, 1].max(), xyzs[:, 2].min(), xyzs[:, 2].max()
        for aug_idx in range(1):
            add_list = []
            positions_aug = positions.copy()
            key_toadd = max(positions.keys()) + 1
            for scarce in ['bathtub', 'bench', 'desk', 'dresser']:
                if scarce == 'bathtub' and add_bathtub == False:
                    continue
                if scarce == 'bench' and add_bench == False:
                    continue
                if scarce == 'desk' and add_desk == False:
                    continue
                if scarce == 'dresser' and add_dresser == False:
                    continue
                this_class_info = np.load(os.path.join(modelnet40_path, scarce, "this_class_info.npy"), allow_pickle=True).item()
                if np.random.rand() > 0.5:
                    _, _, _, dxavg, dyavg, dzavg = obj_prop[MDN_dict[scarce]][0:6]
                else:
                    dxavg, dyavg, dzavg, _, _, _ = obj_prop[MDN_dict[scarce]][0:6]
                dxmin, dymin, dzmin = 0.8 * dxavg, 0.8 * dyavg, 0.8 * dzavg
                dxmax, dymax, dzmax = 1.3 * dxavg, 1.3 * dyavg, 1.3 * dzavg
                dx = dxmin + np.random.rand() * (dxmax - dxmin)
                dy = dymin + np.random.rand() * (dymax - dymin)
                dz = dzmin + np.random.rand() * (dzmax - dzmin)
                ls_ratio = max(dx, dy) / min(dx, dy)
                txt, obj_info = find_nearest_object(ls_ratio, this_class_info, scarce)
                obj_txt = os.path.join(modelnet40_path, scarce, txt)
                pc_txt = np.loadtxt(obj_txt, delimiter=',')
                pc_txt[:, [1, 2]] = pc_txt[:, [2, 1]]
                pc_txt[:, [4, 5]] = pc_txt[:, [4, 5]]
                ddx, ddy, ddz = max(pc_txt[:, 0]) - min(pc_txt[:, 0]), max(pc_txt[:, 1]) - min(pc_txt[:, 1]), max(pc_txt[:, 2]) - min(pc_txt[:, 2])
                scale = (dx * dx * dz / ddx / ddy / ddz)**(1 / 3)
                new_position = [(xmin+np.random.rand()*(xmax-xmin), ymin+np.random.rand()*(ymax-ymin), zmin+np.random.rand()*(zmax-zmin)), (scale, scale, scale), obj_txt, False, np.random.rand() * 360, None, None]
                positions_aug[key_toadd] = new_position
                key_toadd += 1
                add_list.append(scarce)
            if len(add_list) == 0:
                continue
            positions_aug, stage_map = generate_gravity_aware_positions(positions_aug, fw_points[0])
            try:
                positions_aug = generate_collision_aware_positions(positions_aug, stage_map, fw_points[0])
            except:
                try:
                    positions_aug = generate_collision_aware_positions(positions_aug, stage_map, fw_points[0])
                except:
                    continue
            positions_aug['floor_points'] = fw_points[0]
            print(add_list)
            aug_scan_name = scan_name[:5] + "_aug%d_" % aug_idx + scan_name[5:]
            np.save('/path/to/BackToReality/data_generation/Matterport3D/augment_random_positions_matterport/' + aug_scan_name + '.npy', positions_aug)
    
    
