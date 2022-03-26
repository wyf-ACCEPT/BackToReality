import open3d as o3d
import numpy as np
from tqdm import tqdm
import cv2
import os, sys
from sklearn.cluster import KMeans


## note that in the scene, axis-z is the direction of gravity
## however in modelnet40, axis-y takes the place


# 10000 points per object
def txt_to_pcd(txt_path):
    pc_txt = np.loadtxt(txt_path, delimiter=',')
    xyz = pc_txt[:, 0:3]
    xyz[:, [1, 2]] = xyz[:, [2, 1]]
    nxnynz = pc_txt[:, 3:6]
    nxnynz[:, [1, 2]] = nxnynz[:, [2, 1]]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.normals = o3d.utility.Vector3dVector(nxnynz)
    return pcd


# MER: ((long, short), theta)
# theta: rotate axis-x anticlockwise until it is parallel with the long edge, in (0, 180]
def get_MER(pcd):
    points = np.array(pcd.points)
    xys = points[:, 0:2]
    xys *= 1000
    xys = xys.astype('int')
    rect = cv2.minAreaRect(xys)
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
    return (l_s, theta)


def get_support_z(pcd):
    points = np.array(pcd.points)
    normals = np.array(pcd.normals)
    z_list = []
    num = points.shape[0]
    for i in range(num):
        normal = normals[i]
        if abs(np.dot(normal, [0, 0, 1])) > 0.88:
            z_list.append(points[i][2])
    z_list.sort()
    z_num = len(z_list)
    return np.mean(z_list[z_num * 4 // 5:z_num * 9 // 10])


def get_horizontal_area(points):
    xys = points[:,:2]
    xys *= 1000
    xys = xys.astype('int')
    hull = cv2.convexHull(xys, clockwise=True, returnPoints=True)
    area = cv2.contourArea(hull)
    return area/1000000


def Is_support(pcd, support_z, MER):
    # If support surface's area is much smaller than the MER's area
    # this object is not supportable
    points = np.array(pcd.points)
    zs = points[:, 2]
    dz = zs.max() - zs.min()
    support_points = points[abs(points[:, 2] - support_z) < dz / 10]
    try:
        k_means = KMeans(n_clusters=2)
        k_means.fit(support_points)
        cluster_label = k_means.predict(support_points)
        points0 = support_points[cluster_label == 0]
        points1 = support_points[cluster_label == 1]
        MER_area = MER[0][0] * MER[0][1]
        surface_area = get_horizontal_area(points0) + get_horizontal_area(points1)
        if surface_area > MER_area * 0.9:
            return True
    except:
        pass
    return False



if __name__ == "__main__":
    modelnet40_path = "./modelnet40_normal_resampled"
    obj_prop = np.load('CONFIG/object40_property.npy', allow_pickle=True, encoding='bytes').item()
    overlap_id = list(obj_prop)
    overlap_id = [id - 1 for id in overlap_id]
    modelnet40_names = np.loadtxt('CONFIG/modelnet40_shape_names.txt', dtype='object')
    overlap_names = modelnet40_names[overlap_id]
    for name in tqdm(overlap_names):
        if name == "sink":
            continue
        now_dir = os.path.join(modelnet40_path, name)
        if os.path.exists(os.path.join(now_dir, 'this_class_info.npy')):
            continue
        txts = os.listdir(now_dir)
        now_dict = {}
        for txt in txts:
            txt_file = os.path.join(now_dir, txt)
            s_id = txt[-8:-4]
            pcd = txt_to_pcd(txt_file)
            support_z = get_support_z(pcd)
            MER = get_MER(pcd)
            now_dict[s_id] = [MER, support_z, Is_support(pcd, support_z, MER)]
        np.save(os.path.join(now_dir, 'this_class_info.npy'), now_dict)
        print(name+" saved!")



