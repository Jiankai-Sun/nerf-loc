# Copyright (c) Facebook, Inc. and its affiliates.


""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os, pickle
import sys, cv2
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points
import copy
import utils.pc_util as pc_util
from utils.random_cuboid import RandomCuboid
from utils.pc_util import shift_scale_points, scale_points
from utils.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
)


MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
DATA_PATH_V1 = "" ## Replace with path to dataset
DATA_PATH_V2 = "" ## Not used in the codebase.

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

class NerfDatasetConfig(object):
    def __init__(self, args=None):
        self.num_angle_bin = 12
        self.max_num_obj = 64
        # self.type2class = {
        #     "bike": 0,
        #     "book": 1,
        #     "bottle": 2,
        #     "camera": 3,
        #     "cereal_box": 4,
        #     "chair": 5,
        #     "cup": 6,
        #     "laptop": 7,
        #     "shoe": 8,
        #     "9": 9,
        # }
        self.type2class = {
            "chair": 0,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        # self.type2onehotclass = {
        #     "bike": 0,
        #     "book": 1,
        #     "bottle": 2,
        #     "camera": 3,
        #     "cereal_box": 4,
        #     "chair": 5,
        #     "cup": 6,
        #     "laptop": 7,
        #     "shoe": 8,
        #     "9": 9,
        # }
        self.type2onehotclass = {
            "chair": 0,
        }
        self.num_semcls = len(self.type2class)
        # print(self.num_semcls)
        # raise NotImplementedError
        self.input_type = args.input_type  # rgb or nerf
        self.version = args.version
        self.model_type = args.model_type
        self.pseudo_batch_size = 8
        self.num_scan_samples = 50
        self.sampling_space = 15  # 30  # 4
        self.sampling_shift = 1  # 30  # 4

    def angle2class(self, angle):
        """Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        returns class [0,1,...,N-1] and a residual number such that
            class*(2pi/N) + number = angle
        """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def my_compute_box_3d(self, center, size, heading_angle):
        R = pc_util.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)


class NerfDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        num_points=20000,
        use_color=False,
        use_height=False,
        use_v1=True,
        augment=False,
        use_random_cuboid=True,
        meta_data_dir=None,
        random_cuboid_min_points=30000,
    ):
        assert num_points <= 50000
        assert split_set in ["train", "val", "trainval"]
        self.dataset_config = dataset_config
        self.use_v1 = use_v1
        self.pose_factor = 8
        self.sh = [240, 180, 3]
        self.recenter = True
        self.bd_factor = .75
        self.pseudo_batch_size = dataset_config.pseudo_batch_size  # if >1, pseudo batch, if =1, normal batch
        self.sampling_space = dataset_config.sampling_space
        self.sampling_shift = dataset_config.sampling_shift

        # if root_dir is None:
        #     root_dir = DATA_PATH_V1 if use_v1 else DATA_PATH_V2

        self.data_path = os.path.join(root_dir, "%s" % (split_set))
        self.split_set = split_set
        if split_set in ['train', 'val', 'test']:
            scan_name_path = 'scan_name.pkl'
            if os.path.exists(scan_name_path):
                with open(scan_name_path, 'rb') as f:
                    self.scan_names = pickle.load(f)
                    print('Loading from {}'.format(scan_name_path))
            else:
                scan_names = []
                for class_name in sorted(os.listdir(self.data_path)):
                    class_path = class_name
                    print('class_name: ', class_name)
                    if class_name not in ['chair']: #, 'cup']:
                        continue
                    for batch_id in sorted(os.listdir(os.path.join(self.data_path, class_path))):
                        batch_path = os.path.join(class_path, batch_id)
                        for video_id in sorted(os.listdir(os.path.join(self.data_path, batch_path))):
                            img_path = os.path.join(batch_path, video_id, 'images')
                            for bbox_counter, bbox_id in enumerate(sorted([x for x in os.listdir(os.path.join(self.data_path, img_path)) if '_bbox' in x])):
                                # print('bbox_id: ', bbox_id)  # 000051_bbox.pkl
                                if not os.path.exists(os.path.join(self.data_path, img_path, '{0:06d}_bbox.pkl'.format(int(bbox_id.split('_')[0])+self.pseudo_batch_size * self.sampling_space))):
                                    continue
                                if split_set == 'train' and bbox_counter % (self.pseudo_batch_size * self.sampling_space) == 0:
                                    nerf_model_dir = os.path.join('../nerf-pytorch/logs' + self.data_path, batch_path, video_id, 'objectron_small')
                                    if os.path.exists(nerf_model_dir):  # .pkl
                                        ckpts = [os.path.join(nerf_model_dir, f) for f in
                                                 sorted(os.listdir(os.path.join(nerf_model_dir))) if
                                                 'tar' in f]
                                        if len(ckpts) > 0:
                                            bbox_path = os.path.join(img_path, bbox_id)
                                            scan_names.append(bbox_path)
                                        # else:
                                        #     print('{} is empty'.format(nerf_model_dir))
                                elif split_set in ['val', 'test'] and bbox_counter % (self.pseudo_batch_size * self.sampling_space) == 0:
                                    # print(os.path.join('../nerf-pytorch/logs' + self.data_path, batch_path, video_id))
                                    # print('Modified: ', os.path.join('../nerf-pytorch/logs' + self.data_path.replace('val', 'train'), batch_path, video_id))
                                    nerf_model_dir = os.path.join('../nerf-pytorch/logs' + self.data_path.replace('val', 'train'), batch_path,
                                                                  video_id, 'objectron_small')
                                    if os.path.exists(nerf_model_dir):  # .pkl
                                        ckpts = [os.path.join(nerf_model_dir, f) for f in
                                                 sorted(os.listdir(os.path.join(nerf_model_dir))) if
                                                 'tar' in f]
                                        if len(ckpts) > 0:
                                            bbox_path = os.path.join(img_path, bbox_id)
                                            scan_names.append(bbox_path)
                                        # else:
                                        #     print('{} is empty'.format(nerf_model_dir))
                                # bbox_counter = bbox_counter + 1
                # self.scan_names = scan_names[:self.dataset_config.num_scan_samples]
                print('len(scan_names): ', len(scan_names))
                if self.split_set in ['train', 'val', 'test']:
                    if int(len(scan_names) / (self.dataset_config.num_scan_samples * 0.5)) < 1:
                        self.scan_names = scan_names
                    else:
                        self.scan_names = scan_names[::int(len(scan_names) / (self.dataset_config.num_scan_samples * 0.5))]
                else:
                    if int(len(scan_names) / (self.dataset_config.num_scan_samples * 0.2)) < 1:
                        self.scan_names = scan_names
                    else:
                        self.scan_names = scan_names[::int(len(scan_names) / (self.dataset_config.num_scan_samples * 0.2))]
                # self.scan_names = scan_names[0:1]
                # if split_set == 'val':
                #     print('val: self.scan_names: ', self.scan_names)
                with open(scan_name_path, 'wb') as f:
                    pickle.dump(self.scan_names, f)
                    print('Save to {}'.format(scan_name_path))

        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points,
            aspect=0.75,
            min_crop=0.75,
            max_crop=1.0,
        )
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 1

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        if scan_name.startswith("/"):
            scan_path = scan_name
        else:
            if self.split_set == 'train':
                scan_path = os.path.join(self.data_path, scan_name)
            else:
                scan_path = os.path.join(self.data_path.replace('val', 'train'), scan_name)
        # point_cloud = np.load(scan_path + "_pc.npz")["pc"]  # Nx6
        # print(scan_path)
        # if self.dataset_config.input_type == 'nerf':
        #     point_cloud = np.load(scan_path + ".npy")  # Nx6  # (240, 180, 128, 4)
        # elif self.dataset_config.input_type == 'rgb':
        #     point_cloud = cv2.imread(scan_path + ".png")  # Nx6  # (240, 180, 3)
        img_path, bbox_id_name = os.path.split(scan_path)
        video_path = '/'.join(img_path.split('/')[:-1])
        bbox_id = bbox_id_name.split('_')[0]
        corner_bboxes_path_list = []
        if self.pseudo_batch_size == 1:
            with open(scan_path, 'rb') as handle:  # (8, 3)
                data = pickle.load(handle)
            corner_bboxes = data['corner_bbox']
            counter = 1
        else:
            corner_bboxes_list = []
            counter = 0
            if self.split_set in ['train']: #, 'val']:
                for i in range(int(bbox_id), int(bbox_id) + self.pseudo_batch_size * self.sampling_space, self.sampling_space):
                    corner_bboxes_path = os.path.join(img_path, '{0:06d}_bbox.pkl'.format(i))
                    try:
                        with open(corner_bboxes_path, 'rb') as handle:  # (8, 3)
                            data = pickle.load(handle)
                        corner_bboxes = data['corner_bbox'][:, :8]  # (1, 26, 3)
                        # print('{} shape: {}'.format(corner_bboxes_path, corner_bboxes.shape))
                        # assert corner_bboxes.shape == [8, 3], '{} shape: {}'.format(corner_bboxes_path, corner_bboxes.shape)
                        corner_bboxes_list.append(copy.deepcopy(corner_bboxes))
                        corner_bboxes_path_list.append(corner_bboxes_path)
                        counter += 1
                    except:
                        print('{} does not exist'.format(corner_bboxes_path))
            else:
                for i in range(int(bbox_id) + self.sampling_space - self.sampling_shift, int(bbox_id) + self.sampling_space - self.sampling_shift + self.pseudo_batch_size * self.sampling_space, self.sampling_space):
                    corner_bboxes_path = os.path.join(img_path, '{0:06d}_bbox.pkl'.format(i))
                    try:
                        with open(corner_bboxes_path, 'rb') as handle:  # (8, 3)
                            data = pickle.load(handle)
                        corner_bboxes = data['corner_bbox'][:, :8]  # (1, 26, 3)
                        # print('{} shape: {}'.format(corner_bboxes_path, corner_bboxes.shape))
                        # assert corner_bboxes.shape == [8, 3], '{} shape: {}'.format(corner_bboxes_path, corner_bboxes.shape)
                        corner_bboxes_list.append(copy.deepcopy(corner_bboxes))
                        corner_bboxes_path_list.append(corner_bboxes_path)
                        counter += 1
                    except:
                        print('{} does not exist'.format(corner_bboxes_path))
            scan_path = corner_bboxes_path
            corner_bboxes = np.asarray(corner_bboxes_list)
            # print('corner_bboxes: ', corner_bboxes.shape)
        semantic_class = np.asarray([0] * counter)  # data['category']
        try:
            bboxes = data['csa_bbox']
        except:
            bboxes = np.zeros((1 * counter, 8))  # data['csa_bbox']

        # print('video_path: {}, bbox_id: {}'.format(video_path, bbox_id))
        # _load_data
        poses_arr = np.load(os.path.join(video_path, "poses_bounds.npy"))  # Nx17  # (331, 17)
        # print('1 poses_arr.shape: ', poses_arr.shape)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        poses[:2, 4, :] = np.array(self.sh[:2]).reshape([2, 1])
        poses[2, 4, :] = poses[2, 4, :] * 1. / self.pose_factor

        # _load_llff_data
        poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)
        sc = 1. if self.bd_factor is None else 1. / (bds.min() * self.bd_factor)
        poses[:, :3, 3] *= sc
        # bds *= sc
        if self.recenter:
            poses = recenter_poses(poses)
        # print('2 poses.shape: ', poses.shape)
        if self.split_set in ['train']:  # , 'val']:
            poses = poses[int(bbox_id):(int(bbox_id) + counter * self.sampling_space):self.sampling_space].astype(np.float32)
        else:
            poses = poses[(int(bbox_id) + self.sampling_space - self.sampling_shift):(int(bbox_id) + self.sampling_space - self.sampling_shift + self.pseudo_batch_size * self.sampling_space):self.sampling_space].astype(
                np.float32)
        # print('3 poses.shape: ', poses.shape)
        point_cloud = poses
        nerf_model_dir = os.path.join('../nerf-pytorch/logs' + video_path, 'objectron_small')
        ckpts = [os.path.join(nerf_model_dir, f) for f in sorted(os.listdir(os.path.join(nerf_model_dir))) if
                 'tar' in f]
        # print(ckpts[-1])
        # point_cloud = point_cloud[:1, :1].reshape(-1, point_cloud.shape[-1])
        # bboxes = np.load(scan_path + "_bbox.npy")  # K,8  # (1, 8)
        # print('bboxes.shape: ', bboxes.shape)
        # ------------------------------- LABELS ------------------------------
        angle_classes = np.zeros((self.max_num_obj * counter,), dtype=np.float32)
        angle_residuals = np.zeros((self.max_num_obj * counter,), dtype=np.float32)
        raw_angles = np.zeros((self.max_num_obj * counter,), dtype=np.float32)
        # raw_sizes = np.zeros((self.max_num_obj, point_cloud.shape[0], point_cloud.shape[1], point_cloud.shape[2], 4), dtype=np.float32)
        raw_sizes = np.zeros((self.max_num_obj * counter, 3), dtype=np.float32)
        label_mask = np.zeros((self.max_num_obj * counter))
        label_mask[0 : poses.shape[0]] = 1
        # max_bboxes = np.zeros((self.max_num_obj, 8))
        # max_bboxes[0 : bboxes.shape[0], :] = bboxes

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((self.max_num_obj * counter, 6))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            raw_angles[i] = bbox[6] % 2 * np.pi
            box3d_size = bbox[3:6] * 2
            raw_sizes[i, :] = box3d_size
            angle_class, angle_residual = self.dataset_config.angle2class(bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            corners_3d = self.dataset_config.my_compute_box_3d(
                bbox[0:3], bbox[3:6], bbox[6]
            )
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ]
            )
            target_bboxes[i, :] = target_bbox

        # point_cloud, choices = pc_util.random_sampling(
        #     point_cloud, self.num_points, return_choices=True
        # )

        point_cloud_dims_min = point_cloud
        point_cloud_dims_max = point_cloud
        # print('point_cloud_dims_min.shape: {}, point_cloud.shape: {}'
        #       .format(point_cloud_dims_min.shape, point_cloud.shape))

        # mult_factor = point_cloud_dims_max - point_cloud_dims_min
        # box_sizes_normalized = scale_points(
        #     raw_sizes.astype(np.float32)[None, ...],
        #     mult_factor=1.0 / mult_factor[None, ...],
        # )
        # box_sizes_normalized = box_sizes_normalized.squeeze(0)
        box_sizes_normalized = bboxes[:, 3:6]

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        # box_centers_normalized = shift_scale_points(
        #     box_centers[None, ...],
        #     src_range=[
        #         point_cloud_dims_min[None, ...],
        #         point_cloud_dims_max[None, ...],
        #     ],
        #     dst_range=self.center_normalizing_range,
        # )
        # box_centers_normalized = box_centers_normalized.squeeze(0)
        # box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        box_centers_normalized = box_centers

        # re-encode angles to be consistent with VoteNet eval
        angle_classes = angle_classes.astype(np.int64)
        angle_residuals = angle_residuals.astype(np.float32)
        raw_angles = self.dataset_config.class2angle_batch(
            angle_classes, angle_residuals
        )

        # box_corners = self.dataset_config.box_parametrization_to_corners_np(
        #     box_centers[None, ...],
        #     raw_sizes.astype(np.float32)[None, ...],
        #     raw_angles.astype(np.float32)[None, ...],
        # )
        # box_corners = bboxes
        # box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = corner_bboxes.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        target_bboxes_semcls = np.zeros((self.max_num_obj * counter))
        if self.split_set in ['train']:  # , 'val']:
            target_bboxes_semcls[0: counter] = semantic_class  # bboxes[int(bbox_id):self.sampling_space:(int(bbox_id) + counter * self.sampling_space), -1]  # from 0 to 9
        else:
            target_bboxes_semcls[0: counter] = semantic_class  # bboxes[(int(bbox_id) + self.sampling_space - 1):self.sampling_space:(int(bbox_id) + self.sampling_space - 1 + self.pseudo_batch_size * self.sampling_space), -1]  # from 0 to 9

        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max
        ret_dict["scan_path"] = scan_path
        ret_dict["nerf_ckpt_path"] = ckpts[-1]
        # ret_dict["corner_bboxes_path"] = corner_bboxes_path_list
        # print(np.array(data['intrinsic_matrix']).shape) # (9,)
        ret_dict["intrinsic_matrix"] = np.array(data['intrinsic_matrix']).astype(np.float32)
        ret_dict["cam_to_world"] = np.array(data['cam_to_world']).astype(np.float32)
        # print('self.max_num_obj, target_bboxes_semcls.shape: ', self.max_num_obj, target_bboxes_semcls.shape)
        return ret_dict
