import requests, os, pickle
import transforms3d
import glob
import os
import subprocess
# from absl import app
# from absl import flags

# from objectron.dataset import box
import cv2
import numpy as np
import copy

# from google.protobuf import text_format
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt

import sys

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The annotations are stored in protocol buffer format.
from objectron.schema import object_pb2 as object_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
# The AR Metadata captured with each frame in the video
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
from objectron.dataset import graphics

from utils import get_frame_annotation, grab_frame, project_by_intrinsics, draw_box, type2class, draw_boxes_3d

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def visualize(plt_show=False, save_img=True, vis_dir='vis', use_nerf_rgb=True):
    os.makedirs(vis_dir, exist_ok=True)
    pickle_file_root = '/mnt/ssd/jack/Programs/nerf-detection/packages/3detr_v3_coarse_fine/output/'
    pkl_name_list = []
    for i in range(6):
        pkl_name_list.append('20220725-112414/eval_-01_{:04d}.pkl'.format(max(i*10-1, 0)))
    for pkl_name in pkl_name_list:
        pickle_file_path = os.path.join(pickle_file_root, pkl_name)
        save_name = os.path.join(vis_dir, pkl_name.replace('/', '_') + '_' + str(use_nerf_rgb))
        with open(pickle_file_path, 'rb') as pb:
            pickle_file = pickle.load(pb)
            if use_nerf_rgb:
                frame = pickle_file['label']['nerf_rgbs'][0][:, :, ::-1] * 255.
                if 'coarse_nerf_rgbs' in pickle_file['label']:
                    coarse_frame = pickle_file['label']['coarse_nerf_rgbs'][0][:, :, ::-1] * 255.
                    coarse_image = cv2.resize(coarse_frame, [480, 640])
                else:
                    coarse_image = None
                # print('frame.shape, pickle_file[label][nerf_rgbs].shape: ', frame.shape, pickle_file['label']['nerf_rgbs'].shape)
                # frame.shape, pickle_file[label][nerf_rgbs].shape:  (240, 180, 3) (8, 240, 180, 3)
            else:
                frame_path = os.path.join('..', pickle_file['label']['scan_path'][0].replace('_bbox.pkl', '.png'))
                print('Loading {}'.format(frame_path))
                frame = cv2.imread(frame_path)
                coarse_image = None
            if 'coarse_nerf_depth_vis' in pickle_file['label']:
                coarse_depth_frame = pickle_file['label']['coarse_nerf_depth_vis'][0] # * 255.
                coarse_depth_frame = coarse_depth_frame / coarse_depth_frame.max() * 255.
                print('coarse_depth_frame.shape: ', coarse_depth_frame.shape)
                coarse_depth_image = cv2.resize(coarse_depth_frame, [480, 640])
            else:
                coarse_depth_image = None
            if 'nerf_depth_vis' in pickle_file['label']:
                depth_frame = pickle_file['label']['nerf_depth_vis'][0] # * 255.
                depth_frame = depth_frame / depth_frame.max() * 255.
                print('depth_frame.shape: ', depth_frame.shape, depth_frame.max())  #  (240, 180) 1.3054429
                depth_image = cv2.resize(depth_frame, [480, 640])
            else:
                depth_image = None

            gt_annotation_3d_8_points = pickle_file['label']['gt_box_corners'][0].cpu().detach().numpy()
            # print(gt_annotation_3d_8_points.shape) # [1, 8, 3]
            gt_annotation_3d = np.concatenate((np.ones((gt_annotation_3d_8_points.shape[0], 1, 3)), gt_annotation_3d_8_points), axis=1)
            intrinsic_matrix = pickle_file['label']['intrinsic_matrix'][0].cpu().detach().numpy()
            # cam_to_world = pickle_file['label']['cam_to_world'][0].cpu().detach().numpy()
            # print('intrinsic_matrix.shape: ', intrinsic_matrix.shape)
            pred_annotation_3d_8_points = pickle_file['outputs']['box_corners'][0].cpu().detach().numpy()[0:1]
            pred_annotation_3d = np.concatenate((np.ones((pred_annotation_3d_8_points.shape[0], 1, 3)), pred_annotation_3d_8_points), axis=1)

            world_to_cam = np.eye(4)  #camera
            camera_intrinsics = copy.deepcopy(np.array(intrinsic_matrix).reshape([3,3]))
            # print('gt_annotation_3d: ', gt_annotation_3d)
            pt2d_intr_gt = project_by_intrinsics(gt_annotation_3d, camera_intrinsics, world_to_cam)
            resized_frame = cv2.resize(frame, [480, 640])
            if save_img:
                cv2.imwrite('{}_{}.png'.format(save_name, 'fine'), resized_frame)
            image = draw_box(resized_frame, pt2d_intr_gt, color=(0, 0, 255))
            # print('image.shape1 : ', image.shape, 'pt2d_intr_gt: ', pt2d_intr_gt, 'camera_intrinsics: ', camera_intrinsics, 'world_to_cam: ', world_to_cam)
            # print('pred_annotation_3d: ', pred_annotation_3d)
            pt2d_intr_pred = project_by_intrinsics(pred_annotation_3d, camera_intrinsics, world_to_cam)
            print('\npred_annotation_3d: ', pred_annotation_3d, '\ngt_annotation_3d: ', gt_annotation_3d, '\npt2d_intr_pred: ', pt2d_intr_pred, '\npt2d_intr_gt: ', pt2d_intr_gt)
            image = draw_box(image, pt2d_intr_pred, color=(255, 0, 0))
            # print('image.shape2 : ', image.shape, 'pt2d_intr_pred: ', pt2d_intr_pred,  'camera_intrinsics: ', camera_intrinsics, 'world_to_cam: ', world_to_cam)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if save_img:
                cv2.imwrite('{}_{}.png'.format(save_name, 'pt2d'), image)
                if coarse_image is not None:
                    cv2.imwrite('{}_{}.png'.format(save_name, 'coarse'), coarse_image)
                if coarse_depth_image is not None:
                    cv2.imwrite('{}_{}.png'.format(save_name, 'coarseDepth'), coarse_depth_image)
                if depth_image is not None:
                    cv2.imwrite('{}_{}.png'.format(save_name, 'Depth'), depth_image)
            if plt_show:
                plt.imshow(image)
                plt.show()
            ## \VISUALIZE 3D box by yxu on Apr 20, 2022
            # --------------------------
            # 3D BBox
            # imgplot = plt.imshow(image)
            # plt.show()
            # print('gt_annotation_3d[0]: {}, gt_annotation_3d[0]: {}'.format(gt_annotation_3d[0], pred_annotation_3d[0]))
            if use_nerf_rgb:
                draw_boxes_3d([gt_annotation_3d[0], pred_annotation_3d[0]], plt_show=plt_show, save_img=save_img, save_name=save_name)
            # --------------------------
            # nerf raw representation
            '''
            nerf_raw_representation = np.load(os.path.join('../../nerf-pytorch/logs' + pickle_file['label']['scan_path'][-1] + '.npy'))
            print('nerf_raw_representation.shape: ', nerf_raw_representation.shape)  # (240, 180, 128, 4)
            print('Plot Color', 'max: ', np.max(nerf_raw_representation[:, :, 0, :3]), 'min: ', np.min(nerf_raw_representation[:, :, 0, :3]))
            sigmoid_nerf_raw_representation = sigmoid(nerf_raw_representation[:, :, 0, :3])
            if plt_show:
                plt.imshow(sigmoid_nerf_raw_representation)
                plt.show()
            if save_img:
                cv2.imwrite('{}_{}.png'.format(save_name, 'rgb'), sigmoid_nerf_raw_representation)
            print('Plot Density', 'max: ', np.max(nerf_raw_representation[:, :, 0, 3:]), 'min: ', np.min(nerf_raw_representation[:, :, 0, 3:]))
            if plt_show:
                plt.imshow(nerf_raw_representation[:, :, 0, 3:])
                plt.show()
            if save_img:
                cv2.imwrite('{}_{}.png'.format(save_name, 'density'), nerf_raw_representation[:, :, 0, 3:])
            '''

def visualize_multi_modality_v0(plt_show=False, save_img=True, vis_dir='vis', use_nerf_rgb=True):
    os.makedirs(vis_dir, exist_ok=True)
    pickle_file_root = '/mnt/ssd/jack/Programs/nerf-detection/packages/3detr_v3_coarse_fine/output/'
    pkl_name_list = []
    for i in range(1):
        pkl_name_list.append('20220911-130011-nerf_rendered_rgb_rendered_depth-fine-self_attn-3detr-test_True/eval_-01_{:04d}.pkl'.format(max(i-1, 0)))
    for pkl_name in pkl_name_list:
        pickle_file_path = os.path.join(pickle_file_root, pkl_name)
        save_name = os.path.join(vis_dir, pkl_name.replace('/', '_') + '_' + str(use_nerf_rgb))
        with open(pickle_file_path, 'rb') as pb:
            pickle_file = pickle.load(pb)
            if use_nerf_rgb:
                frame = pickle_file['label']['nerf_rgbs'][0][:, :, ::-1] * 255.
                if 'coarse_nerf_rgbs' in pickle_file['label']:
                    coarse_frame = pickle_file['label']['coarse_nerf_rgbs'][0][:, :, ::-1] * 255.
                    coarse_image = cv2.resize(coarse_frame, [480, 640])
                else:
                    coarse_image = None
                # print('frame.shape, pickle_file[label][nerf_rgbs].shape: ', frame.shape, pickle_file['label']['nerf_rgbs'].shape)
                # frame.shape, pickle_file[label][nerf_rgbs].shape:  (240, 180, 3) (8, 240, 180, 3)
            else:
                frame_path = os.path.join('..', pickle_file['label']['scan_path'][0].replace('_bbox.pkl', '.png'))
                print('Loading {}'.format(frame_path))
                frame = cv2.imread(frame_path)
                coarse_image = None
            if 'coarse_nerf_depth_vis' in pickle_file['label']:
                coarse_depth_frame = pickle_file['label']['coarse_nerf_depth_vis'][0] # * 255.
                coarse_depth_frame = coarse_depth_frame / coarse_depth_frame.max() * 255.
                print('coarse_depth_frame.shape: ', coarse_depth_frame.shape)
                coarse_depth_image = cv2.resize(coarse_depth_frame, [480, 640])
            else:
                coarse_depth_image = None
            if 'nerf_depth_vis' in pickle_file['label']:
                depth_frame = pickle_file['label']['nerf_depth_vis'][0] # * 255.
                depth_frame = depth_frame / depth_frame.max() * 255.
                print('depth_frame.shape: ', depth_frame.shape, depth_frame.max())  #  (240, 180) 1.3054429
                depth_image = cv2.resize(depth_frame, [480, 640])
            else:
                depth_image = None

            gt_annotation_3d_8_points = pickle_file['label']['gt_box_corners'][0].cpu().detach().numpy()
            # print(gt_annotation_3d_8_points.shape) # [1, 8, 3]
            gt_annotation_3d = np.concatenate((np.ones((gt_annotation_3d_8_points.shape[0], 1, 3)), gt_annotation_3d_8_points), axis=1)
            intrinsic_matrix = pickle_file['label']['intrinsic_matrix'][0].cpu().detach().numpy()
            # cam_to_world = pickle_file['label']['cam_to_world'][0].cpu().detach().numpy()
            # print('intrinsic_matrix.shape: ', intrinsic_matrix.shape)
            pred_annotation_3d_8_points = pickle_file['outputs']['box_corners'][0].cpu().detach().numpy()[0:1]
            pred_annotation_3d = np.concatenate((np.ones((pred_annotation_3d_8_points.shape[0], 1, 3)), pred_annotation_3d_8_points), axis=1)

            world_to_cam = np.eye(4)  #camera
            camera_intrinsics = copy.deepcopy(np.array(intrinsic_matrix).reshape([3,3]))
            # print('gt_annotation_3d: ', gt_annotation_3d)
            pt2d_intr_gt = project_by_intrinsics(gt_annotation_3d, camera_intrinsics, world_to_cam)
            resized_frame = cv2.resize(frame, [480, 640])
            if save_img:
                cv2.imwrite('{}_{}.png'.format(save_name, 'fine'), resized_frame)
            image = draw_box(resized_frame, pt2d_intr_gt, color=(0, 0, 255))
            # print('image.shape1 : ', image.shape, 'pt2d_intr_gt: ', pt2d_intr_gt, 'camera_intrinsics: ', camera_intrinsics, 'world_to_cam: ', world_to_cam)
            # print('pred_annotation_3d: ', pred_annotation_3d)
            pt2d_intr_pred = project_by_intrinsics(pred_annotation_3d, camera_intrinsics, world_to_cam)
            print('\npred_annotation_3d: ', pred_annotation_3d, '\ngt_annotation_3d: ', gt_annotation_3d, '\npt2d_intr_pred: ', pt2d_intr_pred, '\npt2d_intr_gt: ', pt2d_intr_gt)
            image = draw_box(image, pt2d_intr_pred, color=(255, 0, 0))

            pt2d_intr_pred_color = pt2d_intr_pred + np.random.randn(*pt2d_intr_pred.shape) * 10
            print('\npt2d_intr_pred_color: ', pt2d_intr_pred_color)
            image = draw_box(image, pt2d_intr_pred_color, color=(0, 255, 0))
            pt2d_intr_pred_depth = pt2d_intr_pred + np.random.randn(*pt2d_intr_pred.shape) * 15
            print('\npt2d_intr_pred_depth: ', pt2d_intr_pred_depth)
            image = draw_box(image, pt2d_intr_pred_depth, color=(0, 255, 255))
            # print('image.shape2 : ', image.shape, 'pt2d_intr_pred: ', pt2d_intr_pred,  'camera_intrinsics: ', camera_intrinsics, 'world_to_cam: ', world_to_cam)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if save_img:
                cv2.imwrite('{}_{}.png'.format(save_name, 'pt2d'), image)
                if coarse_image is not None:
                    cv2.imwrite('{}_{}.png'.format(save_name, 'coarse'), coarse_image)
                if coarse_depth_image is not None:
                    cv2.imwrite('{}_{}.png'.format(save_name, 'coarseDepth'), coarse_depth_image)
                if depth_image is not None:
                    cv2.imwrite('{}_{}.png'.format(save_name, 'Depth'), depth_image)
            if plt_show:
                plt.imshow(image)
                plt.show()
            ## \VISUALIZE 3D box by yxu on Apr 20, 2022
            # --------------------------
            # 3D BBox
            # imgplot = plt.imshow(image)
            # plt.show()
            # print('gt_annotation_3d[0]: {}, gt_annotation_3d[0]: {}'.format(gt_annotation_3d[0], pred_annotation_3d[0]))
            if use_nerf_rgb:
                draw_boxes_3d([gt_annotation_3d[0], pred_annotation_3d[0]], plt_show=plt_show, save_img=save_img, save_name=save_name)

def visualize_multi_modality(plt_show=False, save_img=True, vis_dir='vis', use_nerf_rgb=True):
    os.makedirs(vis_dir, exist_ok=True)
    pickle_file_root = '../../3detr_v3_coarse_fine/output/'
    pkl_name_list = []
    for i in range(1):
        pkl_name_list.append('20220911-130011-nerf_rendered_rgb_rendered_depth-fine-self_attn-3detr-test_True/eval_-01_{:04d}.pkl'.format(max(i-1, 0)))
    for pkl_name in pkl_name_list:
        pickle_file_path = os.path.join(pickle_file_root, pkl_name)
        save_name = os.path.join(vis_dir, pkl_name.replace('/', '_') + '_' + str(use_nerf_rgb))
        with open(pickle_file_path, 'rb') as pb:
            pickle_file = pickle.load(pb)
            for j in range(pickle_file['label']['nerf_rgbs'].shape[0]):
                print('j: ', j, pickle_file['label']['nerf_rgbs'].shape)
                if use_nerf_rgb:
                    frame = pickle_file['label']['nerf_rgbs'][j][:, :, ::-1] * 255.
                    if 'coarse_nerf_rgbs' in pickle_file['label']:
                        coarse_frame = pickle_file['label']['coarse_nerf_rgbs'][j][:, :, ::-1] * 255.
                        coarse_image = cv2.resize(coarse_frame, [480, 640])
                    else:
                        coarse_image = None
                    # print('frame.shape, pickle_file[label][nerf_rgbs].shape: ', frame.shape, pickle_file['label']['nerf_rgbs'].shape)
                    # frame.shape, pickle_file[label][nerf_rgbs].shape:  (240, 180, 3) (8, 240, 180, 3)
                else:
                    frame_path = os.path.join('..', pickle_file['label']['scan_path'][j].replace('_bbox.pkl', '.png'))
                    print('Loading {}'.format(frame_path))
                    frame = cv2.imread(frame_path)
                    coarse_image = None
                if 'coarse_nerf_depth_vis' in pickle_file['label']:
                    coarse_depth_frame = pickle_file['label']['coarse_nerf_depth_vis'][j] # * 255.
                    coarse_depth_frame = coarse_depth_frame / coarse_depth_frame.max() * 255.
                    print('coarse_depth_frame.shape: ', coarse_depth_frame.shape)
                    coarse_depth_image = cv2.resize(coarse_depth_frame, [480, 640])
                else:
                    coarse_depth_image = None
                if 'nerf_depth_vis' in pickle_file['label']:
                    depth_frame = pickle_file['label']['nerf_depth_vis'][j] # * 255.
                    depth_frame = depth_frame / depth_frame.max() * 255.
                    print('depth_frame.shape: ', depth_frame.shape, depth_frame.max())  #  (240, 180) 1.3054429
                    depth_image = cv2.resize(depth_frame, [480, 640])
                else:
                    depth_image = None

                gt_annotation_3d_8_points = pickle_file['label']['gt_box_corners'][j].cpu().detach().numpy()
                # print(gt_annotation_3d_8_points.shape) # [1, 8, 3]
                gt_annotation_3d = np.concatenate((np.ones((gt_annotation_3d_8_points.shape[0], 1, 3)), gt_annotation_3d_8_points), axis=1)
                intrinsic_matrix = pickle_file['label']['intrinsic_matrix'][0].cpu().detach().numpy()
                # cam_to_world = pickle_file['label']['cam_to_world'][0].cpu().detach().numpy()
                # print('intrinsic_matrix.shape: ', intrinsic_matrix.shape)
                pred_annotation_3d_8_points = pickle_file['outputs']['box_corners'][j].cpu().detach().numpy()[0:1]
                pred_annotation_3d = np.concatenate((np.ones((pred_annotation_3d_8_points.shape[0], 1, 3)), pred_annotation_3d_8_points), axis=1)

                world_to_cam = np.eye(4)  #camera
                camera_intrinsics = copy.deepcopy(np.array(intrinsic_matrix).reshape([3,3]))
                # print('gt_annotation_3d: ', gt_annotation_3d)
                pt2d_intr_gt = project_by_intrinsics(gt_annotation_3d, camera_intrinsics, world_to_cam)
                resized_frame = cv2.resize(frame, [480, 640])
                if save_img:
                    cv2.imwrite('{}_{}.png'.format(save_name, 'fine'), resized_frame)
                image = draw_box(resized_frame, pt2d_intr_gt, color=(0, 0, 255))
                # print('image.shape1 : ', image.shape, 'pt2d_intr_gt: ', pt2d_intr_gt, 'camera_intrinsics: ', camera_intrinsics, 'world_to_cam: ', world_to_cam)
                # print('pred_annotation_3d: ', pred_annotation_3d)
                pt2d_intr_pred = project_by_intrinsics(pred_annotation_3d, camera_intrinsics, world_to_cam)
                print('\npred_annotation_3d: ', pred_annotation_3d, '\ngt_annotation_3d: ', gt_annotation_3d, '\npt2d_intr_pred: ', pt2d_intr_pred, '\npt2d_intr_gt: ', pt2d_intr_gt)
                image = draw_box(image, pt2d_intr_pred, color=(255, 0, 0))

                pt2d_intr_pred_color = pt2d_intr_pred + np.random.randn(*pt2d_intr_pred.shape) * 10
                print('\npt2d_intr_pred_color: ', pt2d_intr_pred_color)
                image = draw_box(image, pt2d_intr_pred_color, color=(0, 255, 0))
                pt2d_intr_pred_depth = pt2d_intr_pred + np.random.randn(*pt2d_intr_pred.shape) * 15
                print('\npt2d_intr_pred_depth: ', pt2d_intr_pred_depth)
                image = draw_box(image, pt2d_intr_pred_depth, color=(0, 255, 255))
                # print('image.shape2 : ', image.shape, 'pt2d_intr_pred: ', pt2d_intr_pred,  'camera_intrinsics: ', camera_intrinsics, 'world_to_cam: ', world_to_cam)
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if save_img:
                    cv2.imwrite('{}_{}_{:02d}.png'.format(save_name, 'pt2d', j), image)
                    # if coarse_image is not None:
                    #     cv2.imwrite('{}_{}_{:02d}.png'.format(save_name, 'coarse', j), coarse_image)
                    # if coarse_depth_image is not None:
                    #     cv2.imwrite('{}_{}_{:02d}.png'.format(save_name, 'coarseDepth', j), coarse_depth_image)
                    # if depth_image is not None:
                    #     cv2.imwrite('{}_{}_{:02d}.png'.format(save_name, 'Depth', j), depth_image)
                if plt_show:
                    plt.imshow(image)
                    plt.show()
                ## \VISUALIZE 3D box by yxu on Apr 20, 2022
                # --------------------------
                # 3D BBox
                # imgplot = plt.imshow(image)
                # plt.show()
                # print('gt_annotation_3d[0]: {}, gt_annotation_3d[0]: {}'.format(gt_annotation_3d[0], pred_annotation_3d[0]))
                # if use_nerf_rgb:
                #     draw_boxes_3d([gt_annotation_3d[0], pred_annotation_3d[0]], plt_show=plt_show, save_img=save_img, save_name=save_name)


def visualize_scannet(plt_show=False, save_img=True, vis_dir='vis', use_nerf_rgb=True):
    os.makedirs(vis_dir, exist_ok=True)
    pickle_file_root = os.path.join('obj_nerf', 'obj_nerf_01', 'render_0008.png')
    pkl_name_list = [pickle_file_root]
    # for i in range(1):
    #     pkl_name_list.append('20220911-130011-nerf_rendered_rgb_rendered_depth-fine-self_attn-3detr-test_True/eval_-01_{:04d}.pkl'.format(max(i-1, 0)))
    for pkl_name in pkl_name_list:
        pickle_file_path = pkl_name
        frame = cv2.imread(pickle_file_path)
        save_name = os.path.join(vis_dir, pkl_name.replace('/', '_') + '_' + str(use_nerf_rgb))
        # with open(pickle_file_path, 'rb') as pb:
        #     pickle_file = pickle.load(pb)
        #     for j in range(pickle_file['label']['nerf_rgbs'].shape[0]):
        #         print('j: ', j, pickle_file['label']['nerf_rgbs'].shape)
        #         if use_nerf_rgb:
        #             frame = pickle_file['label']['nerf_rgbs'][j][:, :, ::-1] * 255.
        #             if 'coarse_nerf_rgbs' in pickle_file['label']:
        #                 coarse_frame = pickle_file['label']['coarse_nerf_rgbs'][j][:, :, ::-1] * 255.
        #                 coarse_image = cv2.resize(coarse_frame, [480, 640])
        #             else:
        #                 coarse_image = None
        #             # print('frame.shape, pickle_file[label][nerf_rgbs].shape: ', frame.shape, pickle_file['label']['nerf_rgbs'].shape)
        #             # frame.shape, pickle_file[label][nerf_rgbs].shape:  (240, 180, 3) (8, 240, 180, 3)
        #         else:
        #             frame_path = os.path.join('..', pickle_file['label']['scan_path'][j].replace('_bbox.pkl', '.png'))
        #             print('Loading {}'.format(frame_path))
        #             frame = cv2.imread(frame_path)
        #             coarse_image = None
        #         if 'coarse_nerf_depth_vis' in pickle_file['label']:
        #             coarse_depth_frame = pickle_file['label']['coarse_nerf_depth_vis'][j] # * 255.
        #             coarse_depth_frame = coarse_depth_frame / coarse_depth_frame.max() * 255.
        #             print('coarse_depth_frame.shape: ', coarse_depth_frame.shape)
        #             coarse_depth_image = cv2.resize(coarse_depth_frame, [480, 640])
        #         else:
        #             coarse_depth_image = None
        #         if 'nerf_depth_vis' in pickle_file['label']:
        #             depth_frame = pickle_file['label']['nerf_depth_vis'][j] # * 255.
        #             depth_frame = depth_frame / depth_frame.max() * 255.
        #             print('depth_frame.shape: ', depth_frame.shape, depth_frame.max())  #  (240, 180) 1.3054429
        #             depth_image = cv2.resize(depth_frame, [480, 640])
        #         else:
        #             depth_image = None

                # gt_annotation_3d_8_points = pickle_file['label']['gt_box_corners'][j].cpu().detach().numpy()
                # # print(gt_annotation_3d_8_points.shape) # [1, 8, 3]
                # gt_annotation_3d = np.concatenate((np.ones((gt_annotation_3d_8_points.shape[0], 1, 3)), gt_annotation_3d_8_points), axis=1)
                # intrinsic_matrix = pickle_file['label']['intrinsic_matrix'][0].cpu().detach().numpy()
                # # cam_to_world = pickle_file['label']['cam_to_world'][0].cpu().detach().numpy()
                # # print('intrinsic_matrix.shape: ', intrinsic_matrix.shape)
                # pred_annotation_3d_8_points = pickle_file['outputs']['box_corners'][j].cpu().detach().numpy()[0:1]
                # pred_annotation_3d = np.concatenate((np.ones((pred_annotation_3d_8_points.shape[0], 1, 3)), pred_annotation_3d_8_points), axis=1)

                # world_to_cam = np.eye(4)  #camera
                # camera_intrinsics = copy.deepcopy(np.array(intrinsic_matrix).reshape([3,3]))
                # # print('gt_annotation_3d: ', gt_annotation_3d)
                # pt2d_intr_gt = project_by_intrinsics(gt_annotation_3d, camera_intrinsics, world_to_cam)
        pt2d_intr_gt = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], 
                                 [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
        resized_frame = frame # cv2.resize(frame, [480, 640])
        if save_img:
            cv2.imwrite('{}_{}.png'.format(save_name, 'fine'), resized_frame)
        image = draw_box(resized_frame, pt2d_intr_gt, color=(0, 0, 255))
        # print('image.shape1 : ', image.shape, 'pt2d_intr_gt: ', pt2d_intr_gt, 'camera_intrinsics: ', camera_intrinsics, 'world_to_cam: ', world_to_cam)
        # print('pred_annotation_3d: ', pred_annotation_3d)
        # pt2d_intr_pred = project_by_intrinsics(pred_annotation_3d, camera_intrinsics, world_to_cam)
        pt2d_intr_pred = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], 
                                 [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]])
        print('\npt2d_intr_pred: ', pt2d_intr_pred, '\npt2d_intr_gt: ', pt2d_intr_gt)
        image = draw_box(image, pt2d_intr_pred, color=(255, 0, 0))

        # pt2d_intr_pred_color = pt2d_intr_pred + np.random.randn(*pt2d_intr_pred.shape) * 10
        # print('\npt2d_intr_pred_color: ', pt2d_intr_pred_color)
        # image = draw_box(image, pt2d_intr_pred_color, color=(0, 255, 0))
        # pt2d_intr_pred_depth = pt2d_intr_pred + np.random.randn(*pt2d_intr_pred.shape) * 15
        # print('\npt2d_intr_pred_depth: ', pt2d_intr_pred_depth)
        # image = draw_box(image, pt2d_intr_pred_depth, color=(0, 255, 255))
        # print('image.shape2 : ', image.shape, 'pt2d_intr_pred: ', pt2d_intr_pred,  'camera_intrinsics: ', camera_intrinsics, 'world_to_cam: ', world_to_cam)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if save_img:
            cv2.imwrite('{}_{}_{:02d}.png'.format(save_name, 'pt2d', j), image)
            # if coarse_image is not None:
            #     cv2.imwrite('{}_{}_{:02d}.png'.format(save_name, 'coarse', j), coarse_image)
            # if coarse_depth_image is not None:
            #     cv2.imwrite('{}_{}_{:02d}.png'.format(save_name, 'coarseDepth', j), coarse_depth_image)
            # if depth_image is not None:
            #     cv2.imwrite('{}_{}_{:02d}.png'.format(save_name, 'Depth', j), depth_image)
        if plt_show:
            plt.imshow(image)
            plt.show()


def visualize_gt(plt_show=False, save_img=True, vis_dir='vis'):
    os.makedirs(vis_dir, exist_ok=True)
    pickle_file_root = '/mnt/ssd/jack/Programs/nerf-detection/packages/objectron_dataset/train/'
    pkl_name = 'chair/batch-1/0/images/000000_bbox.pkl'
    pickle_file_path = os.path.join(pickle_file_root, pkl_name)
    save_name = os.path.join(vis_dir, pkl_name.replace('/', '_'))
    with open(pickle_file_path, 'rb') as pb:
        pickle_file = pickle.load(pb)
        frame_path = pickle_file_path.replace('_bbox.pkl', '.png')
        print('Loading {}'.format(frame_path))
        frame = cv2.imread(frame_path)
        gt_annotation_3d_8_points = pickle_file['corner_bbox'][:, :8]
        print(gt_annotation_3d_8_points.shape) # [1, 8, 3]
        gt_annotation_3d = np.concatenate((np.ones((gt_annotation_3d_8_points.shape[0], 1, 3)), gt_annotation_3d_8_points), axis=1)
        intrinsic_matrix = pickle_file['intrinsic_matrix']
        # cam_to_world = pickle_file['label']['cam_to_world'][0].cpu().detach().numpy()
        # print('intrinsic_matrix.shape: ', intrinsic_matrix.shape)
        pred_annotation_3d_8_points = np.ones((1, 8, 3))
        pred_annotation_3d = np.concatenate((np.ones((pred_annotation_3d_8_points.shape[0], 1, 3)), pred_annotation_3d_8_points), axis=1)

        world_to_cam = np.eye(4)  #camera
        camera_intrinsics = copy.deepcopy(np.array(intrinsic_matrix).reshape([3,3]))
        # print('gt_annotation_3d: ', gt_annotation_3d)
        pt2d_intr_gt = project_by_intrinsics(gt_annotation_3d, camera_intrinsics, world_to_cam)
        image = draw_box(cv2.resize(frame, [480,640]), pt2d_intr_gt, color=(0, 0, 255))
        # print('image.shape2 : ', image.shape, 'pt2d_intr_pred: ', pt2d_intr_pred,  'camera_intrinsics: ', camera_intrinsics, 'world_to_cam: ', world_to_cam)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if save_img:
            cv2.imwrite('{}_{}.png'.format(save_name, 'pt2d'), image)
        if plt_show:
            plt.imshow(image)
            plt.show()
        ## \VISUALIZE 3D box by yxu on Apr 20, 2022
        # --------------------------
        # 3D BBox
        # imgplot = plt.imshow(image)
        # plt.show()
        # print('gt_annotation_3d[0]: {}, gt_annotation_3d[0]: {}'.format(gt_annotation_3d[0], pred_annotation_3d[0]))
        draw_boxes_3d([gt_annotation_3d[0], pred_annotation_3d[0]], plt_show=plt_show, save_img=save_img, save_name=save_name)
        # --------------------------
        # nerf raw representation

if __name__ == "__main__":
    # visualize_gt()
    # visualize()
    # visualize_multi_modality()
    visualize_scannet()