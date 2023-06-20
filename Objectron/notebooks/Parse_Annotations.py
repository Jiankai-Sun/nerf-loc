import requests, os, pickle
import transforms3d

import glob
import os
import subprocess
from absl import app
from absl import flags

# import box as Box
import cv2
import numpy as np
from tqdm import tqdm

from google.protobuf import text_format
from IPython.core.display import display, HTML
import matplotlib.pyplot as plt

import sys
import struct

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The annotations are stored in protocol buffer format.
from objectron.schema import object_pb2 as object_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
# The AR Metadata captured with each frame in the video
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
from objectron.dataset import box as Box
from objectron.dataset import graphics
from utils import load_frame_data, grab_frame, get_frame_annotation, transform_box_format, draw_box, make_poses_bounds_array, type2class

batch_num = 24
video_num = 33
# type2class = {
#             "chair": 0,
#         }

def prepare_dataset(save_dir = '../../objectron_dataset', label_dir='../index/', download_dir = './videos',
                annotation_dir = './annotations', public_url = "https://storage.googleapis.com/objectron",
                image_dir = 'images', resize_factor=8):
    os.makedirs(save_dir, exist_ok=True)
    splits = ['train', 'test']
    origin_label_files = sorted([i for i in os.listdir(label_dir) if ('annotations' in i and 'test' not in i and 'train' not in i)])
    label_files = []
    for i in origin_label_files:
        if ("chair" in i): # or ("cup" in i) or ('bike' in i) or ('camera' in i) or ('bottle' in i) or ('shoe' in i) or ('book' in i) or ('laptop' in i) or ('cereal_box' in i):
            label_files.append(i)
    # label_files = origin_label_files[:2]

    cat_set = set()
    pbar = tqdm(total=15000, ascii=' >=')
    for split in splits:
        os.makedirs(os.path.join(save_dir, split), exist_ok=True)
        for each_label in label_files:
            label_filename = '{}_{}'.format(each_label, split)
            label_file = os.path.join(label_dir, label_filename)
            f = open(label_file)
            lines = f.readlines()
            for video_id in lines:
                pbar.update(1)
                video_id = video_id.split()[0]
                print('line: ', video_id)
                save_path = os.path.join(save_dir, split, video_id)
                if os.path.exists(os.path.join(save_path,)):  # or 'chair/batch-17/26' not in save_path:
                    print('Skip {}'.format(save_path))
                    continue
                os.makedirs(os.path.join(download_dir, video_id), exist_ok=True)
                class_name, batch_i, _ = video_id.split('/')
                os.makedirs(os.path.join(annotation_dir, '{}/{}'.format(class_name, batch_i)), exist_ok=True)
                urls = [f'{public_url}/videos/{video_id}/video.MOV',
                        f'{public_url}/videos/{video_id}/geometry.pbdata']
                for url in urls:
                    with open(os.path.join(download_dir, video_id, os.path.basename(url)), "wb") as f:
                        f.write(requests.get(url).content)
                for url in [f'{public_url}/annotations/{video_id}.pbdata']:
                    with open(os.path.join(annotation_dir, video_id + os.path.splitext(url)[1]), "wb") as f:
                        f.write(requests.get(url).content)

                annotation_file = 'annotations/{}.pbdata'.format(video_id)
                # Along with the video.MOV file, there is a geometry.pbdata file that contains
                # the geometry information of the scene (such as camera poses, point-clouds, and surfaces).
                # There is a copy of this container within each annotation protobuf too.
                geometry_filename = 'videos/{}/geometry.pbdata'.format(video_id)  # a.k.a. AR metadata
                video_filename = 'videos/{}/video.MOV'.format(video_id)

                frame_id = 0

                frame_data = load_frame_data(geometry_filename)

                camera = frame_data[frame_id].camera
                cam_to_world = np.array(camera.transform).reshape(4,4)  # [:3], # 3 x 4
                # JaxNeRF assumes x,y focal lengths are equal.
                intrinsic_matrix = camera.intrinsics
                os.makedirs(save_path, exist_ok=True)
                poses = make_poses_bounds_array(frame_data, near=0.2, far=10)
                np.save(os.path.join(save_path, 'poses_bounds.npy'), poses)
                os.makedirs(os.path.join(save_path, image_dir), exist_ok=True)
                with open(annotation_file, 'rb') as pb:
                    sequence = annotation_protocol.Sequence()
                    sequence.ParseFromString(pb.read())
                    # print('annotation_2d: {}, cat: {}\nnum_keypoints: {}, types: {}\nannotation_3d: {}'
                    #       .format(np.array(annotation_2d), cat, num_keypoints, types, annotation_3d))
                    # print(np.array(annotation_2d).mean(0), np.array(annotation_3d).mean(0))
                    for frame_i in list(range(len(sequence.frame_annotations))):
                        try:
                            frame = grab_frame(video_filename, [frame_i])[0]  # (1920, 1440, 3)
                        except:
                            continue
                        annotation_2d, cat, num_keypoints, types, annotation_3d = get_frame_annotation(sequence, frame_i)
                        annotation_3d_array = np.array(annotation_3d)
                        point_3d = np.array(annotation_3d)
                        camera_intrinsics = np.array(intrinsic_matrix).reshape([3, 3])
                        # print('1. frame.shape: ', frame.shape)
                        frame = cv2.resize(frame, (int(frame.shape[1] / 8), int(frame.shape[0] / 8)))
                        # print('frame.shape', frame.shape)
                        # Points are arranged as 1 + 8 (center
                        #         keypoint + 8 box vertices) matrix.
                        # https://github.com/google-research-datasets/Objectron/blob/master/objectron/dataset/box.py#L130-L131
                        try:
                            print('cat: ', cat)
                            cat_set.add(cat[0])
                        except:
                            continue
                        bbox = {'corner_bbox': point_3d[1:][None, ...],
                                'intrinsic_matrix': camera_intrinsics, 'cam_to_world': np.array(cam_to_world),
                                'category': cat[0],
                                # 'category': type2class[cat[0]]  # motobike
                                }
                        with open(os.path.join(save_path, image_dir, '{0:06d}_bbox.pkl'.format(frame_i)), 'wb') as handle:
                            pickle.dump(bbox, handle)
                        save_img_path = os.path.join(save_path, image_dir, '{0:06d}.png'.format(frame_i))
                        cv2.imwrite(save_img_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        print('Save to {}'.format(save_img_path))

                os.remove(annotation_file)
                # Along with the video.MOV file, there is a geometry.pbdata file that contains
                # the geometry information of the scene (such as camera poses, point-clouds, and surfaces).
                # There is a copy of this container within each annotation protobuf too.
                os.remove(geometry_filename)
                os.remove(video_filename)

    print('cat_set: ', cat_set)
    print('Saved to {}'.format(save_dir))
    pbar.close()

if __name__ == "__main__":
    prepare_dataset()
