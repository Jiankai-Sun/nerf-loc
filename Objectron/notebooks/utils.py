import numpy as np
import sys, os
import cv2
import struct
import transforms3d
import matplotlib.pyplot as plt
import copy

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

# https://github.com/google-research-datasets/Objectron#dataset-format
type2class = {
            "bike": 0,
            "book": 1,
            "bottle": 2,
            "camera": 3,
            "cereal_box": 4,
            "chair": 5,
            "cup": 6,
            "laptop": 7,
            "shoe": 8,
        }

EDGES = (
    [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
)

def draw_boxes_3d(boxes=[], clips=[], colors=['r', 'b', 'g', 'k'], plt_show=False, save_img=False, save_name=''):
    """Draw a list of boxes.

        The boxes are defined as a list of vertices
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    for i, b in enumerate(boxes):
        # print('b.shape: ', b.shape)  #  (9, 3)
        x, y, z = b[:, 0], b[:, 1], b[:, 2]
        # print('x.shape: ', x.shape)  # (9, )
        ax.scatter(x, y, z, c=colors[i % len(colors)])
        for e in EDGES:
            ax.plot(x[e], y[e], z[e], linewidth=2, c=colors[i % len(colors)])

    if (len(clips)):
        points = np.array(clips)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=100, c='k')

    plt.gca().patch.set_facecolor('white')
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_zaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))

    # rotate the axes and update
    ax.view_init(30, 12)
    plt.draw()
    if plt_show:
        plt.show()
    if save_img:
        plt.savefig('{}_{}.png'.format(save_name, 'boxes_3d'))

def get_frame_annotation(sequence, frame_id):
  """Grab an annotated frame from the sequence."""
  # print(len(sequence.frame_annotations))
  data = sequence.frame_annotations[frame_id]
  object_id = 0
  object_keypoints_2d = []
  object_keypoints_3d = []
  object_rotations = []
  object_translations = []
  object_scale = []
  num_keypoints_per_object = []
  object_categories = []
  annotation_types = []
  # Get the camera for the current frame. We will use the camera to bring
  # the object from the world coordinate to the current camera coordinate.
  camera = np.array(data.camera.transform).reshape(4, 4)

  for obj in sequence.objects:
    rotation = np.array(obj.rotation).reshape(3, 3)
    translation = np.array(obj.translation)
    object_scale.append(np.array(obj.scale))
    transformation = np.identity(4)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation
    obj_cam = np.matmul(camera, transformation)
    object_translations.append(obj_cam[:3, 3])
    object_rotations.append(obj_cam[:3, :3])
    object_categories.append(obj.category)
    annotation_types.append(obj.type)

  keypoint_size_list = []
  for annotations in data.annotations:
    num_keypoints = len(annotations.keypoints)
    keypoint_size_list.append(num_keypoints)
    for keypoint_id in range(num_keypoints):
      keypoint = annotations.keypoints[keypoint_id]
      object_keypoints_2d.append(
          (keypoint.point_2d.x, keypoint.point_2d.y, keypoint.point_2d.depth))
      object_keypoints_3d.append(
          (keypoint.point_3d.x, keypoint.point_3d.y, keypoint.point_3d.z))
    num_keypoints_per_object.append(num_keypoints)
    object_id += 1
  return (object_keypoints_2d, object_categories, keypoint_size_list,
          annotation_types, object_keypoints_3d)

def load_frame_data(geometry_filename):
    # See get_geometry_data in objectron-geometry-tutorial.ipynb
    frame_data = []
    with open(geometry_filename, 'rb') as pb:
        proto_buf = pb.read()

        i = 0
        while i < len(proto_buf):
            msg_len = struct.unpack('<I', proto_buf[i:i + 4])[0]
            i += 4
            message_buf = proto_buf[i:i + msg_len]
            i += msg_len
            frame = ar_metadata_protocol.ARFrame()
            frame.ParseFromString(message_buf)
            frame_data.append(frame)
    return frame_data

def grab_frame(video_file, frame_ids):
    """Grab an image frame from the video file."""
    frames = []
    capture = cv2.VideoCapture(video_file)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap = cv2.VideoCapture(video_file)
    print('height: {}, width: {}, frame_size: {}'.format(height, width, (height, width, 3)))
    # height: 1920, width: 1440, frame_size: 8294400
    # fps: 30.005475451724767, frame_count: 274.0

    for frame_id in frame_ids:
        # Get the frames per second
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Get the total numer of frames in the video.
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print('fps: {}, frame_count: {}'.format(fps, frame_count))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        # Read the next frame from the video. If you set frame 749 above then the code will return the last frame.
        ret, current_frame = cap.read()
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        frames.append(current_frame)
    return frames

def transform_box_format(annotation_3d_array, annotation_3d, semantic_class):
    '''
                    x                              x
        1 + + + + + + + + 5                 .-------
        +\                +\                |\
        + \ y             + \             z | \ y
        +  \              +  \              |  \
        +   3 + + + + + + + + 7
        z + +             +   +
        +   +             +   +
        +   +     C=0     +   +
        +   +             +   +
        2 + + + + + + + + 6   +
        \   +              \  +
         \  +               \ +
          \ +                \+
            4 + + + + + + + + 8
        '''
    cx = annotation_3d_array[0, 0]
    cy = annotation_3d_array[0, 1]
    cz = annotation_3d_array[0, 2]
    l = np.linalg.norm(annotation_3d_array[1] - annotation_3d[5])  # x
    h = np.linalg.norm(annotation_3d_array[1] - annotation_3d[2])  # z
    w = np.linalg.norm(annotation_3d_array[1] - annotation_3d[3])  # y

    trans_o2c=annotation_3d_array[0]
    mean_5678 = np.mean(annotation_3d_array[5:9], axis=0)-trans_o2c
    mean_3478 = np.mean(annotation_3d_array[[3,4,7,8]], axis=0)-trans_o2c
    mean_2468 = np.mean(annotation_3d_array[[2,4,6,8]], axis=0)-trans_o2c
    s_rot_o2c= np.stack([mean_5678,mean_3478,mean_2468], axis=0).T
    scale=((s_rot_o2c.T@s_rot_o2c).diagonal())**0.5
    rot_o2c = s_rot_o2c/scale
    ax, angle=transforms3d.axangles.mat2axangle(rot_o2c, unit_thresh=1e-05)

    bbox = np.array([cx, cy, cz, l, h, w, angle, semantic_class])[None, :]
    # import pdb; pdb.set_trace()
    # print(bbox)

    return bbox

def draw_box(img, arranged_points, color=(0, 0, 255)):
    """
    plot arranged_points on img.
    arranged_points: list of points [[x, y]] in image coordinate.
    """
    RADIUS = 10
    COLOR = color
    EDGES = [
      [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
      [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
      [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
    ]
    for i in range(arranged_points.shape[0]):
        x, y = arranged_points[i]
        # print(x, y)
        cv2.circle(img, (int(x), int(y)), RADIUS, COLOR, -10)
    for edge in EDGES:
        start_points = arranged_points[edge[0]]
        start_x = int(start_points[0])
        start_y = int(start_points[1])
        end_points = arranged_points[edge[1]]
        end_x = int(end_points[0])
        end_y = int(end_points[1])
        cv2.line(img, (start_x, start_y), (end_x, end_y), COLOR, 2)
    # plt.imshow(img)
    return img

def project_by_intrinsics(point_3d, camera_intrinsics, world_to_cam):
    """
    Project using camera intrinsics.
    Objectron frame (x down, y right, z in);
    H-Z frame (x right, y down, z out);
    Objectron intrinsics has px and py swapped;
    px and py are from original image size (1440, 1920);
    cam_to_world: 4x4
    Approach 1:
    To transform Objectron frame to H-Z frame,
    we need to negate z and swap x and y;
    To modify intrinsics, we need to swap px, py.
    Or alternatively, approach 2:
    we change the sign for z and swap x and y after projection.
    Reference
    https://github.com/google-research-datasets/Objectron/issues/39#issuecomment-835509430
    https://amytabb.com/ts/2019_06_28/
    """
    vertices_3d = point_3d.reshape(9, 3)
    vertices_3d_homg = np.concatenate((vertices_3d, np.ones_like(vertices_3d[:, :1])), axis=-1).T
    # vertices_2d_proj = np.matmul(np.linalg.inv(cam_to_world.reshape(4, 4)), vertices_3d_homg)  # 4x9
    vertices_2d_proj = np.matmul(world_to_cam.reshape(4, 4), vertices_3d_homg)  # 4x9
    # import pdb; pdb.set_trace()
    # print('vertices_2d_proj.T: ', vertices_2d_proj.T)
    vertices_3d = vertices_2d_proj.T[:, :3]  # (9, 3)
    # Objectron to H-Z frame
    vertices_3d[:, 2] = - vertices_3d[:, 2]
    intr = copy.deepcopy(camera_intrinsics).reshape(3, 3)
    # scale intrinsics from (1920, 1440) to (640, 480)
    intr[:2, :] = intr[:2, :] / np.array([[1920], [1440]]) * np.array([[640], [480]])
    point_2d = intr @ vertices_3d.T  # 3x9
    point_2d[:2, :] = point_2d[:2, :] / point_2d[2, :]
    # landscape to portrait swap x and y.
    point_2d[[0, 1], :] = point_2d[[1, 0], :]
    arranged_points = point_2d.T[:, :2]
    # print('arranged_points: ', arranged_points)
    return arranged_points

def make_poses_bounds_array(frame_data, near=0.2, far=10):
    # See https://github.com/Fyusion/LLFF#using-your-own-poses-without-running-colmap
    # Returns an array of shape (N, 17).
    rows = []
    for frame in frame_data:
        camera = frame.camera
        cam_to_world = np.array(camera.transform).reshape(4,4)[:3]
        w = camera.image_resolution_width
        h = camera.image_resolution_height
        assert w>=h, (w, h)  # assume landscape
        # JaxNeRF assumes x,y focal lengths are equal.
        row = llff_pose(cam_to_world, image_hw=[w, h],
                        focal=camera.intrinsics[0], near=near, far=far)
        rows.append(row)
    return np.vstack(rows)

def llff_pose(cam_to_world_mat34, image_hw, focal, near, far):
    # Returns an array of shape (17,).
    col = np.concatenate([image_hw, [focal]]).reshape(-1, 1)
    mat35 = np.hstack([cam_to_world_mat34, col])
    return np.concatenate([mat35.ravel(), [near, far]])

# vis_pred
def project_by_camera_projection(point_3d, camera_projection, image_width, image_height):
    """
    Projection using camera projection matrix.

    Reference:
    function project_points in https://github.com/google-research-datasets/Objectron/blob/master/notebooks/objectron-geometry-tutorial.ipynb
    http://www.songho.ca/opengl/gl_projectionmatrix.html
    """
    vertices_3d = point_3d.reshape(9, 3)
    vertices_3d_homg = np.concatenate((vertices_3d, np.ones_like(vertices_3d[:, :1])), axis=-1).T
    vertices_2d_proj = np.matmul(camera_projection.reshape(4, 4), vertices_3d_homg)
    # Project the points
    points2d_ndc = vertices_2d_proj[:-1, :] / vertices_2d_proj[-1, :]
    points2d_ndc = points2d_ndc.T
    # Convert the 2D Projected points from the normalized device coordinates to pixel values
    x = points2d_ndc[:, 1]
    y = points2d_ndc[:, 0]
    pt2d = np.copy(points2d_ndc)
    pt2d[:, 0] = (1 + x) / 2 * image_width
    pt2d[:, 1] = (1 + y) / 2 * image_height
    arranged_points = pt2d[:, :2]
    return arranged_points