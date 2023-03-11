
import numpy as np
import os
import cv2
import torch
from collections import OrderedDict
from torch.nn import functional as F
import json
import yaml
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

actual_screen_size = [1920, 1080]


def build_image_matrix(images, n_rows, n_cols):
    image_shape = images.shape[1:]

    image_matrix = np.zeros((n_rows * image_shape[0], n_cols * image_shape[1], 3), dtype=np.uint8)
    for i in range(n_cols):
        for j in range(n_rows):
            image_matrix[j * image_shape[0] : (j + 1) * image_shape[0], i * image_shape[1] : (i + 1) * image_shape[1]] = images[j * n_cols + i]

    return image_matrix

def adjust_learning_rate(optimizers, decay, number_decay, base_lr):
    lr = base_lr * (decay ** number_decay)
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    return lr

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.log = []
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.log.append(self.avg)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class RunningStatistics(object):
    def __init__(self):
        self.losses = OrderedDict()

    def add(self, key, value):
        if key not in self.losses:
            self.losses[key] = []
        self.losses[key].append(value)

    def means(self):
        return OrderedDict([
            (k, np.mean(v)) if len(v) > 0 else (k, v) for k, v in self.losses.items()
        ])

    def reset(self):
        for key in self.losses.keys():
            self.losses[key] = []

    def __str__(self):
        fmtstr = ', '.join(['%s: %.6f (%.6f)' % (k, v[-1], self.means()[k]) for k, v in self.losses.items()])
        return fmtstr


def recover_images(x):
    # Every specified iterations save sample images
    # Note: We're doing this separate to Tensorboard to control which input
    #       samples we visualize, and also because Tensorboard is an inefficient
    #       way to store such images.
    x = x.cpu().numpy()
    x = (x + 1.0) * (255.0 / 2.0)
    x = np.clip(x, 0, 255)  # Avoid artifacts due to slight under/overflow
    x = x.astype(np.uint8)
    if len(x.shape) == 4:
        x = np.transpose(x, [0, 2, 3, 1])  # CHW to HWC
        x = x[:, :, :, ::-1]  # RGB to BGR for OpenCV
    else:
        x = np.transpose(x, [1, 2, 0])  # CHW to HWC
        x = x[:, :, ::-1]  # RGB to BGR for OpenCV
    return x


def send_data_dict_to_gpu(data1, data2, device):
    data = {}
    for k, v in data1.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    for k, v in data2.items():
        if isinstance(v, torch.Tensor):
            data[k] = v.detach().to(device, non_blocking=True)
    return data


class convert_dict(object):
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
               setattr(self, a, [convert_dict(x) if isinstance(x, dict) else x for x in b])
            else:
               setattr(self, a, convert_dict(b) if isinstance(b, dict) else b)

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) ==0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)


def np_pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out


def np_vector_to_pitchyaw(vectors):
    n = vectors.shape[0]
    out = np.empty((n, 2))
    vectors = np.divide(vectors, np.linalg.norm(vectors, axis=1).reshape(n, 1))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


radians_to_degrees = 180.0 / np.pi


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = np_pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = np_pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    similarity = np.clip(similarity, a_min=-1.0 + 1e-6, a_max=1.0 - 1e-6)

    return np.arccos(similarity) * radians_to_degrees


def pitchyaw_to_vector(a):
    if a.shape[1] == 2:
        sin = torch.sin(a)
        cos = torch.cos(a)
        return torch.stack([cos[:, 0] * sin[:, 1], sin[:, 0], cos[:, 0] * cos[:, 1]], dim=1)
    elif a.shape[1] == 3:
        return F.normalize(a)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def vector_to_pitchyaw(a):
    if a.shape[1] == 2:
        return a
    elif a.shape[1] == 3:
        a = a.view(-1, 3)
        norm_a = torch.div(a, torch.norm(a, dim=1).view(-1, 1) + 1e-7)
        return torch.stack([
            torch.asin(norm_a[:, 1]),
            torch.atan2(norm_a[:, 0], norm_a[:, 2]),
        ], dim=1)
    else:
        raise ValueError('Do not know how to convert tensor of size %s' % a.shape)


def apply_transformation(T, vec):
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
    vec = vec.reshape(-1, 3, 1)
    h_vec = F.pad(vec, pad=(0, 0, 0, 1), value=1.0)
    return torch.matmul(T, h_vec)[:, :3, 0]


def apply_rotation(T, vec):
    if vec.shape[1] == 2:
        vec = pitchyaw_to_vector(vec)
    vec = vec.reshape(-1, 3, 1)
    R = T[:, :3, :3]
    return torch.matmul(R, vec).reshape(-1, 3)


nn_plane_normal = None
nn_plane_other = None


def get_intersect_with_zero(o, g):
    """Intersects a given gaze ray (origin o and direction g) with z = 0."""
    global nn_plane_normal, nn_plane_other
    if nn_plane_normal is None:
        nn_plane_normal = torch.tensor([0, 0, 1], dtype=torch.float32, device=device).view(1, 3, 1)
        nn_plane_other = torch.tensor([1, 0, 0], dtype=torch.float32, device=device).view(1, 3, 1)

    # Define plane to intersect with
    n = nn_plane_normal
    a = nn_plane_other
    g = g.view(-1, 3, 1)
    o = o.view(-1, 3, 1)
    numer = torch.sum(torch.mul(a - o, n), dim=1)

    # Intersect with plane using provided 3D origin
    denom = torch.sum(torch.mul(g, n), dim=1) + 1e-7
    t = torch.div(numer, denom).view(-1, 1, 1)
    return (o + torch.mul(t, g))[:, :2, 0]


def to_screen_coordinates(origin, direction, rotation, inv_camera_transformation, pixels_per_millimeter):

    direction = pitchyaw_to_vector(direction)

    # Negate gaze vector back (to camera perspective)
    direction = -direction

    # De-rotate gaze vector
    inv_rotation = torch.transpose(rotation, 1, 2)
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(inv_rotation, direction)

    # Transform values
    direction = apply_rotation(inv_camera_transformation, direction)
    origin = apply_transformation(inv_camera_transformation, origin)

    # Intersect with z = 0
    recovered_target_2D = get_intersect_with_zero(origin, direction)
    PoG_mm = recovered_target_2D

    # Convert back from mm to pixels
    ppm_w = pixels_per_millimeter[:, 0]
    ppm_h = pixels_per_millimeter[:, 1]
    PoG_px = torch.stack([
        torch.clamp(recovered_target_2D[:, 0] * ppm_w,
                    0.0, float(actual_screen_size[0])),
        torch.clamp(recovered_target_2D[:, 1] * ppm_h,
                    0.0, float(actual_screen_size[1]))
    ], axis=-1)

    return PoG_mm, PoG_px


def save_configs(save_path, config):
    target_dir = save_path + '/configs'
    if not os.path.isdir(target_dir):
        os.makedirs(target_dir)
    fpath = os.path.relpath(target_dir + '/params.yaml')
    with open(fpath, 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)


def set_logger(save_path):

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(save_path, 'training.log'))
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

    return logging


def get_R(x,y,z):
    ''' Get rotation matrix from three rotation angles (radians). right-handed.
    Args:
        angles: [3,]. x, y, z angles
    Returns:
        R: [3, 3]. rotation matrix.
    '''
    # x
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(x), -np.sin(x)],
                   [0, np.sin(x), np.cos(x)]])
    # y
    Ry = np.array([[np.cos(y), 0, np.sin(y)],
                   [0, 1, 0],
                   [-np.sin(y), 0, np.cos(y)]])
    # z
    Rz = np.array([[np.cos(z), -np.sin(z), 0],
                   [np.sin(z), np.cos(z), 0],
                   [0, 0, 1]])

    R = Rz.dot(Ry.dot(Rx))
    return R


def get_original_gaze(direction, rotation):
    direction = pitchyaw_to_vector(direction)

    # Negate gaze vector back (to camera perspective)
    direction = -direction

    # De-rotate gaze vector
    inv_rotation = torch.transpose(rotation, 1, 2)
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(inv_rotation, direction).reshape(-1, 3)

    # to user perspective back
    direction = -direction
    direction = vector_to_pitchyaw(direction)

    return direction


def calculate_combined_gaze_direction(avg_origin, avg_PoG, head_rotation,
                                      camera_transformation):
    # NOTE: PoG is assumed to be in mm and in the screen-plane
    avg_PoG_3D = F.pad(avg_PoG, (0, 1))

    # Bring to camera-specific coordinate system, where the origin is
    avg_PoG_3D = apply_transformation(camera_transformation, avg_PoG_3D)
    direction = avg_PoG_3D - avg_origin

    # Rotate gaze vector back
    direction = direction.reshape(-1, 3, 1)
    direction = torch.matmul(head_rotation, direction)

    # Negate gaze vector back (to user perspective)
    direction = -direction

    direction = vector_to_pitchyaw(direction)
    return direction



def pitchyaw_to_rotation(a):
    if a.shape[1] == 3:
        a = vector_to_pitchyaw(a)

    cos = torch.cos(a)
    sin = torch.sin(a)
    ones = torch.ones_like(cos[:, 0])
    zeros = torch.zeros_like(cos[:, 0])
    matrices_1 = torch.stack([ones, zeros, zeros,
                              zeros, cos[:, 0], sin[:, 0],
                              zeros, -sin[:, 0], cos[:, 0]
                              ], dim=1)
    matrices_2 = torch.stack([cos[:, 1], zeros, sin[:, 1],
                              zeros, ones, zeros,
                              -sin[:, 1], zeros, cos[:, 1]
                              ], dim=1)
    matrices_1 = matrices_1.view(-1, 3, 3)
    matrices_2 = matrices_2.view(-1, 3, 3)
    matrices = torch.matmul(matrices_2, matrices_1)
    return matrices
