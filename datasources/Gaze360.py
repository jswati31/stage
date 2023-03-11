import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import os
import math
import torch.nn as nn


def make_dataset(source_path, file_name, mapping):

    images = []
    enable_time = True
    print(file_name)
    skip_count = 0
    with open(file_name, 'r') as f:
        for line in f:
            line = line[:-1]
            line = line.replace("\t", " ")
            line = line.replace("  ", " ")
            split_lines = line.split(" ")
            if (len(split_lines) > 3):
                frame_number = int(split_lines[0].split('/')[-1][:-4])
                lists_sources = []
                lists_sources2 = []
                # import pdb
                # pdb.set_trace()
                for j in range(-30 + 1, 1):
                    new_frame_number = int(frame_number + j * int(enable_time))
                    name_frame = '/'.join(split_lines[0].split('/')[:-1] + ['%0.6d.jpg' % (new_frame_number)])
                    lists_sources.append(name_frame)

                gaze = np.zeros((3))

                gaze[0] = float(split_lines[1])
                gaze[1] = float(split_lines[2])
                gaze[2] = float(split_lines[3])

                if lists_sources[-1] not in mapping:
                    skip_count += 1
                    continue

                if not any([nm in mapping for nm in lists_sources]):
                    skip_count += 1
                    continue

                item = (lists_sources, gaze)
                images.append(item)
                # print(images)

    if skip_count > 0:
        print('Skipped', skip_count / (skip_count + len(images)))

    return images


class Gaze360Loader(data.Dataset):
    def __init__(self, source_path, config, subset, transforms=None, angle=None):

        assert subset in ["train", "test", "validation"]

        self.transforms = transforms
        file_name = os.path.join(source_path, subset + '.txt')

        if subset == "validation":
            label_file = 'val.label'
        else:
            label_file = subset + '.label'
        label_file = os.path.join(source_path, 'FaceBased/Label/' + label_file)

        with open(label_file) as infile:
            lines = infile.readlines()
            header = lines.pop(0)
        self.img2label_mapping = {}
        all_face_paths = []
        for K in lines:
            face_path = os.path.join(source_path, 'FaceBased', 'Image', K.split(' ')[0])

            orig_path = K.split(' ')[3]
            all_face_paths.append(face_path)

            label = K.strip().split(' ')[4].split(",")
            label = np.array(label).astype('float')

            label2 = K.strip().split(' ')[5].split(",")
            label2 = np.array(label2).astype('float')

            if angle is not None:

                if abs((label2[0]*180/np.pi)) > angle or abs((label2[1]*180/np.pi)) > angle:
                    continue

            self.img2label_mapping[orig_path] = {}
            self.img2label_mapping[orig_path]['face_path'] = face_path
            self.img2label_mapping[orig_path]['2D_gaze'] = label2[::-1]  # originally [yaw,pitch], store as [pitch,yaw]
            self.img2label_mapping[orig_path]['3D_gaze'] = label

        print(len(self.img2label_mapping))

        prev_imgs = make_dataset(os.path.join(source_path, "imgs"), file_name, self.img2label_mapping)
        imgs = []
        for x in prev_imgs:
            if len(x) != 0:
                if imgs.count(x) >= 1:
                    continue
                else:
                    imgs.append(x)

        self.all_avail_image_labels = list(self.img2label_mapping.keys())

        self.source_path = source_path
        self.file_name = file_name
        self.imgs = imgs
        self.config = config

    def preprocess_frames(self, frames):
        if self.transforms is not None:
            imgss = [self.transforms(frames[k]) for k in range(frames.shape[0])]
            return torch.stack(imgss, dim=0)
        else:
            # Expected input:  N x H x W x C
            # Expected output: N x C x H x W
            frames = np.transpose(frames, [0, 3, 1, 2])
            frames = frames.astype(np.float32)
            frames *= 2.0 / 255.0
            frames -= 1.0
            return frames

    def __getitem__(self, index):
        path_source, _ = self.imgs[index]

        subentry = {}

        source_video = []
        validity = []
        gaze_pitchyaw = []

        count = 0
        for i, frame_path in enumerate(path_source):
            # print(frame_path)
            if frame_path in self.all_avail_image_labels:
                face_path = self.img2label_mapping[frame_path]['face_path']
                img = Image.open(face_path).convert('RGB')
                img = img.resize((self.config.face_size[0], self.config.face_size[1]))
                source_video.append(np.array(img))
                validity.append(0)
                gaze_pitchyaw.append(self.img2label_mapping[frame_path]['2D_gaze'])
                count += 1
            else:
                if count == 0:
                    continue
                else:
                    source_video.append(source_video[-1].copy())
                    validity.append(0)
                    gaze_pitchyaw.append(gaze_pitchyaw[-1].copy())
                    count += 1

        if len(source_video) == 0:
            return None

        validity[-1] = 1
        source_video = np.array(source_video).astype(np.float32)
        gaze_pitchyaw = np.array(gaze_pitchyaw).astype(np.float32)
        validity = np.array(validity).astype(np.int)

        source_video = self.preprocess_frames(source_video)

        subentry['face_patch'] = source_video
        subentry['face_g_tobii'] = gaze_pitchyaw
        subentry['face_g_tobii_validity'] = validity

        for key, value in subentry.items():
            if value.shape[0] < self.config.max_sequence_len:
                pad_len = self.config.max_sequence_len - value.shape[0]

                if pad_len > 0:
                    subentry[key] = np.pad(
                        value,
                        pad_width=[(0, pad_len if i == 0 else 0) for i in range(value.ndim)],
                        mode='constant',
                        constant_values=(False if value.dtype is np.bool else 0.0),
                    )

        torch_entry = dict([
            (k, torch.from_numpy(a)) if isinstance(a, np.ndarray) else (k, a)
            for k, a in subentry.items()
        ])

        return torch_entry

    def __len__(self):
        return len(self.imgs)

