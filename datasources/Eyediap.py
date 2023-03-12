
import torch.utils.data as data
from PIL import Image
import torch
import numpy as np
import os


class EyediapLoader(data.Dataset):
    def __init__(self, source_path, config, transforms=None, person_id=None, no_padding=False):
        self.transforms = transforms

        self.config = config
        self.source_path = source_path
        self.no_padding = no_padding

        self.all_eyediap_mapping = {}

        all_label_files_list = [f for f in os.listdir(source_path + '/EyeDiap_face/Label/') if
                                f.endswith('label')]

        if person_id is not None:
            all_label_files_list = [p for p in all_label_files_list if p.split('.')[0] in person_id]

        for ff in all_label_files_list:

            pid = ff.split('.')[0]

            with open(source_path + '/EyeDiap_face/Label/' + ff) as infile:
                lines = infile.readlines()
                header = lines.pop(0)

            eyediap_img_mapping = {}

            for K in lines:
                face_path = os.path.join(source_path, 'EyeDiap_face/Image', K.split(' ')[0])

                orig_path = K.strip().split(' ')[3]
                session_str = "_".join(orig_path.split('_')[:-1])
                timestamp = int(orig_path.split('_')[-1])

                assert (timestamp - 1) % 15 == 0

                if session_str not in eyediap_img_mapping:
                    eyediap_img_mapping[session_str] = {}
                    eyediap_img_mapping[session_str]['timestamps'] = []
                    time_sections = []

                label = K.strip().split(' ')[4].split(",")
                label = np.array(label).astype('float')

                label2 = K.strip().split(' ')[6].split(",")
                label2 = np.array(label2).astype('float')[::-1]  # originally [yaw,pitch], store as [pitch,yaw]

                head_label2 = K.strip().split(' ')[7].split(",")
                head_label2 = np.array(head_label2).astype('float')[::-1]  # originally [yaw,pitch], store as [pitch,yaw]

                time_sections.append(timestamp)
                if len(time_sections) == 30:
                    eyediap_img_mapping[session_str]['timestamps'].append(time_sections)
                    time_sections = []

                eyediap_img_mapping[session_str][timestamp] = {'face_path': face_path, '2D_gaze': label2,
                                                               '3D_gaze': label, '2D_head': head_label2}

            self.all_eyediap_mapping[pid] = eyediap_img_mapping

        self.meta_data = []

        for person_id, data1 in self.all_eyediap_mapping.items():
            for session, data2 in data1.items():
                for tt in data2['timestamps']:
                    self.meta_data.append({'session': session,
                                           'timestamp_window': tt,
                                           'person': person_id})
    def preprocess_frames(self, frames):
        if self.no_padding:
            return frames
        if self.transforms is not None:
            imgss = [self.transforms(frames[k]) for k in range(frames.shape[0])]
            return torch.stack(imgss, dim=0)
        else:
            # Expected input:  N x H x W x C
            # Expected output: N x C x H x W
            frames = np.transpose(frames, [0, 3, 1, 2])
            # frames = np.transpose(frames, [2, 0, 1])
            frames = frames.astype(np.float32)
            # frames *= 1.0 / 255.0
            frames *= 2.0 / 255.0
            frames -= 1.0
            return frames

    def __getitem__(self, index):
        meta_source = self.meta_data[index]

        all_timestamps = meta_source['timestamp_window']
        session_ = meta_source['session']
        person_ = meta_source['person']

        entry_data = self.all_eyediap_mapping[person_][session_]

        subentry = {}

        source_video = []
        validity = []
        gaze_pitchyaw = []

        count = 0
        for i, timestep in enumerate(all_timestamps):
            if timestep in entry_data.keys():
                face_path = entry_data[timestep]['face_path']
                img = Image.open(face_path).convert('RGB')
                img = img.resize((self.config.face_size[0], self.config.face_size[1]))
                img = img.resize((self.config.face_size[0], self.config.face_size[1]))
                source_video.append(np.array(img))
                validity.append(1)
                gaze_pitchyaw.append(entry_data[timestep]['2D_gaze'])
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
        return len(self.meta_data)
