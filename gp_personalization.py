"""
script for GP personalization on EYEDIAP dataset
"""

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import warnings
warnings.filterwarnings('ignore')
import gc
import json
import torch
import numpy as np
import os
import random
import argparse
import gpytorch
from models.losses import AngularLoss
from models import create_model
from argparse import Namespace
from torch.utils.data import DataLoader
from utils.core_utils import RunningStatistics
from utils.train_utils import my_collate
import torch.utils.data as data
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EyediapLoader(data.Dataset):
    def __init__(self, source_path, config, transforms=None, person_id=None, train=False, no_padding=False):
        self.transforms = transforms

        self.config = config
        self.source_path = source_path
        self.no_padding = no_padding

        self.all_eyediap_mapping = {}
        self.selected_samples = None

        all_label_files_list = [f for f in os.listdir(source_path + '/EyeDiap_face/Label/') if
                                f.endswith('label')]

        self.all_samples = []

        if person_id is not None:
            if train:
                all_label_files_list = [p for p in all_label_files_list if p.split('.')[0] != person_id]
            else:
                all_label_files_list = [p for p in all_label_files_list if p.split('.')[0] == person_id]

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

                self.all_samples.append([pid, session_str, timestamp])

            self.all_eyediap_mapping[pid] = eyediap_img_mapping

        self.meta_data = []

        for person_id, data1 in self.all_eyediap_mapping.items():
            for session, data2 in data1.items():
                for tt in data2['timestamps']:
                    self.meta_data.append({'session': session,
                                           'timestamp_window': tt,
                                           'person': person_id})

    def get_few_samples(self, num_samples):
        subentry = {}

        all_samples = self.all_samples
        selected_samples_indices = np.random.choice([i for i in range(len(all_samples))], num_samples, replace=False)
        selected_samples = [all_samples[k] for k in selected_samples_indices]
        self.selected_samples = selected_samples

        gaze_pitchyaw = np.zeros((len(selected_samples), self.config.max_sequence_len, 2)).astype(np.float32)
        source_video = np.zeros((len(selected_samples), self.config.max_sequence_len, self.config.face_size[0], self.config.face_size[1], 3)).astype(
            np.float32)
        validity = np.zeros((len(selected_samples), self.config.max_sequence_len)).astype(np.int)

        count = 0
        for sample in selected_samples:
            face_path = self.all_eyediap_mapping[sample[0]][sample[1]][sample[2]]['face_path']
            img = Image.open(face_path).convert('RGB')
            img = img.resize((self.config.face_size[0], self.config.face_size[1]))
            gaze_pitchyaw[count, 0] = self.all_eyediap_mapping[sample[0]][sample[1]][sample[2]]['2D_gaze']
            source_video[count, 0] = img
            validity[count, 0] = 1
            count += 1

        source_video = np.transpose(source_video, [0, 1, 4, 2, 3])
        source_video = source_video.astype(np.float32)
        source_video *= 2.0 / 255.0
        source_video -= 1.0

        subentry['face_patch'] = source_video
        subentry['face_g_tobii'] = gaze_pitchyaw
        subentry['face_g_tobii_validity'] = validity

        torch_entry = dict([
            (k, torch.from_numpy(a)) if isinstance(a, np.ndarray) else (k, a)
            for k, a in subentry.items()
        ])

        return torch_entry

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
            if timestep in entry_data.keys() and self.selected_samples is not None \
                    and [person_, session_, timestep] not in self.selected_samples:

                        face_path = entry_data[timestep]['face_path']
                        img = Image.open(face_path).convert('RGB')
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

# Define the Model
class BatchedGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, shape):
        super(BatchedGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([shape]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([shape])),
            batch_shape=torch.Size([shape])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


def evaluation_adapt(test_data, trainer_model, curr_step, device, train_data):
    test_losses = RunningStatistics()
    trainer_model.eval()
    torch.cuda.empty_cache()

    epochs = 15

    gp_model = None
    likelihood = None

    loss_func = AngularLoss()

    try:
        for param in trainer_model.module.parameters():
            param.requires_grad = False
    except:
        for param in trainer_model.parameters():
            param.requires_grad = False

    for module in trainer_model.children():
        module.train(False)

    for batch_num, input_data in enumerate(test_data):
        # Move tensors to GPU
        for k, v in input_data.items():
            if isinstance(v, torch.Tensor):
                input_data[k] = v.detach().to(device, non_blocking=True)

        with torch.no_grad():
            if torch.cuda.device_count() > 1:
                v_loss_dict, v_out_dict = trainer_model.module.compute_losses(input_data, only_3D=True)
            else:
                v_loss_dict, v_out_dict = trainer_model.compute_losses(input_data, only_3D=True)

            pred_global_gaze = v_out_dict['pred'].detach().squeeze(0)
            x_feats = v_out_dict['feats'].detach().squeeze(0)

        # train GP model for first batch on k samples
        if batch_num == 0:

            for k, v in train_data.items():
                if isinstance(v, torch.Tensor):
                    train_data[k] = v.detach().to(device, non_blocking=True)

            with torch.no_grad():
                if torch.cuda.device_count() > 1:
                    _, v_out_dict_tr = trainer_model.module.compute_losses(train_data, only_3D=True)
                else:
                    _, v_out_dict_tr = trainer_model.compute_losses(train_data, only_3D=True)

            y_pred_gp = v_out_dict_tr['pred'].detach().permute(1, 0, 2)[0]
            y_train_gp = train_data['face_g_tobii'].detach().permute(1, 0, 2)[0]
            x_train_gp = v_out_dict_tr['feats'].detach().permute(1, 0, 2)[0]

            gp_model_input_true = y_train_gp - y_pred_gp

            train_x = x_train_gp
            train_y = gp_model_input_true

            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
            gp_model = BatchedGP(train_x, train_y, likelihood, 2)
            gp_model.load_state_dict(torch.load(config.gp_model_path))

            gp_model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.001)

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

            gp_model = gp_model.to(device)
            likelihood = likelihood.to(device)
            mll = mll.to(device)

            # Define training helper function
            def epoch_train(ii):
                optimizer.zero_grad()  # Zero gradients
                output = gp_model(train_x)  # Compute noise-free output
                loss = -mll(output, train_y).sum()  # Compute batched loss
                loss.backward()  # Compute gradients with backpropagation
                print("Iter %d/%d - Loss: %.3f" % (ii + 1, epochs, loss.item()))

                optimizer.step()  # Update weights with gradients
                optimizer.zero_grad()  # Zero gradients
                gc.collect()  # Used to ensure there is no memory leak

            # Run training
            for i in range(epochs):
                epoch_train(i)

            torch.cuda.empty_cache()

        # testing
        gp_model.eval()
        likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            preds = likelihood(gp_model(x_feats))
            pred_mean = preds.mean
            lower, upper = preds.confidence_region()

        pred_gaze_person = pred_global_gaze + pred_mean

        v_loss_dict['final_personalized_gaze_error'] = loss_func(pred_gaze_person.unsqueeze(0), 'face_g_tobii', input_data)

        for k, v in v_loss_dict.items():
            test_losses.add('%s' % k, v.detach().cpu().numpy())

    test_loss_means = test_losses.means()
    print('Test Losses at [%s]: %s' %
                 (str(curr_step), ', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))

    return test_loss_means['total_loss'], test_loss_means['final_personalized_gaze_error']


def test_eyediap(config, model):

    print(' -------- Adapting on Eyediap Test participants ------------- ')

    over_10_mean = []
    for seed in range(10):

        random.seed(2*seed)
        np.random.seed(2*seed)
        torch.manual_seed(2*seed)

        overall_before_adaptation = []
        overall_after_adaptation = []

        for part_id in range(1, 16):
            if part_id in [12, 13]:
                continue

            pid = 'p' + str(part_id)

            test_dataset = EyediapLoader(source_path=config.eyediap_path, config=config,
                                         transforms=None, person_id=pid, train=False)

            train_data = test_dataset.get_few_samples(config.k)

            test_dataloader = DataLoader(test_dataset,
                                      batch_size=1,
                                      shuffle=True,
                                      drop_last=False,
                                      num_workers=config.train_data_workers,
                                      pin_memory=True,
                                      collate_fn=my_collate
                                      )

            per_person_error_before, per_person_error_after = evaluation_adapt(test_dataloader, model, part_id, device, train_data)

            overall_before_adaptation.append(per_person_error_before)
            overall_after_adaptation.append(per_person_error_after)

        print('Eyediap: Before Adaptation at iteration %s %s' % (seed, np.mean(overall_before_adaptation)))
        print('Eyediap: After Adaptation at iteration %s %s' % (seed, np.mean(overall_after_adaptation)))

        over_10_mean.append(np.mean(overall_after_adaptation))

    print('Eyediap: Overall 10 iterations Adaptation %s +- %s' % (np.mean(over_10_mean),
                                                                  np.std(over_10_mean)/np.sqrt(10)))


def main(config):

    gpu_devices = torch.cuda.device_count()

    print(config.model_type)
    model = create_model(config)
    if gpu_devices > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    # call tester
    print('-------- Loading ' + str(config.load_checkpoint_path) + ' model ------------- ')
    assert os.path.isfile(config.load_checkpoint_path)
    weights = torch.load(config.load_checkpoint_path)
    try:
        model.load_state_dict(weights)
    except:
        # If was stored using DataParallel but being read on 1 GPU
        if torch.cuda.device_count() == 1:
            if next(iter(weights.keys())).startswith('module.'):
                weights = dict([(k[7:], v) for k, v in weights.items()])
    model.load_state_dict(weights)

    test_eyediap(config, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GP Personalization on EYEDIAP.')
    parser.add_argument('--config_json', type=str, help='Path to config in JSON format')
    parser.add_argument('--gp_model_path', type=str, help='Path to base GP model')
    parser.add_argument('--k', type=int, default=9, help='few shot sample size')
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help='Path to STAGE checkpoint')
    parser.add_argument('--spatial_model', type=str, default="proposed", choices=['dual', 'cross',
                                                                                  'proposed', 'proposed_nodual'],
                        help='type of spatial model')

    args = parser.parse_args()

    ###############
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    ###############

    default_config = json.load(open('configs/default.json'))

    # load configurations
    assert os.path.isfile(args.config_json)
    print('Loading ' + args.config_json)

    model_specific_config = json.load(open(args.config_json))
    args = vars(args)

    # merging configurations
    config = {**default_config, **model_specific_config, **args}

    config = Namespace(**config)
    config.tanh = True

    main(config)
