"""
Main script to run within-dataset experiments
"""

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['WANDB_CACHE_DIR'] = "cache/"
import warnings
warnings.filterwarnings('ignore')
import json
import wandb
import numpy as np
import argparse
from models import create_model
from argparse import Namespace
from tensorboardX import SummaryWriter
import core.tester as Tester
from utils.core_utils import set_logger, save_configs
from utils.checkpoints_manager import CheckpointsManager
from utils.core_utils import RunningStatistics
from utils.train_utils import get_training_batches
import torch.nn as nn
from utils.train_utils import my_collate
from torch.utils.data import DataLoader, Subset
import torch.utils.data as data
from PIL import Image
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_dataset(source_path, file_name, mapping):
    """
    Here, we predict the gaze for the last frame. Usually (in make_dataset()), we predict for middle frame.
    """
    # assert enable_time is True

    # assert seq_len > 0
    # assert seq_len % 2 == 1 or enable_time is False
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

                if not all([nm in mapping for nm in lists_sources]):
                    skip_count += 1
                    continue

                item = (lists_sources, gaze)
                images.append(item)

    if skip_count > 0:
        print('Skipped', skip_count / (skip_count + len(images)))

    return images


class Gaze360Loader(data.Dataset):
    def __init__(self, source_path, config, subset, transforms=None, angle=None):

        assert subset in ["train", "test", "validation"]

        self.transforms = transforms
        self.subset = subset
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

        if self.subset == "train":
            validity_value = 1
        else:
            validity_value = 0

        count = 0
        for i, frame_path in enumerate(path_source):
            # print(frame_path)
            if frame_path in self.all_avail_image_labels:
                face_path = self.img2label_mapping[frame_path]['face_path']
                img = Image.open(face_path).convert('RGB')
                img = img.resize((self.config.face_size[0], self.config.face_size[1]))
                source_video.append(np.array(img))
                validity.append(validity_value)
                gaze_pitchyaw.append(self.img2label_mapping[frame_path]['2D_gaze'])
                count += 1
            else:
                if count == 0:
                    continue
                else:
                    source_video.append(source_video[-1].copy())
                    validity.append(validity_value)
                    gaze_pitchyaw.append(gaze_pitchyaw[-1].copy())
                    count += 1

        if len(source_video) == 0:
            return None

        if self.subset != "train":
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


def step_modulo(current, interval_size):
    return current % interval_size == (interval_size - 1)


def evaluation(test_data, trainer_model, curr_step, tensorboard, device, logger):
    test_losses = RunningStatistics()

    try:
        for param in trainer_model.module.parameters():
            param.requires_grad = False
    except:
        for param in trainer_model.parameters():
            param.requires_grad = False

    for module in trainer_model.children():
        module.train(False)

    torch.cuda.empty_cache()

    for tag, data_dict in test_data.items():
        with torch.no_grad():
            for i, input_data in enumerate(data_dict['dataloader']):

                # Move tensors to GPU
                for k, v in input_data.items():
                    if isinstance(v, torch.Tensor):
                        input_data[k] = v.detach().to(device, non_blocking=True)

                if torch.cuda.device_count() > 1:
                    v_loss_dict, v_out_dict = trainer_model.module.compute_losses(input_data, only_3D=True)
                else:
                    v_loss_dict, v_out_dict = trainer_model.compute_losses(input_data, only_3D=True)

                for k, v in v_loss_dict.items():
                    test_losses.add('%s' % k, v.detach().cpu().numpy())

    test_loss_means = test_losses.means()
    logger.info('Test Losses at [%7d]: %s' %
                 (curr_step, ', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))

    for k, v in test_loss_means.items():
        tensorboard.add_scalar('test/%s' % k, v, curr_step)

    return test_loss_means['total_loss']


def trainer(trainer_model, train_data, test_data, logger, config, optimizer, scheduler, checkpoint_manager, tensorboard,
            device, wandb_logger):

    logger.info('Training')
    running_losses = RunningStatistics()

    max_dataset_len = np.amax([len(data_dict['dataset']) for data_dict in train_data.values()])

    val_loss_best = float('inf')

    for current_step in range(config.load_step, config.num_iter):

        torch.cuda.empty_cache()

        if current_step % config.save_interval == 0 and current_step != config.load_step:
            checkpoint_manager.save_checkpoint(current_step)

        current_epoch = (current_step * config.batch_size) / max_dataset_len  # fractional value

        try:
            for param in trainer_model.module.parameters():
                param.requires_grad = True
        except:
            for param in trainer_model.parameters():
                param.requires_grad = True

        for module in trainer_model.children():
            module.train(True)

        trainer_model.zero_grad()

        # get input data
        full_input_dict = get_training_batches(train_data, device)
        assert len(full_input_dict) == 1  # for now say there's 1 training data source
        full_input_dict = next(iter(full_input_dict.values()))

        if torch.cuda.device_count() > 1:
            loss_dict, output_dict = trainer_model.module.compute_losses(full_input_dict, only_3D=True)
        else:
            loss_dict, output_dict = trainer_model.compute_losses(full_input_dict, only_3D=True)

        loss = loss_dict['total_loss']

        # backward
        loss.backward()

        # Maybe clip gradients
        if config.do_gradient_clipping:
            if config.gradient_clip_by == 'norm':
                clip_func = nn.utils.clip_grad_norm_
            elif config.gradient_clip_by == 'value':
                clip_func = nn.utils.clip_grad_value_
            clip_amount = config.gradient_clip_amount
            clip_func(trainer_model.parameters(), clip_amount)

        optimizer.step()
        scheduler.step()

        for k, v in loss_dict.items():
            running_losses.add('%s' % k, v.item())

        if current_step != 0 and (current_step % config.print_freq_train == 0):
            logger.info('Losses at Step [%8d|%8d] Epoch [%.2f]: %s' % (current_step, config.num_iter,
                                                                       current_epoch, running_losses))
            # log to tensorboard
            for k, v in running_losses.means().items():
                tensorboard.add_scalar('train/%s' % k, v, current_step)
                wandb_logger.log({'train/%s' % k: v})

        if current_step % config.print_freq_test == 0 and current_step > 0:
            torch.cuda.empty_cache()
            # test
            val_loss = evaluation(test_data, trainer_model, current_step, tensorboard, device, logger)
            if val_loss < val_loss_best:
                checkpoint_manager.save_best_checkpoint()
                val_loss_best = val_loss
            torch.cuda.empty_cache()

    logger.info('Finished Training')
    checkpoint_manager.save_checkpoint(config.num_iter)


def main(config):

    gpu_devices = torch.cuda.device_count()

    print(config.model_type)
    model = create_model(config)
    if gpu_devices > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)

    checkpoint_manager = CheckpointsManager(network=model, output_dir=config.save_path)

    if not config.skip_training:

        logger = set_logger(config.save_path)

        ###############
        # load datasets
        logger.info('------- Initializing dataloaders --------')

        train_dataset = Gaze360Loader(source_path=config.gaze360_path, config=config, subset="train", transforms=None)

        train_dataloader = DataLoader(train_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=True,
                                     drop_last=True,
                                     num_workers=config.train_data_workers,
                                     pin_memory=True,
                                     collate_fn=my_collate
                                     )

        val_dataset = Gaze360Loader(source_path=config.gaze360_path, config=config, subset="validation", transforms=None)

        val_dataset.original_full_dataset = val_dataset
        # then subsample datasets for quicker testing
        num_subset = config.test_num_samples
        if len(val_dataset) > num_subset:
            subset = Subset(val_dataset, sorted(np.random.permutation(len(val_dataset))[:num_subset]))
            subset.original_full_dataset = val_dataset
            val_dataset = subset

        val_dataloader = DataLoader(val_dataset,
                                     batch_size=config.batch_size,
                                     shuffle=False,
                                     drop_last=False,
                                     num_workers=config.test_data_workers,
                                     pin_memory=True,
                                     collate_fn=my_collate
                                     )

        train_data = {}
        train_data['gaze360_train'] = {}
        train_data['gaze360_train']['dataset'] = train_dataset
        train_data['gaze360_train']['dataloader'] = train_dataloader

        val_data = {}
        val_data['gaze360_val'] = {}
        val_data['gaze360_val']['dataset'] = val_dataset
        val_data['gaze360_val']['dataloader'] = val_dataloader
        ###############
        # initialize network and checkpoint manager

        logger.info('------- Initializing model --------')

        wandb.watch(model, log='all')

        # Print model details
        parameters_ = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters_]) / 1_000_000
        logger.info('Trainable Parameters: %.3fM' % parameters)

        if config.load_step != 0:
            logger.info('Loading available model at step {}'.format(config.load_step))
            checkpoint_manager.load_checkpoint(config.load_step)

        #####################################################
        # initialize optimizer, scheduler and tensorboard

        tensorboard = SummaryWriter(logdir=config.save_path)

        if config.opt == 'adam':
            optimizer = torch.optim.Adam(
                    params=model.parameters(),
                    lr=config.learning_rate,
                    weight_decay=config.weight_decay)
        else:
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_iter,
                                                               eta_min=0, last_epoch=-1)
        #####################################################
        # call trainer

        trainer(model, train_data, val_data, logger, config, optimizer,
                scheduler, checkpoint_manager, tensorboard, device, wandb)

    #####################################################
    # call tester
    if config.load_checkpoint_path is not None:
        checkpoint_manager.load_checkpoint(config.load_checkpoint_path)
        with open(os.path.join(config.save_path, "test_errors.txt"), "a") as text_file:
            text_file.write('Checkpoint {}'.format(config.load_checkpoint_path))
    else:
        checkpoint_manager.load_checkpoint(config.num_iter)
        with open(os.path.join(config.save_path, "test_errors.txt"), "a") as text_file:
            text_file.write('Checkpoint {}'.format(config.num_iter))

    print(' -------- Testing on Gaze360 Test dataset ------------- ')
    Tester.gaze360_test(args=config, trainer=model, device=device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a video-based gaze estimation.')
    parser.add_argument('--config_json', type=str, help='Path to config in JSON format')
    parser.add_argument('--skip_training', action='store_true', help='skip_training')
    parser.add_argument('--opt', default="sgd", type=str, help='optimizer')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save files')
    parser.add_argument('--load_checkpoint_path', type=int, default=None, help='Path to test checkpoint')
    parser.add_argument('--load_step', type=int, default=0, help='Path to test checkpoint')
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

    config.learning_rate = config.base_learning_rate * config.batch_size
    config.tanh = False

    if not config.skip_training:
        os.makedirs(config.save_path, exist_ok=True)
        # writing config
        save_configs(config.save_path, config)
        print('Written Config file at %s' % config.save_path)
        ###############

        run = wandb.init(entity="name", project="STAGE", name=config.save_path, sync_tensorboard=True)
        wandb.config.update(vars(config))
        run.log_code(".")

        ###############

    main(config)
