"""
Main script to run cross-dataset experiments
"""

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
warnings.filterwarnings('ignore')
import os
import json
import torch
import numpy as np
import argparse
from models import create_model
from argparse import Namespace
from tensorboardX import SummaryWriter
import core.trainer as T
import core.tester as Tester
from utils.core_utils import set_logger, save_configs
from utils.train_utils import init_datasets
from utils.checkpoints_manager import CheckpointsManager
from datasources.EVE import EVEDataset_train, EVEDataset_val

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        train_data_class = EVEDataset_train
        val_data_class = EVEDataset_val

        train_dataset_paths = [
            ('eve_train', train_data_class, config.datasrc_eve, config.train_stimuli, config.train_cameras),
        ]
        validation_dataset_paths = [
            ('eve_val', val_data_class, config.datasrc_eve, config.test_stimuli, config.test_cameras),
        ]

        train_data, test_data = init_datasets(train_dataset_paths, validation_dataset_paths, config, logger,
                                              data_transforms=None)

        ###############
        # initialize network and checkpoint manager

        logger.info('------- Initializing model --------')

        # Print model details
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
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
        elif config.opt == 'adamw':
            optimizer = torch.optim.AdamW(params=model.parameters(),
                                          lr=config.learning_rate,
                                          weight_decay=config.weight_decay,
                                          betas=config.betas)
        else:
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
                momentum=0.9)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_iter, eta_min=0, last_epoch=-1)
        #####################################################
        # call trainer
        T.trainer(model, train_data, test_data, logger, config, optimizer,
                  scheduler, checkpoint_manager, tensorboard, device)

    #####################################################
    # call tester
    if config.load_checkpoint_path is not None:
        checkpoint_manager.load_checkpoint_frompath(config.load_checkpoint_path)
        with open(os.path.join(config.save_path, "test_errors.txt"), "a") as text_file:
            text_file.write('Checkpoint {}'.format(config.load_checkpoint_path))
    else:
        checkpoint_manager.load_checkpoint(config.num_iter)
        with open(os.path.join(config.save_path, "test_errors.txt"), "a") as text_file:
            text_file.write('Checkpoint {}'.format(config.num_iter))

    print(' -------- Testing on EVE Test dataset ------------- ')
    _losses = Tester.eve_test(args=config, trainer=model, device=device)
    acc_str = ', '.join(['%s: %.6f\n' % (k, _losses[k]) for k, v in _losses.items()])
    with open(os.path.join(config.save_path, "test_errors.txt"), "a") as text_file:
        text_file.write('EVE dataset: {}'.format(acc_str))

    print(' -------- Testing on Eyediap Test dataset ------------- ')
    _losses = Tester.eyediap_test(args=config, trainer=model, device=device)
    acc_str = ', '.join(['%s: %.6f\n' % (k, _losses[k]) for k, v in _losses.items()])
    with open(os.path.join(config.save_path, "test_errors.txt"), "a") as text_file:
        text_file.write('Eyediap dataset: {}'.format(acc_str))

    print(' -------- Testing on Gaze360 Test dataset ------------- ')
    Tester.gaze360_test(args=config, trainer=model, device=device)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train a video-based gaze estimation.')
    parser.add_argument('--config_json', type=str, help='Path to config in JSON format')
    parser.add_argument('--skip_training', action='store_true', help='skip_training')
    parser.add_argument('--opt', default="sgd", type=str, help='optimizer')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save files')
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help='Path to test checkpoint')
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
    config.tanh = True

    if not config.skip_training:
        os.makedirs(config.save_path, exist_ok=True)
        # writing config
        save_configs(config.save_path, config)
        print('Written Config file at %s' % config.save_path)
        ###############

    main(config)
