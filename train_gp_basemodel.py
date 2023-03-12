"""
script to train GP base model

"""

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import warnings
warnings.filterwarnings('ignore')
import gc
import os
import json
import torch
import numpy as np
import argparse
import gpytorch
from models import create_model
from argparse import Namespace
from torch.utils.data import DataLoader
from utils.train_utils import my_collate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Define the Model
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


def train_gp(train_data, trainer_model, device, config):
    trainer_model.eval()
    torch.cuda.empty_cache()

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)
    gp_model = BatchedGP(None, None, likelihood, 2)

    # Find optimal model hyperparameters
    gp_model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(gp_model.parameters(), lr=0.1)

    gp_model = gp_model.to(device)
    likelihood = likelihood.to(device)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, gp_model)

    mll = mll.to(device)

    epochs = 2
    iter_num = 0
    for param in trainer_model.module.parameters():
        param.requires_grad = False

    total_iters = 50000
    for ep in range(epochs):
        # train for iterations = 50000
        if iter_num == total_iters:
            break

        for batch_num, input_data in enumerate(train_data):
            if input_data is None:
                continue
            if iter_num == total_iters:
                break

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
            true_gaze = input_data['face_g_tobii'].detach().squeeze(0)
            x_feats = v_out_dict['feats'].detach().squeeze(0)

            x_train_gp = x_feats
            y_train_gp = true_gaze
            y_pred_gp = pred_global_gaze

            gp_model_input_true = y_train_gp - y_pred_gp

            train_x = x_train_gp
            train_y = gp_model_input_true

            gp_model.set_train_data(train_x, train_y, strict=False)

            # Define training helper function
            optimizer.zero_grad()  # Zero gradients
            output = gp_model(train_x)  # Compute noise-free output
            loss = -mll(output, train_y).sum()  # Compute batched loss
            loss.backward()  # Compute gradients with backpropagation
            print("Iter %d/%d/%d - Loss: %.3f" % (batch_num + 1, ep, epochs, loss.item()))

            optimizer.step()  # Update weights with gradients
            optimizer.zero_grad()  # Zero gradients
            gc.collect()  # Used to ensure there is no memory leak

            torch.cuda.empty_cache()

            iter_num += 1

    torch.save(gp_model.state_dict(), config.gp_model_name + '.pth')


def train_gp_eve(config, model):
    from datasources.EVE import EVEDataset_train

    train_dataset = EVEDataset_train(config.datasrc_eve,
                                  config=config,
                                  cameras_to_use=config.train_cameras,
                                  types_of_stimuli=config.train_stimuli, transforms=None)

    train_dataloader = DataLoader(train_dataset,
                                 batch_size=config.batch_size,
                                 shuffle=True,
                                 drop_last=False,
                                 num_workers=config.test_data_workers,
                                 pin_memory=True,
                                 collate_fn=my_collate
                                  )

    # train GP base model on EVE train
    train_gp(train_dataloader, model, device, config)


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

    # EVE GP training
    train_gp_eve(config, model)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train GP base model.')

    parser.add_argument('--config_json', type=str, help='Path to config in JSON format')
    parser.add_argument('--gp_model_name', type=str, help='name of base GP model')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
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
