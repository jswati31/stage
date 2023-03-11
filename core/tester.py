
import os
import torch
from utils.core_utils import RunningStatistics
from datasources.EVE import EVEDataset_test
from datasources.Gaze360 import Gaze360Loader
from datasources.Eyediap import EyediapLoader
from torch.utils.data import DataLoader
from utils.train_utils import my_collate
from tqdm import tqdm


def eve_test(args, trainer, device):

    test_dataset = EVEDataset_test(args.datasrc_eve,
                                   config=args,
                                   cameras_to_use=args.train_cameras,
                                   types_of_stimuli=args.train_stimuli, transforms=None)

    test_dataloader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=args.test_data_workers,
                            pin_memory=True,
                            collate_fn=my_collate
                            )

    test_losses = RunningStatistics()
    try:
        for param in trainer.module.parameters():
            param.requires_grad = False
    except:
        for param in trainer.parameters():
            param.requires_grad = False

    for module in trainer.children():
        module.train(False)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, input_data in tqdm(enumerate(test_dataloader)):

            # Move tensors to GPU
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    input_data[k] = v.detach().to(device, non_blocking=True)

            if torch.cuda.device_count() > 1:
                v_loss_dict, v_out_dict = trainer.module.compute_losses(input_data)
            else:
                v_loss_dict, v_out_dict = trainer.compute_losses(input_data)

            for k, v in v_loss_dict.items():
                test_losses.add('%s' % k, v.detach().cpu().numpy())

    test_loss_means = test_losses.means()
    print('Test Losses for EVE test data %s' % (', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))

    return test_loss_means


def gaze360_test(args, trainer, device):

    print("**** Testing for FULL Gaze360******")
    test_dataset = Gaze360Loader(source_path=args.gaze360_path,
                                 config=args,
                                 subset='test'
                                 )

    test_dataloader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=args.test_data_workers,
                            pin_memory=True,
                            collate_fn=my_collate
                            )

    test_losses = RunningStatistics()
    try:
        for param in trainer.module.parameters():
            param.requires_grad = False
    except:
        for param in trainer.parameters():
            param.requires_grad = False

    for module in trainer.children():
        module.train(False)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, input_data in tqdm(enumerate(test_dataloader)):

            if input_data is None:
                continue

            # Move tensors to GPU
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    input_data[k] = v.detach().to(device, non_blocking=True)

            if torch.cuda.device_count() > 1:
                v_loss_dict, v_out_dict = trainer.module.compute_losses(input_data, only_3D=True)
            else:
                v_loss_dict, v_out_dict = trainer.compute_losses(input_data, only_3D=True)

            for k, v in v_loss_dict.items():
                test_losses.add('%s' % k, v.detach().cpu().numpy())

    test_loss_means = test_losses.means()

    acc_str = ', '.join(['%s: %.6f\n' % (k, test_loss_means[k]) for k, v in test_loss_means.items()])
    print('Test Losses for full Gaze360 test data %s' % (acc_str))
    with open(os.path.join(args.save_path, "test_errors.txt"), "a") as text_file:
        text_file.write('Full Gaze360 dataset: {}'.format(acc_str))

    # del test_losses

    print("**** Testing for Front Facing Gaze360 180 ******")
    test_dataset = Gaze360Loader(source_path=args.gaze360_path,
                                 config=args,
                                 subset='test',
                                 angle=90)

    test_dataloader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=args.test_data_workers,
                            pin_memory=True,
                            collate_fn=my_collate
                            )

    test_losses = RunningStatistics()
    try:
        for param in trainer.module.parameters():
            param.requires_grad = False
    except:
        for param in trainer.parameters():
            param.requires_grad = False

    for module in trainer.children():
        module.train(False)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, input_data in tqdm(enumerate(test_dataloader)):

            if input_data is None:
                continue

            # Move tensors to GPU
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    input_data[k] = v.detach().to(device, non_blocking=True)

            if torch.cuda.device_count() > 1:
                v_loss_dict, v_out_dict = trainer.module.compute_losses(input_data, only_3D=True)
            else:
                v_loss_dict, v_out_dict = trainer.compute_losses(input_data, only_3D=True)

            for k, v in v_loss_dict.items():
                test_losses.add('%s' % k, v.detach().cpu().numpy())

    test_loss_means = test_losses.means()

    acc_str = ', '.join(['%s: %.6f\n' % (k, test_loss_means[k]) for k, v in test_loss_means.items()])
    print('Test Losses for range 180  Gaze360 test data %s' % (acc_str))
    with open(os.path.join(args.save_path, "test_errors.txt"), "a") as text_file:
        text_file.write('range 180  Gaze360 dataset: {}'.format(acc_str))

    del test_losses

    print("**** Testing for Gaze360 with 20 ******")
    test_dataset = Gaze360Loader(source_path=args.gaze360_path,
                                 config=args,
                                 subset='test',
                                 angle=20)

    test_dataloader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=args.test_data_workers,
                            pin_memory=True,
                            collate_fn=my_collate
                            )

    test_losses = RunningStatistics()
    try:
        for param in trainer.module.parameters():
            param.requires_grad = False
    except:
        for param in trainer.parameters():
            param.requires_grad = False

    for module in trainer.children():
        module.train(False)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, input_data in tqdm(enumerate(test_dataloader)):

            if input_data is None:
                continue

            # Move tensors to GPU
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    input_data[k] = v.detach().to(device, non_blocking=True)

            if torch.cuda.device_count() > 1:
                v_loss_dict, v_out_dict = trainer.module.compute_losses(input_data, only_3D=True)
            else:
                v_loss_dict, v_out_dict = trainer.compute_losses(input_data, only_3D=True)

            for k, v in v_loss_dict.items():
                test_losses.add('%s' % k, v.detach().cpu().numpy())

    test_loss_means = test_losses.means()

    acc_str = ', '.join(['%s: %.6f\n' % (k, test_loss_means[k]) for k, v in test_loss_means.items()])
    print('Test Losses for range 20  Gaze360 test data %s' % (acc_str))
    with open(os.path.join(args.save_path, "test_errors.txt"), "a") as text_file:
        text_file.write('range 20  Gaze360 dataset: {}'.format(acc_str))

    return test_loss_means


def eyediap_test(args, trainer, device):

    test_dataset = EyediapLoader(source_path=args.eyediap_path,
                                 config=args, transforms=None)

    test_dataloader = DataLoader(test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            num_workers=args.test_data_workers,
                            pin_memory=True,
                            collate_fn=my_collate
                            )

    test_losses = RunningStatistics()
    try:
        for param in trainer.module.parameters():
            param.requires_grad = False
    except:
        for param in trainer.parameters():
            param.requires_grad = False

    for module in trainer.children():
        module.train(False)
    torch.cuda.empty_cache()

    with torch.no_grad():
        for i, input_data in tqdm(enumerate(test_dataloader)):

            # Move tensors to GPU
            for k, v in input_data.items():
                if isinstance(v, torch.Tensor):
                    input_data[k] = v.detach().to(device, non_blocking=True)

            if torch.cuda.device_count() > 1:
                v_loss_dict, v_out_dict = trainer.module.compute_losses(input_data, only_3D=True)
            else:
                v_loss_dict, v_out_dict = trainer.compute_losses(input_data, only_3D=True)

            for k, v in v_loss_dict.items():
                test_losses.add('%s' % k, v.detach().cpu().numpy())

    test_loss_means = test_losses.means()
    print('Test Losses for Eyediap test data %s' % (', '.join(['%s: %.6f' % v for v in test_loss_means.items()])))

    return test_loss_means
