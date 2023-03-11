import torch
import numpy as np
from utils.core_utils import RunningStatistics
from utils.train_utils import get_training_batches
import torch.nn as nn

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
        full_input_dict = get_training_batches(train_data, device, config.augmentation)
        assert len(full_input_dict) == 1  # for now say there's 1 training data source
        full_input_dict = next(iter(full_input_dict.values()))

        if torch.cuda.device_count() > 1:
            loss_dict, output_dict = trainer_model.module.compute_losses(full_input_dict)
        else:
            loss_dict, output_dict = trainer_model.compute_losses(full_input_dict)

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
        scheduler.step(current_step+1)

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
            evaluation(test_data, trainer_model, current_step, tensorboard, device, logger)

    logger.info('Finished Training')
    checkpoint_manager.save_checkpoint(config.num_iter)
