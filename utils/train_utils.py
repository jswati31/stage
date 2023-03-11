
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader, Subset
import numpy as np
from .core_utils import my_collate

def get_training_batches(train_data_dicts, device, augm=False):
    """Get training batches of data from all training data sources."""
    out = {}
    for tag, data_dict in train_data_dicts.items():
        if 'data_iterator' not in data_dict:
            data_dict['data_iterator'] = iter(data_dict['dataloader'])
        # Try to get data
        while True:
            try:
                out[tag] = next(data_dict['data_iterator'])
                break
            except StopIteration:
                del data_dict['data_iterator']
                torch.cuda.empty_cache()
                if augm is True:
                    data_dict['dataloader'].dataset.increment_difficulty()
                data_dict['data_iterator'] = iter(data_dict['dataloader'])

        # Move tensors to GPU
        for k, v in out[tag].items():
            if isinstance(v, torch.Tensor):
                out[tag][k] = v.detach()
                if k != 'screen_full_frame':
                    out[tag][k] = out[tag][k].to(device, non_blocking=True)
            else:
                out[tag][k] = v
    return out


def init_datasets(train_specs, test_specs, config, logger, data_transforms=None):

    # Initialize training datasets
    train_data = OrderedDict()
    for tag, dataset_class, path, stimuli, cameras in train_specs:
        dataset = dataset_class(path, config=config,
                                transforms=data_transforms,
                                cameras_to_use=cameras,
                                types_of_stimuli=stimuli)
        dataset.original_full_dataset = dataset
        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                drop_last=True,
                                num_workers=config.train_data_workers,
                                pin_memory=True,
                                collate_fn=my_collate
                                )

        train_data[tag] = {
            'dataset': dataset,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use training dataset: %s' % tag)
        logger.info('          with number of videos: %d' % len(dataset))

    # Initialize test datasets
    test_data = OrderedDict()
    for tag, dataset_class, path, stimuli, cameras in test_specs:
        # Get the full dataset
        dataset = dataset_class(path, config=config,
                                transforms=data_transforms,
                                cameras_to_use=cameras,
                                types_of_stimuli=stimuli,
                                live_validation=True)

        dataset.original_full_dataset = dataset
        # then subsample datasets for quicker testing
        num_subset = config.test_num_samples
        if len(dataset) > num_subset:
            subset = Subset(dataset, sorted(np.random.permutation(len(dataset))[:num_subset]))
            subset.original_full_dataset = dataset
            dataset = subset

        dataloader = DataLoader(dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                num_workers=config.test_data_workers,
                                pin_memory=True,
                                collate_fn=my_collate,
                                )

        test_data[tag] = {
            'dataset': dataset,
            'dataset_class': dataset_class,
            'dataset_path': path,
            'dataloader': dataloader,
        }
        logger.info('> Ready to use evaluation dataset: %s' % tag)
        logger.info('           with number of entries: %d' % len(dataset.original_full_dataset))
        if dataset.original_full_dataset != dataset:
            logger.info('     of which we evaluate on just: %d' % len(dataset))

    return train_data, test_data
