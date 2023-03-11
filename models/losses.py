import numpy as np
import torch
import torch.nn.functional as F
from utils.core_utils import pitchyaw_to_vector


def nn_angular_distance(a, b):
    sim = F.cosine_similarity(a, b, dim=1, eps=1e-6)
    sim = F.hardtanh(sim, -1.0, 1.0)
    return torch.acos(sim) * (180 / np.pi)


def gaze_angular_loss(y, y_hat):
    y = pitchyaw_to_vector(y)
    y_hat = pitchyaw_to_vector(y_hat)
    loss = nn_angular_distance(y, y_hat)
    return torch.mean(loss)


class BaseLossWithValidity(object):

    def calculate_loss(self, predictions, ground_truth):
        raise NotImplementedError('Must implement BaseLossWithValidity::calculate_loss')

    def calculate_mean_loss(self, predictions, ground_truth):
        return torch.mean(self.calculate_loss(predictions, ground_truth))

    def __call__(self, predictions, gt_key, reference_dict):
        # Since we deal with sequence data, assume B x T x F (if ndim == 3)
        batch_size = predictions.shape[0]

        individual_entry_losses = []
        num_valid_entries = 0

        for b in range(batch_size):
            # Get sequence data for predictions and GT
            entry_predictions = predictions[b]
            entry_ground_truth = reference_dict[gt_key][b]

            # If validity values do not exist, return simple mean
            # NOTE: We assert for now to catch unintended errors,
            #       as we do not expect a situation where these flags do not exist.
            validity_key = gt_key + '_validity'
            assert(validity_key in reference_dict)
            # if validity_key not in reference_dict:
            #     individual_entry_losses.append(torch.mean(
            #         self.calculate_mean_loss(entry_predictions, entry_ground_truth)
            #     ))
            #     continue

            # Otherwise, we need to set invalid entries to zero
            validity = reference_dict[validity_key][b].float()
            losses = self.calculate_loss(entry_predictions, entry_ground_truth)

            # Some checks to make sure that broadcasting is not hiding errors
            # in terms of consistency in return values
            assert(validity.ndim == losses.ndim)
            assert(validity.shape[0] == losses.shape[0])

            # Make sure to scale the accumulated loss correctly
            num_valid = torch.sum(validity)
            accumulated_loss = torch.sum(validity * losses)
            if num_valid > 1:
                accumulated_loss /= num_valid
            num_valid_entries += 1
            individual_entry_losses.append(accumulated_loss)
            # print(validity * losses, accumulated_loss, num_valid)

        # Merge all loss terms to yield final single scalar
        return torch.sum(torch.stack(individual_entry_losses)) / float(num_valid_entries)

class AngularLoss(BaseLossWithValidity):

    _to_degrees = 180. / np.pi

    def calculate_loss(self, a, b):
        a = pitchyaw_to_vector(a)
        b = pitchyaw_to_vector(b)
        sim = F.cosine_similarity(a, b, dim=1, eps=1e-8)
        sim = F.hardtanh_(sim, min_val=-1+1e-8, max_val=1-1e-8)
        return torch.acos(sim) * self._to_degrees


class L1LossWithValidity(object):

    def calculate_loss(self, predictions, ground_truth):
        return torch.nn.L1Loss(reduction='none')(predictions, ground_truth)

    def calculate_mean_loss(self, predictions, ground_truth):
        return torch.mean(self.calculate_loss(predictions, ground_truth))

    def __call__(self, predictions, ground_truth_key, all_dict):
        # Since we deal with sequence data, assume B x T x F (if ndim == 3)
        all_validity = all_dict[ground_truth_key+'_validity']
        ground_truth = all_dict[ground_truth_key]

        batch_size = predictions.shape[0]

        individual_entry_losses = []
        num_valid_entries = 0

        for b in range(batch_size):
            # Get sequence data for predictions and GT
            entry_predictions = predictions[b]
            entry_ground_truth = ground_truth[b]

            validity = all_validity[b]

            # Otherwise, we need to set invalid entries to zero
            validity = validity.float()
            losses = self.calculate_loss(entry_predictions, entry_ground_truth)
            losses = losses.sum(1)

            # Some checks to make sure that broadcasting is not hiding errors
            # in terms of consistency in return values
            assert(validity.ndim == losses.ndim)
            assert(validity.shape[0] == losses.shape[0])

            # Make sure to scale the accumulated loss correctly
            num_valid = torch.sum(validity)
            accumulated_loss = torch.sum(validity * losses)
            if num_valid > 1:
                accumulated_loss /= num_valid
            num_valid_entries += 1
            individual_entry_losses.append(accumulated_loss)

        # Merge all loss terms to yield final single scalar
        return torch.sum(torch.stack(individual_entry_losses)) / float(num_valid_entries)
