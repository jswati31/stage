
import torch
import torch.nn as nn
import torchvision
from .common import load_gazeclr_pretrained_weights, Identity, SequenceWise
from .losses import AngularLoss, L1LossWithValidity
from utils.core_utils import to_screen_coordinates
import numpy as np
from .sam_modules import *
from .transformer_blocks import GPT2Model


half_pi = 0.5 * np.pi

gazeclr_weights_path = "gazeclr_weights/gazeclr_inv_equiv.pth.tar"


def build_model():

    cnn = torchvision.models.resnet18(pretrained=False)
    cnn.fc = Identity()

    print("*******Loading pretrained GazeCLR model********")
    gazeclr_weights = load_gazeclr_pretrained_weights(gazeclr_weights_path)
    cnn.load_state_dict(gazeclr_weights)

    layers = [
        cnn.conv1,
        cnn.bn1,
        cnn.relu,
        cnn.maxpool,
    ]

    for i in range(3):
        name = 'layer%d' % (i + 1)
        layers.append(getattr(cnn, name))
    model = torch.nn.Sequential(*layers)
    return model


class STAGE_Transformer(nn.Module):

    def __init__(self, config):
        super(STAGE_Transformer, self).__init__()

        self.config = config
        self.input_type = config.camera_frame_type

        self.conv_feature_extractor = build_model()

        if self.config.spatial_model == "dual":
            self.spatial_info = AddSpatialInfo()
            self.spatial_attention_difference = Dual_SAM(2*256+4, 64)
            embed_dim_inp = 3 * 256 + 6

        elif self.config.spatial_model == "cross":
            self.spatial_attention_difference = Cross_SAM(feature_dim=256, dropout=0.1, h=8, w=8)
            embed_dim_inp = 512*3

        elif self.config.spatial_model == "proposed":
            self.spatial_attention_difference = Proposed_SAM(feature_dim=256, dropout=0.1, h=8, w=8)
            embed_dim_inp = 512*3

        elif self.config.spatial_model == "proposed_nodual":
            self.spatial_attention_difference = Proposed_SAM_NoDual(feature_dim=256, dropout=0.1, h=8, w=8)
            embed_dim_inp = 512*3

        else:
            raise NotImplemented("Define spatial model from [dual, cross, proposed, proposed_nodual]")

        self.embed = nn.Sequential(
            nn.Linear(embed_dim_inp, config.input_num_features),
            nn.ReLU()
        )

        inp_num_features = config.input_num_features if not self.config.use_head_pose else config.input_num_features+2
        self.encoder = GPT2Model(num_frames=config.max_sequence_len,
                                 embedding_dim=inp_num_features,
                                 n_layers=config.n_layers,
                                 dk=config.key_dim,
                                 n_head=config.n_heads,
                                 args=config)

        if config.tanh:
            fc_to_gaze = nn.Sequential(
                nn.Linear(inp_num_features, inp_num_features),
                nn.SELU(inplace=True),
                nn.Linear(inp_num_features, 2, bias=False),
                nn.Tanh(),
            )

            nn.init.zeros_(fc_to_gaze[-2].weight)

        else:

            fc_to_gaze = nn.Sequential(
                nn.Linear(inp_num_features, inp_num_features),
                nn.SELU(inplace=True),
                nn.Linear(inp_num_features, 2),
            )

        self.gaze_output_layer = SequenceWise(fc_to_gaze)

        self.angular_criterion = AngularLoss()
        self.l1_criterion = L1LossWithValidity()

    def frame_features(self, x):
        b, t, c, h, w = x.size()

        x = x.view(b * t, c, h, w)
        x = self.conv_feature_extractor(x)
        inter_Feats = x

        if self.config.spatial_model == "dual":
            # positional embeddings
            x = self.spatial_info(x)

        _, K, h1, w1 = x.shape
        x = x.view(b, t, K, h1, w1)
        x_t_1 = torch.cat((x[:, 0, :, :, :].unsqueeze(1), x[:, :-1, :, :, :]), dim=1)

        return x_t_1, x, inter_Feats

    def forward(self, input_dict, output_dict):

        x = input_dict['face_patch']

        b, t, c, h, w = x.size()
        x_t_1, x_t, inter_Feats = self.frame_features(x)    #(x_(t-1), x_t, intermediate_feats)

        output_dict['feats'] = inter_Feats.contiguous().view(b, t, -1)

        x = self.spatial_attention_difference(x_t_1, x_t)

        x = self.embed(x)

        if self.config.use_head_pose:
            x = torch.cat((x, input_dict['face_h']), dim=-1)

        x, _ = self.encoder(x)
        x = self.gaze_output_layer(x)

        # For gaze, the range of output values are limited by a tanh and scaling
        # (use TanhH- depends on dataset trained on)
        gaze_prediction = half_pi * x if self.config.tanh else x
        output_dict['pred'] = gaze_prediction

        return output_dict

    def get_pog_predictions(self, input_dict, output_dict):

        _, t, _ = output_dict['pred'].shape

        g_origin = input_dict['left_o'] if self.input_type == 'eyes' else input_dict['face_o']
        g_rot = input_dict['left_R'] if self.input_type == 'eyes' else input_dict['face_R']

        all_pog_mm = []
        all_pog_px = []
        for j in range(t):
            pog_mm, pog_px = to_screen_coordinates(g_origin[:, j], output_dict['pred'][:, j], g_rot[:, j],
                                                   input_dict['inv_camera_transformation'][:, j],
                                                   input_dict['pixels_per_millimeter'][:, j])
            all_pog_mm.append(pog_mm)
            all_pog_px.append(pog_px)

        all_pog_mm = torch.stack(all_pog_mm, dim=1)
        all_pog_px = torch.stack(all_pog_px, dim=1)

        assert all_pog_mm.shape[1] == t

        output_dict['pred_pog_mm'] = all_pog_mm
        output_dict['pred_pog_px'] = all_pog_px
        output_dict['pred_pog_cm'] = 0.1*all_pog_mm

        return output_dict

    def compute_losses(self, input_dict, only_3D=False):

        losses_dict = {}
        output_dict = {}

        output_dict = self.forward(input_dict, output_dict)

        gt_key = 'face_g_tobii'
        pog_gt_key = 'face_PoG_tobii'

        loss = self.angular_criterion(output_dict['pred'], gt_key, input_dict)
        losses_dict['3D_gaze_loss'] = loss

        total_loss = loss

        if only_3D:
            losses_dict['total_loss'] = losses_dict['3D_gaze_loss']
            return losses_dict, output_dict

        # 2D PoG loss
        if self.config.w_pog_loss > 0:
            output_dict = self.get_pog_predictions(input_dict, output_dict)

            input_dict['face_PoG_cm_tobii'] = torch.mul(
                input_dict[pog_gt_key],
                0.1 * input_dict['millimeters_per_pixel'], ).detach()
            input_dict['face_PoG_cm_tobii_validity'] = input_dict[pog_gt_key + '_validity']

            pog_loss = self.l1_criterion(output_dict['pred_pog_cm'], 'face_PoG_cm_tobii',
                                          input_dict)

            total_loss = total_loss + self.config.w_pog_loss * pog_loss
            losses_dict['pog_loss'] = pog_loss

        losses_dict['total_loss'] = total_loss

        return losses_dict, output_dict
