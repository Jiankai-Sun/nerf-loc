# Copyright (c) Facebook, Inc. and its affiliates.
import copy

import math
import os
from functools import partial
import random
import time, cv2, pickle
import numpy as np
import torch
import torch.nn as nn
# from third_party.pointnet2.pointnet2_modules import PointnetSAModuleVotes
# from third_party.pointnet2.pointnet2_utils import furthest_point_sample
from utils.pc_util import scale_points, shift_scale_points
from utils.nerf_utils import create_nerf_model, load_nerf_model, render_path
from typing import List, Tuple

from models.helpers import GenericMLP
from models.position_embedding import PositionEmbeddingCoordsSine
from models.transformer import (MaskedTransformerEncoder, TransformerDecoder,
                                TransformerDecoderLayer, TransformerEncoder,
                                TransformerEncoderLayer)
import torchvision.models as models
# import queue

class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        # print('query_xyz.shape: ', query_xyz.shape)  # torch.Size([8, 128, 4])
        center_unnormalized = query_xyz[:, :, :3] + center_offset
        # center_normalized = shift_scale_points(
        #     center_unnormalized, src_range=point_cloud_dims
        # )
        center_normalized = center_unnormalized
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = size_normalized
        # size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(
                2, pred_angle_class.unsqueeze(-1)
            ).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(
        self, box_center_unnorm, box_size_unnorm, box_angle
    ):
        return self.dataset_config.box_parametrization_to_corners(
            box_center_unnorm, box_size_unnorm, box_angle
        )


class Model3DETR(nn.Module):
    """
    Main 3DETR model. Consists of the following learnable sub-models
    - pre_encoder: takes raw point cloud, subsamples it and projects into "D" dimensions
                Input is a Nx3 matrix of N point coordinates
                Output is a N'xD matrix of N' point features
    - encoder: series of self-attention blocks to extract point features
                Input is a N'xD matrix of N' point features
                Output is a N''xD matrix of N'' point features.
                N'' = N' for regular encoder; N'' = N'//2 for masked encoder
    - query computation: samples a set of B coordinates from the N'' points
                and outputs a BxD matrix of query features.
    - decoder: series of self-attention and cross-attention blocks to produce BxD box features
                Takes N''xD features from the encoder and BxD query features.
    - mlp_heads: Predicts bounding box parameters and classes from the BxD box features
    """

    def __init__(
        self,
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        render_kwargs_test=None,
        encoder_dim=256,
        decoder_dim=256,
        position_embedding="fourier",
        mlp_dropout=0.3,
        num_queries=256,
        use_buffer=True,
        test_only=False,
        args=None,
    ):
        super().__init__()
        self.render_kwargs_test = render_kwargs_test
        self.nerf_ckpt_path = None
        self.use_buffer = use_buffer
        self.nerf_buffer_type = 'dict'  # 'list'
        self.test_only = test_only
        self.args = args
        self.total_rendering_time = 0
        self.rendering_time_per_batch = 0
        self.coarse_total_rendering_time = 0
        self.coarse_rendering_time_per_batch = 0
        self.split = 'train'
        if self.nerf_buffer_type == 'list':
            self.nerf_buffer = []  # queue.Queue(self.buffer_his_length)
            self.buffer_his_length = 300
        elif self.nerf_buffer_type == 'dict':
            self.nerf_buffer = {}
            self.buffer_his_length = min(dataset_config.num_scan_samples+2, 300)
        self.forward_idx = 0
        if self.args.arch_type not in ['coarse']:
            if dataset_config.model_type == 'resnet18':
                self.resnet = models.resnet18()
                num_ftrs = self.resnet.fc.in_features
                self.resnet.fc = nn.Linear(num_ftrs, decoder_dim)
                if dataset_config.input_type == 'nerf':
                    # print(self.resnet)
                    # num_ftrs = self.resnet.conv1.in_features
                    self.resnet.conv1 = nn.Conv2d(128 * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.corner_head = nn.Linear(decoder_dim, 8 * 3)
                self.semcls_head = nn.Linear(decoder_dim, dataset_config.num_semcls + 1)
            elif dataset_config.model_type == 'mlp':
                self.pre_encoder = pre_encoder
                self.encoder = encoder
                if hasattr(self.encoder, "masking_radius"):
                    hidden_dims = [encoder_dim]
                else:
                    hidden_dims = [encoder_dim, encoder_dim]
                self.encoder_to_decoder_projection = GenericMLP(
                    input_dim=encoder_dim,
                    hidden_dims=hidden_dims,
                    output_dim=decoder_dim,
                    norm_fn_name="bn1d",
                    activation="relu",
                    use_conv=True,
                    output_use_activation=True,
                    output_use_norm=True,
                    output_use_bias=False,
                )
                self.avgpool = nn.AdaptiveAvgPool1d((1, ))
                self.corner_head = nn.Linear(decoder_dim, 8 * 3)
                self.semcls_head = nn.Linear(decoder_dim, dataset_config.num_semcls + 1)
            else:
                self.pre_encoder = pre_encoder
                self.encoder = encoder
                self.pos_embedding = PositionEmbeddingCoordsSine(
                    d_pos=decoder_dim, pos_type=position_embedding, normalize=True
                )
                self.query_projection = GenericMLP(
                    input_dim=decoder_dim,
                    hidden_dims=[decoder_dim],
                    output_dim=decoder_dim,
                    use_conv=True,
                    output_use_activation=True,
                    hidden_use_bias=True,
                )
                self.decoder = decoder
                self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)

                self.box_processor = BoxProcessor(dataset_config)

        if self.args.arch_type in ['coarse_fine', 'coarse', 'coarse2fine']: # , 'coarse', 'coarse2fine']:
            if dataset_config.model_type == 'resnet18':
                self.coarse_resnet = models.resnet18()
                num_ftrs = self.coarse_resnet.fc.in_features
                self.coarse_resnet.fc = nn.Linear(num_ftrs, decoder_dim)
                if dataset_config.input_type == 'nerf':
                    self.coarse_resnet.conv1 = nn.Conv2d(128 * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                  bias=False)
                if self.args.arch_type in ['coarse_fine', 'coarse2fine']:
                    if self.args.fusion_type in ['self_attn']:
                        self.norm1 = nn.LayerNorm(args.enc_dim)
                        self.norm3 = nn.LayerNorm(args.enc_dim)
                        # Implementation of Feedforward model
                        self.linear1 = nn.Linear(args.enc_dim, args.enc_dim)
                        self.linear2 = nn.Linear(args.enc_dim, args.enc_dim)
                        self.activation = nn.ReLU()
                        self.fuse_layer = nn.MultiheadAttention(args.enc_dim, args.enc_nhead, dropout=args.enc_dropout)
                    elif self.args.fusion_type in ['mlp']:
                        self.fuse_layer = GenericMLP(
                            input_dim=2,
                            hidden_dims=[2, ],
                            output_dim=1,
                            use_conv=False,
                            output_use_activation=False,
                            hidden_use_bias=True,
                        )
                if self.args.arch_type in ['coarse', 'coarse2fine']:
                    self.coarse_corner_head = nn.Linear(decoder_dim, 8 * 3)
                    self.coarse_semcls_head = nn.Linear(decoder_dim, dataset_config.num_semcls + 1)
            elif dataset_config.model_type == 'mlp':
                self.coarse_pre_encoder = pre_encoder
                self.coarse_encoder = encoder
                if hasattr(self.encoder, "masking_radius"):
                    hidden_dims = [encoder_dim]
                else:
                    hidden_dims = [encoder_dim, encoder_dim]
                self.coarse_encoder_to_decoder_projection = GenericMLP(
                    input_dim=encoder_dim,
                    hidden_dims=hidden_dims,
                    output_dim=decoder_dim,
                    norm_fn_name="bn1d",
                    activation="relu",
                    use_conv=True,
                    output_use_activation=True,
                    output_use_norm=True,
                    output_use_bias=False,
                )
                self.coarse_avgpool = nn.AdaptiveAvgPool1d((1,))
                if self.args.arch_type in ['coarse_fine', 'coarse2fine']:
                    if self.args.fusion_type in ['self_attn']:
                        self.norm1 = nn.LayerNorm(args.enc_dim)
                        self.norm3 = nn.LayerNorm(args.enc_dim)
                        # Implementation of Feedforward model
                        self.linear1 = nn.Linear(args.enc_dim, args.enc_dim)
                        self.linear2 = nn.Linear(args.enc_dim, args.enc_dim)
                        self.activation = nn.ReLU()
                        self.fuse_layer = nn.MultiheadAttention(args.enc_dim, args.enc_nhead, dropout=args.enc_dropout)
                    elif self.args.fusion_type in ['mlp']:
                        self.fuse_layer = GenericMLP(
                            input_dim=2,
                            hidden_dims=[2, ],
                            output_dim=1,
                            use_conv=False,
                            output_use_activation=False,
                            hidden_use_bias=True,
                        )
            else:
                self.coarse_pre_encoder = pre_encoder
                self.coarse_encoder = encoder
                self.coarse_pos_embedding = PositionEmbeddingCoordsSine(
                    d_pos=decoder_dim, pos_type=position_embedding, normalize=True
                )
                self.coarse_query_projection = GenericMLP(
                    input_dim=decoder_dim,
                    hidden_dims=[decoder_dim],
                    output_dim=decoder_dim,
                    use_conv=True,
                    output_use_activation=True,
                    hidden_use_bias=True,
                )
                self.coarse_decoder = decoder
                self.build_mlp_heads(dataset_config, decoder_dim, mlp_dropout)
                self.coarse_mlp_heads = copy.deepcopy(self.mlp_heads)
                self.box_processor = BoxProcessor(dataset_config)
                if self.args.arch_type in ['coarse_fine', 'coarse2fine']:
                    if self.args.fusion_type in ['self_attn']:
                        self.norm1 = nn.LayerNorm(args.enc_dim)
                        self.norm3 = nn.LayerNorm(args.enc_dim)
                        # Implementation of Feedforward model
                        self.linear1 = nn.Linear(args.enc_dim, args.enc_dim)
                        self.linear2 = nn.Linear(args.enc_dim, args.enc_dim)
                        self.activation = nn.ReLU()
                        self.fuse_layer = nn.MultiheadAttention(args.enc_dim, args.enc_nhead, dropout=args.enc_dropout)
                    elif self.args.fusion_type in ['mlp']:
                        self.fuse_layer = GenericMLP(
                            input_dim=2,
                            hidden_dims=[2, ],
                            output_dim=1,
                            use_conv=False,
                            output_use_activation=False,
                            hidden_use_bias=True,
                        )

        self.num_queries = num_queries
        self.version = dataset_config.version
        self.dataset_config = dataset_config

    def build_mlp_heads(self, dataset_config, decoder_dim, mlp_dropout):
        mlp_func = partial(
            GenericMLP,
            norm_fn_name="bn1d",
            activation="relu",
            use_conv=True,
            hidden_dims=[decoder_dim, decoder_dim],
            dropout=mlp_dropout,
            input_dim=decoder_dim,
        )

        # Semantic class of the box
        # add 1 for background/not-an-object class
        semcls_head = mlp_func(output_dim=dataset_config.num_semcls + 1)
        # semcls_head = nn.Conv1d(decoder_dim, dataset_config.num_semcls + 1, 1, )

        # geometry of the box
        if dataset_config.version == 'v1_csa':
            center_head = mlp_func(output_dim=3)
            size_head = mlp_func(output_dim=3)
            angle_cls_head = mlp_func(output_dim=dataset_config.num_angle_bin)
            angle_reg_head = mlp_func(output_dim=dataset_config.num_angle_bin)

            mlp_heads = [
                ("sem_cls_head", semcls_head),
                ("center_head", center_head),
                ("size_head", size_head),
                ("angle_cls_head", angle_cls_head),
                ("angle_residual_head", angle_reg_head),
            ]
        elif dataset_config.version == 'v2_corner':
            corner_head = mlp_func(output_dim=3 * 8)
            # corner_head = nn.Conv1d(decoder_dim, 3 * 8, 1, )

            mlp_heads = [
                ("sem_cls_head", semcls_head),
                ("corner_head", corner_head),
            ]
        self.mlp_heads = nn.ModuleDict(mlp_heads)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        # query_inds = furthest_point_sample(encoder_xyz, self.num_queries)
        # query_inds = query_inds.long()
        # query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds) for x in range(3)]
        query_xyz = encoder_xyz  # [:, self.num_queries]
        # query_xyz = torch.stack(query_xyz)
        # print('query_xyz.shape: ', query_xyz.shape)  # [8, 240, 180, 128, 4]
        # query_xyz = query_xyz.permute(1, 2, 0)

        # Gater op above can be replaced by the three lines below from the pointnet2 codebase
        # xyz_flipped = encoder_xyz.transpose(1, 2).contiguous()
        # query_xyz = gather_operation(xyz_flipped, query_inds.int())
        # query_xyz = query_xyz.transpose(1, 2)
        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def coarse_get_query_embeddings(self, encoder_xyz, point_cloud_dims):
        query_xyz = encoder_xyz  # [:, self.num_queries]
        pos_embed = self.coarse_pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.coarse_query_projection(pos_embed)
        return query_xyz, query_embed

    def _break_up_pc(self, pc):
        # pc may contain color/normals.

        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None
        return xyz, features

    def run_encoder(self, point_clouds):
        # xyz, features = self._break_up_pc(point_clouds)
        xyz = point_clouds
        features = point_clouds
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        # pre_enc_features = point_clouds.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        # if enc_inds is None:
        #     # encoder does not perform any downsampling
        #     enc_inds = pre_enc_inds
        # else:
        #     # use gather here to ensure that it works for both FPS and random sampling
        #     enc_inds = torch.gather(pre_enc_inds, 1, enc_inds)
        # enc_xyz = point_clouds
        enc_inds = None
        enc_features = enc_features.permute(1, 0, 2)
        return enc_xyz, enc_features, enc_inds

    def run_coarse_encoder(self, point_clouds):
        xyz = point_clouds
        features = point_clouds
        # print('xyz: {}, features: {}'.format(xyz.shape, features.shape))  # xyz: torch.Size([8, 240, 180, 64, 4]), features: torch.Size([8, 240, 180, 64, 4])
        pre_enc_xyz, pre_enc_features, pre_enc_inds = self.coarse_pre_encoder(xyz, features)
        # xyz: batch x npoints x 3
        # features: batch x channel x npoints
        # inds: batch x npoints

        # nn.MultiHeadAttention in encoder expects npoints x batch x channel features
        # pre_enc_features = point_clouds.permute(2, 0, 1)

        # xyz points are in batch x npointx channel order
        enc_xyz, enc_features, enc_inds = self.coarse_encoder(
            pre_enc_features, xyz=pre_enc_xyz
        )
        enc_inds = None
        enc_features = enc_features.permute(1, 0, 2)
        return enc_xyz, enc_features, enc_inds

    def get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        # print('box_features.shape: ', box_features.shape)  # box_features.shape:  torch.Size([1, 128, 8, 256])
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)[:, :, 0:1, :]
        # print('cls_logits.shape: ', cls_logits.shape)  # cls_logits.shape:  torch.Size([1, 8, 64, 2])
        if self.version == 'v1_csa':
            center_offset = (
                self.mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
            )
            size_normalized = (
                self.mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
            )
            angle_logits = self.mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
            angle_residual_normalized = self.mlp_heads["angle_residual_head"](
                box_features
            ).transpose(1, 2)
            center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
            size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
            angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
            angle_residual_normalized = angle_residual_normalized.reshape(
                num_layers, batch, num_queries, -1
            )
            angle_residual = angle_residual_normalized * (
                    np.pi / angle_residual_normalized.shape[-1]
            )
        elif self.version == 'v2_corner':
            box_corners = self.mlp_heads["corner_head"](box_features).transpose(1, 2)
            box_corners = box_corners.reshape(num_layers, batch, num_queries, 8, 3)[:, :, 0:1, :, :]
        # print('box_corners.shape: ', box_corners.shape)
        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            if self.version == 'v1_csa':
                (
                    center_normalized,
                    center_unnormalized,
                ) = self.box_processor.compute_predicted_center(
                    center_offset[l], query_xyz, point_cloud_dims
                )
                angle_continuous = self.box_processor.compute_predicted_angle(
                    angle_logits[l], angle_residual[l]
                )
                size_unnormalized = self.box_processor.compute_predicted_size(
                    size_normalized[l], point_cloud_dims
                )
                box_corners = self.box_processor.box_parametrization_to_corners(
                    center_unnormalized, size_unnormalized, angle_continuous
                )
                # print(box_corners.shape) # torch.Size([8, 128, 8, 3])
            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])
            # print(box_corners[l].shape)
            if self.version == 'v1_csa':
                box_prediction = {
                    "sem_cls_logits": cls_logits[l],
                    "center_normalized": center_normalized.contiguous(),
                    "center_unnormalized": center_unnormalized,
                    "size_normalized": size_normalized[l],
                    "size_unnormalized": size_unnormalized,
                    "angle_logits": angle_logits[l],
                    "angle_residual": angle_residual[l],
                    "angle_residual_normalized": angle_residual_normalized[l],
                    "angle_continuous": angle_continuous,
                    "objectness_prob": objectness_prob,
                    "sem_cls_prob": semcls_prob,
                    "box_corners": box_corners,
                }
            elif self.version == 'v2_corner':
                box_prediction = {
                    "sem_cls_logits": cls_logits[l],
                    "objectness_prob": objectness_prob,
                    "sem_cls_prob": semcls_prob,
                    "box_corners": box_corners[l],
                }
                # print('cls_logits[l].shape, box_corners[l].shape: ', cls_logits[l].shape, box_corners[l].shape)
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        # aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            # "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def coarse_get_box_predictions(self, query_xyz, point_cloud_dims, box_features):
        """
        Parameters:
            query_xyz: batch x nqueries x 3 tensor of query XYZ coords
            point_cloud_dims: List of [min, max] dims of point cloud
                              min: batch x 3 tensor of min XYZ coords
                              max: batch x 3 tensor of max XYZ coords
            box_features: num_layers x num_queries x batch x channel
        """
        # box_features change to (num_layers x batch) x channel x num_queries
        # print('box_features.shape: ', box_features.shape)  # box_features.shape:  torch.Size([1, 128, 8, 256])
        box_features = box_features.permute(0, 2, 3, 1)
        num_layers, batch, channel, num_queries = (
            box_features.shape[0],
            box_features.shape[1],
            box_features.shape[2],
            box_features.shape[3],
        )
        box_features = box_features.reshape(num_layers * batch, channel, num_queries)

        # mlp head outputs are (num_layers x batch) x noutput x nqueries, so transpose last two dims
        cls_logits = self.coarse_mlp_heads["sem_cls_head"](box_features).transpose(1, 2)
        # reshape outputs to num_layers x batch x nqueries x noutput
        cls_logits = cls_logits.reshape(num_layers, batch, num_queries, -1)
        if self.version == 'v1_csa':
            center_offset = (
                self.coarse_mlp_heads["center_head"](box_features).sigmoid().transpose(1, 2) - 0.5
            )
            size_normalized = (
                self.coarse_mlp_heads["size_head"](box_features).sigmoid().transpose(1, 2)
            )
            angle_logits = self.coarse_mlp_heads["angle_cls_head"](box_features).transpose(1, 2)
            angle_residual_normalized = self.coarse_mlp_heads["angle_residual_head"](
                box_features
            ).transpose(1, 2)
            center_offset = center_offset.reshape(num_layers, batch, num_queries, -1)
            size_normalized = size_normalized.reshape(num_layers, batch, num_queries, -1)
            angle_logits = angle_logits.reshape(num_layers, batch, num_queries, -1)
            angle_residual_normalized = angle_residual_normalized.reshape(
                num_layers, batch, num_queries, -1
            )
            angle_residual = angle_residual_normalized * (
                    np.pi / angle_residual_normalized.shape[-1]
            )
        elif self.version == 'v2_corner':
            box_corners = self.coarse_mlp_heads["corner_head"](box_features).transpose(1, 2)
            box_corners = box_corners.reshape(num_layers, batch, num_queries, 8, 3)

        outputs = []
        for l in range(num_layers):
            # box processor converts outputs so we can get a 3D bounding box
            if self.version == 'v1_csa':
                (
                    center_normalized,
                    center_unnormalized,
                ) = self.box_processor.compute_predicted_center(
                    center_offset[l], query_xyz, point_cloud_dims
                )
                angle_continuous = self.box_processor.compute_predicted_angle(
                    angle_logits[l], angle_residual[l]
                )
                size_unnormalized = self.box_processor.compute_predicted_size(
                    size_normalized[l], point_cloud_dims
                )
                box_corners = self.box_processor.box_parametrization_to_corners(
                    center_unnormalized, size_unnormalized, angle_continuous
                )
                # print(box_corners.shape) # torch.Size([8, 128, 8, 3])
            # below are not used in computing loss (only for matching/mAP eval)
            # we compute them with no_grad() so that distributed training does not complain about unused variables
            with torch.no_grad():
                (
                    semcls_prob,
                    objectness_prob,
                ) = self.box_processor.compute_objectness_and_cls_prob(cls_logits[l])
            # print(box_corners[l].shape)
            if self.version == 'v1_csa':
                box_prediction = {
                    "sem_cls_logits": cls_logits[l],
                    "center_normalized": center_normalized.contiguous(),
                    "center_unnormalized": center_unnormalized,
                    "size_normalized": size_normalized[l],
                    "size_unnormalized": size_unnormalized,
                    "angle_logits": angle_logits[l],
                    "angle_residual": angle_residual[l],
                    "angle_residual_normalized": angle_residual_normalized[l],
                    "angle_continuous": angle_continuous,
                    "objectness_prob": objectness_prob,
                    "sem_cls_prob": semcls_prob,
                    "box_corners": box_corners,
                }
            elif self.version == 'v2_corner':
                box_prediction = {
                    "sem_cls_logits": cls_logits[l],
                    "objectness_prob": objectness_prob,
                    "sem_cls_prob": semcls_prob,
                    "box_corners": box_corners[l],
                }
                # print('cls_logits[l].shape, box_corners[l].shape: ', cls_logits[l].shape, box_corners[l].shape)
            outputs.append(box_prediction)

        # intermediate decoder layer outputs are only used during training
        aux_outputs = outputs[:-1]
        outputs = outputs[-1]

        return {
            "outputs": outputs,  # output from last layer of decoder
            "aux_outputs": aux_outputs,  # output from intermediate layers of decoder
        }

    def forward(self, inputs, encoder_only=False, net_device='cuda:0', save_feat=True):
        poses = inputs["point_clouds"]
        nerf_ckpt_path = inputs['nerf_ckpt_path']
        scan_path = inputs['scan_path'][0]
        # print('scan_path: ', scan_path)
        # print('len(self.nerf_buffer): ', len(self.nerf_buffer))
        with torch.no_grad():
            if self.nerf_buffer_type == 'list':
                if self.forward_idx % 10 == 0 or self.dataset_config.pseudo_batch_size == 1:
                    if nerf_ckpt_path != self.nerf_ckpt_path:
                        self.render_kwargs_test, start = load_nerf_model(self.render_kwargs_test, ckpt_path=nerf_ckpt_path)
                        self.nerf_ckpt_path = nerf_ckpt_path
                    args_chunk = 1024 * 32
                    # print('poses.shape: ', poses.shape)  # poses.shape:  torch.Size([8, 1, 3, 5])  # torch.Size([1, 8, 3, 5])
                    poses = poses.reshape(-1, 3, 5)
                    hwf = poses[0, :3, -1]
                    render_poses = poses[:, :3, :4]
                    # render_poses = np.array(poses)
                    # render_poses = torch.Tensor(render_poses).to(device)
                    H, W, focal = hwf.cpu().numpy()
                    H, W = int(H), int(W)
                    hwf = [H, W, focal]
                    K = np.array([
                        [focal, 0, 0.5 * W],
                        [0, focal, 0.5 * H],
                        [0, 0, 1]
                    ])
                    rgbs, disps, raws = render_path(render_poses, hwf, K, args_chunk, self.render_kwargs_test, savedir=None,
                                      render_factor=0)
                    if len(self.nerf_buffer) > self.buffer_his_length:
                         self.nerf_buffer = self.nerf_buffer[1:]
                    # print('inputs[gt_box_corners].shape, inputs[gt_box_sem_cls_label].shape: ', inputs["gt_box_corners"].shape, inputs['gt_box_sem_cls_label'].shape)
                    # inputs[gt_box_corners].shape, inputs[gt_box_sem_cls_label].shape:  torch.Size([1, 1, 8, 3]) torch.Size([1, 1])
                    self.nerf_buffer.append({self.forward_idx: {'raws': raws, 'nerf_ckpt_path': nerf_ckpt_path, 'scan_path': scan_path, "gt_box_corners": inputs["gt_box_corners"].cpu().numpy(),
                                                                "gt_box_sem_cls_label": inputs['gt_box_sem_cls_label'].cpu().numpy().astype(np.int64),
                                                                'gt_box_present': inputs['gt_box_present'].cpu().numpy(),
                                                                'gt_box_angles': inputs['gt_box_angles'].cpu().numpy(),
                                                                }})
                    return_gt = {}
                else:
                    nerf_data = list(random.sample(self.nerf_buffer, 1)[0].values())[0]
                    return_gt = {"gt_box_corners": torch.Tensor(nerf_data["gt_box_corners"]).to(net_device),
                                 "gt_box_sem_cls_label": torch.LongTensor(nerf_data['gt_box_sem_cls_label']).to(net_device),
                                 'gt_box_present': torch.Tensor(nerf_data['gt_box_present']).to(net_device),
                                 'gt_box_angles': torch.Tensor(nerf_data['gt_box_angles']).to(net_device),
                                 }
                    raws = nerf_data['raws']
            elif self.nerf_buffer_type == 'dict':
                if scan_path not in self.nerf_buffer or self.dataset_config.pseudo_batch_size == 1:
                    if nerf_ckpt_path != self.nerf_ckpt_path:
                        self.render_kwargs_test, start = load_nerf_model(self.render_kwargs_test,
                                                                         ckpt_path=nerf_ckpt_path)
                        self.nerf_ckpt_path = nerf_ckpt_path
                    args_chunk = 1024 * 32
                    # print('poses.shape: ', poses.shape)  # poses.shape:  torch.Size([8, 1, 3, 5])  # torch.Size([1, 8, 3, 5])
                    poses = poses.reshape(-1, 3, 5)

                    hwf = poses[0, :3, -1]
                    render_poses = poses[:, :3, :4]
                    # render_poses = np.array(poses)
                    # render_poses = torch.Tensor(render_poses).to(device)
                    H, W, focal = hwf.cpu().numpy()
                    H, W = int(H), int(W)
                    hwf = [H, W, focal]
                    K = np.array([
                        [focal, 0, 0.5 * W],
                        [0, focal, 0.5 * H],
                        [0, 0, 1]
                    ])
                    # print('render_poses[0]: ', render_poses[0])
                    rendering_time_per_batch_start = time.time()
                    rgbs, disps_vis, raws, disps = render_path(render_poses, hwf, K, args_chunk, self.render_kwargs_test,
                                                    savedir=None,
                                                    render_factor=0)
                    self.rendering_time_per_batch = time.time() - rendering_time_per_batch_start
                    self.total_rendering_time = self.total_rendering_time + self.rendering_time_per_batch
                    # print('K: {}, hwf: {}'.format(K, hwf))
                    # print('render_poses: ', render_poses)
                    if self.dataset_config.input_type in ['rendered_depth', 'nerf_rendered_depth', 'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
                        if np.isnan(disps).any():
                            disps = np.nan_to_num(disps, nan=1.)
                            disps_vis = np.nan_to_num(disps_vis, nan=1.)
                    tmp_dir = 'tmp'
                    os.makedirs(tmp_dir, exist_ok=True)
                    for i in range(rgbs.shape[0]):
                        resized_frame = cv2.resize(rgbs[i][:, :, ::-1] * 255., [480, 640])
                        cv2.imwrite(os.path.join(tmp_dir, '{:06d}_{:02d}_{}_{}.png'.format(self.forward_idx, i, self.split, 'fine')), resized_frame)
                    if save_feat:
                        pickle_content = {'rgbs': rgbs, 'depth': disps_vis, 'raws': raws, 'disps': disps,
                                          'gt_box_corners': inputs['gt_box_corners'].cpu().numpy(), 'K': K, 'rendered_pose': poses.cpu().numpy()}
                        with open(os.path.join(tmp_dir, '{:06d}_{}_{}.pkl'.format(self.forward_idx, self.split, 'fine')), 'wb') as handle:
                            pickle.dump(pickle_content, handle)
                    # print('raws.shape, disps.shape, rgbs.shape: ', raws.shape, disps.shape, rgbs.shape)
                    # raws.shape, disps.shape, rgbs.shape:  (8, 240, 180, 64, 4) (8, 240, 180) (8, 240, 180, 3)
                    if self.dataset_config.input_type == 'rendered_rgb':
                        raws = rgbs
                    if self.dataset_config.input_type == 'rendered_depth':
                        raws = disps[..., None]
                    if self.dataset_config.input_type == 'nerf_rendered_rgb':
                        raws = np.concatenate((raws, np.repeat(rgbs[..., None], 4, axis=-1)), axis=-2)
                    if self.dataset_config.input_type == 'nerf_rendered_depth':
                        raws = np.concatenate((raws, np.repeat(disps[..., None, None], 4, axis=-1)), axis=-2)
                    if self.dataset_config.input_type == 'nerf_rendered_rgb_rendered_depth':
                        raws = np.concatenate((raws, np.repeat(rgbs[..., None], 4, axis=-1),
                                          np.repeat(disps[..., None, None], 4, axis=-1)), axis=-2)
                    if self.dataset_config.input_type == 'rendered_rgb_rendered_depth':
                        raws = np.concatenate((np.repeat(rgbs[..., None], 4, axis=-1),
                                          np.repeat(disps[..., None, None], 4, axis=-1)), axis=-2)
                    if self.args.arch_type in ['coarse_fine', 'coarse', 'coarse2fine']:
                        coarse_K = np.array([
                            [focal / self.args.focal_length, 0, 0.5 * W],
                            [0, focal / self.args.focal_length, 0.5 * H],
                            [0, 0, 1]
                        ])
                        # coarse_K = np.array([
                        #     [focal, 0, 0.5 * W * int(self.args.focal_length)],
                        #     [0, focal, 0.5 * H * int(self.args.focal_length)],
                        #     [0, 0, 1]
                        # ])
                        coarse_hwf = copy.copy(hwf)
                        # coarse_hwf[0] *= int(self.args.focal_length)
                        # coarse_hwf[1] *= int(self.args.focal_length)
                        # print('coarse_K: {}, coarse_hwf: {}'.format(coarse_K, coarse_hwf))
                        # coarse_K[0, 0]: 183.589599609375, coarse_hwf: [480, 360, 183.5896]
                        coarse_rendering_time_per_batch_start = time.time()
                        coarse_rgbs, coarse_disps_vis, coarse_raws, coarse_disps = render_path(render_poses, coarse_hwf, coarse_K, args_chunk, self.render_kwargs_test,
                                                        savedir=None,
                                                        render_factor=0)
                        self.coarse_rendering_time_per_batch = time.time() - coarse_rendering_time_per_batch_start
                        self.coarse_total_rendering_time = self.coarse_total_rendering_time + self.coarse_rendering_time_per_batch
                        # print('coarse_rendering_time_per_batch: {}, self.coarse_total_rendering_time: {}'.format(
                        #     self.coarse_rendering_time_per_batch, self.coarse_total_rendering_time))
                        # print('coarse_rgbs.shape: {}, coarse_disps_vis.shape: {}, coarse_raws.shape: {}, coarse_disps.shape: {}'
                        #       .format(coarse_rgbs.shape, coarse_disps_vis.shape, coarse_raws.shape, coarse_disps.shape))
                        # coarse_rgbs = coarse_rgbs[:, ::int(self.args.focal_length), ::int(self.args.focal_length), ]
                        # coarse_disps_vis = coarse_disps_vis[:, ::int(self.args.focal_length), ::int(self.args.focal_length), ]
                        # coarse_raws = coarse_raws[:, ::int(self.args.focal_length), ::int(self.args.focal_length), ]
                        # coarse_disps = coarse_disps[:, ::int(self.args.focal_length), ::int(self.args.focal_length), ]
                        # print(
                        #     'coarse_rgbs.shape: {}, coarse_disps_vis.shape: {}, coarse_raws.shape: {}, coarse_disps.shape: {}'
                        #     .format(coarse_rgbs.shape, coarse_disps_vis.shape, coarse_raws.shape, coarse_disps.shape))
                        # coarse_rgbs.shape: (8, 480, 360, 3), coarse_disps_vis.shape: (8, 480, 360), coarse_raws.shape: (8, 480, 360, 64, 4), coarse_disps.shape: (8, 480, 360)
                        # coarse_rgbs.shape: (8, 240, 180, 3), coarse_disps_vis.shape: (8, 240, 180), coarse_raws.shape: (8, 240, 180, 64, 4), coarse_disps.shape: (8, 240, 180)
                        if self.dataset_config.input_type in ['rendered_depth', 'nerf_rendered_depth',
                                                              'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
                            if np.isnan(coarse_disps).any():
                                coarse_disps = np.nan_to_num(coarse_disps, nan=1.)
                                coarse_disps_vis = np.nan_to_num(coarse_disps_vis, nan=1.)
                        for i in range(coarse_rgbs.shape[0]):
                            coarse_resized_frame = cv2.resize(coarse_rgbs[i][:, :, ::-1] * 255., [480, 640])
                            cv2.imwrite(os.path.join(tmp_dir, '{:06d}_{:02d}_{}_{}.png'.format(self.forward_idx, i, self.split, 'coarse')), coarse_resized_frame)
                        if save_feat:
                            pickle_content = {'coarse_rgbs': coarse_rgbs, 'coarse_depth': coarse_disps_vis,
                                              'coarse_raws': coarse_raws, 'coarse_disps': coarse_disps,
                                              'gt_box_corners': inputs['gt_box_corners'].cpu().numpy(),
                                              'K': coarse_K, 'rendered_pose': poses.cpu().numpy()}
                            with open(os.path.join(tmp_dir, '{:06d}_{}_{}.pkl'.format(self.forward_idx, self.split, 'coarse')), 'wb') as handle:
                                pickle.dump(pickle_content, handle)
                        if self.dataset_config.input_type == 'rendered_rgb':
                            coarse_raws = coarse_rgbs
                        if self.dataset_config.input_type == 'rendered_depth':
                            coarse_raws = disps[..., None]
                        if self.dataset_config.input_type == 'nerf_rendered_rgb':
                            coarse_raws = np.concatenate((coarse_raws, np.repeat(coarse_rgbs[..., None], 4, axis=-1)), axis=-2)
                        if self.dataset_config.input_type == 'nerf_rendered_depth':
                            coarse_raws = np.concatenate((coarse_raws, np.repeat(coarse_disps[..., None, None], 4, axis=-1)), axis=-2)
                        if self.dataset_config.input_type == 'nerf_rendered_rgb_rendered_depth':
                            coarse_raws = np.concatenate((coarse_raws, np.repeat(coarse_rgbs[..., None], 4, axis=-1),
                                                     np.repeat(coarse_disps[..., None, None], 4, axis=-1)), axis=-2)
                        if self.dataset_config.input_type == 'rendered_rgb_rendered_depth':
                            coarse_raws = np.concatenate((np.repeat(coarse_rgbs[..., None], 4, axis=-1),
                                                     np.repeat(coarse_disps[..., None, None], 4, axis=-1)), axis=-2)
                        # print('coarse_raws.shape: ', coarse_raws.shape)  # (8, 240, 180, 64, 4)
                    if len(self.nerf_buffer) >= self.buffer_his_length:
                        pop_obj = self.nerf_buffer.popitem()
                    # print('inputs[gt_box_corners].shape, inputs[gt_box_sem_cls_label].shape: ', inputs["gt_box_corners"].shape, inputs['gt_box_sem_cls_label'].shape)
                    # inputs[gt_box_corners].shape, inputs[gt_box_sem_cls_label].shape:  torch.Size([1, 1, 8, 3]) torch.Size([1, 1])
                    self.nerf_buffer[scan_path] = {'raws': raws, 'forward_idx': self.forward_idx,
                                                            "gt_box_corners": inputs["gt_box_corners"].cpu().numpy(),
                                                            "gt_box_sem_cls_label": inputs['gt_box_sem_cls_label'].cpu().numpy().astype(
                                                                    np.int64),
                                                            'gt_box_present': inputs['gt_box_present'].cpu().numpy(),
                                                            'gt_box_angles': inputs['gt_box_angles'].cpu().numpy(),
                                                            }
                    if self.args.arch_type in ['coarse_fine', 'coarse', 'coarse2fine']:
                        coarse_raws_buffer = coarse_raws  # [:, :int(coarse_raws.shape[1] / 2), :int(coarse_raws.shape[2] / 2), ]
                        self.nerf_buffer[scan_path]['coarse_raws'] = copy.deepcopy(coarse_raws_buffer)  # [:, :int(coarse_raws.shape[1] / 2), :int(coarse_raws.shape[2] / 2), ]
                    if self.test_only: # or True:
                        self.nerf_buffer[scan_path]['nerf_rgbs'] = copy.deepcopy(rgbs)
                        return_gt = {'nerf_rgbs': copy.deepcopy(rgbs)}
                        if self.dataset_config.input_type in ['rendered_depth', 'nerf_rendered_depth', 'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
                            self.nerf_buffer[scan_path]['nerf_depth'] = copy.deepcopy(disps)
                            return_gt['nerf_depth'] = copy.deepcopy(disps)
                            self.nerf_buffer[scan_path]['nerf_depth_vis'] = copy.deepcopy(disps_vis)
                            return_gt['nerf_depth_vis'] = copy.deepcopy(disps_vis)
                        if self.args.arch_type in ['coarse_fine', 'coarse', 'coarse2fine']:
                            self.nerf_buffer[scan_path]['coarse_nerf_rgbs'] = copy.deepcopy(coarse_rgbs)  # [:, :int(coarse_rgbs.shape[1] / 2), :int(coarse_rgbs.shape[2] / 2), ]
                            return_gt['coarse_nerf_rgbs'] = copy.deepcopy(coarse_rgbs)
                            if self.dataset_config.input_type in ['rendered_depth', 'nerf_rendered_depth',
                                                                  'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
                                self.nerf_buffer[scan_path]['coarse_nerf_depth'] = copy.deepcopy(coarse_disps)
                                return_gt['coarse_nerf_depth'] = copy.deepcopy(coarse_disps)
                                self.nerf_buffer[scan_path]['coarse_nerf_depth_vis'] = copy.deepcopy(coarse_disps_vis)
                                return_gt['coarse_nerf_depth_vis'] = copy.deepcopy(coarse_disps_vis)
                    else:
                        return_gt = {}
                    # print('batch_data_label[nerf_rgbs].shape: ', rgbs.shape) # batch_data_label[nerf_rgbs].shape:  (8, 240, 180, 3)
                else:
                    nerf_data = self.nerf_buffer[scan_path]
                    return_gt = {"gt_box_corners": torch.Tensor(nerf_data["gt_box_corners"]).to(net_device),
                                 "gt_box_sem_cls_label": torch.LongTensor(nerf_data['gt_box_sem_cls_label']).to(
                                     net_device),
                                 'gt_box_present': torch.Tensor(nerf_data['gt_box_present']).to(net_device),
                                 'gt_box_angles': torch.Tensor(nerf_data['gt_box_angles']).to(net_device),
                                 }
                    raws = nerf_data['raws']
                    if self.args.arch_type in ['coarse_fine', 'coarse', 'coarse2fine']:
                        coarse_raws = nerf_data['coarse_raws']
                    else:
                        coarse_raws = None
                    # print('batch_data_label[nerf_rgbs].shape: ', nerf_data['nerf_rgbs'].shape)
                    if self.test_only: # or True:
                        return_gt['nerf_rgbs'] = copy.deepcopy(nerf_data['nerf_rgbs'])
                        if self.dataset_config.input_type in ['rendered_depth', 'nerf_rendered_depth', 'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
                            return_gt['nerf_depth'] = copy.deepcopy(nerf_data['nerf_depth'])
                            return_gt['nerf_depth_vis'] = copy.deepcopy(nerf_data['nerf_depth_vis'])
                        if self.args.arch_type in ['coarse_fine', 'coarse', 'coarse2fine']:
                            return_gt['coarse_nerf_rgbs'] = copy.deepcopy(nerf_data['coarse_nerf_rgbs'])
                            if self.dataset_config.input_type in ['rendered_depth', 'nerf_rendered_depth',
                                                                  'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
                                return_gt['coarse_nerf_depth'] = copy.deepcopy(nerf_data['coarse_nerf_depth'])
                                return_gt['coarse_nerf_depth_vis'] = copy.deepcopy(nerf_data['coarse_nerf_depth_vis'])
        # print('raws.shape: ', raws.shape)  # raws.shape:  (8, 240, 180, 64, 4)
        # device = 'cuda:0'
        print('rendering_time_per_batch: {}, self.total_rendering_time: {}'.format(self.rendering_time_per_batch, self.total_rendering_time))
        point_clouds = torch.Tensor(raws).to(net_device)  # inputs["point_clouds"]
        if self.args.arch_type in ['coarse2fine', 'coarse']:
            if self.dataset_config.model_type == 'resnet18':
                coarse_raws = torch.Tensor(coarse_raws).to(net_device)
                if self.dataset_config.input_type == 'nerf':
                    coarse_raws = coarse_raws.reshape(coarse_raws.shape[0], coarse_raws.shape[1], coarse_raws.shape[2], -1)
                coarse_raws = coarse_raws.permute(0, 3, 1, 2)
                coarse_resnet_feat = self.coarse_resnet(coarse_raws)

                # print(resnet_output.shape)
                # box_corners = box_corners.reshape(-1, 8, 3).unsqueeze(1)  # (B, 1, 8, 3)
                coarse_bbox_corner = self.coarse_corner_head(coarse_resnet_feat)
                coarse_box_corners = coarse_bbox_corner.reshape(-1, 8, 3).unsqueeze(1)  # (B, 1, 8, 3)
                coarse_sem_cls_logits = self.semcls_head(coarse_resnet_feat).unsqueeze(1)
                coarse_outputs = {
                        "sem_cls_logits": coarse_sem_cls_logits,  # torch.ones((box_corners.shape[0], 1)).to(box_corners.device),
                        "objectness_prob": torch.ones((coarse_box_corners.shape[0], 1)).to(coarse_box_corners.device),
                        "sem_cls_prob": torch.ones((coarse_box_corners.shape[0], 1, 1)).to(coarse_box_corners.device),
                        "box_corners": coarse_box_corners,
                    }
                coarse_box_predictions = {
                    "outputs": coarse_outputs,  # output from last layer of decoder
                }
            elif self.dataset_config.model_type == 'mlp':
                coarse_raws = torch.Tensor(coarse_raws).to(net_device)
                coarse_enc_xyz, coarse_enc_features, coarse_enc_inds = self.run_coarse_encoder(coarse_raws)
                # enc_features: [128, 8, 256]
                coarse_enc_features = coarse_enc_features.permute(1, 2, 0)
                # print('enc_features.shape: ', enc_features.shape)
                coarse_feat = self.coarse_encoder_to_decoder_projection(coarse_enc_features)
                # print('feat.shape: ', feat.shape)
                coarse_pool = self.coarse_avgpool(coarse_feat).squeeze(-1)
                # print('pool.shape: ', pool.shape)
                coarse_bbox = self.corner_head(coarse_pool)
                # print('bbox.shape: ', bbox.shape)
                coarse_box_corners = coarse_bbox.reshape(-1, 8, 3).unsqueeze(1)  # (B, 1, 8, 3)
                coarse_sem_cls_logits = self.semcls_head(coarse_pool).unsqueeze(1)
                # print('resnet_output.shape: ', resnet_output.shape)
                coarse_outputs = {
                    "sem_cls_logits": coarse_sem_cls_logits,  #  torch.ones((resnet_output.shape[0], 1, 2)).to(resnet_output.device),
                    "objectness_prob": torch.ones((coarse_box_corners.shape[0], 1)).to(coarse_box_corners.device),
                    "sem_cls_prob": torch.ones((coarse_box_corners.shape[0], 1, 1)).to(coarse_box_corners.device),
                    "box_corners": coarse_box_corners,
                }
                coarse_box_predictions = {
                    "outputs": coarse_outputs,  # output from last layer of decoder
                }
            else:
                coarse_raws = torch.Tensor(coarse_raws).to(net_device)  # inputs["point_clouds"]
                # Fine Stream
                coarse_enc_xyz, coarse_enc_features, coarse_enc_inds = self.run_coarse_encoder(coarse_raws)
                point_cloud_dims = [
                    inputs["point_cloud_dims_min"],
                    inputs["point_cloud_dims_max"],
                ]
                coarse_query_xyz, coarse_query_embed = self.coarse_get_query_embeddings(coarse_enc_xyz, point_cloud_dims)
                # query_embed: batch x channel x npoint
                coarse_enc_pos = self.coarse_pos_embedding(coarse_enc_xyz, input_range=point_cloud_dims)
                # decoder expects: npoints x batch x channel
                coarse_enc_pos = coarse_enc_pos.permute(2, 0, 1)
                coarse_query_embed = coarse_query_embed.permute(2, 0, 1)
                coarse_tgt = torch.zeros_like(coarse_query_embed)
                coarse_box_features = self.coarse_decoder(
                    coarse_tgt, coarse_enc_features, query_pos=None, pos=None
                )[0]
                # coarse_box_features = self.coarse_decoder(
                #     coarse_tgt, coarse_enc_features, query_pos=coarse_query_embed, pos=coarse_enc_pos
                # )[0]
                coarse_box_predictions = self.coarse_get_box_predictions(
                    coarse_query_xyz, point_cloud_dims, coarse_box_features
                )
                # Coarse Stream
        elif self.args.arch_type in ['coarse_fine']:
            if self.dataset_config.model_type == 'resnet18':
                coarse_raws = torch.Tensor(coarse_raws).to(net_device)  # inputs["point_clouds"]
                if self.dataset_config.input_type == 'nerf':
                    coarse_raws = coarse_raws.reshape(coarse_raws.shape[0], coarse_raws.shape[1], coarse_raws.shape[2], -1)
                coarse_raws = coarse_raws.permute(0, 3, 1, 2)
                coarse_resnet_feat = self.coarse_resnet(coarse_raws)
            elif self.dataset_config.model_type == 'mlp':
                coarse_enc_xyz, coarse_enc_features, coarse_enc_inds = self.run_coarse_encoder(point_clouds)
                # enc_features: [128, 8, 256]
                coarse_enc_features = coarse_enc_features.permute(1, 2, 0)
                # print('enc_features.shape: ', enc_features.shape)
                coarse_feat = self.coarse_encoder_to_decoder_projection(coarse_enc_features)
                coarse_pool = self.avgpool(coarse_feat).squeeze(-1)
                # # print('pool.shape: ', pool.shape)
            else:
                coarse_raws = torch.Tensor(coarse_raws).to(net_device)  # inputs["point_clouds"]
                # Fine Stream
                coarse_enc_xyz, coarse_enc_features, coarse_enc_inds = self.run_coarse_encoder(coarse_raws)
                # point_cloud_dims = [
                #     inputs["point_cloud_dims_min"],
                #     inputs["point_cloud_dims_max"],
                # ]
                # coarse_query_xyz, coarse_query_embed = self.coarse_get_query_embeddings(coarse_enc_xyz, point_cloud_dims)
                # query_embed: batch x channel x npoint
                # coarse_enc_pos = self.coarse_pos_embedding(coarse_enc_xyz, input_range=point_cloud_dims)
                # decoder expects: npoints x batch x channel
                # coarse_enc_pos = coarse_enc_pos.permute(2, 0, 1)
                # coarse_query_embed = coarse_query_embed.permute(2, 0, 1)
                # coarse_tgt = torch.zeros_like(coarse_query_embed)
        if not self.args.arch_type in ['coarse']:
            if self.dataset_config.model_type == 'resnet18':
                if self.dataset_config.input_type == 'nerf':
                    point_clouds = point_clouds.reshape(point_clouds.shape[0], point_clouds.shape[1], point_clouds.shape[2], -1)
                point_clouds = point_clouds.permute(0, 3, 1, 2)
                resnet_feat = self.resnet(point_clouds)

                if self.args.arch_type in ['coarse_fine']:
                    if self.args.fusion_type in ['self_attn']:
                        # Self Attn
                        tgt = resnet_feat
                        q = k = coarse_resnet_feat
                        value = self.norm1(resnet_feat)
                        resnet_feat = self.fuse_layer(q, k, value=value, attn_mask=None,
                                              key_padding_mask=None)[0]

                        tgt = tgt + resnet_feat
                        tgt2 = self.norm3(tgt)
                        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
                        resnet_feat = tgt + tgt2
                    elif self.args.fusion_type in ['mlp']:
                        fuse_features = torch.cat((coarse_resnet_feat[..., None], resnet_feat[..., None]), -1)
                        # print('coarse_enc_features.shape: {}, enc_features.shape: {}, fuse_features.shape: {}'
                        #       .format(coarse_enc_features.shape, enc_features.shape, fuse_features.shape))
                        # coarse_enc_features.shape: torch.Size([64, 8, 256]), enc_features.shape: torch.Size([64, 8, 256]), fuse_features.shape: torch.Size([64, 8, 256, 2])
                        resnet_feat = self.fuse_layer(fuse_features).squeeze(-1)

                # print(resnet_output.shape)
                # box_corners = box_corners.reshape(-1, 8, 3).unsqueeze(1)  # (B, 1, 8, 3)
                bbox_corner = self.corner_head(resnet_feat)
                box_corners = bbox_corner.reshape(-1, 8, 3).unsqueeze(1)  # (B, 1, 8, 3)
                sem_cls_logits = self.semcls_head(resnet_feat).unsqueeze(1)
                outputs = {
                        "sem_cls_logits": sem_cls_logits,  # torch.ones((box_corners.shape[0], 1)).to(box_corners.device),
                        "objectness_prob": torch.ones((box_corners.shape[0], 1)).to(box_corners.device),
                        "sem_cls_prob": torch.ones((box_corners.shape[0], 1, 1)).to(box_corners.device),
                        "box_corners": box_corners,
                    }
                box_predictions = {
                    "outputs": outputs,  # output from last layer of decoder
                }
            elif self.dataset_config.model_type == 'mlp':
                enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
                # enc_features: [128, 8, 256]
                enc_features = enc_features.permute(1, 2, 0)
                # print('enc_features.shape: ', enc_features.shape)
                feat = self.encoder_to_decoder_projection(enc_features)
                # print('feat.shape: ', feat.shape)
                pool = self.avgpool(feat).squeeze(-1)
                if self.args.arch_type in ['coarse_fine']:
                    # coarse_enc_features = coarse_enc_features
                    # enc_features = enc_features
                    if self.args.fusion_type in ['self_attn']:
                        # Self Attn
                        tgt = pool
                        q = k = coarse_pool
                        value = self.norm1(pool)
                        pool = self.fuse_layer(q, k, value=value, attn_mask=None,
                                              key_padding_mask=None)[0]
                        tgt = tgt + pool
                        tgt2 = self.norm3(tgt)
                        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
                        pool = tgt + tgt2
                    elif self.args.fusion_type in ['mlp']:
                        fuse_features = torch.cat((coarse_pool[..., None], pool[..., None]), -1)
                        # print('coarse_enc_features.shape: {}, enc_features.shape: {}, fuse_features.shape: {}'
                        #       .format(coarse_enc_features.shape, enc_features.shape, fuse_features.shape))
                        # coarse_enc_features.shape: torch.Size([64, 8, 256]), enc_features.shape: torch.Size([64, 8, 256]), fuse_features.shape: torch.Size([64, 8, 256, 2])
                        pool = self.fuse_layer(fuse_features).squeeze(-1)
                # print('pool.shape: ', pool.shape)
                bbox = self.corner_head(pool)
                # print('bbox.shape: ', bbox.shape)
                box_corners = bbox.reshape(-1, 8, 3).unsqueeze(1)  # (B, 1, 8, 3)
                sem_cls_logits = self.semcls_head(pool).unsqueeze(1)
                # print('resnet_output.shape: ', resnet_output.shape)
                outputs = {
                    "sem_cls_logits": sem_cls_logits,  #  torch.ones((resnet_output.shape[0], 1, 2)).to(resnet_output.device),
                    "objectness_prob": torch.ones((box_corners.shape[0], 1)).to(box_corners.device),
                    "sem_cls_prob": torch.ones((box_corners.shape[0], 1, 1)).to(box_corners.device),
                    "box_corners": box_corners,
                }
                box_predictions = {
                    "outputs": outputs,  # output from last layer of decoder
                }
            else:
                enc_xyz, enc_features, enc_inds = self.run_encoder(point_clouds)
                point_cloud_dims = [
                    inputs["point_cloud_dims_min"],
                    inputs["point_cloud_dims_max"],
                ]
                query_xyz, query_embed = self.get_query_embeddings(enc_xyz, point_cloud_dims)
                # query_embed: batch x channel x npoint
                # enc_pos = self.pos_embedding(enc_xyz, input_range=point_cloud_dims)
                # decoder expects: npoints x batch x channel
                # enc_pos = enc_pos.permute(2, 0, 1)
                query_embed = query_embed.permute(2, 0, 1)
                tgt = torch.zeros_like(query_embed)
                if self.args.arch_type in ['coarse_fine']:
                    # coarse_enc_features = coarse_enc_features
                    # enc_features = enc_features
                    if self.args.fusion_type in ['self_attn']:
                        # Self Attn
                        tgt = enc_features
                        q = k = coarse_enc_features
                        value = self.norm1(enc_features)
                        enc_features = self.fuse_layer(q, k, value=value, attn_mask=None,
                                              key_padding_mask=None)[0]

                        tgt = tgt + enc_features
                        tgt2 = self.norm3(tgt)
                        tgt2 = self.linear2(self.activation(self.linear1(tgt2)))
                        enc_features = tgt + tgt2
                    elif self.args.fusion_type in ['mlp']:
                        fuse_features = torch.cat((coarse_enc_features[..., None], enc_features[..., None]), -1)
                        # print('coarse_enc_features.shape: {}, enc_features.shape: {}, fuse_features.shape: {}'
                        #       .format(coarse_enc_features.shape, enc_features.shape, fuse_features.shape))
                        # coarse_enc_features.shape: torch.Size([64, 8, 256]), enc_features.shape: torch.Size([64, 8, 256]), fuse_features.shape: torch.Size([64, 8, 256, 2])
                        enc_features = self.fuse_layer(fuse_features).squeeze(-1)
                        # print('2 enc_features: {}'.format(enc_features.shape))
                        # 2 enc_features: torch.Size([64, 8, 256, 1])

                box_features = self.decoder(
                    tgt, enc_features, query_pos=None, pos=None
                )[0]
                # box_features = self.decoder(
                #     tgt, enc_features, query_pos=query_embed, pos=enc_pos
                # )[0]
                box_predictions = self.get_box_predictions(
                    query_xyz, point_cloud_dims, box_features
                )
        self.forward_idx = self.forward_idx + 1
        if self.args.arch_type in ['coarse2fine']:
            return [box_predictions, coarse_box_predictions], return_gt
        elif self.args.arch_type in ['coarse']:
            return coarse_box_predictions, return_gt
        else:  # coarse_fine, fine
            return box_predictions, return_gt


class PreEncoder(nn.Module):
    ''' Modified based on _PointnetSAModuleBase and PointnetSAModuleMSG
    with extra support for returning point indices for getting their GT votes '''

    def __init__(
            self,
            *,
            mlp: List[int],
            npoint: int = None,
            radius: float = None,
            nsample: int = None,
            bn: bool = True,
            use_xyz: bool = True,
            pooling: str = 'max',
            sigma: float = None,   # for RBF pooling
            normalize_xyz: bool = False,   # noramlize local XYZ with radius
            sample_uniformly: bool = False,
            ret_unique_cnt: bool = False,
            args=None,
    ):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.pooling = pooling
        self.mlp_module = None
        self.use_xyz = use_xyz
        self.sigma = sigma
        if self.sigma is None:
            self.sigma = self.radius/2
        self.normalize_xyz = normalize_xyz
        self.ret_unique_cnt = ret_unique_cnt
        self.input_type = args.input_type  # rgb or nerf
        self.args = args

        # (240, 180, 128, 4)
        # print(self.mlp_module)
        if args.preencoder_type == 'resnet18':
            self.resnet = models.resnet18()
            num_ftrs = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(num_ftrs, args.dec_dim)
            if self.input_type in ['nerf']:
                # print(self.resnet)
                # num_ftrs = self.resnet.conv1.in_features
                self.resnet.conv1 = nn.Conv2d(128 * 2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                              bias=False)
            self.xyz_layer = nn.Linear(256, 4)
        else:
            if self.input_type in ['nerf', 'nerf_rendered_rgb', 'nerf_rendered_depth', 'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
                self.mlp_module = nn.ModuleList(
                    [nn.Conv3d(in_dim, out_dim, (6, 5, 1), stride=(3, 3, 1)) for i, (in_dim, out_dim) in
                     enumerate(zip(mlp[:-1], mlp[1:]))])
                self.xyz_layer = nn.Linear(256, 4)
            elif self.input_type in ['rgb', 'rendered_rgb', 'depth', 'rendered_depth']:
                self.mlp_module = nn.ModuleList(
                    [nn.Conv2d(in_dim, out_dim, (6, 5), stride=(3, 3)) for i, (in_dim, out_dim) in
                     enumerate(zip(mlp[:-1], mlp[1:]))])
                self.xyz_layer = nn.Linear(256, 4)

    def forward(self, xyz: torch.Tensor,
                features: torch.Tensor = None,
                inds: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features
        inds : torch.Tensor
            (B, npoint) tensor that stores index to the xyz points (values in 0-N-1)

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B, \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        inds: torch.Tensor
            (B, npoint) tensor of the inds
        """

        # xyz_flipped = xyz.transpose(1, 2).contiguous()
        # if inds is None:
        #     inds = pointnet2_utils.furthest_point_sample(xyz, self.npoint)
        # else:
        #     assert(inds.shape[1] == self.npoint)
        # new_xyz = pointnet2_utils.gather_operation(
        #     xyz_flipped, inds
        # ).transpose(1, 2).contiguous() if self.npoint is not None else None

        # if not self.ret_unique_cnt:
        #     grouped_features, grouped_xyz = self.grouper(
        #         xyz, new_xyz, features
        #     )  # (B, C, npoint, nsample)
        # else:
        #     grouped_features, grouped_xyz, unique_cnt = self.grouper(
        #         xyz, new_xyz, features
        #     )  # (B, C, npoint, nsample), (B,3,npoint,nsample), (B,npoint)
        # print('1 features.shape: ', features.shape)  # [8, 128, 4]
        # new_xyz = self.mlp_module(features)  # (B, mlp[-1], npoint, nsample)
        if self.args.preencoder_type == 'resnet18':
            if self.input_type in ['nerf']:
                features = features.reshape(features.shape[0], features.shape[1], features.shape[2],
                                                    -1)
            features = features.permute(0, 3, 1, 2)
            features = self.resnet(features)
            features = features.unsqueeze(1)
            print('features.shape: {}'.format(features.shape))  # torch.Size([8, 1， 256])
            new_xyz = self.xyz_layer(features)
            print('new_xyz.shape: {}'.format(new_xyz.shape))  # torch.Size([8, 1， 4])
        else:
            if self.input_type in ['nerf', 'nerf_rendered_rgb', 'nerf_rendered_depth', 'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
                features = features.permute(0, 4, 1, 2, 3)
            elif self.input_type in ['rgb', 'rendered_rgb', 'depth', 'rendered_depth']:
                features = features.permute(0, 3, 1, 2)
            for i, m in enumerate(self.mlp_module):
                features = m(features)
                # print(i, features.shape)
            if self.input_type in ['nerf', 'nerf_rendered_rgb', 'nerf_rendered_depth', 'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
                features = features.reshape(features.shape[0], features.shape[1], features.shape[4]).permute(0, 2, 1)
            elif self.input_type in ['rgb', 'rendered_rgb', 'depth', 'rendered_depth']:
                # print(features.shape)  # torch.Size([8, 256, 25, 19])
                features = features.reshape(features.shape[0], features.shape[1], -1).permute(0, 2, 1)
                # print(features.shape)  #  torch.Size([8, 475, 256])
            # torch.Size([8, 128, 256])
            new_xyz = self.xyz_layer(features)  # torch.Size([8, 128, 4])
            # print('new_xyz.shape: {}, features.shape: {}'.format(new_xyz.shape, features.shape))
            # new_xyz.shape: torch.Size([8, 64, 4]), features.shape: torch.Size([8, 64, 256])
        return new_xyz, features, inds

def build_nerf(args):
    # mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    # mlp_dims = [128, args.enc_dim]
    render_kwargs_train, render_kwargs_test, start = create_nerf_model(args, ckpt_path=None,)
    return render_kwargs_test

def build_preencoder(args):
    # mlp_dims = [3 * int(args.use_color), 64, 128, args.enc_dim]
    # mlp_dims = [128, args.enc_dim]
    if args.input_type in ['nerf', 'nerf_rendered_rgb', 'nerf_rendered_depth', 'nerf_rendered_rgb_rendered_depth', 'rendered_rgb_rendered_depth']:
        mlp_dims = [4, 4, 4, 4, args.enc_dim]
    elif args.input_type in ['rgb', 'rendered_rgb']:
        mlp_dims = [3, 4, args.enc_dim]
    elif args.input_type in ['depth', 'rendered_depth']:
        mlp_dims = [1, 4, args.enc_dim]
    # elif args.input_type in ['nerf_rendered_rgb']:
    #     mlp_dims = [4+3, 4, 4, 4, args.enc_dim]
    # elif args.input_type in ['nerf_rendered_depth']:
    #     mlp_dims = [4+1, 4, 4, 4, args.enc_dim]
    # elif args.input_type in ['nerf_rendered_rgb_rendered_depth']:
    #     mlp_dims = [4+3+1, 4, 4, 4, args.enc_dim]
    # mlp_dims = [4, 4]
    preencoder = PreEncoder(
        radius=0.2,
        nsample=64,
        npoint=args.preenc_npoints,
        mlp=mlp_dims,
        normalize_xyz=True,
        args=args,
    )
    # preencoder = nn.ModuleList([nn.Linear(4, args.enc_dim) for i in range(1)])
    return preencoder


def build_encoder(args):
    if args.enc_type == "vanilla":
        encoder_layer = TransformerEncoderLayer(
            d_model=args.enc_dim,
            nhead=args.enc_nhead,
            dim_feedforward=args.enc_ffn_dim,
            dropout=args.enc_dropout,
            activation=args.enc_activation,
            use_mlp=args.use_mlp_in_attention,
        )
        encoder = TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=args.enc_nlayers
        )

    # elif args.enc_type in ["masked"]:
    #     encoder_layer = TransformerEncoderLayer(
    #         d_model=args.enc_dim,
    #         nhead=args.enc_nhead,
    #         dim_feedforward=args.enc_ffn_dim,
    #         dropout=args.enc_dropout,
    #         activation=args.enc_activation,
    #     )
    #     interim_downsampling = PointnetSAModuleVotes(
    #         radius=0.4,
    #         nsample=32,
    #         npoint=args.preenc_npoints // 2,
    #         mlp=[args.enc_dim, 256, 256, args.enc_dim],
    #         normalize_xyz=True,
    #     )
    #
    #     masking_radius = [math.pow(x, 2) for x in [0.4, 0.8, 1.2]]
    #     encoder = MaskedTransformerEncoder(
    #         encoder_layer=encoder_layer,
    #         num_layers=3,
    #         interim_downsampling=interim_downsampling,
    #         masking_radius=masking_radius,
    #     )
    else:
        raise ValueError(f"Unknown encoder type {args.enc_type}")
    return encoder


def build_decoder(args):
    decoder_layer = TransformerDecoderLayer(
        d_model=args.dec_dim,
        nhead=args.dec_nhead,
        dim_feedforward=args.dec_ffn_dim,
        dropout=args.dec_dropout,
        use_mlp=args.use_mlp_in_attention,
    )
    decoder = TransformerDecoder(
        decoder_layer, num_layers=args.dec_nlayers, return_intermediate=False
    )
    return decoder


def build_3detr(args, dataset_config):
    nerf = build_nerf(args)
    pre_encoder = build_preencoder(args)
    encoder = build_encoder(args)
    decoder = build_decoder(args)
    model = Model3DETR(
        pre_encoder,
        encoder,
        decoder,
        dataset_config,
        render_kwargs_test=nerf,
        encoder_dim=args.enc_dim,
        decoder_dim=args.dec_dim,
        mlp_dropout=args.mlp_dropout,
        num_queries=args.nqueries,
        test_only=args.test_only,
        args=args
    )
    # print('args.nqueries: ', args.nqueries)
    output_processor = BoxProcessor(dataset_config)
    return model, output_processor
