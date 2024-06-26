# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob

from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear

@HEADS.register_module()
class StreamPETRHead(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 stride=16,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner3D',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0)),),
                 test_cfg=dict(max_per_img=100),
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start = 1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 1.0,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is StreamPETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.LID = LID
        self.depth_start = depth_start
        self.stride=stride

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split 

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        self.velo = False
        super(StreamPETRHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        if self.velo:
            # 加入速度100个整数分类头
            cls_branch_vx = []
            for _ in range(self.num_reg_fcs):
                cls_branch_vx.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch_vx.append(nn.LayerNorm(self.embed_dims))
                cls_branch_vx.append(nn.ReLU(inplace=True))
            if self.normedlinear:
                cls_branch_vx.append(NormedLinear(self.embed_dims, 100))
            else:
                cls_branch_vx.append(Linear(self.embed_dims, 100))
            fc_cls_vx = nn.Sequential(*cls_branch_vx)

            cls_branch_vy = []
            for _ in range(self.num_reg_fcs):
                cls_branch_vy.append(Linear(self.embed_dims, self.embed_dims))
                cls_branch_vy.append(nn.LayerNorm(self.embed_dims))
                cls_branch_vy.append(nn.ReLU(inplace=True))
            if self.normedlinear:
                cls_branch_vy.append(NormedLinear(self.embed_dims, 100))
            else:
                cls_branch_vy.append(Linear(self.embed_dims, 100))
            fc_cls_vy = nn.Sequential(*cls_branch_vy)

            # 加入速度小数部分回归头
            reg_branch_v = []
            for _ in range(self.num_reg_fcs):
                reg_branch_v.append(Linear(self.embed_dims, self.embed_dims))
                reg_branch_v.append(nn.ReLU())
            reg_branch_v.append(Linear(self.embed_dims, 2))
            reg_branch_v = nn.Sequential(*reg_branch_v)


            self.cls_branches_vx = nn.ModuleList(
                [fc_cls_vx for _ in range(self.num_pred)])
            self.cls_branches_vy = nn.ModuleList(
                [fc_cls_vy for _ in range(self.num_pred)])
            self.reg_branches_v = nn.ModuleList(
                [reg_branch_v for _ in range(self.num_pred)])



        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, self.embed_dims*4),
                nn.ReLU(),
                nn.Linear(self.embed_dims*4, self.embed_dims),
            )

        self.memory_embed = nn.Sequential(
                nn.Linear(self.in_channels, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        
        # can be replaced with MLN
        self.featurized_pe = SELayer_Linear(self.embed_dims)

        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)


        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(8)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)
            if self.velo:
                for m in self.cls_branches_vx:
                    nn.init.constant_(m[-1].bias, bias_init)        
                for m in self.cls_branches_vy:
                    nn.init.constant_(m[-1].bias, bias_init) 

    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    def pre_update_memory(self, data):
        x = data['prev_exists']  # (bs,)
        # if `prev_exists` is True, current frame is not the first frame in a sub group
        # clear the memory when prev_exists is False at corresponding batch id
        # sometimes the training iter i will use training iter i-1's memory, don't worry,
        # the inputs for iter i and iter i-1 are continuous, because the data sampler only shuffle between clips
        # and donot shuffle frames in the same clip, and different samples in a mini-batch are always from different clips
        B = x.size(0)
        # refresh the memory when the scene changes
        # 每一个query包含内容编码，参考点坐标，时间戳，自身位姿，速度
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
        else:
            # ego_pose: lidar2global, ego_pose_inv: global2lidar, (bs, 4, 4)
            # up to here, all memory pose or coord are `to global` or `in global`, then we can use current data's `ego_pose_inv` to convert it
            # memory_timestamp is `- each history timestamp`, then we can get delta t between current timestamp and each history timestamp
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)  # `current timestamp` + `- each history timestamp`
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose  # K(global to current lidar) @ K(history lidar to global) = history lidar to current lidar
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)  # convert memory_reference_point from global to current lidar
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)  # pop oldest query & set query to 0 when prev_exists is False
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)
        
        # for the first frame, padding pseudo_reference_points (non-learnable)
        # 对于第一帧，256个从bank中拿出来拼接给当前帧query的参考坐标是生成的伪参考坐标
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3] # (128, 3)
            # if x == 1(prev exists is True), self.memory_reference_point and self.memory_egopose remains unchanged
            # if x == 0(prev exists is False), add pseudo_reference_points or eye top num `self.num_propagated`.
            # self.pseudo_reference_points is nn.Embedding and not learnable
            
            # 前128/512 是继承的参考点
            self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        # 他的ego_pose都没取出来，能对吗
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        # 128 + 512 = 640
        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)  # history lidar to current lidar, here rec_ego_pose is eye
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)  # current lidar coordinate
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)  # convert from current lidar to global
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)  # convert to `- each history timestamp`, then in pre_update_memory, convert to `current timestamp` - `each history timestamp`
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose  # K(current lidar to global) @ K(history lidar to current lidar) = history lidar to global

    def position_embeding(self, data, memory_centers, topk_indexes, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)  #[2,4224,2]
        LEN = intrinsic.size(1)

        num_sample_tokens = topk_indexes.size(1) if topk_indexes is not None else LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        # memory_center就是location，即网格中心点
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        
        # 把所有token按照topk索引排序后，对每个toke复制深度64份
        topk_centers = topk_gather(memory_centers, topk_indexes).repeat(1, 1, D, 1)  # (B, LEN, D, 2)
        # 为每个token赋予64个深度位置的具体值
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)  # (B, LEN, D, 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)  # (B, LEN, D, 3)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)  # (B, LEN, D, 4)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)  # (u*z, v*z, z, 1)

        coords = coords.unsqueeze(-1)  # (B, LEN, D, 4, 1)

        img2lidars = data['lidar2img'].inverse()
        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)
        img2lidars = topk_gather(img2lidars, topk_indexes)

        """
        下面这段代码是位置编码部分，对6个图像所有的token的D个深度在通过内参变换为三维坐标后，对坐标进行归一化，
        并对每个token的D个深度的3个坐标值通道压在一起得到(B, LEN, D*3)，再做反sigmoid生成编码后纬度变为256，
        得到所有token的位置编码
        """

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]  # (B, LEN, D, 3) in lidar coord
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])  # norm coord in lidar to 0~1
        coords3d = coords3d.reshape(B, -1, D*3)  # (B, LEN, D*3), 0~1 norm
      
        # 位置编码使用对归一化后的3d位置进行反sigmoid
        pos_embed  = inverse_sigmoid(coords3d)  # why PE uses inverse_sigmoid but spatial alignment uses sigmoid?
        coords_position_embeding = self.position_encoder(pos_embed)  # (B, LEN, embed_dims)
        intrinsic = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        # 对每个token构造cone特征，是内参和该token对应两个深度位置形成的射线的拼接特征
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)  # select two points along tracing ray (B, LEN, 2+6), (fx, fy, x1,y1,z1, x2,y2,z2)

        return coords_position_embeding, cone

    def temporal_alignment(self, query_pos, tgt, reference_points):
        """
            query_pos: (bs, aug_gt_num + num_query, embed_dims)
            tgt: : (bs, aug_gt_num + num_query, embed_dims)
            reference_points: (bs, aug_gt_num + num_query, 3)
        """
        B = query_pos.size(0)

        # 时序的query有512个
        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        
        # temp_pos是时序query的位置编码，temp_memory是时序query的特征编码
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point)) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            # current frame query alignment
            # 这里为什么用zeros_like
            # 生成当前query的运动信息编码
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)  # (bs, aug_gt_num + num_query, 3 + 12)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)  # (bs, aug_gt_num + num_query, 15*12)
            # 用运动信息编码通过MLN去更新query的特征编码和位置编码
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            
            # 时序的query同理使用MLN
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()  # (bs, memory_len, 2+1+12)
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)  # (bs, memory_len, 15*12)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        # why adding time embedding here?
        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))  # (bs, aug_gt_num + num_query, embed_dims)
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())  # (bs, memory_len, embed_dims)

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            # 这个rec_ego_pose怀疑是多加了一个num_propagated
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]
            
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        # the augmented gt ref points will compute HungarianAssigner independently
        if self.training and self.with_dn:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in targets]
        
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])
        
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                # 抖动范围在框的尺度的一半之内
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                # 生成-1到1的随机值
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                # 对框的中心点进行抖动
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                # 将抖动后的中心点位置进行归一化
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
                # reference_points are also initialized in range [0, 1]

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                # 对抖动太大的进行过滤
                mask = torch.norm(rand_prob, 2, 1) > self.split  # ignore too large aug
                known_labels[mask] = self.num_classes
            
            single_pad = int(max(known_num))  # pad to max gt num
            pad_size = int(single_pad * self.scalar)  # noised gt
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)  # (max_gt_num * scalar, 3)
            # 330个dn的点和300个参考点
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)  # (bs, max_gt_num*scalar + 300, 3)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                """ 0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                    14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
                    28,  29,  30,  31,  32,|   0,   1,   2,   3,   4,   5,   6,   7,   8,
                    9,  10,  11,||  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,
                    44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,
                    58,  59,  60,  61,  62,  63,  64,  65,|  33,  34,  35,  36,  37,  38,
                    39,  40,  41,  42,  43,  44,||  66,  67,  68,  69,  70,  71,  72,  73,
                    74,  75,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,
                    88,  89,  90,  91,  92,  93,  94,  95,  96,  97,  98,  66,  67,  68,
                    69,  70,  71,  72,  73,  74,  75,  76,  77,  99, 100, 101, 102, 103,"""
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()  # inserted indices(0~max_gt_num*scalar) for each gt box in each sample considering its repeatance
            if len(known_bid):
                # 450个框是由45个gt框，每个框抖动10个得到。
                # 45个框中，33个来自bs0，12来自bs1，捋清楚之后再往paddded_reference_point中填，最终只影响[2,630,3]中的[2,330,3]
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)  # there exists zero padding ref points

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct, since in inference there is no gt, attention should not dependant on input gt
            # 掩码为true的地方是不能看到的地方
            # dn框和预测框之间相互不能看见
            attn_mask[pad_size:, :pad_size] = True
            
            # reconstruct cannot see each other
            # 10组dn框，组与组之间不能看见
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
             
            # update dn mask for temporal modeling
            query_size = pad_size + self.num_query + self.num_propagated    #330+300+128
            tgt_size = pad_size + self.num_query + self.memory_len  #330+300+512
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            # dn框和非dn框之间相互不能看见
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask  # modify here when traing single-frame model

            mask_dict = { #假设一个batch有两个sample，共有33+12个gt
                'known_indice': torch.as_tensor(known_indice).long(),  # 则为0-44循环10次
                'batch_idx': torch.as_tensor(batch_idx).long(), # 则为33个0和12个1
                'map_known_indice': torch.as_tensor(map_known_indice).long(), # 上面那个很长的举例注释
                'known_lbs_bboxes': (known_labels, known_bboxs),    #450个dn框的标签和框
                'know_idx': know_idx,   #33个1，和12个1
                'pad_size': pad_size    #取决于一个batch中gt最多的那个sample的十倍
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is StreamPETRHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    

    def forward(self, memory_center, img_metas, topk_indexes=None,  **data):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # zero init the memory bank
        self.pre_update_memory(data)

        x = data['img_feats']
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        
        # memory是根据置信度排名的图像token
        memory = topk_gather(memory, topk_indexes)

        # memory_center是所有图像token的中心位置，topk_indices是图像token按置信度排名的索引
        # 返回的pos_embed就是PETR的位置编码的示意图的下面那条黄色的支路，表示融合了3d信息的位置编码
        pos_embed, cone = self.position_embeding(data, memory_center, topk_indexes, img_metas)  # (B, LEN, embed_dims), (B, LEN, 2+6)

        # 对所有token进行特征编码
        # 图像token的特征编码叫memory，位置编码叫pos_embed
        memory = self.memory_embed(memory)  # 2d feature, (B, LEN, embed_dims)

        # spatial_alignment in focal petr, why not just use pos_embed?
        # 使用所有token的特征编码和cone进行空间对齐（spatial_alignment）
        # 这里是空间对齐的部分，输入的x是所有token的编码，c是cone，空间对齐采用mlp生成加权系数gamma和beta来生成输出
        memory = self.spatial_alignment(memory, cone)  # see focal petr

        # 将融合了3d信息的位置编码和2d的特征编码进行融合，得到变量pos_embed，这里都是PETR的思想
        pos_embed = self.featurized_pe(pos_embed, memory)  # se channel attention

        # [300, 3]
        reference_points = self.reference_points.weight
        
        # reference_points: (bs, aug_gt_num + num_query, 3)
        # attn_mask: (aug_gt_num + num_query + propagated, aug_gt_num + num_query + memory_len)
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)
        #query的位置编码是query_pos,内容编码是tgt
        query_pos = self.query_embedding(pos2posemb3d(reference_points))  # (bs, aug_gt_num + num_query, embed_dims)
        tgt = torch.zeros_like(query_pos)  # query_context

        # prepare for the tgt and query_pos using mln.
        
        """
        输入为：
            reference_points, query_pos, tgt 分别是参考点(query)的位置，位置编码，内容编码
        输出为：
            主要是在对当前帧的query和memory的query分别做MLN
            当前帧的query经过MLN后:
                tgt, query_pos: (bs, aug_gt_num + num_query + propagated, 256)
                reference_points: (bs, aug_gt_num + num_query + propagated, 3), in sigmoid format
            memery的query经过MLN后:
                temp_pos, temp_memory 分别是memory的query的位置编码，内容编码
                temp_memory, temp_pos: (bs, memory_len - propagated, 256)
                rec_ego_pose: (bs, aug_gt_num + num_query + propagated, 4, 4), torch.eye
        """
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)

        # transformer here is a little different from PETR
        # 输入：
        # memory图像特征编码[2,4224,256]，pos_embed图像位置编码[2,4224,256]，
        # tgt是query内容编码[2,758,256]，query_pos是query位置编码[2,758,256]
        # temp_memory是时序query特征编码[2,384,256], temp_pos是时序query位置编码[2,384,256]
        # attn_mask[758,1142]
        outs_dec, _ = self.transformer(memory, tgt, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)  # (num_layers, bs, aug_gt_num + num_query + propagated, embed_dims)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        if self.velo:
            outputs_classes_vx = []
            outputs_classes_vy = []
            outputs_coords_v = []

        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            if self.velo:
                outputs_class_vx = self.cls_branches_vx[lvl](outs_dec[lvl])
                outputs_class_vy = self.cls_branches_vy[lvl](outs_dec[lvl])
                tmp = self.reg_branches_v[lvl](outs_dec[lvl])
                tmp[..., :] = tmp[..., :].sigmoid()

                outputs_coord_v = tmp
                outputs_classes_vx.append(outputs_class_vx)
                outputs_classes_vy.append(outputs_class_vy)
                outputs_coords_v.append(outputs_coord_v)


        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        
        if self.velo:
            all_cls_scores_vx = torch.stack(outputs_classes_vx)
            all_cls_scores_vy = torch.stack(outputs_classes_vy)
            all_bbox_preds_velo = torch.stack(outputs_coords_v)

        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            
            if self.velo:
                output_known_class_vx = all_cls_scores_vx[:, :, :mask_dict['pad_size'], :]
                output_known_class_vy = all_cls_scores_vx[:, :, :mask_dict['pad_size'], :]
                output_known_coord_v = all_bbox_preds_velo[:, :, :mask_dict['pad_size'], :]
                outputs_class_vx = all_cls_scores_vx[:, :, mask_dict['pad_size']:, :]
                outputs_class_vy = all_cls_scores_vy[:, :, mask_dict['pad_size']:, :]
                outputs_coord_v = all_bbox_preds_velo[:, :, mask_dict['pad_size']:, :]

                mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord, output_known_class_vx, output_known_class_vy, output_known_coord_v)
                outs = {
                    'all_cls_scores': outputs_class,
                    'all_bbox_preds': outputs_coord,
                    'all_cls_scores_vx': outputs_class_vx,
                    'all_cls_scores_vy': outputs_class_vy,
                    'all_bbox_preds_v': outputs_coord_v,
                    'dn_mask_dict':mask_dict,
                }
            else:
                mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
                outs = {
                    'all_cls_scores': outputs_class,
                    'all_bbox_preds': outputs_coord,
                    'dn_mask_dict':mask_dict,

            }
        else:
            if self.velo:
                outs = {
                    'all_cls_scores': all_cls_scores,
                    'all_bbox_preds': all_bbox_preds,
                    'all_cls_scores_vx': all_cls_scores_vx,
                    'all_cls_scores_vy': all_cls_scores_vy,
                    'all_bbox_preds_v': all_bbox_preds_velo,
                    'dn_mask_dict':None,
                }
            else:                
                outs = {
                    'all_cls_scores': all_cls_scores,
                    'all_bbox_preds': all_bbox_preds,             
                    'dn_mask_dict':None,
                }

        return outs
    
    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes'][:2]
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]  # juse like `known_bid`, including bg ref point
        if len(output_known_class) > 0:
            # extrace valid output: from (num_layers, bs, aug_gt_num, dim), some indices are padded zeros
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)  # (valid_num, num_layers, dim)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        # gt labels and its corresponding prediction in each decoder layer for augmented ref points
        # each augmented prediction's has a corresponding gtboxes3d before the augmentation
        # its ref points are augmented gtboxes
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indexes for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indexes for each image.
                - neg_inds (Tensor): Sampled negative indexes for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                                gt_labels, gt_bboxes_ignore, self.match_costs, self.match_with_velo)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        if sampling_result.num_gts > 0:
            bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        return (labels, label_weights, bbox_targets, bbox_weights, 
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single, cls_scores_list, bbox_preds_list,
             gt_labels_list, gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_scores,
                    bbox_preds,
                    cls_scores_vx,
                    cls_scores_vy,
                    velo_preds,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # single decoder layer loss at the same timestamp for all sample in a mini-batch
        num_imgs = cls_scores.size(0)  # (bs, num_query + propagated, dim)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list, 
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights


        # 实验3: 1-5
        # weight_mask = torch.clone(bbox_weights)
        # weight_mask[normalized_bbox_targets < 10] = 1
        # weight_mask[normalized_bbox_targets >= 10] = normalized_bbox_targets[normalized_bbox_targets >= 10] / 10
        # weight_mask[:, :8] = 1
        # bbox_weights = bbox_weights * weight_mask


        # a = bbox_preds.clone()
        # 指对数解码-方案1
        # mask_pos = bbox_preds[:,8:10] >= 0
        # mask_neg = bbox_preds[:,8:10] < 0
        # bbox_preds_velo = (10 ** bbox_preds[:, 8:10] - 1) * mask_pos + (1 - 10 ** (-1 * bbox_preds[:, 8:10])) * mask_neg
        
        # bbox_preds_velo[bbox_preds_velo > 50] = 50
        # bbox_preds_velo[bbox_preds_velo < -50] = -50
        # a = bbox_preds_velo.max()
        # if a > 50:
        #     print(a)
        # bbox_predictions = torch.cat((bbox_preds[:, :8], bbox_preds_velo), dim=1)
        

        # 指对数解码-方案2
        # mask_pos_en = normalized_bbox_targets[:,8:10] >= 0
        # mask_neg_en = normalized_bbox_targets[:,8:10] < 0
        # encode_velo = (torch.log2(normalized_bbox_targets[:, 8:10] * mask_pos_en + 1)) * mask_pos_en - (torch.log2(1 - normalized_bbox_targets[:, 8:10] * mask_neg_en)) * mask_neg_en
        # have_nan = (~torch.isfinite(encode_velo)).sum()
        # if have_nan:
        #     print("这里有nan值")
        # normalized_bbox_targets_en = torch.cat((normalized_bbox_targets[:, :8], encode_velo), dim=1)


        # 累计高斯分布解码-方案1
        # bbox_preds_velo = 0 + 15 * torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 * torch.clamp(bbox_preds[:, 8:10], min=0.0001, max=0.9999) - 1)
        # bbox_predictions = torch.cat((bbox_preds[:, :8], bbox_preds_velo), dim=1)
     
        # 累计高斯分布解码-方案2
        encode_velo = 0.5 * (1 + torch.erf((normalized_bbox_targets[:, 8:10] - 0) / (15 * torch.sqrt(torch.tensor(2.0)))))
        normalized_bbox_targets_en = torch.cat((normalized_bbox_targets[:, :8], encode_velo), dim=1)



        # loss_bbox = self.loss_bbox(
        #         bbox_predictions[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)
                
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets_en[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        # loss_bbox = self.loss_bbox(
        #         bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)


        if self.velo:
            # 制作100类速度标签，对于每一个query的速度，
            #  v >= 0 时，v // 1 + 50 作为他的label，例如0.5 和49.5 的label为50和99
            #  v < 0 时，v // 1 + 49 作为他的label，例如-0.5和-49.5 的label为49和0
            
            vx_gt = bbox_targets[:, 7]
            mask_pos_vx = vx_gt >= 0
            mask_neg_vx = vx_gt < 0
            vx_label = (vx_gt // 1 + 50) * mask_pos_vx + (vx_gt // 1 + 49) * mask_neg_vx
            vx_label_weights = label_weights.clone()
            vx_label_weights[bbox_targets[:, 7] > 5] = 10
            cls_scores_vx = cls_scores_vx.reshape(-1, cls_scores_vx.size(-1))

            loss_cls_vx = self.loss_cls(
                cls_scores_vx, vx_label.long(), vx_label_weights, avg_factor=cls_avg_factor)


            vy_gt = bbox_targets[:, 8]
            mask_pos_vy = vy_gt >= 0
            mask_neg_vy = vy_gt < 0
            vy_label = (vy_gt // 1 + 50) * mask_pos_vy + (vy_gt // 1 + 49) * mask_neg_vy
            vy_label_weights = label_weights.clone()
            vy_label_weights[bbox_targets[:, 7] > 5] = 10
            cls_scores_vy = cls_scores_vy.reshape(-1, cls_scores_vy.size(-1))

            loss_cls_vy = self.loss_cls(
                cls_scores_vy, vy_label.long(), vy_label_weights, avg_factor=cls_avg_factor)

            # 速度的小数部分回归头
            gt_velo = bbox_targets[:, 7:9]
            mask_pos_v = gt_velo >= 0
            mask_neg_v = gt_velo < 0
            gt_v = (gt_velo % 1) * mask_pos_v + (1 + (gt_velo) % 1) * mask_neg_v
            v_weights = bbox_weights.clone()[:,:2]
            velo_preds = velo_preds.reshape(-1, velo_preds.size(-1))

            loss_v = self.loss_bbox(
                velo_preds[isnotnan], gt_v[isnotnan], v_weights[isnotnan], avg_factor=num_total_pos)


            loss_cls_vx = torch.nan_to_num(loss_cls_vx)
            loss_cls_vy = torch.nan_to_num(loss_cls_vy)
            loss_v = torch.nan_to_num(loss_v)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        if self.velo:
            return loss_cls, loss_bbox, loss_cls_vx, loss_cls_vy, loss_v 
        else:
            return loss_cls, loss_bbox, [], [], []
   
    def dn_loss_single(self,
                    cls_scores,
                    bbox_preds,
                    known_bboxs,
                    known_labels,
                    num_total_pos=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        # single decoder layer loss at the same timestamp for all sample in a mini-batch
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 3.14159 / 6 * self.split * self.split  * self.split ### positive rate
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)
        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, known_labels.long(), label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)

        bbox_weights = bbox_weights * self.code_weights

        
        loss_bbox = self.loss_bbox(
                bbox_preds[isnotnan, :10], normalized_bbox_targets[isnotnan, :10], bbox_weights[isnotnan, :10], avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        
        return self.dn_weight * loss_cls, self.dn_weight * loss_bbox
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             preds_dicts,
             gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indexes for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']  # (num_layers, bs, num_query + propagated, dim)
        all_bbox_preds = preds_dicts['all_bbox_preds']  # (num_layers, bs, num_query + propagated, dim)

        if self.velo:
            all_cls_scores_vx = preds_dicts['all_cls_scores_vx']
            all_cls_scores_vy = preds_dicts['all_cls_scores_vy']
            all_velo_preds = preds_dicts['all_bbox_preds_v']  # (num_layers, bs, num_query + propagated, dim)
        else:
            all_cls_scores_vx, all_cls_scores_vy, all_velo_preds = torch.zeros_like(all_cls_scores), torch.zeros_like(all_cls_scores), torch.zeros_like(all_cls_scores)
        
        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        gt_bboxes_list = [torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
            dim=1).to(device) for gt_bboxes in gt_bboxes_list]  # [gt_boxes] * bs

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox, losses_cls_vx, losses_cls_vy, losses_v = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds, 
            all_cls_scores_vx, all_cls_scores_vy, all_velo_preds, 
            all_gt_bboxes_list, all_gt_labels_list, 
            all_gt_bboxes_ignore_list)

        loss_dict = dict()

        # loss_dict['size_loss'] = size_loss
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        if self.velo:
            loss_dict['loss_cls_vx'] = losses_cls_vx[-1]
            loss_dict['loss_cls_vy'] = losses_cls_vy[-1]
            loss_dict['loss_v'] = losses_v[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1],
                                           losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1

        if self.velo:
            num_dec_layer = 0
            for loss_cls_vx_i, loss_cls_vy_i, loss_v_i in zip(losses_cls_vx[:-1],losses_cls_vy[:-1],
                                               losses_v[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls_vx'] = loss_cls_vx_i
                loss_dict[f'd{num_dec_layer}.loss_cls_vy'] = loss_cls_vy_i
                loss_dict[f'd{num_dec_layer}.loss_v'] = loss_v_i
                num_dec_layer += 1

        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            all_known_bboxs_list = [known_bboxs for _ in range(num_dec_layers)]
            all_known_labels_list = [known_labels for _ in range(num_dec_layers)]
            all_num_tgts_list = [
                num_tgt for _ in range(num_dec_layers)
            ]
            
            dn_losses_cls, dn_losses_bbox = multi_apply(
                self.dn_loss_single, output_known_class, output_known_coord,
                all_known_bboxs_list, all_known_labels_list, 
                all_num_tgts_list)
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i
                num_dec_layer += 1
                
        elif self.with_dn:  # never be exectued
            # dn_losses_cls, dn_losses_bbox = multi_apply(
            #     self.loss_single, all_cls_scores, all_bbox_preds,
            #     all_gt_bboxes_list, all_gt_labels_list, 
            #     all_gt_bboxes_ignore_list)

            dn_losses_cls, dn_losses_bbox, _, _, _ = multi_apply(
                self.loss_single, all_cls_scores, all_bbox_preds,
                all_cls_scores_vx, all_cls_scores_vy, all_velo_preds, 
                all_gt_bboxes_list, all_gt_labels_list, 
                all_gt_bboxes_ignore_list)

            loss_dict['dn_loss_cls'] = dn_losses_cls[-1].detach()
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1].detach()     
            num_dec_layer = 0
            for loss_cls_i, loss_bbox_i in zip(dn_losses_cls[:-1],
                                            dn_losses_bbox[:-1]):
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i.detach()     
                loss_dict[f'd{num_dec_layer}.dn_loss_bbox'] = loss_bbox_i.detach()     
                num_dec_layer += 1

        return loss_dict


    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
