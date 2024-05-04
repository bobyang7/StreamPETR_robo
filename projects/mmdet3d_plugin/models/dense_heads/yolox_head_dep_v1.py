import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      bias_init_with_prob)
from mmcv.ops.nms import batched_nms
from mmcv.runner import force_fp32

from mmdet.core import (MlvlPointGenerator, bbox_xyxy_to_cxcywh,
                        build_assigner, build_sampler, multi_apply,
                        reduce_mean)
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.dense_heads.dense_test_mixins import BBoxTestMixin

# from .focal_head import DepthNet
from torch.cuda.amp.autocast_mode import autocast
from ..depth_predictor import DenseDepthNet
from ..depth_predictor.ddn_loss import DDNLoss
from projects.mmdet3d_plugin.models.utils.misc import MLN

@HEADS.register_module()
class YOLOXHead_Depth_v1(BaseDenseHead, BBoxTestMixin):
    """YOLOXHead head used in `YOLOX <https://arxiv.org/abs/2107.08430>`_.
    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of hidden channels in stacking convs.
            Default: 256
        stacked_convs (int): Number of stacking convs of the head.
            Default: 2.
        strides (tuple): Downsample factor of each feature map.
        use_depthwise (bool): Whether to depthwise separable convolution in
            blocks. Default: False
        dcn_on_last_conv (bool): If true, use dcn in the last layer of
            towers. Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by
            the norm_cfg. Bias of conv will be set as True if `norm_cfg` is
            None, otherwise False. Default: "auto".
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer. Default: None.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_obj (dict): Config of objectness loss.
        loss_l1 (dict): Config of L1 loss.
        train_cfg (dict): Training config of anchor head.
        test_cfg (dict): Testing config of anchor head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 strides=[8, 16, 32],
                 use_depthwise=False,
                 dcn_on_last_conv=False,
                 conv_bias='auto',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='IoULoss',
                     mode='square',
                     eps=1e-16,
                     reduction='sum',
                     loss_weight=5.0),
                 loss_obj=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     reduction='sum',
                     loss_weight=1.0),
                 loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 loss_centers2d=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=dict(
                     type='Kaiming',
                     layer='Conv2d',
                     a=math.sqrt(5),
                     distribution='uniform',
                     mode='fan_in',
                     nonlinearity='leaky_relu'),
                 pred_with_depth=False,
                 depthnet_config={},
                 reg_depth_level='p4',
                 loss_depth_weight=1.0,
                 sample_with_score=True,
                 threshold_score=0.05,
                 topk_proposal=None,    # filter proposal num
                 return_context_feat=False, # get_bbox return valid 2d context feature for future Q_feat
                 embedding_cam=False, #embedding cam parameters into feats for depth prediction
                *args, 
                **kwargs,
                 ):

        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.use_depthwise = use_depthwise
        self.dcn_on_last_conv = dcn_on_last_conv
        assert conv_bias == 'auto' or isinstance(conv_bias, bool)
        self.conv_bias = conv_bias
        self.use_sigmoid_cls = True

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_obj = build_loss(loss_obj)
        self.loss_centers2d = build_loss(loss_centers2d)

        self.use_l1 = True  # This flag will be modified by hooks.
        self.loss_l1 = build_loss(loss_l1)

        self.prior_generator = MlvlPointGenerator(strides, offset=0)

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
            self.sampler_ = build_sampler(sampler_cfg, context=self)

        self.pred_with_depth = pred_with_depth
        self.depthnet_config = depthnet_config
        self.reg_depth_level = reg_depth_level  # specify one level to regress depth
        self.sample_with_score = sample_with_score
        self.threshold_score = threshold_score
        self.topk_proposal = topk_proposal
        self.return_context_feat = return_context_feat
        self.embedding_cam = embedding_cam

        self.fp16_enabled = False
        
        self.grid_config = depthnet_config['grid_config']
        self.depth_channels = depthnet_config['depth_channels']
        self.loss_depth_weight = loss_depth_weight
        self._init_layers()


    def _init_layers(self):
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        self.multi_level_conv_centers2d = nn.ModuleList()
        for _ in self.strides:
            self.multi_level_cls_convs.append(self._build_stacked_convs())
            self.multi_level_reg_convs.append(self._build_stacked_convs())
            conv_cls, conv_reg, conv_obj, conv_centers2d = self._build_predictor()
            self.multi_level_conv_cls.append(conv_cls)
            self.multi_level_conv_reg.append(conv_reg)
            self.multi_level_conv_obj.append(conv_obj)
            self.multi_level_conv_centers2d.append(conv_centers2d)

        # depth and uncertainty related
        if self.pred_with_depth:
            self.depthnet = nn.ModuleList()
            for _ in self.strides:
                self.depthnet.append(DenseDepthNet(**self.depthnet_config))

    def _build_stacked_convs(self):
        """Initialize conv layers of a single level head."""
        conv = DepthwiseSeparableConvModule \
            if self.use_depthwise else ConvModule
        stacked_convs = []
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            stacked_convs.append(
                conv(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    bias=self.conv_bias))
        return nn.Sequential(*stacked_convs)

    def _build_predictor(self):
        """Initialize predictor layers of a single level head."""
        conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        conv_reg = nn.Conv2d(self.feat_channels, 4, 1)
        conv_obj = nn.Conv2d(self.feat_channels, 1, 1)
        conv_centers2d = nn.Conv2d(self.feat_channels, 2, 1)
        return conv_cls, conv_reg, conv_obj, conv_centers2d

    def init_weights(self):
        super(YOLOXHead_Depth_v1, self).init_weights()
        # Use prior in model initialization to improve stability
        bias_init = bias_init_with_prob(0.01)
        for conv_cls, conv_obj in zip(self.multi_level_conv_cls,
                                      self.multi_level_conv_obj):
            conv_cls.bias.data.fill_(bias_init)
            conv_obj.bias.data.fill_(bias_init)
      

    def forward_single(self, x, cls_convs, reg_convs, conv_cls, conv_reg,
                       conv_obj, conv_centers2d):
        """Forward feature of a single scale level."""
        if x.dim() == 5:
            bs, n, c, h, w= x.shape
            x = x.reshape(bs*n, c, h, w)

        cls_feat = cls_convs(x)
        reg_feat = reg_convs(x)

        cls_score = conv_cls(cls_feat)
        bbox_pred = conv_reg(reg_feat)
        objectness = conv_obj(reg_feat)
        centers2d_offset = conv_centers2d(reg_feat)

        return cls_score, bbox_pred, objectness, centers2d_offset

    @force_fp32(apply_to=('img', 'img_feats'))
    def forward(self, locations, **data):
        """Forward features from the upstream network.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """
        feats = data['img_feats']
        cls_scores, bbox_preds, objectnesses, centers2d_offsets= multi_apply(self.forward_single, feats,
                           self.multi_level_cls_convs,
                           self.multi_level_reg_convs,
                           self.multi_level_conv_cls,
                           self.multi_level_conv_reg,
                           self.multi_level_conv_obj,
                           self.multi_level_conv_centers2d,
                           )
        out = {
            'enc_cls_scores': cls_scores,
            'enc_bbox_preds': bbox_preds,
            'pred_centers2d_offset': centers2d_offsets,
            'objectnesses':objectnesses,
            'topk_indexes':None
        }
        
        depth_logits_, _ = multi_apply(self.depth_func, [feat.flatten(0, 1) for feat in feats], self.depthnet)
        # len=4
        # [12, 60, 16000], [12, 6, 4000] [12, 6, 1000] [12, 6, 250]
        depth_logits = [depth_logit.flatten(2, 3) for depth_logit in depth_logits_]  # 3x(BN, D, HW)
        pred_depths = [depth_logit.softmax(dim=1) for depth_logit in depth_logits]  # 3x(BN, D, HW)
        depth_logits = torch.cat(depth_logits, dim=-1)
        pred_depths = torch.cat(pred_depths, dim=-1)    # (BN, D, sum(HW))
        out.update(depth_logit=depth_logits, pred_depth=pred_depths, pred_depth_single_stage=depth_logits_[1])
        return out

    def depth_func(self, feat, net):
        return net(feat), None

    @torch.no_grad()
    def get_2dpred_height(self, bbox_preds, level_idx, HW):
        featmap_size = HW
        priors = self.prior_generator.single_level_grid_priors(featmap_size, level_idx, dtype=bbox_preds.dtype,
                                                                    device=bbox_preds.device, with_stride=True)
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]
        hs = whs[..., 1]    # (BN, HW)
        return hs
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses'))
    def get_bboxes(self, preds_dicts,
                   img_metas=None,
                   cfg=None,
                   rescale=False,
                   with_nms=True,
                   threshold_score=0.1,
                   **data
                   ):
        """Transform network outputs of a batch into bbox results.
        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            img_metas (list[dict], Optional): Image meta info. Default None.
            cfg (mmcv.Config, Optional): Test / postprocessing configuration,
                if None, test_cfg would be used.  Default None.
            rescale (bool): If True, return boxes in original image space.
                Default False.
            with_nms (bool): If True, do nms before return boxes.
                Default True.
        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of
                the corresponding box.
        """
        # if self.return_context_feat:
        #     # list(feat with shape (B N C H W)) -> (BN HW C)
        #     fpn_feats = torch.cat([fpn_feat.flatten(0, 1).flatten(2, 3).permute(0, 2, 1) \
        #                            for fpn_feat in data['img_feats']], dim=1)
        if self.sample_with_score:
            threshold_score = self.threshold_score
        else:
            topk_proposal = self.topk_proposal
        cls_scores = preds_dicts['enc_cls_scores']      # shape 3x(BN num_cls Hi Wi), they are logits
        bbox_preds = preds_dicts['enc_bbox_preds']      # shape 3x(BN 4 Hi Wi)
        objectnesses = preds_dicts['objectnesses']      # shape 3x(BN 1 Hi Wi)
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]

        mlvl_priors = self.prior_generator.grid_priors(featmap_sizes, dtype=cls_scores[0].dtype, device=cls_scores[0].device, with_stride=True)       # 3x(Hi*Wi, 4)
        
        assert len(cls_scores) == len(bbox_preds) == len(objectnesses)
        cfg = self.test_cfg if cfg is None else cfg
        # scale_factors = np.array(
        #     [img_meta['scale_factor'] for img_meta in img_metas])

        # flatten cls_scores, bbox_preds and objectness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]

        valid_indices_list = []
        # for obj in objectnesses:
        for i in range(len(objectnesses)):
        # for cls_score in cls_scores:
            # sample_weight = cls_scores[i].topk(1, dim=1).values.sigmoid()       # (BN, 1, Hi, Wi)
            sample_weight = objectnesses[i].sigmoid() * cls_scores[i].topk(1, dim=1).values.sigmoid()
            sample_weight_nms = nn.functional.max_pool2d(sample_weight, (3, 3), stride=1, padding=1)
            sample_weight_nms = sample_weight_nms.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)  # (BN, Hi*Wi, 1)
            sample_weight_ = sample_weight.permute(0, 2, 3, 1).reshape(num_imgs, -1, 1)
            sample_weight = sample_weight_ * (sample_weight_ == sample_weight_nms).float()  # (BN, Hi*Wi, 1)
            valid_indices_list.append(sample_weight)
        valid_indices = torch.cat(valid_indices_list, dim=1)
        flatten_sample_weight = valid_indices.clone()       # (BN, sum(Hi*Wi), 1)

        if self.sample_with_score:
            valid_indices = valid_indices > threshold_score
        else:
            scores, topk_indexes = torch.topk(valid_indices.view(num_imgs // 6, 6, -1, 1), topk_proposal, dim=2)   # 每次选择一个bs里的前topk个 
            valid_indices_ = torch.zeros_like(valid_indices.view(num_imgs // 6, 6, -1, 1))
            valid_indices_[np.arange(num_imgs // 6)[:, np.newaxis, np.newaxis], np.arange(6)[np.newaxis,:,np.newaxis], topk_indexes[:, :, :, 0]] = 1
            # threshold_score = scores[:, :, -1:, :].flatten(0, 1)
            valid_indices = valid_indices_.flatten(0, 1).to(torch.bool)


        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1).sigmoid()     # (BN, sum(Hi*Wi), num_cls)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)               # (BN, sum(Hi*Wi), 4)
        flatten_objectness = torch.cat(flatten_objectness, dim=1).sigmoid()     # (BN, sum(Hi*Wi))
        flatten_priors = torch.cat(mlvl_priors)     # (sum(Hi*Wi), 4)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)  # (BN, sum(Hi*Wi), 4)
        # if rescale:
        #     flatten_bboxes[..., :4] /= flatten_bboxes.new_tensor(
        #         scale_factors).unsqueeze(1)

        result_list = []
        for i in range(num_imgs):
            if self.sample_with_score:
                pred_bbox = flatten_bboxes[i][valid_indices[i].repeat(1, 4)].reshape(-1, 4)     # (M, 4)
            else:
                # pred_bbox = torch.gather(flatten_bboxes[i], 0, topk_indexes[i].repeat(1, 4))
                pred_bbox = flatten_bboxes[i][valid_indices[i].repeat(1, 4)].reshape(-1, 4)     # (M, 4)

            bbox = bbox_xyxy_to_cxcywh(pred_bbox)
            result_list.append(bbox)
        # for i in range(7):
        #     # print(len(img_metas))
        #     bbox = img_metas[0]['offline_2d'][i]
        #     bbox = torch.tensor(bbox, device=flatten_cls_scores.device).reshape(-1, 6)
        #     if bbox is not None:
        #         bbox = bbox_xyxy_to_cxcywh(bbox[..., :4])
        #     result_list.append(bbox)

        bbox2d_scores = flatten_sample_weight[valid_indices].reshape(-1, 1)  # (M, 1)
        outs = {
            'bbox_list': result_list,
            'bbox2d_scores': bbox2d_scores,
            'valid_indices': valid_indices
        }
        # if self.return_context_feat:
        #     # filter context feature and return
        #     _dim = fpn_feats.shape[-1]
        #     context_feat = fpn_feats[valid_indices.repeat(1, 1, _dim)].reshape(-1, _dim)
        #     outs['context_feat'] = context_feat

        # if self.multi_level_pred:
        #     valid_depth_list = []
        #     if self.sample_with_score:
        #         pred_depths = torch.argmax(preds_dicts['pred_depth'].permute(0, 2, 1), dim=-1, keepdim=True) # (BN, HW, 1)
        #         for ith in range(num_imgs):
        #             valid_depth = pred_depths[ith][valid_indices[ith]].reshape(-1, 1).detach()
        #             valid_depth_list.append(valid_depth)   # BN x (Mi, 1)
        #         outs['valid_depth_list'] = valid_depth_list

        # return result_list
        return outs

    def _bbox_decode(self, priors, bbox_preds):
        xys = (bbox_preds[..., :2] * priors[:, 2:]) + priors[:, :2]
        whs = bbox_preds[..., 2:].exp() * priors[:, 2:]

        tl_x = (xys[..., 0] - whs[..., 0] / 2)
        tl_y = (xys[..., 1] - whs[..., 1] / 2)
        br_x = (xys[..., 0] + whs[..., 0] / 2)
        br_y = (xys[..., 1] + whs[..., 1] / 2)

        decoded_bboxes = torch.stack([tl_x, tl_y, br_x, br_y], -1)
        return decoded_bboxes
    
    def _centers2d_decode(self, priors, centers2d):
        centers2d = (centers2d[..., :2] * priors[:, 2:]) + priors[:, :2]
        return centers2d

    def _bboxes_nms(self, cls_scores, bboxes, score_factor, cfg):
        max_scores, labels = torch.max(cls_scores, 1)
        valid_mask = score_factor * max_scores >= cfg.score_thr

        bboxes = bboxes[valid_mask]
        scores = max_scores[valid_mask] * score_factor[valid_mask]
        labels = labels[valid_mask]

        if labels.numel() == 0:
            return bboxes, labels
        else:
            dets, keep = batched_nms(bboxes, scores, labels, cfg.nms)
            return dets, labels[keep]

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'objectnesses', 'centers2d'))
    def loss(self,
             gt_bboxes2d_list,
             gt_labels2d_list,
             centers2d,
             depths,
             preds_dicts,
             img_metas, #len=B
             gt_bboxes_ignore=None):
        """Compute loss of the head.`
        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_priors * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_priors * 4.
            objectnesses (list[Tensor], Optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, 1, H, W).
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
        """
        cls_scores = preds_dicts['enc_cls_scores']
        bbox_preds = preds_dicts['enc_bbox_preds']
        objectnesses = preds_dicts['objectnesses']
        centers2d_offset = preds_dicts['pred_centers2d_offset']
        num_imgs = cls_scores[0].shape[0]
        featmap_sizes = [cls_score.shape[2:] for cls_score in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=cls_scores[0].dtype,
            device=cls_scores[0].device,
            with_stride=True)
            
        flatten_cls_preds = [
            cls_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1,
                                                 self.cls_out_channels)
            for cls_pred in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(num_imgs, -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_objectness = [
            objectness.permute(0, 2, 3, 1).reshape(num_imgs, -1)
            for objectness in objectnesses
        ]
        flatten_centers2d_offset = [
            center2d_offset.permute(0, 2, 3, 1).reshape(num_imgs, -1, 2)
            for center2d_offset in centers2d_offset
        ]

        flatten_cls_preds = torch.cat(flatten_cls_preds, dim=1)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)
        flatten_objectness = torch.cat(flatten_objectness, dim=1)
        flatten_centers2d_offset = torch.cat(flatten_centers2d_offset, dim=1)
        flatten_priors = torch.cat(mlvl_priors)
        flatten_bboxes = self._bbox_decode(flatten_priors, flatten_bbox_preds)

        gt_bboxes = [bboxes2d for i in gt_bboxes2d_list for bboxes2d in i[0]]
        gt_labels = [labels2d for i in gt_labels2d_list for labels2d in i[0]]
        centers2d = [center2d for i in centers2d for center2d in i[0]]

        (pos_masks, cls_targets, obj_targets, bbox_targets, l1_targets, centers2d_target,
         num_fg_imgs) = multi_apply(
             self._get_target_single, flatten_cls_preds.detach(),
             flatten_objectness.detach(),
             flatten_priors.unsqueeze(0).repeat(num_imgs, 1, 1),
             flatten_bboxes.detach(), gt_bboxes, gt_labels, centers2d)

        # The experimental results show that ‘reduce_mean’ can improve
        # performance on the COCO dataset.
        num_pos = torch.tensor(
            sum(num_fg_imgs),
            dtype=torch.float,
            device=flatten_cls_preds.device)
        num_total_samples = max(reduce_mean(num_pos), 1.0)

        pos_masks = torch.cat(pos_masks, 0)
        cls_targets = torch.cat(cls_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)
        centers2d_target = torch.cat(centers2d_target, 0)

        loss_bbox = self.loss_bbox(
            flatten_bboxes.view(-1, 4)[pos_masks],
            bbox_targets) / num_total_samples
        loss_obj = self.loss_obj(flatten_objectness.view(-1, 1),
                                 obj_targets) / num_total_samples
        loss_cls = self.loss_cls(
            flatten_cls_preds.view(-1, self.num_classes)[pos_masks],
            cls_targets) / num_total_samples
        loss_centers2d = self.loss_centers2d(
            flatten_centers2d_offset.view(-1, 2)[pos_masks],
            centers2d_target) / num_total_samples

        loss_dict = dict(
            enc_loss_cls=loss_cls, enc_loss_iou=loss_bbox, enc_loss_obj=loss_obj, enc_loss_centers2d=loss_centers2d)

        if self.use_l1:
            loss_l1 = self.loss_l1(
                flatten_bbox_preds.view(-1, 4)[pos_masks],
                l1_targets) / num_total_samples
            loss_dict.update(enc_loss_bbox=loss_l1)
        # depth related
        if self.pred_with_depth:
            # MonoDETR style: LID and using 3D projective depth as supervision
            device = preds_dicts['enc_cls_scores'][0].device
            depth_loss = self.get_depth_loss(preds_dicts['pred_depth'], depths)
            loss_dict.update(
                depth_loss
            )
            

        return loss_dict


    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        if True:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:,
                                                                           1:]
        return gt_depths.float()

    @force_fp32()
    def get_depth_loss(self, depth_preds, depth_labels):
        
        multi_level = len(depth_labels[0][0])
        bs = len(depth_labels)
        gt_depths = []

        for level in range(multi_level):
            gt_depths_ = []
            for b in range(bs):
                gt_depths_.append(depth_labels[b][0][level])
            gt_depths_ = torch.cat(gt_depths_, dim=0)
            gt_depths_ = self.get_downsampled_gt_depth(gt_depths_)

            gt_depths.append(gt_depths_)

        gt_depths = torch.cat(gt_depths, dim=0)
        depth_preds = depth_preds.permute(0, 2, 1).contiguous().view(-1, self.depth_channels)
        fg_mask = torch.max(gt_depths, dim=1).values > 0.0
        depth_labels = gt_depths[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return dict(loss_depth=self.loss_depth_weight * depth_loss)


    @torch.no_grad()
    def _get_target_single(self, cls_preds, objectness, priors, decoded_bboxes,
                    gt_bboxes, gt_labels, centers2d):
        """Compute classification, regression, and objectness targets for
        priors in a single image.
        Args:
            cls_preds (Tensor): Classification predictions of one image,
                a 2D-Tensor with shape [num_priors, num_classes]
            objectness (Tensor): Objectness predictions of one image,
                a 1D-Tensor with shape [num_priors]
            priors (Tensor): All priors of one image, a 2D-Tensor with shape
                [num_priors, 4] in [cx, xy, stride_w, stride_y] format.
            decoded_bboxes (Tensor): Decoded bboxes predictions of one image,
                a 2D-Tensor with shape [num_priors, 4] in [tl_x, tl_y,
                br_x, br_y] format.
            gt_bboxes (Tensor): Ground truth bboxes of one image, a 2D-Tensor
                with shape [num_gts, 4] in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth labels of one image, a Tensor
                with shape [num_gts].
        """

        num_priors = priors.size(0)
        num_gts = gt_labels.size(0)
        gt_bboxes = gt_bboxes.to(decoded_bboxes.dtype)
        centers2d = centers2d.to(decoded_bboxes.dtype)
        # No target
        if num_gts == 0:
            cls_target = cls_preds.new_zeros((0, self.num_classes))
            bbox_target = cls_preds.new_zeros((0, 4))
            l1_target = cls_preds.new_zeros((0, 4))
            obj_target = cls_preds.new_zeros((num_priors, 1))
            foreground_mask = cls_preds.new_zeros(num_priors).bool()
            centers2d_target = cls_preds.new_zeros((0, 2))
            return (foreground_mask, cls_target, obj_target, bbox_target,
                    l1_target, centers2d_target, 0)

        # YOLOX uses center priors with 0.5 offset to assign targets,
        # but use center priors without offset to regress bboxes.
        offset_priors = torch.cat(
            [priors[:, :2] + priors[:, 2:] * 0.5, priors[:, 2:]], dim=-1)

        assign_result = self.assigner.assign(
            cls_preds.sigmoid() * objectness.unsqueeze(1).sigmoid(),
            offset_priors, decoded_bboxes, gt_bboxes, gt_labels)

        sampling_result = self.sampler.sample(assign_result, priors, gt_bboxes)
        sampling_result_centers2d = self.sampler_.sample(assign_result, priors, centers2d)
        pos_inds = sampling_result.pos_inds
        num_pos_per_img = pos_inds.size(0)

        pos_ious = assign_result.max_overlaps[pos_inds]
        # IOU aware classification score
        cls_target = F.one_hot(sampling_result.pos_gt_labels,
                               self.num_classes) * pos_ious.unsqueeze(-1)
        obj_target = torch.zeros_like(objectness).unsqueeze(-1)
        obj_target[pos_inds] = 1
        bbox_target = sampling_result.pos_gt_bboxes
        l1_target = cls_preds.new_zeros((num_pos_per_img, 4))
        if self.use_l1:
            l1_target = self._get_l1_target(l1_target, bbox_target, priors[pos_inds])
        foreground_mask = torch.zeros_like(objectness).to(torch.bool)
        foreground_mask[pos_inds] = 1

        #centers2d target

        centers2d_labels = sampling_result_centers2d.pos_gt_bboxes
        centers2d_target = cls_preds.new_zeros((num_pos_per_img, 2))
        centers2d_target = self._get_centers2d_target(centers2d_target, centers2d_labels, priors[pos_inds])
        return (foreground_mask, cls_target, obj_target, bbox_target,
                l1_target, centers2d_target, num_pos_per_img)

    def _get_l1_target(self, l1_target, gt_bboxes, priors, eps=1e-8):
        """Convert gt bboxes to center offset and log width height."""
        gt_cxcywh = bbox_xyxy_to_cxcywh(gt_bboxes)
        l1_target[:, :2] = (gt_cxcywh[:, :2] - priors[:, :2]) / priors[:, 2:]
        l1_target[:, 2:] = torch.log(gt_cxcywh[:, 2:] / priors[:, 2:] + eps)
        return l1_target
    
    def _get_centers2d_target(self, centers2d_target, centers2d_labels, priors):
        centers2d_target = (centers2d_labels - priors[:, :2]) / priors[:, 2:]
        return centers2d_target
