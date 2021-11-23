import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmseg.ops import resize
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .stdc_head import STDCNetLightCNN8
from .lpls_utils import Lap_Pyramid_Conv


class BiSeNetOutput(nn.Module):

    def __init__(self, conv_cfg, norm_cfg, act_cfg, in_c, mid_c, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()

        self.conv_bn_relu = ConvModule(
            in_c,
            mid_c,
            3,
            stride=1,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        self.conv_out = nn.Conv2d(mid_c, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv_out(x)
        return x

class SRDecoder(nn.Module):
    # super resolution decoder
    def __init__(self, conv_cfg, norm_cfg, act_cfg, channels=128, up_list=[2, 2, 2]):
        super(SRDecoder, self).__init__()
        self.conv1 = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.up1 = nn.Upsample(scale_factor=up_list[0])
        self.conv2 = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.up2 = nn.Upsample(scale_factor=up_list[1])
        self.conv3 = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.up3 = nn.Upsample(scale_factor=up_list[2])
        self.conv_sr = BiSeNetOutput(conv_cfg, norm_cfg, act_cfg, channels, channels // 2, 3, kernel_size=1)

    def forward(self, x, fa=False):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        feats = self.conv3(x)
        outs = self.conv_sr(feats)
        if fa:
            return feats, outs
        else:
            return outs


class ChannelAtt(nn.Module):
    def __init__(self, in_channels, out_channels, conv_cfg, norm_cfg, act_cfg, s_ratio=2):
        super(ChannelAtt, self).__init__()
        self.conv_bn_relu = ConvModule(in_channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg, act_cfg=act_cfg)
        # self.conv_head = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg,
                                   norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x, fre=False):
        """Forward function."""
        feat = self.conv_bn_relu(x)
        if fre:
            h, w = feat.size()[2:]
            h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h - 1, :], 2)
            w_tv = torch.pow(feat[..., 1:] - feat[..., :w - 1], 2)
            atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
        else:
            atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_1x1(atten)
        # atten = atten.sigmoid()
        return feat, atten


class ChannelAffinity(nn.Module):
    def __init__(self, channels, conv_cfg, norm_cfg, act_cfg, ext=2, r=16):
        super(ChannelAffinity, self).__init__()
        self.r = r
        self.g1 = nn.Parameter(torch.zeros(1))
        self.g2 = nn.Parameter(torch.zeros(1))
        # self.g1 = 0
        # self.g2 = 0
        self.spatial_mlp = nn.Sequential(nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels))
        self.spatial_att = ChannelAtt(channels * ext, channels, conv_cfg, norm_cfg, act_cfg)
        # self.spatial_head = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.context_mlp = nn.Sequential(*[nn.Linear(channels * 2, channels), nn.ReLU(), nn.Linear(channels, channels)])
        self.context_att = ChannelAtt(channels, channels, conv_cfg, norm_cfg, act_cfg)
        self.context_head = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)
        self.smooth = ConvModule(channels, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg,
                                 act_cfg=None)

    def forward(self, sp_feat, co_feat):
        # xx_att: B x C x 1 x 1
        s_feat, s_att = self.spatial_att(sp_feat)
        c_feat, c_att = self.context_att(co_feat)
        s_att_residual = s_att
        c_att_residual = c_att
        # channel affinity
        b, c, h, w = s_att.size()
        s_att_split = s_att.view(b, self.r, c // self.r)
        c_att_split = c_att.view(b, self.r, c // self.r)
        chl_affinity = torch.bmm(s_att_split, c_att_split.permute(0, 2, 1))
        chl_affinity = chl_affinity.view(b, -1)
        sp_mlp_out = F.relu(self.spatial_mlp(chl_affinity))
        co_mlp_out = F.relu(self.context_mlp(chl_affinity))
        re_s_att = torch.sigmoid(s_att + self.g1 * sp_mlp_out.unsqueeze(-1).unsqueeze(-1))
        re_c_att = torch.sigmoid(c_att + self.g2 * co_mlp_out.unsqueeze(-1).unsqueeze(-1))
        c_feat = torch.mul(c_feat, re_c_att)
        s_feat = torch.mul(s_feat, re_s_att)
        c_feat = F.interpolate(c_feat, s_feat.size()[2:], mode='bilinear', align_corners=False)
        c_feat = self.context_head(c_feat)
        out = self.smooth(s_feat + c_feat)
        return s_feat, c_feat, out


@HEADS.register_module()
class ISDNet(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels, **kwargs):
        super(ISDNet, self).__init__(**kwargs)
        self.down_ratio = down_ratio
        self.fuse8 = ChannelAffinity(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=2)
        self.fuse16 = ChannelAffinity(self.channels, self.conv_cfg, self.norm_cfg, self.act_cfg, ext=4)
        # self.stdc_net = STDCNetLightCNN8(in_channels=6, pretrain_model="/code/mmckpts/STDCNet813M_73.91.tar")
        self.stdc_net = STDCNetLightCNN8(in_channels=6)
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        self.conv_seg = BiSeNetOutput(self.conv_cfg, self.norm_cfg, self.act_cfg, self.channels,
                                      self.channels // 2, self.num_classes, kernel_size=1)
        self.sr_decoder = SRDecoder(self.conv_cfg, self.norm_cfg, self.act_cfg, channels=self.channels,
                                    up_list=[4, 2, 2])

    def forward(self, inputs, prev_output, train_flag=True):
        """Forward function."""
        pyr_list = self.lap_prymaid_conv.pyramid_decom(inputs)
        h1 = pyr_list[0]
        h2 = F.interpolate(pyr_list[1], pyr_list[0].size()[2:], mode='bilinear', align_corners=False)
        high_fre_input = torch.cat([h1, h2], dim=1)
        # the os8, os16 feature of full size high frequency map
        stdc_8, stdc_16 = self.stdc_net(high_fre_input)
        # output by deeplabv3(backbone: resnet18)
        pre_features = prev_output[0]
        # stage 1
        red_stdc_16, feat_arm_16, feat_16 = self.fuse16(stdc_16, pre_features)
        # stage 2
        _, feat_arm_8, feat_8 = self.fuse8(stdc_8, feat_16)
        output = self.cls_seg(feat_8)
        if train_flag:
            feats, output_sr = self.sr_decoder(pre_features, True)
            losses_re = self.image_recon_loss(inputs, output_sr, re_weight=0.1)
            losses_fa = self.feature_affinity_loss(pre_features, feats, fa_weight=1.)
            return output, losses_re, losses_fa
        else:
            return output

    def image_recon_loss(self, img, pred, re_weight=0.5):
        loss = dict()
        if pred.size()[2:] != img.size()[2:]:
            pred = F.interpolate(pred, img.size()[2:], mode='bilinear', align_corners=False)
        recon_loss = F.mse_loss(pred, img) * re_weight
        loss['recon_losses'] = recon_loss
        return loss

    def feature_affinity_loss(self, seg_feats, sr_feats, fa_weight=1., eps=1e-6):
        # seg_feats and sr_feats: B x 128 x (x32) x (x32), B x 128 x (x1) x (x1),
        if seg_feats.size()[2:] != sr_feats.size()[2:]:
            sr_feats = F.interpolate(sr_feats, seg_feats.size()[2:], mode='bilinear', align_corners=False)
        # seg_feats = F.interpolate(seg_feats, [h1*4, w1*4], mode='bilinear', align_corners=False)
        loss = dict()
        # H1 and W1 are the height and width of these two features
        # flatten:
        seg_feats_flatten = torch.flatten(seg_feats, start_dim=2)
        sr_feats_flatten = torch.flatten(sr_feats, start_dim=2)
        # L2 norm
        seg_norm = torch.norm(seg_feats_flatten, p=2, dim=2, keepdim=True)
        sr_norm = torch.norm(sr_feats_flatten, p=2, dim=2, keepdim=True)
        # similiarity
        seg_feats_flatten = seg_feats_flatten / (seg_norm + eps)
        sr_feats_flatten = sr_feats_flatten / (sr_norm + eps)
        seg_sim = torch.matmul(seg_feats_flatten.permute(0, 2, 1), seg_feats_flatten)
        sr_sim = torch.matmul(sr_feats_flatten.permute(0, 2, 1), sr_feats_flatten)
        # L1 loss
        loss['fa_loss'] = F.l1_loss(seg_sim, sr_sim.detach()) * fa_weight
        # loss['fa_loss'] = F.kl_div(act_seg_feats, act_sr_feats.detach(),reduction='mean')
        return loss

    def feature_affinity_loss(self, seg_feats, sr_feats, fa_weight=1., eps=1e-6):
        # seg_feats and sr_feats: B x 128 x (x32) x (x32), B x 128 x (x1) x (x1),
        if seg_feats.size()[2:] != sr_feats.size()[2:]:
            sr_feats = F.interpolate(sr_feats, seg_feats.size()[2:], mode='bilinear', align_corners=False)
        loss = dict()
        # H1 and W1 are the height and width of these two features
        # flatten:
        seg_feats_flatten = torch.flatten(seg_feats, start_dim=2)
        sr_feats_flatten = torch.flatten(sr_feats, start_dim=2)
        # L2 norm
        seg_norm = torch.norm(seg_feats_flatten, p=2, dim=2, keepdim=True)
        sr_norm = torch.norm(sr_feats_flatten, p=2, dim=2, keepdim=True)
        # similiarity
        seg_feats_flatten = seg_feats_flatten / (seg_norm + eps)
        sr_feats_flatten = sr_feats_flatten / (sr_norm + eps)
        seg_sim = torch.matmul(seg_feats_flatten.permute(0, 2, 1), seg_feats_flatten)
        sr_sim = torch.matmul(sr_feats_flatten.permute(0, 2, 1), sr_feats_flatten)
        # L1 loss
        loss['fa_loss'] = F.l1_loss(seg_sim, sr_sim.detach()) * fa_weight
        return loss

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg):
        seg_logits, losses_re, losses_fa = self.forward(inputs, prev_output)
        losses = self.losses(seg_logits, gt_semantic_seg)
        return losses, losses_re, losses_fa

    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        """Forward function for testing.

        Args:
            inputs (list[Tensor]): List of multi-level img features.
            prev_output (Tensor): The output of previous decode head.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Output segmentation map.
        """
        return self.forward(inputs, prev_output, False)

