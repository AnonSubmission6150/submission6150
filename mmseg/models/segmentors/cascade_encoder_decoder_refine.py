from torch import nn

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class CascadeEncoderDecoderRefine(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 down_ratio,
                 refine_input_ratio,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        self.down_scale = down_ratio
        self.refine_input_ratio = refine_input_ratio
        self.num_stages = num_stages
        super(CascadeEncoderDecoderRefine, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = nn.ModuleList()
        for i in range(self.num_stages - 1):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.refine_head = builder.build_head(decode_head[-1])
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        img_downsample = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])

        if self.refine_input_ratio == 1.:
            img_refine = img
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2] * self.refine_input_ratio), int(img.shape[-1] * self.refine_input_ratio)])
        x = self.extract_feat(img_downsample)
        out = self.decode_head[0].forward_test(x, img_metas, self.test_cfg)
        for i in range(1, self.num_stages - 1):
            out, prev_outputs = self.decode_head[i].forward_test(x, out, img_metas,
                                                   self.test_cfg)
        out = self.refine_head.forward_test(img_refine, prev_outputs, img_metas, self.test_cfg)

        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    def forward_train(self, img, img_metas, gt_semantic_seg):
        img_downsample = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        if self.refine_input_ratio == 1.:
            img_refine = img
        elif self.refine_input_ratio < 1.:
            img_refine = nn.functional.interpolate(img, size=[int(img.shape[-2] * self.refine_input_ratio), int(img.shape[-1] * self.refine_input_ratio)])
        losses = dict()
        x = self.extract_feat(img_downsample)
        loss_decode = self._decode_head_forward_train(x, img_refine, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def _decode_head_forward_train(self, x, img, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode_0'))
        for i in range(1, self.num_stages - 1):
            # forward test again, maybe unnecessary for most methods.
            prev_outputs = self.decode_head[i - 1].forward_test(
                x, img_metas, self.test_cfg)
            loss_decode, prev_features = self.decode_head[i].forward_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_decode, f'decode_{i}'))
        loss_refine, loss_refine_aux16, loss_refine_aux8, *loss_contrsative_list = self.refine_head.forward_train(
                 img, prev_features, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_refine, 'refine'))
        losses.update(add_prefix(loss_refine_aux16, 'refine_aux16'))
        losses.update(add_prefix(loss_refine_aux8, 'refine_aux8'))
        j = 1
        for loss_aux in loss_contrsative_list:
            losses.update(add_prefix(loss_aux, 'contrastive_' + str(j)))
            j += 1
        return losses
