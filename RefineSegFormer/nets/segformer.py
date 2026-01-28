import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class SegFormerHeadRefineFPN(nn.Module):
    def __init__(self, num_classes=20, in_channels=[32, 64, 160, 256], embedding_dim=256, dropout_ratio=0.1, num_recursions=2):
        super().__init__()
        assert num_recursions in [1, 2], "Only support 1 or 2 recursions for efficiency"
        self.num_recursions = num_recursions
        c1_in, c2_in, c3_in, c4_in = in_channels

        self.lateral_c1 = nn.Conv2d(c1_in, embedding_dim, 1)
        self.lateral_c2 = nn.Conv2d(c2_in, embedding_dim, 1)
        self.lateral_c3 = nn.Conv2d(c3_in, embedding_dim, 1)
        self.lateral_c4 = nn.Conv2d(c4_in, embedding_dim, 1)

        self.smooth_c3 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)
        self.smooth_c2 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)
        self.smooth_c1 = nn.Conv2d(embedding_dim, embedding_dim, 3, padding=1)

        if num_recursions == 2:
            self.feedback_offset_conv = nn.Conv2d(embedding_dim, 18, kernel_size=3, padding=1)
            self.feedback_deform_conv = DeformConv2d(
                in_channels=c4_in,
                out_channels=c4_in,
                kernel_size=3,
                padding=1
            )
            self.feedback_act = nn.ReLU(inplace=False) 

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def _fpn_forward(self, c1, c2, c3, c4):
        p4 = self.lateral_c4(c4)
        p3 = self.lateral_c3(c3)
        p2 = self.lateral_c2(c2)
        p1 = self.lateral_c1(c1)

        p3 = p3 + F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=True)
        p3 = self.smooth_c3(p3)

        p2 = p2 + F.interpolate(p3, size=p2.shape[2:], mode='bilinear', align_corners=True)
        p2 = self.smooth_c2(p2)

        p1 = p1 + F.interpolate(p2, size=p1.shape[2:], mode='bilinear', align_corners=True)
        p1 = self.smooth_c1(p1)

        return p1, p4  

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        if self.num_recursions == 1:
            p1, _ = self._fpn_forward(c1, c2, c3, c4)
        else:
            p1_1, _ = self._fpn_forward(c1, c2, c3, c4)
            guidance = F.interpolate(p1_1, size=c4.shape[2:], mode='bilinear', align_corners=True)  
            offsets = self.feedback_offset_conv(guidance)       
            feedback = self.feedback_deform_conv(c4, offsets)  
            feedback = self.feedback_act(feedback)
            c4_enhanced = c4 + feedback
            p1, _ = self._fpn_forward(c1, c2, c3, c4_enhanced)
        x = self.dropout(p1)
        x = self.linear_pred(x)
        return x
    
class SegFormer(nn.Module):
    def __init__(self, num_classes=21, phi='b0', pretrained=False, num_recursions=2):
        super(SegFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512],
            'b2': [64, 128, 320, 512], 'b3': [64, 128, 320, 512],
            'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[phi](pretrained)
        self.embedding_dim = 256 

        self.decode_head = SegFormerHeadRefineFPN(
            num_classes=num_classes,
            in_channels=self.in_channels,
            embedding_dim=self.embedding_dim,
            dropout_ratio=0.1,
            num_recursions=num_recursions
        )

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        feats = self.backbone(inputs)  
        x = self.decode_head(feats)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x