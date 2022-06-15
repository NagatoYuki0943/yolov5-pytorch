import torch
import torch.nn as nn

from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet
from nets.Swin_transformer import Swin_transformer_Tiny

#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 64
        base_depth          = max(round(dep_mul * 3), 1)  # 3
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone_name  = backbone
        if backbone == "cspdarknet":
            #---------------------------------------------------#
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            #---------------------------------------------------#
            self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained)
        else:
            #---------------------------------------------------#
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            #---------------------------------------------------#
            self.backbone       = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            in_channels         = {
                'convnext_tiny'         : [192, 384, 768],
                'convnext_small'        : [192, 384, 768],
                'swin_transfomer_tiny'  : [192, 384, 768],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)

        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        #-------------------------------------------#
        #   P5开始的的1x1Conv,得到 P5
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        #-------------------------------------------#
        #   P5_upsample,feat2 拼接之后的 CSPLayer
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        #-------------------------------------------#
        #   拼接 P5_upsample,feat2 后的CSPLayer的1x1Conv,得到P4
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        #-------------------------------------------#
        #   拼接 P4_upsample,feat1 后的 CPSLayer,得到 P3_out
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        #-------------------------------------------#
        #   P3_out下采样
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        #-------------------------------------------#
        #   拼接 P3_downsample, P4 后的 CPSLayer,得到 P4_out
        #   40, 40, 512 -> 40, 40, 512
        #-------------------------------------------#
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        #-------------------------------------------#
        #   P4_out下采样
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        #-------------------------------------------#
        #   拼接 P4_downsample, P5 后的 CPSLayer,得到 P5_out
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        #-------------------------------------------#
        #   yolo_head
        #   3个1x1Conv
        #-------------------------------------------#
        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

    def forward(self, x):
        #-------------------------------------------#
        #   feat1: 80, 80, 256
        #   feat2: 40, 40, 512
        #   feat3: 20, 20, 1024
        #-------------------------------------------#
        feat1, feat2, feat3 = self.backbone(x)
        if self.backbone_name != "cspdarknet":
            feat1 = self.conv_1x1_feat1(feat1)
            feat2 = self.conv_1x1_feat2(feat2)
            feat3 = self.conv_1x1_feat3(feat3)

        """第一次上采样"""
        #-------------------------------------------#
        #   P5开始的的1x1Conv,得到 P5
        #   20, 20, 1024 -> 20, 20, 512
        #-------------------------------------------#
        P5          = self.conv_for_feat3(feat3)
        #-------------------------------------------#
        #   P5上采样
        #   20, 20, 512 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.upsample(P5)
        #-------------------------------------------#
        #   拼接 P5_upsample,feat2
        #   40, 40, 512 + 40, 40, 512 -> 40, 40, 1024
        #-------------------------------------------#
        P5_upsample = torch.cat([P5_upsample, feat2], 1)
        #-------------------------------------------#
        #   P5_upsample,feat2 拼接之后的CSPLayer
        #   40, 40, 1024 -> 40, 40, 512
        #-------------------------------------------#
        P5_upsample = self.conv3_for_upsample1(P5_upsample)

        """第二次上采样"""
        #-------------------------------------------#
        #   拼接 P5_upsample,feat2 后的CSPLayer的1x1Conv,得到P4
        #   40, 40, 512 -> 40, 40, 256
        #-------------------------------------------#
        P4          = self.conv_for_feat2(P5_upsample)
        #-------------------------------------------#
        #   P4的上采样
        #   40, 40, 256 -> 80, 80, 256
        #-------------------------------------------#
        P4_upsample = self.upsample(P4)
        #-------------------------------------------#
        #   拼接 P4_upsample,feat1
        #   80, 80, 256 + 80, 80, 256 -> 80, 80, 512
        #-------------------------------------------#
        P4_upsample  = torch.cat([P4_upsample, feat1], 1)
        #-------------------------------------------#
        #   拼接 P4_upsample,feat1 后的 CPSLayer,得到 P3_out
        #   80, 80, 512 -> 80, 80, 256
        #-------------------------------------------#
        P3_out       = self.conv3_for_upsample2(P4_upsample)

        """第一次下采样"""
        #-------------------------------------------#
        #   P3_out下采样
        #   80, 80, 256 -> 40, 40, 256
        #-------------------------------------------#
        P3_downsample = self.down_sample1(P3_out)
        #-------------------------------------------#
        #   拼接 P3_downsample, P4
        #   40, 40, 256 + 40, 40, 256 -> 40, 40, 512
        #-------------------------------------------#
        P3_downsample = torch.cat([P3_downsample, P4], 1)
        #-------------------------------------------#
        #   拼接 P3_downsample, P4 后的 CPSLayer,得到 P4_out
        #   40, 40, 512 -> 40, 40, 512
        #-------------------------------------------#
        P4_out        = self.conv3_for_downsample1(P3_downsample)

        """第二次下采样"""
        #-------------------------------------------#
        #   P4_out下采样
        #   40, 40, 512 -> 20, 20, 512
        #-------------------------------------------#
        P4_downsample = self.down_sample2(P4_out)
        #-------------------------------------------#
        #   拼接 P4_downsample, P5
        #   20, 20, 512 + 20, 20, 512 -> 20, 20, 1024
        #-------------------------------------------#
        P4_downsample = torch.cat([P4_downsample, P5], 1)
        #-------------------------------------------#
        #   拼接 P4_downsample, P5 后的 CPSLayer,得到 P5_out
        #   20, 20, 1024 -> 20, 20, 1024
        #-------------------------------------------#
        P5_out        = self.conv3_for_downsample2(P4_downsample)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3_out)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4_out)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5_out)
        return out0, out1, out2


if __name__ == "__main__":
    model = YoloBody([[6, 7, 8], [3, 4, 5], [0, 1, 2]], 80, 's')
    state_dict = torch.load("F:\BaiduYunDownload\yolov5_s.pth")
    model.load_state_dict(state_dict)
