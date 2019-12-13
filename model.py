import torch
import torch.nn as nn
import torch.nn.functional as F


class PSPNet(nn.Module):

    def __init__(self, n_classes):
        super(PSPNet, self).__init__()

        block_config = [3, 4, 6, 3]
        img_size = 475
        img_size_8 = 60

        self.feature_conv = FeatureMap_convolution()
        self.feature_res_1 = ResidualBlockPSP(
            n_blocks=block_config[0], in_channels=128, mid_channels=64, out_channels=256,
            stride=1, dilation=1)
        self.feature_res_2 = ResidualBlockPSP(
            n_blocks=block_config[1], in_channels=256, mid_channels=128, out_channels=512,
            stride=2, dilation=1)
        self.feature_dilated_res_1 = ResidualBlockPSP(
            n_blocks=block_config[2], in_channels=512, mid_channels=256, out_channels=1024,
            stride=1, dilation=2)
        self.feature_dilated_res_2 = ResidualBlockPSP(
            n_blocks=block_config[3], in_channels=1024, mid_channels=512, out_channels=2048,
            stride=1, dilation=4)

        self.pyramid_pooling = PyramidPooling(
            in_channels=2048, pool_sizes=[6, 3, 2, 1],
            height=img_size_8, width=img_size_8)

        self.decode_feature = DecodePSPFeature(height=img_size, width=img_size, n_classes=n_classes)

        self.aux = AuxiliaryPSPLayers(in_channels=1024, height=img_size, width=img_size, n_classes=n_classes)

    def forward(self, x):
        x = self.feature_conv(x)
        x = self.feature_res_1(x)
        x = self.feature_res_2(x)
        x = self.feature_dilated_res_1(x)

        # Featureモジュールの途中をAuxモジュールへ
        output_aux = self.aux(x)

        x = self.feature_dilated_res_2(x)
        x = self.pyramid_pooling(x)
        output = self.decode_feature(x)

        return (output, output_aux)


class Conv2dBatchNorm(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2dBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batchnorm(x)
        return outputs


class BottleNeckPSP(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(BottleNeckPSP, self).__init__()

        self.cbr_1 = Conv2dBatchNormReLU(in_channels, mid_channels,
                                         kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = Conv2dBatchNormReLU(mid_channels, mid_channels,
                                         kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.cb_3 = Conv2dBatchNorm(mid_channels, out_channels,
                                    kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        # skip結合におけるConvolution
        self.cb_residual = Conv2dBatchNorm(in_channels, out_channels,
                                           kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = self.cb_residual(x)
        return self.relu(conv + residual)


class BottleNeckIdentifyPSP(nn.Module):

    def __init__(self, in_channels, mid_channels, stride, dilation):
        super(BottleNeckIdentifyPSP, self).__init__()

        self.cbr_1 = Conv2dBatchNormReLU(in_channels, mid_channels,
                                         kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.cbr_2 = Conv2dBatchNormReLU(mid_channels, mid_channels,
                                         kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.cb_3 = Conv2dBatchNorm(mid_channels, in_channels,
                                    kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.cb_3(self.cbr_2(self.cbr_1(x)))
        residual = x
        return self.relu(conv + residual)


class Conv2dBatchNormReLU(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2dBatchNormReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        outputs = self.relu(x)

        return outputs


class FeatureMap_convolution(nn.Module):

    def __init__(self):
        super(FeatureMap_convolution, self).__init__()

        self.cbnr_1 = Conv2dBatchNormReLU(3, 64, kernel_size=3, stride=2, padding=1, dilation=1, bias=False)
        self.cbnr_2 = Conv2dBatchNormReLU(64, 64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.cbnr_3 = Conv2dBatchNormReLU(64, 128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.cbnr_1(x)
        x = self.cbnr_2(x)
        x = self.cbnr_3(x)
        outputs = self.maxpool(x)
        return outputs


class ResidualBlockPSP(nn.Sequential):

    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # nn.Sequentialを継承しているのでforward()は自動的に実装されている
        self.add_module('block1', BottleNeckPSP(in_channels, mid_channels, out_channels, stride, dilation))
        for i in range(n_blocks - 1):
            self.add_module('block' + str(i + 2), BottleNeckIdentifyPSP(out_channels, mid_channels, stride, dilation))


class PyramidPooling(nn.Module):

    def __init__(self, in_channels, pool_sizes, height, width):
        super(PyramidPooling, self).__init__()

        self.height = height
        self.width = width

        out_channels = int(in_channels / len(pool_sizes))

        # 各畳み込み層を作成
        # pool_sizes: [6, 3, 2, 1]
        # AdaptiveAvgPool2dはカーネルサイズを指定する代わりに出力サイズを指定する
        # 出力サイズになるようにカーネルサイズは自動計算される
        # 出力サイズが小さいほどカーネルサイズが大きくなる
        # PyramidPoolingはカーネルサイズが異なるPoolingを組み合わせることで様々なサイズの特徴を抽出できる
        self.avpool_1 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[0])
        self.cbr_1 = Conv2dBatchNormReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_2 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[1])
        self.cbr_2 = Conv2dBatchNormReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_3 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[2])
        self.cbr_3 = Conv2dBatchNormReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.avpool_4 = nn.AdaptiveAvgPool2d(output_size=pool_sizes[3])
        self.cbr_4 = Conv2dBatchNormReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

    def forward(self, x):
        out1 = self.cbr_1(self.avpool_1(x))
        out1 = F.interpolate(out1, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out2 = self.cbr_2(self.avpool_2(x))
        out2 = F.interpolate(out2, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out3 = self.cbr_3(self.avpool_3(x))
        out3 = F.interpolate(out3, size=(self.height, self.width), mode='bilinear', align_corners=True)

        out4 = self.cbr_4(self.avpool_4(x))
        out4 = F.interpolate(out4, size=(self.height, self.width), mode='bilinear', align_corners=True)

        output = torch.cat([x, out1, out2, out3, out4], dim=1)

        return output


class DecodePSPFeature(nn.Module):

    def __init__(self, height, width, n_classes):
        super(DecodePSPFeature, self).__init__()

        self.height = height
        self.width = width

        self.cbr = Conv2dBatchNormReLU(in_channels=4096, out_channels=512,
                                       kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        # 全結合層を使わずに出力チャネルがクラス数となるカーネルサイズ1のpointwise convolutionを使う
        # 特徴マップのサイズは変えずにチャネル数だけ変換する手法
        self.classification = nn.Conv2d(in_channels=512, out_channels=n_classes,
                                        kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)
        return output


class AuxiliaryPSPLayers(nn.Module):

    def __init__(self, in_channels, height, width, n_classes):
        super(AuxiliaryPSPLayers, self).__init__()

        self.height = height
        self.width = width

        self.cbr = Conv2dBatchNormReLU(in_channels=in_channels, out_channels=256,
                                       kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(in_channels=256, out_channels=n_classes,
                                        kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cbr(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=True)
        return output
