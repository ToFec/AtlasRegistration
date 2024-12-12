import torch.nn as nn
import torch
import torch.nn.functional as F
from atlas_utils import *
import imageTransformation
from EncoderBrick import EncoderBrick
from DecoderBrick import DecoderBrick
from SelfSupervisionBrick import SelfSupervisionBrick
from numpy import power


class ScalingAndSquaring(nn.Module):
    def __init__(self, num_steps=0):
        super(ScalingAndSquaring, self).__init__()
        self.num_steps = num_steps
        self.scale = 1.0 / (2**self.num_steps)
        self.bilinear = imageTransformation.Transformation()

    def forward(self, flow):
        pos_flow = flow * self.scale
        neg_flow = -flow * self.scale
        for _ in range(self.num_steps):
            pos_deform_field = self.bilinear.getDeformationField(pos_flow)
            neg_deform_field = self.bilinear.getDeformationField(neg_flow)
            pos_flow_1 = self.bilinear.sampleImage(pos_flow, pos_deform_field, paddMode="zeros")
            neg_flow_1 = self.bilinear.sampleImage(neg_flow, neg_deform_field, paddMode="zeros")
            pos_flow = pos_flow_1 + pos_flow
            neg_flow = neg_flow_1 + neg_flow

        return pos_flow, neg_flow


dim = 3
Conv = nn.Conv2d if dim == 2 else nn.Conv3d
MaxPool = nn.MaxPool2d if dim == 2 else nn.MaxPool3d
ConvTranspose = nn.ConvTranspose2d if dim == 2 else nn.ConvTranspose3d
BatchNorm = nn.BatchNorm2d if dim == 2 else nn.BatchNorm3d
conv = F.conv2d if dim == 2 else F.conv3d


class conv_bn_rel(nn.Module):
    """
    conv + bn (optional) + relu

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        active_unit="relu",
        same_padding=False,
        bn=False,
        reverse=False,
        group=1,
        dilation=1,
    ):
        super(conv_bn_rel, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        if not reverse:
            self.conv = Conv(
                in_channels, out_channels, kernel_size, stride, padding=padding, groups=group, dilation=dilation
            )
        else:
            self.conv = ConvTranspose(
                in_channels, out_channels, kernel_size, stride, padding=padding, groups=group, dilation=dilation
            )

        self.bn = BatchNorm(out_channels) if bn else None  # , eps=0.0001, momentum=0, affine=True
        if active_unit == "relu":
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == "elu":
            self.active_unit = nn.ELU(inplace=True)
        elif active_unit == "leaky_relu":
            self.active_unit = nn.LeakyReLU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


class FcRel(nn.Module):
    """
    fc+ relu(option)
    """

    def __init__(self, in_features, out_features, active_unit="relu"):
        super(FcRel, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        if active_unit == "relu":
            self.active_unit = nn.ReLU(inplace=True)
        elif active_unit == "elu":
            self.active_unit = nn.ELU(inplace=True)
        else:
            self.active_unit = None

    def forward(self, x):
        x = self.fc(x)
        if self.active_unit is not None:
            x = self.active_unit(x)
        return x


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()
        self.x = torch.nn.Parameter(torch.randn([1]))

    def forward(self, x):
        xShape = list(x.shape)
        pos_flow = torch.zeros([xShape[0], 3] + xShape[2:], device=x.device)
        neg_flow = torch.zeros([xShape[0], 3] + xShape[2:], device=x.device)
        return pos_flow, neg_flow

    def getShapeForModel(self, shape):
        return torch.tensor(shape)


class SVF_resid(nn.Module):
    def __init__(
        self,
        bn=False,
        scaleSquare=7,
    ):
        super(SVF_resid, self).__init__()
        self.imageSizeModuloVal = 16

        self.scaleAndSquare = ScalingAndSquaring(scaleSquare)

        self.down_path_1 = conv_bn_rel(2, 16, 3, stride=1, active_unit="relu", same_padding=True, bn=False, group=2)
        self.down_path_2_1 = conv_bn_rel(16, 32, 2, stride=2, active_unit="relu", same_padding=False, bn=False, group=2)
        self.down_path_2_2 = conv_bn_rel(32, 32, 3, stride=1, active_unit="relu", same_padding=True, bn=False, group=2)
        self.down_path_2_3 = conv_bn_rel(32, 32, 3, stride=1, active_unit="relu", same_padding=True, bn=bn)
        self.down_path_4_1 = conv_bn_rel(32, 64, 2, stride=2, active_unit="relu", same_padding=False, bn=bn)
        self.down_path_4_2 = conv_bn_rel(64, 64, 3, stride=1, active_unit="relu", same_padding=True, bn=bn)
        self.down_path_4_3 = conv_bn_rel(64, 64, 3, stride=1, active_unit="relu", same_padding=True, bn=bn)
        self.down_path_8_1 = conv_bn_rel(64, 128, 2, stride=2, active_unit="relu", same_padding=False, bn=bn)
        self.down_path_8_2 = conv_bn_rel(128, 128, 3, stride=1, active_unit="relu", same_padding=True, bn=bn)
        self.down_path_8_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit="relu", same_padding=True, bn=bn)
        self.down_path_16_1 = conv_bn_rel(128, 256, 2, stride=2, active_unit="relu", same_padding=False, bn=bn)
        self.down_path_16_2 = conv_bn_rel(256, 256, 3, stride=1, active_unit="relu", same_padding=True, bn=bn)

        # output_size = strides * (input_size-1) + kernel_size - 2*padding
        self.up_path_8_1 = conv_bn_rel(
            256, 128, 2, stride=2, active_unit="leaky_relu", same_padding=False, bn=bn, reverse=True
        )
        self.up_path_8_2 = conv_bn_rel(128 + 128, 128, 3, stride=1, active_unit="leaky_relu", same_padding=True, bn=bn)
        self.up_path_8_3 = conv_bn_rel(128, 128, 3, stride=1, active_unit="leaky_relu", same_padding=True, bn=bn)
        self.up_path_4_1 = conv_bn_rel(
            128, 64, 2, stride=2, active_unit="leaky_relu", same_padding=False, bn=bn, reverse=True
        )
        self.up_path_4_2 = conv_bn_rel(64 + 64, 32, 3, stride=1, active_unit="leaky_relu", same_padding=True, bn=bn)
        self.up_path_4_3 = conv_bn_rel(32, 32, 3, stride=1, active_unit="leaky_relu", same_padding=True, bn=bn)
        self.up_path_2_1 = conv_bn_rel(
            32, 32, 2, stride=2, active_unit="leaky_relu", same_padding=False, bn=bn, reverse=True
        )

        self.up_path_2_2 = conv_bn_rel(32 + 32, 16, 3, stride=1, active_unit="None", same_padding=True)
        self.up_path_2_3 = conv_bn_rel(16, 16, 3, stride=1, active_unit="None", same_padding=True)
        self.up_path_1_1 = conv_bn_rel(16, 16, 2, stride=2, active_unit="None", same_padding=False, bn=bn, reverse=True)
        self.up_path_1_2 = conv_bn_rel(16, 3, 3, stride=1, active_unit="None", same_padding=True)

        self.weights_init()

    def getShapeForModel(self, shape):
        shape = torch.tensor(shape)
        remainderVals = torch.remainder(shape, self.imageSizeModuloVal)
        newShape = shape + ((self.imageSizeModuloVal - remainderVals) * ((remainderVals != 0.0)))
        return newShape

    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                if not m.weight is None:
                    nn.init.xavier_normal_(m.weight.data)
                if not m.bias is None:
                    m.bias.data.zero_()
            # elif classname.find("BatchNorm") != -1:
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        d1 = self.down_path_1(x)
        d2_1 = self.down_path_2_1(d1)
        d2_2 = self.down_path_2_2(d2_1)
        d2_2 = d2_1 + d2_2
        d2_3 = self.down_path_2_3(d2_2)
        d2_3 = d2_2 + d2_3
        d4_1 = self.down_path_4_1(d2_3)
        d4_2 = self.down_path_4_2(d4_1)
        d4_2 = d4_1 + d4_2
        d4_3 = self.down_path_4_3(d4_2)
        d4_3 = d4_2 + d4_3
        d8_1 = self.down_path_8_1(d4_3)
        d8_2 = self.down_path_8_2(d8_1)
        d8_2 = d8_1 + d8_2
        d8_3 = self.down_path_8_3(d8_2)
        d8_3 = d8_2 + d8_3
        d16_1 = self.down_path_16_1(d8_3)
        d16_2 = self.down_path_16_2(d16_1)
        d16_2 = d16_1 + d16_2

        u8_1 = self.up_path_8_1(d16_2)
        u8_2 = self.up_path_8_2(torch.cat((d8_3, u8_1), 1))
        u8_3 = self.up_path_8_3(u8_2)
        u8_3 = u8_2 + u8_3
        u4_1 = self.up_path_4_1(u8_3)
        u4_2 = self.up_path_4_2(torch.cat((d4_3, u4_1), 1))
        u4_3 = self.up_path_4_3(u4_2)
        u4_3 = u4_2 + u4_3
        u2_1 = self.up_path_2_1(u4_3)
        u2_2 = self.up_path_2_2(torch.cat((d2_3, u2_1), 1))
        output = self.up_path_2_3(u2_2)

        flow = self.up_path_1_2(self.up_path_1_1(output))
        pos_flow, neg_flow = self.scaleAndSquare(flow)

        return pos_flow, neg_flow


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=2,
        bn=True,
        concatLayer=True,
        depth=5,
        numberOfFiltersFirstLayer=32,
        useDeepSelfSupervision=False,
        padImg=True,
        scaleSquare=7,
    ):
        super(UNet, self).__init__()

        if depth < 2:
            raise ValueError("minimum depth is 2")

        self.scaleAndSquare = ScalingAndSquaring(scaleSquare)

        self.useBatchNorm = bn
        self.concatLayer = concatLayer
        self.in_channels = in_channels
        self.useDeepSelfSupervision = useDeepSelfSupervision

        self.encoders = []
        self.decoders = []
        self.pools = []
        self.selfSupervisions = []

        self.depth = depth
        for i in range(self.depth):
            currentNumberOfInputChannels = self.in_channels if i == 0 else outputFilterNumber
            outputFilterNumber = numberOfFiltersFirstLayer * (2**i)
            self.encoders.append(
                EncoderBrick(
                    outputFilterNumber, currentNumberOfInputChannels, self.useBatchNorm, self.concatLayer, padImg
                )
            )
            if i < self.depth - 1:
                self.pools.append(nn.AvgPool3d(2, 2))
                self.decoders.append(
                    DecoderBrick(
                        outputFilterNumber, outputFilterNumber * 2, self.useBatchNorm, self.concatLayer, padImg
                    )
                )
            if self.useDeepSelfSupervision:
                self.selfSupervisions.append(SelfSupervisionBrick(in_channels, outputFilterNumber, i, padImg))

        if not self.useDeepSelfSupervision:
            self.selfSupervisions.append(SelfSupervisionBrick(in_channels, numberOfFiltersFirstLayer, 0, padImg))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.pools = nn.ModuleList(self.pools)
        self.selfSupervisions = nn.ModuleList(self.selfSupervisions)

        self.imageSizeModuloVal = 2 ** (self.depth - 1)

        self.receptiveFieldOffsets = [0] * (depth - 1)
        if not padImg:
            offsetBase = 2
            offsetCenter = 2
            for i in range(depth - 1):
                offsetCenter = offsetBase * offsetCenter
                offset = offsetCenter
                for j in range(i):
                    offset += 2 * power(offsetBase, 2 + j)
                self.receptiveFieldOffsets[i] = offset

        self.reset_params()

    def getShapeForModel(self, shape):
        shape = torch.tensor(shape)
        remainderVals = torch.remainder(shape, self.imageSizeModuloVal)
        newShape = shape + ((self.imageSizeModuloVal - remainderVals) * ((remainderVals != 0.0)))
        return newShape

    def forward(self, x):
        encoder_outs = []
        supervisionInputs = list(range(len(self.encoders)))  # python3
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            rFOffSet = self.receptiveFieldOffsets[self.depth - 2 - i]
            encoder_outs.append(
                x[
                    :,
                    :,
                    rFOffSet : x.shape[2] - rFOffSet,
                    rFOffSet : x.shape[3] - rFOffSet,
                    rFOffSet : x.shape[4] - rFOffSet,
                ]
            )
            if i < self.depth - 1:
                x = self.pools[i](x)
            else:
                supervisionInputs[i] = x

        for i in range(len(self.decoders) - 1, -1, -1):
            decoder = self.decoders[i]
            encOut = encoder_outs[i]
            x = decoder(x, encOut)
            supervisionInputs[i] = x

        outputFields = []
        for i in range(len(self.selfSupervisions) - 1, -1, -1):
            decOut = supervisionInputs[i]
            selfSupervision = self.selfSupervisions[i]
            outputFields.append(selfSupervision(decOut))

        x = torch.stack(outputFields)
        flow = torch.sum(x, dim=0)

        pos_flow, neg_flow = self.scaleAndSquare(flow)
        return pos_flow, neg_flow

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
