import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch.nn import init
from collections import OrderedDict
import numpy as np

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_classes = 5,n_channels=3):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        # self.outc = outconv(64, n_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # x = self.outc(x)
        # return F.sigmoid(x)
        
        return x

#==============================================================================================================#

class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        else:
            for i in range(1, n + 1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True), )
                setattr(self, 'conv%d' % i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n + 1):
            conv = getattr(self, 'conv%d' % i)
            x = conv(x)

        return x

def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        return self.conv(outputs0)

class UNet_Nested(nn.Module):

    def __init__(self, in_channels=3, n_classes=1024, feature_scale=2, is_deconv=True, is_batchnorm=True, is_ds=True):
        super(UNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds

        # filters = [64, 128, 256, 512, 1024]
        filters = [32, 64, 128, 256, 512]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat01 = unetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = unetUp(filters[3], filters[2], self.is_deconv)
        # self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)

        self.up_concat02 = unetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = unetUp(filters[2], filters[1], self.is_deconv, 3)
        # self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)

        self.up_concat03 = unetUp(filters[1], filters[0], self.is_deconv, 4)
        # self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)

        # self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)

        # final conv (without any concat)
        # self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_4 = nn.Conv2d(filters[0], n_classes, 1)
        
        # self.avgpool = nn.AdaptiveAvgPool2d((32, 32))
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        # column : 0
        X_00 = self.conv00(inputs)  # 16*512*512
        maxpool0 = self.maxpool(X_00)  # 16*256*256
        X_10 = self.conv10(maxpool0)  # 32*256*256
        maxpool1 = self.maxpool(X_10)  # 32*128*128
        X_20 = self.conv20(maxpool1)  # 64*128*128
        maxpool2 = self.maxpool(X_20)  # 64*64*64
        X_30 = self.conv30(maxpool2)  # 128*64*64
        # maxpool3 = self.maxpool(X_30)  # 128*32*32
        # X_40 = self.conv40(maxpool3)  # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        # X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        # X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        # X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        # X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)

        # final layer
        # final_1 = torch.sigmoid(self.final_1(X_01))
        # final_2 = torch.sigmoid(self.final_2(X_02))
        # final_3 = torch.sigmoid(self.final_3(X_03))
        # print(self.conv00.conv1[0].weight)
        # final_1 = torch.sigmoid(X_01)
        # final_2 = torch.sigmoid(X_02)
        final_3 = torch.sigmoid(X_03)
        # final_4 = torch.sigmoid(X_04)

        final_3 = final_3.permute([0, 2, 3, 1]).contiguous()
        final_3 = final_3.view(final_3.shape[0], final_3.shape[1], final_3.shape[2] * final_3.shape[3])  # Flatten channels into width
        # final_4 = final_4.permute([0, 2, 3, 1]).contiguous()
        # final_4 = final_4.view(final_4.shape[0], final_4.shape[1], final_4.shape[2] * final_4.shape[3])  # Flatten channels into width

        # final_3 = self.avgpool(final_3)
        # final_3 = final_3.view(final_3.size(0), final_3.size(1), -1)
        
        # final_1 = self.final_1(X_01)
        # final_2 = self.final_2(X_02)
        # final_3 = self.final_3(X_03)
        # final_4 = self.final_4(X_04)

        # final = (final_1 + final_2 + final_3 + final_4) / 4

        # if self.is_ds:
        #     return final
        # else:
        #     return final_4
        # print(final_3.shape)
        # print(final_3)
        return final_3


#=================================================== Dense Unet ========================================================
def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        # print(concated_features.shape)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('densenorm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('denserelu1', nn.ReLU(inplace=True)),
        self.add_module('denseconv1', nn.Conv2d(num_input_features, bn_size * growth_rate,
                        kernel_size=1, stride=1, bias=False)),
        self.add_module('densenorm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('denserelu2', nn.ReLU(inplace=True)),
        self.add_module('denseconv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.densenorm1, self.denserelu1, self.denseconv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.denseconv2(self.denserelu2(self.densenorm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features=24, bn_size=4, growth_rate=12, drop_rate=0, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class denseunetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, n_concat=2):
        super(denseunetUp, self).__init__()
        # print(in_size + (n_concat - 2) * out_size)
        self.conv = _DenseLayer(num_input_features = in_size + (n_concat - 2) * out_size,growth_rate = out_size)#unetConv2(in_size + (n_concat - 2) * out_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
        else:
            self.up = nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=2),
                nn.Conv2d(in_size, out_size, 1))

        # initialise the blocks
        for m in self.children():
            # if m.__class__.__name__.find('_DenseBlock') != -1: continue #TODO
            init_weights(m, init_type='kaiming')

    def forward(self, high_feature, *low_feature):
        outputs0 = self.up(high_feature)
        for feature in low_feature:
            outputs0 = torch.cat([outputs0, feature], 1)
        # print(outputs0.shape)
        return self.conv(outputs0)


class DenseUNet_Nested(nn.Module):
    
    def __init__(self, in_channels=1, feature_scale=2,
                 growth_rate=12, block_config=[8, 16, 32, 64], compression=0.5,
                 num_init_features=24, bn_size=4, drop_rate=0,
                 num_classes=10, small_inputs=True, efficient=False,
                 is_deconv=True, is_batchnorm=True, is_ds=True):
        super(DenseUNet_Nested, self).__init__()
        self.in_channels = in_channels
        self.feature_scale = feature_scale
        self.is_deconv = is_deconv
        self.is_batchnorm = is_batchnorm
        self.is_ds = is_ds
        
        # filters = [64, 128, 256, 512, 1024]
        filters = block_config#[32, 64, 128, 256, 512]
        # filters = [int(x / self.feature_scale) for x in filters]
        
        args = (growth_rate,bn_size,drop_rate,efficient)
        
        # downsampling
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv00 = nn.Conv2d(in_channels, filters[0], kernel_size=3, stride=1, padding=1, bias=False)#_DenseLayer(num_input_features=filters[0])#unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.conv10 = _DenseLayer(num_input_features=filters[0],growth_rate=filters[1])#unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.conv20 = _DenseLayer(num_input_features=filters[1],growth_rate=filters[2])#unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.conv30 = _DenseLayer(num_input_features=filters[2],growth_rate=filters[3])#unetConv2(filters[2], filters[3], self.is_batchnorm)
        # self.conv40 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        
        # upsampling
        self.up_concat01 = denseunetUp(filters[1], filters[0], self.is_deconv)
        self.up_concat11 = denseunetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat21 = denseunetUp(filters[3], filters[2], self.is_deconv)
        # self.up_concat31 = unetUp(filters[4], filters[3], self.is_deconv)
        
        self.up_concat02 = denseunetUp(filters[1], filters[0], self.is_deconv, 3)
        self.up_concat12 = denseunetUp(filters[2], filters[1], self.is_deconv, 3)
        # self.up_concat22 = unetUp(filters[3], filters[2], self.is_deconv, 3)
        
        self.up_concat03 = denseunetUp(filters[1], filters[0], self.is_deconv, 4)
        # self.up_concat13 = unetUp(filters[2], filters[1], self.is_deconv, 4)
        
        # self.up_concat04 = unetUp(filters[1], filters[0], self.is_deconv, 5)
        
        # final conv (without any concat)
        # self.final_1 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_2 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_3 = nn.Conv2d(filters[0], n_classes, 1)
        # self.final_4 = nn.Conv2d(filters[0], n_classes, 1)
        
        # self.avgpool = nn.AdaptiveAvgPool2d((32, 32))
        self.shrinkconv = nn.Conv2d(filters[0]*3,filters[0]*3, kernel_size=7, stride=7, padding=1, dilation=5, bias=False)
        
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
    
    def forward(self, inputs):
        # column : 0
        # print(inputs.shape)
        X_00 = self.conv00(inputs)  # 16*512*512
        maxpool0 = self.maxpool(X_00)  # 16*256*256
        X_10 = self.conv10(maxpool0)  # 32*256*256
        maxpool1 = self.maxpool(X_10)  # 32*128*128
        X_20 = self.conv20(maxpool1)  # 64*128*128
        maxpool2 = self.maxpool(X_20)  # 64*64*64
        X_30 = self.conv30(maxpool2)  # 128*64*64
        # maxpool3 = self.maxpool(X_30)  # 128*32*32
        # X_40 = self.conv40(maxpool3)  # 256*32*32
        # column : 1
        X_01 = self.up_concat01(X_10, X_00)
        X_11 = self.up_concat11(X_20, X_10)
        X_21 = self.up_concat21(X_30, X_20)
        # X_31 = self.up_concat31(X_40, X_30)
        # column : 2
        X_02 = self.up_concat02(X_11, X_00, X_01)
        X_12 = self.up_concat12(X_21, X_10, X_11)
        # X_22 = self.up_concat22(X_31, X_20, X_21)
        # column : 3
        X_03 = self.up_concat03(X_12, X_00, X_01, X_02)
        # X_13 = self.up_concat13(X_22, X_10, X_11, X_12)
        # column : 4
        # X_04 = self.up_concat04(X_13, X_00, X_01, X_02, X_03)
        
        # final layer
        # final_1 = torch.sigmoid(self.final_1(X_01))
        # final_2 = torch.sigmoid(self.final_2(X_02))
        # final_3 = torch.sigmoid(self.final_3(X_03))
        # print(self.conv00.conv1[0].weight)
        # X_01 = torch.sigmoid(X_01)
        # X_02 = torch.sigmoid(X_02)
        # X_03 = torch.sigmoid(X_03)
        # final_4 = torch.sigmoid(X_04)
        
        # print(final_3.shape)
        
        # final_3 = final_3.permute([0, 2, 3, 1]).contiguous()
        # final_3 = final_3.view(final_3.shape[0], final_3.shape[1],
        #                        final_3.shape[2] * final_3.shape[3])  # Flatten channels into width
        # final_4 = final_4.permute([0, 2, 3, 1]).contiguous()
        # final_4 = final_4.view(final_4.shape[0], final_4.shape[1], final_4.shape[2] * final_4.shape[3])  # Flatten channels into width
        # print(final_1.shape)
        output = torch.cat([X_01,X_02,X_03],1)
        # output = final_3
        output = self.shrinkconv(output)
        output = torch.sigmoid(output)
        # print(output.shape)
        output = output.view(output.size(0), output.size(1), -1)
        # output = output.permute([0, 2, 1]).contiguous()
        # final_1 = self.final_1(X_01)
        # final_2 = self.final_2(X_02)
        # final_3 = self.final_3(X_03)
        # final_4 = self.final_4(X_04)
        
        # final = (final_1 + final_2 + final_3 + final_4) / 4
        
        # if self.is_ds:
        #     return final
        # else:
        #     return final_4
        # print(final_3.shape)
        # print(final_3)
        return output

