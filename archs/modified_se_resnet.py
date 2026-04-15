# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.hub import load_state_dict_from_url
# from torchvision.models import ResNet
# from archs.se_module import SELayer
#
#
# def conv3x3(in_planes, out_planes, stride=1):
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#
#
# class SEBasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None,
#                  *, reduction=16):
#         super(SEBasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes, 1)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.se = SELayer(planes, reduction)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.se(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class SEBottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None,
#                  *, reduction=16):
#         super(SEBottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(planes * 4)
#         self.relu = nn.ReLU(inplace=True)
#         self.se = SELayer(planes * 4, reduction)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.se(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
#
# class CBAMBlock(nn.Module):
#     def __init__(self, channels, reduction=16):
#         super(CBAMBlock, self).__init__()
#         self.channel_attention = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, channels // reduction, kernel_size=1),
#             nn.ReLU(),
#             nn.Conv2d(channels // reduction, channels, kernel_size=1),
#             nn.Sigmoid()
#         )
#         self.spatial_attention = nn.Sequential(
#             nn.Conv2d(2, 1, kernel_size=7, padding=3),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         # Channel attention
#         x_ca = self.channel_attention(x) * x
#
#         # Spatial attention
#         max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
#         avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
#         x_sa = torch.cat([max_pool, avg_pool], dim=1)
#         x_sa = self.spatial_attention(x_sa) * x_ca
#
#         return x_sa
#
#
# class FPN(nn.Module):
#     def __init__(self):
#         super(FPN, self).__init__()
#         self.conv1_c1 = nn.Conv2d(256, 256, kernel_size=1)
#         self.conv1_c2 = nn.Conv2d(512, 256, kernel_size=1)
#         self.conv1_c3 = nn.Conv2d(1024, 256, kernel_size=1)
#         self.conv1_c4 = nn.Conv2d(2048, 256, kernel_size=1)
#         self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
#
#     def forward(self, c1, c2, c3, c4):
#         p4 = self.conv1_c4(c4)
#         p3 = self.conv1_c3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
#         p2 = self.conv1_c2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
#         p1 = self.conv1_c1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')
#
#         p1 = self.conv2(p1)
#         p2 = self.conv3(p2)
#         p3 = self.conv4(p3)
#
#         return p1, p2, p3, p4
#
#
# def se_resnet50(num_classes=1000, pretrained=False):
#     """Constructs a ResNet-50 model.
#
#     Args:
#         pretrained (bool): If True, returns a model pre-trained on ImageNet
#     """
#     model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
#     model.avgpool = nn.AdaptiveAvgPool2d(1)
#     if pretrained:
#         state_dict = load_state_dict_from_url(
#             "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl")
#         missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
#         print("Missing keys:", missing_keys)
#         print("Unexpected keys:", unexpected_keys)
#     return model
#
#
# class ModifiedSEResNet50(nn.Module):
#     def __init__(self, num_classes=2):
#         super(ModifiedSEResNet50, self).__init__()
#         self.base_model = se_resnet50(pretrained=True)
#         self.base_model.fc = nn.Identity()  # Remove the original classifier
#
#         self.fpn = FPN()
#         self.cbam = CBAMBlock(256)
#         self.classifier = nn.Linear(256 * 4, num_classes)
#
#     def forward(self, x):
#         c1 = self.base_model.layer1(
#             self.base_model.maxpool(self.base_model.relu(self.base_model.bn1(self.base_model.conv1(x)))))
#         c2 = self.base_model.layer2(c1)
#         c3 = self.base_model.layer3(c2)
#         c4 = self.base_model.layer4(c3)
#
#         p1, p2, p3, p4 = self.fpn(c1, c2, c3, c4)
#         p1 = self.cbam(p1)
#         p2 = self.cbam(p2)
#         p3 = self.cbam(p3)
#         p4 = self.cbam(p4)
#
#         p1 = F.adaptive_avg_pool2d(p1, 1).view(p1.size(0), -1)
#         p2 = F.adaptive_avg_pool2d(p2, 1).view(p2.size(0), -1)
#         p3 = F.adaptive_avg_pool2d(p3, 1).view(p3.size(0), -1)
#         p4 = F.adaptive_avg_pool2d(p4, 1).view(p4.size(0), -1)
#
#         features = torch.cat((p1, p2, p3, p4), dim=1)
#         out = self.classifier(features)
#
#         return out








import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet
from archs.se_module import SELayer

# 定义基础卷积函数
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# SE模块基本块
class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# SE模块瓶颈块
class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# CBAM块
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        x_ca = self.channel_attention(x) * x

        # Spatial attention
        max_pool = torch.max(x_ca, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x_ca, dim=1, keepdim=True)
        x_sa = torch.cat([max_pool, avg_pool], dim=1)
        x_sa = self.spatial_attention(x_sa) * x_ca

        return x_sa

# FPN模块
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.conv1_c1 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv1_c2 = nn.Conv2d(512, 256, kernel_size=1)
        self.conv1_c3 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv1_c4 = nn.Conv2d(2048, 256, kernel_size=1)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

    def forward(self, c1, c2, c3, c4):
        p4 = self.conv1_c4(c4)
        p3 = self.conv1_c3(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.conv1_c2(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')
        p1 = self.conv1_c1(c1) + F.interpolate(p2, scale_factor=2, mode='nearest')

        p1 = self.conv2(p1)
        p2 = self.conv3(p2)
        p3 = self.conv4(p3)

        return p1, p2, p3, p4

# SE-ResNet50模型
def se_resnet50(num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)

    # 替换全连接层以匹配新的类别数
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    if pretrained:
        # 加载预训练权重
        # pretrained_state_dict = load_state_dict_from_url(
        #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl",
        #     progress=True)
        pretrained_weight_path = "/data2/wjb/code/pumc-thyroid/code/trained_models/SelfSupervisedAdapt_nofrozen/se_resnet50_self_supervised_400.pth"
        pretrained_state_dict = torch.load(pretrained_weight_path)

        # 创建一个新的状态字典，只包含除了全连接层以外的权重
        new_state_dict = {}
        for name, param in pretrained_state_dict.items():
            if 'fc.' not in name:
                new_state_dict[name] = param

        # 加载新的状态字典到模型中
        model.load_state_dict(new_state_dict, strict=False)

    # if pretrained:
    #     # state_dict = load_state_dict_from_url(
    #     #     "https://github.com/moskomule/senet.pytorch/releases/download/archive/seresnet50-60a8950a85b2b.pkl")
    #     # state_dict = torch.load("/data2/wjb/code/pumc-thyroid/code/trained_models/SelfSupervisedAdapt_nofrozen/se_resnet50_self_supervised_400.pth")
    #     # missing_keys, unexpected_keys = model.load_state_dict(state_dict)
    #     # print("Missing keys:", missing_keys)
    #     # print("Unexpected keys:", unexpected_keys)
    #
    #     # my adapted weigths:
    #     # model.load_state_dict(torch.load("/data2/wjb/code/pumc-thyroid/code/trained_models/SelfSupervisedAdapt_nofrozen/se_resnet50_self_supervised_400.pth"))
    #
    #     pretrained_weight_path = "/data2/wjb/code/pumc-thyroid/code/trained_models/SelfSupervisedAdapt_nofrozen/se_resnet50_self_supervised_400.pth"
    #     pretrained_state_dict = torch.load(pretrained_weight_path)
    #
    #     # 获取当前模型的状态字典
    #     model_state_dict = model.state_dict()
    #
    #     # 遍历预训练权重的键和值
    #     for name, param in pretrained_state_dict.items():
    #         # 如果键在当前模型的状态字典中存在，则加载权重
    #         if name in model_state_dict:
    #             model_state_dict[name].copy_(param)
    #
    #     # 如果希望加载到不匹配层上的权重，可以手动进行匹配
    #     model.fc.weight.data.copy_(pretrained_state_dict['fc.0.weight'].clone())
    #     model.fc.bias.data.copy_(pretrained_state_dict['fc.0.bias'].clone())
    return model

# 修改后的SE-ResNet50模型
class ModifiedSEResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedSEResNet50, self).__init__()
        self.base_model = se_resnet50(pretrained=True)
        # 冻结所有层的参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        # 移除原始的全连接层
        self.base_model.fc = nn.Identity()

        self.fpn = FPN()
        self.cbam = CBAMBlock(256)
        self.classifier = nn.Linear(256 * 4, num_classes)

    def forward(self, x):
        c1 = self.base_model.layer1(
            self.base_model.maxpool(self.base_model.relu(self.base_model.bn1(self.base_model.conv1(x)))))
        c2 = self.base_model.layer2(c1)
        c3 = self.base_model.layer3(c2)
        c4 = self.base_model.layer4(c3)

        p1, p2, p3, p4 = self.fpn(c1, c2, c3, c4)
        p1 = self.cbam(p1)
        p2 = self.cbam(p2)
        p3 = self.cbam(p3)
        p4 = self.cbam(p4)

        p1 = F.adaptive_avg_pool2d(p1, 1).view(p1.size(0), -1)
        p2 = F.adaptive_avg_pool2d(p2, 1).view(p2.size(0), -1)
        p3 = F.adaptive_avg_pool2d(p3, 1).view(p3.size(0), -1)
        p4 = F.adaptive_avg_pool2d(p4, 1).view(p4.size(0), -1)

        features = torch.cat((p1, p2, p3, p4), dim=1)
        out = self.classifier(features)

        return out

