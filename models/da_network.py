import torch
import torch.nn as nn
from torchvision import models
import torch.nn.utils.weight_norm as weigthNorm

vgg_dict = {
    "vgg11":models.vgg11,
    "vgg13":models.vgg13,
    "vgg16":models.vgg16,
    "vgg19":models.vgg19,
}

res_dict={
    "resnet18":models.resnet18,
    "resnet34":models.resnet34,
    "resnet50":models.resnet50,
    "resnet101":models.resnet101,
    "resnet152":models.resnet152,
    "resnext50":models.resnext50_32x4d,
    "resnext101":models.resnext101_32x8d,
}

def init_weights(m):
    """init the weights"""
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1 or classname.find("ConvTranspose2d") != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight,1.0,0.02)
        nn.init.zeros_(m.bias)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class Classifier(nn.Module):
    """netC"""
    def __init__(self,class_num,bottleneck_dim=256,type="bn"):
        super(Classifier,self).__init__()
        if type == "linear":
            self.fc = nn.Linear(bottleneck_dim,class_num)
        else:
            self.fc = weigthNorm(nn.Linear(bottleneck_dim,class_num),name="weight")
        self.fc.apply(init_weights)

    def forward(self,x):
        x = self.fc(x)
        return x

class FeatBootleneck(nn.Module):
    """net B"""
    def __init__(self,feature_dim,bottleneck_dim=256,type="ori",use_tanh=True):
        super(FeatBootleneck,self).__init__()
        self.bottleneck = nn.Linear(feature_dim,bottleneck_dim)
        self.bottleneck.apply(init_weights)

        self.type = type
        self.bn = nn.BatchNorm1d(bottleneck_dim,affine=True)
        self.dropout = nn.Dropout(p=0.5)
        self.use_tanh = use_tanh

    def forward(self,x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
            x = self.dropout(x)
        if self.use_tanh:
            x = torch.tanh(x)
        return x

class VGGEncoder(nn.Module):
    """VGG Baseline"""
    def __init__(self,vgg_name,bottleneck_dim=256,type="ori",use_tanh=True,with_bootle=False):
        super(VGGEncoder, self).__init__()
        self.with_bootle = with_bootle
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i),model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

        if self.with_bootle:
            self.feat_bootleneck = FeatBootleneck(self.in_features,bottleneck_dim,type,use_tanh)

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        if self.with_bootle:
            x = self.feat_bootleneck(x)
        return x

class ResEncoder(nn.Module):
    """Resnet Baseline"""
    def __init__(self,res_name,bottleneck_dim=256,type="ori",use_tanh=True,with_bootle = True):
        super(ResEncoder, self).__init__()
        self.with_bootle = with_bootle
        model_resnet = res_dict[res_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

        if self.with_bootle:
            self.feat_bootleneck = FeatBootleneck(self.in_features,bottleneck_dim,type,use_tanh)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)

        if self.with_bootle:
            x = self.feat_bootleneck(x)
        return x
