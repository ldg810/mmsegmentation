_base_ = './deeplabv3_r50-d8_769x769_80k_karibuilding.py'
model = dict(pretrained='torchvision://resnet50', backbone=dict(type='ResNet'))
