import torch.nn as nn

class Regression(nn.Module):
    def __init__(self, in_size, out_dims, feature_size=256, num_anchors=16, num_classes=15):
        super(Regression, self).__init__()

        self.out_dims = out_dims
        self.num_anchors = num_anchors
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_size, feature_size, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(feature_size)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_size)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(feature_size)
        self.act3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(feature_size)
        self.act4 = nn.ReLU(inplace=True)
        self.output = nn.Conv2d(feature_size, num_anchors*num_classes*out_dims, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.output(x)

        # x is B x C x W x H
        out1 = x.permute(0, 3, 2, 1)
        batch_size, width, height, channels = out1.shape
        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes, self.out_dims)
        return out2.contiguous().view(out2.shape[0], -1, self.num_classes, self.out_dims)
