import torch
import torch.nn as nn

cfg_vgg19 = [
    64, 64, 'M',
    128, 128, 'M',
    256, 256, 256, 256, 'M',
    512, 512, 512, 512, 'M',
    512, 512, 512, 512, 'M'
]

class VGG(nn.Module):
    def __init__(self, cfg=cfg_vgg19, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        return nn.Sequential(*layers)

def vgg19(num_classes=10, pretrained=False):
    """Returns VGG19 model for CIFAR-10."""
    model = VGG(cfg_vgg19, num_classes=num_classes)
    if pretrained:
        try:
            hub_model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg19_bn", pretrained=True, trust_repo=True)
            return hub_model
        except Exception as e:
            print(f"[Info] Could not load pre-trained hub model ({e}). Using initialized VGG19 model.")
    return model
