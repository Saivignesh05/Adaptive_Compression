from .cifar_vgg import vgg19
from .cifar_resnet import resnet18, resnet34

def get_model(name: str, num_classes: int = 10, pretrained: bool = False):
    name_lower = name.lower()
    if "vgg19" in name_lower:
        return vgg19(num_classes=num_classes, pretrained=pretrained)
    elif "resnet18" in name_lower:
        return resnet18(num_classes=num_classes, pretrained=pretrained)
    elif "resnet34" in name_lower:
        return resnet34(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported model name: {name}")
