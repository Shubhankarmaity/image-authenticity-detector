import timm

def build_model(name="efficientnet_b4", num_classes=2, pretrained=True):
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
