import torch
import timm
import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

tf = Compose([Resize(380,380), Normalize(), ToTensorV2()])
def load_model(weights="outputs/checkpoints/best.pt", name="efficientnet_b4"):
    m = timm.create_model(name, pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(weights, map_location="cpu"))
    m.eval()
    return m

def predict(path, weights="outputs/checkpoints/best.pt"):
    m = load_model(weights)
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = tf(image=img)["image"].unsqueeze(0)
    with torch.no_grad():
        logits = m(x)
        probs = torch.softmax(logits, dim=1)[0].numpy()
    idx = int(probs.argmax())
    label = ["Real", "Fake"][idx]
    conf = float(probs[idx])
    return label, conf
