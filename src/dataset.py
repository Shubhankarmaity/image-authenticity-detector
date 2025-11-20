import os, cv2, numpy as np
from albumentations import Compose, Resize, Normalize, HorizontalFlip, RandomBrightnessContrast
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class RealFakeDataset(Dataset):
    def __init__(self, root, size=380, train=True, use_ela=False):
        self.items = []
        for label, cls in enumerate(["Real", "Fake"]):
            folder = os.path.join(root, cls)
            for f in os.listdir(folder):
                if f.lower().endswith((".png",".jpg",".jpeg")):
                    self.items.append((os.path.join(folder, f), label))
        aug = [Resize(size, size)]
        if train:
            aug += [HorizontalFlip(p=0.5), RandomBrightnessContrast(p=0.2)]
        aug += [Normalize(), ToTensorV2()]
        self.tf = Compose(aug)
        self.use_ela = use_ela

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        if self.use_ela:
            from .ela import ela_image
            img = ela_image(path)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tf(image=img)["image"]
        return img, label
