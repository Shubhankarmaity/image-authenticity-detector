import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RealFakeDataset
from model import build_model
from sklearn.metrics import f1_score

def train(train_path="data/train", val_path="data/val", epochs=10, model_name="efficientnet_b4"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = RealFakeDataset(train_path)
    val_ds = RealFakeDataset(val_path, train=False)
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16, shuffle=False)
    net = build_model(model_name).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
    lossf = nn.CrossEntropyLoss()
    best_f1 = 0
    for ep in range(epochs):
        net.train()
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(net(x), y)
            loss.backward()
            opt.step()
        net.eval(); preds = []; targs = []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device)
                preds += net(x).argmax(1).cpu().tolist()
                targs += y.tolist()
        f1 = f1_score(targs, preds, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            torch.save(net.state_dict(), "outputs/checkpoints/best.pt")
    print("Best F1:", best_f1)

if __name__ == "__main__":
    train()
