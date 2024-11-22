import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

#定数
epochs = 5
batch_size = 64
output_path = './NO5/task1_model.pth'

#訓練データセット
train_dir = './data/dataset_1/train'
#テストデータセット
test_dir = './data/dataset_1/test'

#データオーギュメンテーション
train_transform = transforms.Compose([
    transforms.ToTensor()
])
test_transform = transforms.Compose([
    transforms.ToTensor()
])

class NeuralNetwork(nn.Module):
    """モデルを定義"""
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3*96*96, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def train(dataloader, model, loss_fn, optimizer):
    """訓練"""
    size = len(dataloader.dataset)
    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch

        #画像とラベルを取得
        x, y = images.to(device), labels.to(device)
        #ラベルをワンホットエンコーディング
        y_one_hot = F.one_hot(y, num_classes=10).float()
        
        #損失誤差を計算
        pred = model(x)
        loss = loss_fn(pred, y_one_hot)
        
        #バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #100バッチごとに損失を表示
        if batch_idx % 100 == 0:
            loss_value = loss.item()
            current = batch_idx * len(x)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            #画像とラベルを取得
            x, y = images.to(device), labels.to(device)
            # ラベルをワンホットエンコード
            y_one_hot = F.one_hot(y, num_classes=10).float()
            pred = model(x)
            test_loss += loss_fn(pred, y_one_hot).item()
            correct += (pred.argmax(1) == y).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":

    #訓練に際して、可能であればGPU（cuda）を設定します。GPUが搭載されていない場合はCPUを使用します
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    #データセット
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    #データローダー
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size = 64, shuffle=True)

    #モデルを表示
    model = NeuralNetwork().to(device)
    print(model)

    #損失関数：BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss()
    
    #学習の最適化：確率的勾配降下法
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    #訓練・テストの実施
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
    print("Done!")

    #モデルの保存
    torch.save(model.state_dict(), output_path)
    print("Saved PyTorch Model State to model.pth")

    #モデルの読み込み
    model = NeuralNetwork()
    model.load_state_dict(torch.load(output_path))

    #予測結果
    batch = next(iter(test_dataloader))
    images, labels = batch
    model.eval()
    with torch.no_grad():
        #モデルの予測を取得
        outputs = model(images)
    #予測されたクラスのインデックスを取得
    _, predicted = torch.max(outputs, 1)
    #予測クラス名と実際のクラス名
    pred_name = test_dataset.classes[predicted[0].item()]
    actual_name = test_dataset.classes[labels[0].item()]
    #予測されたラベル名と実際のラベル名を表示
    print(f'Predicted: "{pred_name}", Actual: "{actual_name}"')