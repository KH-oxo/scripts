import torch
from torch import nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader

#定数
epochs = 5
batch_size = 64
output_path = './NO6/task1_model.pth'

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

class CNN_1(nn.Module):
    """軽量で精度を重視したCNNモデル"""
    def __init__(self, input_channels=3, num_classes=10):
        super(CNN_1, self).__init__()

        # 全ての特徴抽出レイヤーをまとめる
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            self.conv_block(16, 32),
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            self.conv_block(256, 512)
        )

        self.conv_last = nn.Conv2d(512, 1024, kernel_size=1, bias=False)  #最後の畳み込み層
        self.bn_last = nn.BatchNorm2d(1024)  #バッチ正規化
        self.global_pool = nn.AdaptiveAvgPool2d(1)  #1次元に変換
        self.dropout = nn.Dropout(0.2)  #ドロップアウト
        self.fc = nn.Linear(1024, num_classes)  #出力層

    def conv_block(self, in_channels, out_channels):
        """畳み込みブロック"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False), #畳み込み層
            nn.BatchNorm2d(out_channels), #バッチ正規化
            nn.ReLU(), #活性化関数：ReLu
            nn.AvgPool2d(kernel_size=2, stride=2) #グローバルアベレージプーリング：縦横/２
        )

    def forward(self, x):
        x = self.features(x)  #特徴抽出
        x = self.conv_last(x)  #最後の畳み込み層
        x = self.bn_last(x) #バッチ正規化
        x = self.global_pool(x)  #1次元に変換
        x = torch.flatten(x, 1)  #全結合層
        x = self.dropout(x)  #ドロップアウト
        x = self.fc(x)  #出力層
        return x

def train(dataloader, model, loss_fn, optimizer):
    """訓練"""
    size = len(dataloader.dataset)
    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch
        #画像とラベルを取得
        x, y = images.to(device), labels.to(device)
        
        #損失誤差を計算
        pred = model(x)
        loss = loss_fn(pred, y)
        
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
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
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
    model = CNN_1().to(device)
    print(model)

    #損失関数：クロスエントロピー誤差
    loss_fn = nn.CrossEntropyLoss()

    #学習の最適化：AdamW
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

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
    model = CNN_1()
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