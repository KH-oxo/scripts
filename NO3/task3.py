import os
import torch
import pandas as pd
from torch import nn
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

#定数
epochs = 100
test_size = 0.2
batch_size = 32
global_step = 0
output_path = './NO3/task3_model.pth'
log_dir = './NO3/logs'

class CustomImageDataset(Dataset):
    """データセット"""
    def __init__(self, dataset_df, img_dir, transform=None, target_transform=None):
        self.img_data = dataset_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_data.iloc[idx]['file_name'])
        image = read_image(img_path).float().div(255.0)
        label_id = torch.tensor(self.img_data.iloc[idx]['label_id'], dtype=torch.long)
        label_name = self.img_data.iloc[idx]['label_name']
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label_id = self.target_transform(label_id)
        data_dict = {
            'file_name': self.img_data.iloc[idx]['file_name'],
            'label_id': label_id,
            'label_name': label_name,
            'image': image
        }
        return data_dict

def split_dataset(dataset, test_size):
    """データセットを分割"""
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset

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

def train(dataloader, model, loss_fn, optimizer, writer, global_step):
    """訓練"""
    size = len(dataloader.dataset)
    for batch_idx, batch in enumerate(dataloader):
        # 画像とラベルを取得
        x, y = batch['image'].to(device), batch['label_id'].to(device)
        
        # 損失誤差を計算
        pred = model(x)
        loss = loss_fn(pred, y)
        
        # バックプロパゲーション
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 100バッチごとに損失を表示
        if batch_idx % 100 == 0:
            loss_value, current = loss.item(), batch_idx * len(x)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")

        # TensorBoardに損失を記録（バッチごとにグローバルステップで記録）
        writer.add_scalar('Loss/train', loss, global_step)
        
        # グローバルステップの更新
        global_step += 1
    
    return global_step  # 次のエポックでもグローバルステップを維持する

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            #画像とラベルを取得
            x, y = batch['image'].to(device), batch['label_id'].to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).sum().item()
    test_loss /= size
    correct /= size

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    #TensorBoardのSummaryWriterを初期化
    writer = SummaryWriter(log_dir=log_dir)

    #画像ファイル名＋ラベルIDのtsvファイル、ラベル名のtsvファイルを結合し、データフレームを作成
    train_df = pd.read_csv('./data/train_master.tsv', sep='\t')
    label_df = pd.read_csv('./data/label_master.tsv', sep='\t')
    train_df['label_id'] = train_df['label_id'].astype(int)
    dataset_df = pd.merge(train_df, label_df, on='label_id')
    #画像ファイルパス
    img_dir = './data/train_images'

    #データセット用インスタンスの生成
    dataset = CustomImageDataset(dataset_df, img_dir, transform=None, target_transform=None)
    train_dataset, test_dataset = split_dataset(dataset, test_size)

    #データローダー
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    #訓練に際して、可能であればGPU（cuda）を設定します。GPUが搭載されていない場合はCPUを使用します
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    #モデルを表示
    model = NeuralNetwork().to(device)
    print(model)

    #損失関数
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        global_step = train(train_dataloader, model, loss_fn, optimizer, writer, global_step)
        test(test_dataloader, model)
    print("Done!")

    #モデルの保存
    torch.save(model.state_dict(), output_path)
    print("Saved PyTorch Model State to model.pth")

    writer.close()