import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from task1 import CustomImageDataset, NeuralNetwork
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

#定数
batch_size = 32
test_size = 0.2
input_path = './NO3/task1_model.pth'

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, true_couter, false_counter = 0, 0, 0  # 正解数と不正解数をカウントする変数を用意
    with torch.no_grad():
        for batch in dataloader:
            # 画像とラベルを取得
            x, y = batch['image'].to(device), batch['label_id'].to(device).long()
            pred = model(x)
            
            # 損失を計算
            test_loss += loss_fn(pred, y).item()
            
            # 予測が正しいかどうかを確認
            true_pred = (pred.argmax(1) == y)
            true_couter += true_pred.sum().item()  # 正解数をカウント
            false_counter += (true_pred == False).sum().item()  # 不正解数をカウント
    
    #平均損失と正解率を計算
    test_loss /= size
    accuracy = true_couter / size

    # 結果を表示
    print(f"Test Error: \n Accuracy: {(100 * accuracy):>0.1f}%, Avg loss: {test_loss:>8f}")
    print(f"True: {true_couter}, False: {false_counter}, Total: {size}")

def split_dataset(dataset, test_size):
    """データセットを分割"""
    #データセット全体のインデックスを取得
    indices = list(range(len(dataset)))
    
    #データを8:2に分割
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    
    #トレーニングとテストデータセットを作成
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, test_dataset

if __name__ == "__main__":

    #画像ファイル名＋ラベルIDのtsvファイル、ラベル名のtsvファイルを結合し、データフレームを作成
    train_df = pd.read_csv('./data/train_master.tsv', sep='\t')
    label_df = pd.read_csv('./data/label_master.tsv', sep='\t')
    train_df['label_id'] = train_df['label_id'].astype(int)
    dataset_df = pd.merge(train_df, label_df, on='label_id')
    #画像ファイルパス
    img_dir = './data/train_images'

    #データセット用インスタンスの生成
    dataset = CustomImageDataset(dataset_df, img_dir, transform=None, target_transform=None)
    #訓練・テストデータに分割
    train_dataset, test_dataset = split_dataset(dataset, test_size)

    #データローダー
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False)

    #訓練に際して、可能であればGPU（cuda）を設定します。GPUが搭載されていない場合はCPUを使用します
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    #モデルの読み込み
    model = NeuralNetwork()
    model.load_state_dict(torch.load(input_path))

    #損失関数
    loss_fn = nn.CrossEntropyLoss()

    #モデルテスト
    test(test_dataloader, model, loss_fn)