import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, file_path, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(file_path, sep = '\t') #tsvファイルに対応
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample

if __name__ == "__main__":

    #TSVファイルのパス
    file_path = './data/train_master.tsv'
    #画像ファイルパス
    img_dir = './data/train_images'

    #データセット用インスタンスの生成
    dataset = CustomImageDataset(file_path, img_dir, transform=None, target_transform=None)

    #データローダー
    bs = 5
    train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    #train_dataloaderは全画像からランダムで
    #バッチサイズ分の画像データをテンソル形式で供給するイテレーターとなる。

for batch in train_dataloader: #forループする度にイテレーションされる。
    #__getitem__が辞書で返している↓
    images = batch["image"] # バッチサイズ分の画像を取り出す
    labels = batch["label"] # バッチサイズ分のラベルを取り出す
    print(f"Feature batch shape: {images.size()}")
    print(f"Labels batch shape: {labels.size()}")
    #バッチ分の画像をforループさせる
    for i in range(len(images)):
        img = images[i] #画像リストのi番目を取り出す
        label = labels[i] #ラベルリストのi番目を取り出す
        if img.shape[0] == 1: #グレー画像なら
            plt.imshow(img.squeeze(), cmap="gray")
        else: #カラー画像なら
            plt.imshow(img.permute(1, 2, 0)) #チャンネル順を(H, W, C)に変更
        plt.show()
        print(f"Label: {label}")
        break
    break
    #break２回で画像１枚
    #61行breakでバッチサイズ１回分の画像
    #break無しでループ回数=総画像数÷バッチサイズ