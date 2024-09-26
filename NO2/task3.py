import os
import json
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset): #データセットリストを渡すように変更
    def __init__(self, dataset_list, transform=None, target_transform=None):
        self.img_data = dataset_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx): #リスト辞書を使用する
        img_path = self.img_data[idx]['image_path'] #対応する画像パスを取得
        image = read_image(img_path)
        label_id = self.img_data[idx]['label_id'] #対応するラベルIDを取得
        label_name = self.img_data[idx]['label_name'] #対応するラベル名を取得
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label_id": label_id, "label_name": label_name} #確認用に辞書にラベル名を追加
        return sample

if __name__ == "__main__":

    # データを保存するリスト
    dataset_list = []

    #データセットが格納されているフォルダ
    dataset_folder_path = './NO2/dataset'

    #jsonフォルダのパス
    json_folder = './NO2'
    json_file_path = os.path.join(json_folder, 'translation.json')
    with open(json_file_path, 'r', encoding='utf-8') as f:
        dataset_json = json.load(f)

    #ラベルID
    i = 0
    for folder, label in dataset_json.items(): #jsonファイルから、画像が入ったフォルダ名、ラベル名を取得
        folder_path = os.path.join(dataset_folder_path, folder) #フォルダパスを作成
        for image_name in os.listdir(folder_path): #データセットの各フォルダ内の画像名を全て取り出す。
            image_path = os.path.join(folder_path, image_name) #フォルダパスと画像名をos.path.joinすることで、画像パスとなる。
            #辞書を作成（ラベルID, ラベル名, 画像パス）
            dataset_list.append({
                'label_id': i,
                'label_name': label,
                'image_path': image_path
            })
        i += 1 #フォルダ内の画像を全て取得後インクリメント

    #データセット用インスタンスの生成
    dataset = CustomImageDataset(dataset_list, transform=None, target_transform=None)
    #データローダー
    bs = 5
    train_dataloader = DataLoader(dataset, batch_size=bs, shuffle=True)
    #train_dataloaderは全画像からランダムで
    #バッチサイズ分の画像データをテンソル形式で供給するイテレーターとなる。

for batch in train_dataloader: #forループする度にイテレーションされる。#バッチサイズ5なら1000回？
    #__getitem__が辞書で返している↓
    images = batch["image"] # バッチサイズ分の画像を取り出す
    labels_id = batch["label_id"] # バッチサイズ分のラベルを取り出す
    labels_name = batch["label_name"]
    print(f"Feature batch shape: {images.size()}")
    print(f"Labels batch shape: {labels_id.size()}")
    #バッチ分の画像をforループさせる
    for i in range(len(images)):
        img = images[i] #画像リストのi番目を取り出す
        label_id = labels_id[i] #ラベルリストのi番目を取り出す
        label_name = labels_name[i] #ラベルリストのi番目を取り出す
        if img.shape[0] == 1: #グレー画像なら
            plt.imshow(img.squeeze(), cmap="gray")
        else: #カラー画像なら
            plt.imshow(img.permute(1, 2, 0)) #チャンネル順を(H, W, C)に変更
        plt.show()
        print(f"Label_id: {label_id}, Label_name: {label_name}")
        break
    break
    #break２回で画像１枚
    #94行breakでバッチサイズ１回分の画像
    #break無しでループ回数=総画像数÷バッチサイズ