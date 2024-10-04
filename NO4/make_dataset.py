import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

#画像ファイルパス
img_dir = './data/train_images'
#保存するフォルダのパス
train_dir = './data/dataset_1/train'
test_dir = './data/dataset_1/test'
#ランダム定数
random_state = 42

#フォルダが存在しない場合は作成
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

if __name__ == "__main__":

    #画像ファイル名＋ラベルIDのtsvファイル、ラベル名のtsvファイルを結合し、データフレームを作成
    train_df = pd.read_csv('./data/train_master.tsv', sep='\t')
    label_df = pd.read_csv('./data/label_master.tsv', sep='\t')
    train_df['label_id'] = train_df['label_id'].astype(int)
    dataset_df = pd.merge(train_df, label_df, on='label_id')

    #各ラベルIDごとにデータを分割し、フォルダにコピー
    for label_id in dataset_df['label_id'].unique():

        #該当するラベルのデータを取得
        label_data = dataset_df[dataset_df['label_id'] == label_id]
        #ラベル名を取得
        label_name = label_data['label_name'].iloc[0]
        
        #ラベルごとに訓練データとテストデータを8:2に分割
        train_data, test_data = train_test_split(label_data, test_size=0.2, random_state=random_state)
        
        #ラベルごとにフォルダを作成
        label_folder_name = f"{label_id}.{label_name}" #x.ラベル名
        label_train_dir = os.path.join(train_dir, label_folder_name)
        label_test_dir = os.path.join(test_dir, label_folder_name)
        os.makedirs(label_train_dir, exist_ok=True)
        os.makedirs(label_test_dir, exist_ok=True)
        
        #訓練データセット作成
        for img_name in train_data['file_name']:
            img_path = os.path.join(img_dir, img_name) #画像のパスを作成
            shutil.copy(img_path, label_train_dir) #訓練データフォルダにコピー
        
        #テストデータセット作成
        for img_name in test_data['file_name']:
            img_path = os.path.join(img_dir, img_name) #画像のパスを作成
            shutil.copy(img_path, label_test_dir) #テストデータフォルダにコピー

    print("データの分割とフォルダへのコピーが完了しました。")