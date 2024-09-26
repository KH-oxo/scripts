import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def get_dataset(dataset):

    if dataset not in [datasets.SVHN, datasets.STL10, datasets.CelebA]:
        # 学習用データセットを取得
        training_data = dataset(
            root="data",
            train=True,
            download=True,
            transform=ToTensor()
        )

        # 評価用データセットを取得
        test_data = dataset(
            root="data",
            train=False,
            download=True,
            transform=ToTensor()
        )
    else:
        #splitが必要な学習用データセットを取得
        training_data= dataset(
            root="data",
            split='train',
            download=True,
            transform=ToTensor()
        )

        #splitが必要な評価用データセットを取得
        test_data = dataset(
            root="data",
            split='test',
            download=True,
            transform=ToTensor()
        )

    return training_data, test_data

def show_labels(dataset, labels_map, figsize, shape):
    # この関数ではMatplotlibというライブラリを使用して画像マトリクスを編集＆表示しています
    # 各処理の詳細については下記記事を参考↓
    # https://qiita.com/kenichiro_nishioka/items/8e307e164a4e0a279734

	# マトリクスを作成
    figure = plt.figure(figsize=figsize)
    cols = shape[0]
    rows = shape[1]
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        # 画像の形状に応じて処理
        if img.shape[0] == 1: #テンソル形式の1チャンネル目はグレー画像なら１
          # グレースケール画像の場合
          # img.squeezeでテンソル形状(C, H, W)のCを削除し、テンソル形式(H, W)としてグレー画像として扱う。
          plt.imshow(img.squeeze(), cmap="gray")
        else:
          # カラー画像の場合
          # plt.imshow等のMatplotlibの関数はNumpy配列を入力として受け取る。
          # テンソル形状(C, H, W)を、(H, W, C)に変えることで、Numpy配列として扱うことができる。
          plt.imshow(img.permute(1, 2, 0))
          #imgの形状を考える。
        print(type(img))
    #plt.imshowの中で
    # 完成したマトリクスを表示
    plt.show()


if __name__ == "__main__":
    # 各番号に紐づくラベル名を設定
    FashionMNIST = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    CIFAR10 = {
        0: "plane",
        1: "car",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck",
    }

    MNIST = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }

    SVHN = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }

    CIFAR100 = {i:i for i in range(100)}

    STL10 = {i:i for i in range(100)}

    CelebA = {i:i for i in range(100)}

    # 学習用・評価用データセットを取得
    dataset_key_list = ["FashionMNIST", "CIFAR10", "MNIST", "CIFAR100", "SVHN", "STL10"] # ここを変更してデータセットを選択する

    for  dataset_key in dataset_key_list:

        if "FashionMNIST" == dataset_key :
            dataset = datasets.FashionMNIST
            labels_map = FashionMNIST
        elif "CIFAR10" == dataset_key :
            dataset = datasets.CIFAR10
            labels_map = CIFAR10
        elif "MNIST" == dataset_key :
            dataset = datasets.MNIST
            labels_map = MNIST
        elif "CIFAR100" == dataset_key :
            dataset = datasets.CIFAR100
            labels_map = CIFAR100
        elif "SVHN" == dataset_key :
            dataset = datasets.SVHN
            labels_map = SVHN
        elif "STL10" == dataset_key :
            dataset = datasets.STL10
            labels_map = STL10
        elif "CelebA" == dataset_key :
            dataset = datasets.CelebA
            labels_map = CelebA   
        else:
            print("データセットはありません")

        training_data, test_data = get_dataset(dataset) # type: ignore
        #classを取得する

        # データセットからランダムに画像を取得して表示する
        show_labels(training_data, labels_map, figsize=(8,8), shape=(3,3) ) # type: ignore