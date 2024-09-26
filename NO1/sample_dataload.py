import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def get_dataset(dataset):

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
        plt.imshow(img.squeeze(), cmap="gray")

    # 完成したマトリクスを表示
    plt.show()


if __name__ == "__main__":
    # 学習用・評価用データセットを取得
    training_data, test_data = get_dataset(datasets.FashionMNIST)

    # 各番号に紐づくラベル名を設定
    labels_map = {
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

    # データセットからランダムに画像を取得して表示する
    show_labels(training_data, labels_map, figsize=(8,8), shape=(3,3) )
