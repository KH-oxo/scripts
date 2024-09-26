from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

def get_dataset(dataset_class):
    # 学習用データセットを取得
    training_data = dataset_class(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    
    # 評価用データセットを取得
    test_data = dataset_class(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    return training_data, test_data

if __name__ == "__main__":

    # 学習用・評価用データセットを取得
    training_data, test_data = get_dataset(datasets.FashionMNIST)

    # データローダー
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    for train_features, train_labels in train_dataloader:
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0].squeeze()
        label = train_labels[0]
        plt.imshow(img, cmap="gray")
        plt.show()
        print(f"Label: {label}")
        break