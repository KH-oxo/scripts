import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#定数
batch_size = 10

#訓練データセット
train_dir = './data/dataset_1/train'

#データオーギュメンテーション
transform = transforms.Compose([
    transforms.GaussianBlur(7, sigma=(2.0, 5.0)),
    transforms.ToTensor()
])

def show_labels(dataloader, figsize, shape):

	#マトリクスを作成
    figure = plt.figure(figsize=figsize)
    cols = shape[0] #列
    rows = shape[1] #行

    batch = next(iter(dataloader))
    images, labels = batch

    for i in range(1, rows * cols + 1):
        img, label = images[i], labels[i]
        figure.add_subplot(rows, cols, i)
        plt.title(dataloader.dataset.classes[label.item()])
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0))
    plt.show()

if __name__ == "__main__":

    #データセット
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    #データローダー
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    #データセットからランダムに画像を取得して表示する
    show_labels(train_dataloader, figsize=(8,8), shape=(3,3) )