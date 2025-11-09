import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # CNNの層を定義する
        # Conv2d(入力チャネル, 出力チャネル, カーネルサイズ)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2) # 2x2のプーリング
        # 16チャネル(conv1) -> 32チャネル
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        
        # 全結合層 (分類器)
        # 32チャネル * 7x7画像サイズ -> 10クラス (0-9の数字)
        # (28x28 -> pool1 -> 14x14 -> pool2 -> 7x7)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        # データがモデルを流れる順番を定義する
        x = self.conv1(x) # -> [batch_size, 16, 28, 28]
        x = self.relu(x)
        x = self.pool(x)  # -> [batch_size, 16, 14, 14]
        
        x = self.conv2(x) # -> [batch_size, 32, 14, 14]
        x = self.relu(x)
        x = self.pool(x)  # -> [batch_size, 32, 7, 7]
        
        # 全結合層に入力するために、データを1次元に「平坦化」する
        x = x.view(-1, 32 * 7 * 7) # .view でテンソルの形状を変更
        
        x = self.fc1(x) # -> [batch_size, 10]
        return x

model = SimpleCNN()

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("学習開始...")
num_epochs = 3 # 3回だけデータ全体を学習する (お試し)

for epoch in range(num_epochs):
    # (A) 学習フェーズ
    model.train() # モデルを「訓練モード」に
    for images, labels in train_loader:
        # images: [64, 1, 28, 28] のテンソル
        # labels: [64] のテンソル (正解の数字 0-9)
        
        # (1) 予測
        outputs = model(images)
        
        # (2) 損失の計算
        loss = loss_fn(outputs, labels)
        
        # (3) 勾配のリセット
        optimizer.zero_grad()
        
        # (4) 勾配の計算
        loss.backward()
        
        # (5) パラメータの更新
        optimizer.step()
    
    # (B) 評価フェーズ (1エポックごとに正解率を計算)
    model.eval() # モデルを「評価モード」に
    correct = 0
    total = 0
    with torch.no_grad(): # 勾配計算をオフ
        for images, labels in test_loader:
            outputs = model(images)
            # torch.max で、予測結果 (10クラス) のうち一番確率が高いものを取得
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%")

print("学習完了！")