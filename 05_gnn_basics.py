import torch
import torch.nn.functional as F
# PyTorch Geometric (PyG) のライブラリをインポート
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# --- 1. データセットの準備 (Cora) ---
# Cora: 論文の引用グラフデータセット
# ノード = 論文
# エッジ = 論文Aが論文Bを引用している
# タスク = 各論文を7つのカテゴリに分類する
dataset = Planetoid(root='./data/Cora', name='Cora')
data = dataset[0] # データセットの最初のグラフを取得

# --- 2. モデルの定義 (GCN) ---
# GCN (Graph Convolutional Network)
class SimpleGCN(torch.nn.Module):
    def __init__(self):
        super(SimpleGCN, self).__init__()
        # GCNの層を定義する
        # GCNConv(入力特徴量, 出力特徴量)
        # Coraデータセットの各ノード(論文)は1433次元の特徴を持つ
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        # 16特徴量 -> 7クラス (論文のカテゴリ数)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        # データから必要な情報を取り出す
        x, edge_index = data.x, data.edge_index
        
        # GCN層を通す
        # x (ノード特徴) と edge_index (グラフ構造) の両方を渡す
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # GCN層 2
        x = self.conv2(x, edge_index)
        
        # CrossEntropyLossを使うため、Softmaxはここでは不要
        return x

model = SimpleGCN()

# --- 3. 損失関数と最適化アルゴリズム ---
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# --- 4. 学習ループ ---
print("学習開始...")
model.train() # モデルを訓練モードに

for epoch in range(200):
    # (1) 予測
    # GNNでは、モデルにデータ全体(data)を一度に渡すことが多い
    outputs = model(data)
    
    # (2) 損失の計算
    # data.train_mask を使って「訓練用ノード」の損失だけを計算
    loss = loss_fn(outputs[data.train_mask], data.y[data.train_mask])
    
    # (3) 勾配のリセット
    optimizer.zero_grad()
    
    # (4) 勾配の計算
    loss.backward()
    
    # (5) パラメータの更新
    optimizer.step()
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")

# --- 5. 評価 ---
print("学習完了！ テストデータで評価...")
model.eval() # モデルを評価モードに
with torch.no_grad():
    outputs = model(data)
    # 予測結果 (7クラス) のうち一番確率が高いものを取得
    _, predicted = torch.max(outputs.data, 1)
    
    # 「テスト用ノード」で正解率を計算
    correct = (predicted[data.test_mask] == data.y[data.test_mask]).sum().item()
    total = data.test_mask.sum().item()
    accuracy = 100 * correct / total
    
    print(f"Test Accuracy: {accuracy:.2f}%")