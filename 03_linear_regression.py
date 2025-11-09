import torch

x_train = torch.tensor([[1.0],[2.0],[3.0],[4.0]])
y_train = torch.tensor([[2.0],[4.0],[6.0],[8.0]])

model = torch.nn.Linear(1, 1)

loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("学習開始...")

for epoch in range(100):
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch[{epoch + 1}], Loss:{loss.item():.4f}")

print("学習終了！")

model.eval()

with torch.no_grad():
    test_input = torch.tensor([[5.0]])
    predicted_output = model(test_input)
    print(f"x = 5.0 のときの予測 y = {predicted_output.item():.4f}")

for param in model.parameters():
    print(param)