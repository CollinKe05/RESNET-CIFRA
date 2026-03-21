import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# ==========================================
# 1. 硬件设置：有显卡用显卡，没显卡用 CPU
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算设备: {device}")

# ==========================================
# 2. 数据准备：加载 CIFAR-10 数据集
# ==========================================
# 定义数据预处理方式：转为张量并进行标准化 (为了加速训练和提高精度)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64 # 每次喂给模型 64 张图片

# 训练集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

# 测试集
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# ==========================================
# 3. 模型搭建：召唤 ResNet-18
# ==========================================
model = models.resnet18(weights=None) # 不使用预训练权重，从头开始学
num_ftrs = model.fc.in_features       # 获取原网络最后一层的输入维度
model.fc = nn.Linear(num_ftrs, 10)    # 将最后一层修改为输出 10 个类别
model = model.to(device)              # 将模型搬运到 GPU 或 CPU

# ==========================================
# 4. 设定训练规则：损失函数与优化器
# ==========================================
criterion = nn.CrossEntropyLoss() # 分类任务最常用的交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam 优化器，学习率设为 0.001

# ==========================================
# 5. 核心环节：开始训练循环！
# ==========================================
epochs = 5 # 为了让你快速看到结果，我们先只跑 5 轮

print("\n--- 🚀 开始训练 ---")
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # 取出数据并搬运到对应设备
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # 核心 5 步曲：
        optimizer.zero_grad()             # a. 梯度清零 (防止上一次的梯度累积)
        outputs = model(inputs)           # b. 前向传播 (让模型做预测)
        loss = criterion(outputs, labels) # c. 计算损失 (看看预测得准不准)
        loss.backward()                   # d. 反向传播 (计算每个参数该往哪边调整)
        optimizer.step()                  # e. 更新参数 (正式修改模型权重)

        # 打印训练进度
        running_loss += loss.item()
        if i % 100 == 99: # 每 100 个批次打印一次状态
            print(f'[第 {epoch + 1} 轮, 批次 {i + 1}] 实时损失 Loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print("--- 🎉 训练完成！ ---")

# ==========================================
# 6. 最终考核：在测试集上评估准确率
# ==========================================
correct = 0
total = 0
# 测试阶段不需要更新参数，所以关闭梯度计算，这能大大节省显存和算力
with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        # torch.max 返回最大值和最大值所在的索引，索引就是模型预测的类别
        _, predicted = torch.max(outputs.data, 1) 
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'\n🏆 模型在 10000 张测试集图片上的整体准确率: {100 * correct / total:.2f} %')