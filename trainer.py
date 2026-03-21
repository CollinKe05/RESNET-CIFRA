import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

def run_training(dataset_name='CIFAR10', model_name='resnet18', epochs=5, batch_size=64, lr=0.001):
    """
    核心训练框架
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*40)
    print(f"🚀 启动实验 | 模型: {model_name} | 数据集: {dataset_name}")
    print(f"⚙️ 超参数 | Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")
    print("="*40)

    # 1. 动态选择数据集 (CIFAR10 或 CIFAR100)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # 随机裁剪
        transforms.RandomHorizontalFlip(),    # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    if dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        num_classes = 10
    elif dataset_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        num_classes = 100
    else:
        raise ValueError("目前只支持 CIFAR10 或 CIFAR100 哦！")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    # 2. 动态召唤模型 (resnet18, resnet34 等)
    # getattr 可以根据字符串名字，直接从 models 库里把对应的模型抓出来
    model_class = getattr(models, model_name)
    model = model_class(weights=None)
    
    # 修改最后一层以匹配数据集的类别数
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    # 3. 设置优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 4. 训练循环
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 199: # 每 200 批次打印一次，让终端清爽点
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 200:.3f}')
                running_loss = 0.0

    # 5. 测试评估
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'🏆 最终测试集准确率: {acc:.2f} %\n')
    return acc