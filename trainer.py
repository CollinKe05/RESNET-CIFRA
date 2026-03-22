import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import classification_report
from tqdm import tqdm

def run_training(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 解析超参数
    dataset_name = config.get('dataset', 'CIFAR10')
    model_name = config.get('model', 'resnet18')
    epochs = config.get('epochs', 10)
    batch_size = config.get('batch_size', 128)
    lr = config.get('lr', 0.001)

    # 🚀 核心：自动生成标准化的实验名称
    exp_name = f"{dataset_name}_{model_name}_ep{epochs}_bs{batch_size}_lr{lr}"

    print("\n" + "="*60)
    print(f"🚀 开始执行: {exp_name}")
    print("="*60)

    # 2. 数据处理与加载
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    if dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
        target_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    else:
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
        target_names = [str(i) for i in range(100)]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. 模型与优化器
    model_class = getattr(models, model_name)
    #是否使用预训练权重
    model = model_class(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')

    epoch_losses = []

    # 4. 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        epoch_losses.append(running_loss / len(trainloader))

        # ==========================================
        # 💾 新增：Checkpoint 阶段性存档机制
        # ==========================================
        # 每隔 10 轮（或者你觉得合适的轮数），存一次当前状态
        if (epoch + 1) % 10 == 0:
            ckpt_path = f"result/weights/{exp_name}_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_path)
            # tqdm 的 write 方法可以在不打断进度条的情况下打印信息
            tqdm.write(f"💾 自动存档: 已保存第 {epoch+1} 轮权重至 {ckpt_path}")
    # 5. 测试与评估
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    report = classification_report(all_targets, all_preds, target_names=target_names, digits=4)
    acc = sum(1 for p, t in zip(all_preds, all_targets) if p == t) / len(all_targets) * 100

    # ==========================================
    # 💾 核心新增：自动化硬盘 I/O (保存权重与日志)
    # ==========================================
    # 保存网络参数权重
    weight_path = f"result/weights/{exp_name}.pth"
    torch.save(model.state_dict(), weight_path)
    
    # 保存完整的测试报告 txt
    log_path = f"result/log/{exp_name}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment Configuration:\n{config}\n\n")
        f.write(f"Final Accuracy: {acc:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    print(f"✅ 权重已保存至: {weight_path}")
    print(f"✅ 日志已保存至: {log_path}")
    
    return exp_name, epoch_losses, acc