import os
import itertools
import matplotlib.pyplot as plt
from trainer import run_training # 注意这里改成你实际的文件名

# 必须把所有执行动作包裹在这个 if 语句里！
if __name__ == '__main__':
    # ==========================================
    # 1. 环境初始化：自动创建结果文件夹
    # ==========================================
    folders = ['result/log', 'result/diagram', 'result/weights']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    print(f"📁 结果存储目录已就绪: {folders}")

    # ==========================================
    # 2. 定义网格搜索空间
    # ==========================================
    param_grid = {
        'dataset': ['CIFAR10'],
        'model': ['resnet18', 'resnet34'],
        'lr': [0.0005,0.001, 0.01, 0.1],
        'batch_size': [128],
        'epochs': [5, 50, 100]  # 测试先跑5个epoch，后续再增加到50和100
    }

    keys = param_grid.keys()
    experiments = [dict(zip(keys, combo)) for combo in itertools.product(*param_grid.values())]

    # ==========================================
    # 3. 运行实验并生成独立图表
    # ==========================================
    results = {}

    for config in experiments:
        exp_name, losses, final_acc = run_training(config)
        results[exp_name] = {'losses': losses, 'accuracy': final_acc}
        
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(losses) + 1), losses, marker='o', color='b', linewidth=2)
        plt.title(f'Training Loss: {exp_name}', fontsize=12)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        diagram_path = f"result/diagram/{exp_name}_Loss.png"
        plt.savefig(diagram_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"📈 专属 Loss 图已保存至: {diagram_path}")

    print("\n🎉 全部实验结束！正在生成全局对比图...")

    # ==========================================
    # 4. 生成【网格搜索全局对比图】
    # ==========================================
    plt.figure(figsize=(12, 7))

    for exp_name, data in results.items():
        losses = data['losses']
        acc = data['accuracy']
        plt.plot(range(1, len(losses) + 1), losses, marker='s', markersize=5, linewidth=2, 
                 label=f"{exp_name} (Acc: {acc:.2f}%)")

    plt.title('Grid Search: Loss Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.legend(fontsize=10, loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)

    final_diagram_path = "result/diagram/GridSearch_Comparison.png"
    plt.savefig(final_diagram_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 全局对比大图已保存至: {final_diagram_path}")