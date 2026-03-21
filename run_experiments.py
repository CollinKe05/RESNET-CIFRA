# 从我们的核心引擎中导入封装好的函数
from trainer import run_training

# ==========================================
# 实验 1：复现我们刚才的基准实验
# ==========================================
'''
run_training(
    dataset_name='CIFAR10',
    model_name='resnet18',
    epochs=5,
    batch_size=64,
    lr=0.001
)
'''

# ==========================================
# 实验 2：加深网络，看看 ResNet-34 会不会更强？
# ==========================================
# 你可以把这段代码取消注释来运行它
run_training(
    dataset_name='CIFAR10',
    model_name='resnet34',
    epochs=5,
    batch_size=64,
    lr=0.001
)

# ==========================================
# 实验 3：挑战地狱难度，用更深的网络打 CIFAR-100！
# 因为难度变大，我们把学习率稍微调小一点，多跑几轮
# ==========================================
"""
run_training(
    dataset_name='CIFAR100',
    model_name='resnet34',
    epochs=10,
    batch_size=128,
    lr=0.0005
)
"""

# ==========================================
# 实验 4：ResNet-18 在 CIFAR-100 上的表现
# ==========================================
'''
run_training(
    dataset_name='CIFAR100',
    model_name='resnet18',
    epochs=5,
    batch_size=64,
    lr=0.001
)
'''