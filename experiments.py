# 从我们的核心引擎中导入封装好的函数
from trainer import run_training

# ==========================================
# 实验 1：ResNet-18 在 CIFAR-10 上的表现
# ==========================================
#不同epoches
'''
run_training(
    dataset_name='CIFAR10',
    model_name='resnet18',
    epochs=5,
    batch_size=64,
    lr=0.001
)
'''
'''
run_training(
    dataset_name='CIFAR10',
    model_name='resnet18',
    epochs=50,
    batch_size=64,
    lr=0.001
)
'''
'''
run_training(
    dataset_name='CIFAR10',
    model_name='resnet18',
    epochs=200,
    batch_size=64,
    lr=0.001
)
'''
#不同batch_size
'''
run_training(
    dataset_name='CIFAR10',
    model_name='resnet18',
    epochs=5,
    batch_size=64,
    lr=0.001
)
'''
#不同lr
'''
run_training(
    dataset_name='CIFAR10',
    model_name='resnet18',
    epochs=5,
    batch_size=64,
    lr=0.01
)
'''
#best



# ==========================================
# 实验 2：ResNet-34 在 CIFAR-10 上的表现
# ==========================================
'''
run_training(
    dataset_name='CIFAR10',
    model_name='resnet34',
    epochs=5,
    batch_size=64,
    lr=0.001
)
'''
#best


# ==========================================
# 实验 3：ResNet-18 在 CIFAR-100 上的表现
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

# ==========================================
# 实验 4：ResNet-34 在 CIFAR-100 上的表现
# ==========================================

run_training(
    dataset_name='CIFAR100',
    model_name='resnet34',
    epochs=10,
    batch_size=128,
    lr=0.0005
)


