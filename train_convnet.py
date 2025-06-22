# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
import random
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet
from src.util import *
import time


# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 初始化网络
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1
epochs = 10
iter_per_epoch = max(train_size // batch_size, 1)
iters_num = iter_per_epoch * epochs

train_loss_list = []
train_acc_list = []
test_acc_list = []
total_elapsed_time = 0

for i in range(iters_num):
    # 随机选取 batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 计算梯度
    start = time.time()
    grad = network.gradient(x_batch, t_batch)
    elapsed = time.time() - start
    total_elapsed_time += elapsed
    # 更新参数
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        network.params[key] -= learning_rate * grad[key]

    # 记录损失
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 每轮计算一次准确率
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("epoch {:2d} | train acc: {:.4f}, test acc: {:.4f}".format(i // iter_per_epoch, train_acc, test_acc))

print("训练总耗时：{:.2f} 秒".format(total_elapsed_time))

# 绘制 loss 曲线
plt.figure()
plt.plot(train_loss_list, label='Train Loss')
plt.title('Loss Curve')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 绘制 accuracy 曲线
epochs = np.arange(len(train_acc_list))
plt.figure()
plt.plot(epochs, train_acc_list, label='Train Accuracy')
plt.plot(epochs, test_acc_list, label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
