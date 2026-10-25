# Deep Learning Lab · 从梯度计算到图像分类

[简体中文](README.md) | [English](README_EN.md)

围绕“**梯度如何计算、网络如何训练、框架如何组织完整流程**”展开的深度学习实验。以 NumPy 理解底层算子，再使用 MindSpore 与 PyTorch 构建图像分类模型，同时记录 DeepSeek 蒸馏模型的本地部署方法。

**技术栈：** Python · NumPy · SymPy · MindSpore · PyTorch · Jupyter

## 核心内容

### 1. 梯度计算与反向传播

- 使用 SymPy 进行符号求导。
- 实现前向差分与中心差分，对照解析导数理解数值精度。
- 通过计算图示例展示前向和反向链式求导。
- 使用数值梯度对照反向传播结果，检查网络参数梯度。

### 2. NumPy 神经网络基础

`util.py` 包含 Sigmoid、稳定 Softmax、交叉熵、SGD、数值梯度和相对误差等工具。卷积部分提供：

- 朴素卷积与卷积反向传播；
- `im2col` / `col2im` 数据重排；
- 将卷积转换为矩阵乘法的实现；
- stride、padding 和卷积输出尺寸实验。

Notebook 进一步展示图像灰度转换、边缘滤波，以及全连接网络和卷积网络训练流程。

### 3. MindSpore：MNIST 手写数字识别

构建数据下载、标准化、分批、模型训练、测试和预测可视化流程。

| 配置 | 内容 |
| --- | --- |
| 模型 | LeNet5：两层卷积、平均池化、三层全连接 |
| 激活函数 | ReLU |
| 损失 / 优化器 | Cross-Entropy / SGD |
| Batch size | 64 |
| Learning rate / epochs | 0.01 / 15 |
| 观察指标 | 训练损失、测试准确率、预测标签 |

### 4. PyTorch：CIFAR-10 分类

使用 `nn.Module` 与 `nn.Sequential` 表达不同网络，并通过统一训练、验证函数进行实验：

- `TwoLayerFC`：两层全连接网络；
- `ThreeLayerConvNet`：两层卷积与全连接分类器；
- `CustomCNN`：三组卷积、BatchNorm、池化，加 Dropout 与全连接层。

数据划分为 49,000 个训练样本、1,000 个验证样本，以及独立测试集。自定义 CNN 使用 Adam，训练调用配置为 9 epochs；代码按 CUDA 可用性选择 GPU 或 CPU。

### 5. DeepSeek 本地部署记录

`1.deepseek.ipynb` 介绍使用 Ollama 运行 DeepSeek R1 蒸馏模型，以及通过 Open WebUI 提供浏览器交互。该部分是部署操作说明，与前面的图像分类代码分开组织。

## Notebook 导航

| 文件 | 主题 |
| --- | --- |
| `0.install.ipynb` | Linux、Conda、Jupyter 环境准备 |
| `1.deepseek.ipynb` | Ollama / Open WebUI 部署说明 |
| `2.mindspore.ipynb` | MindSpore + MNIST + LeNet5 |
| `3.gradient.ipynb` | 符号求导、差分和链式求导 |
| `4.fc_network.ipynb` | 全连接网络、梯度检查 |
| `5.conv_network.ipynb` | 卷积、滤波、梯度检查 |
| `6.pytorch.ipynb` | PyTorch + CIFAR-10 分类 |
| `util.py` | NumPy 算子与数据工具 |
| `train_fcnet.py` / `train_convnet.py` | MNIST 训练脚本 |

根目录还保留了实验报告，记录实验过程与分析。

## 开始使用

建议使用独立 Conda 环境。以下命令准备 Notebook 和基础科学计算依赖：

```bash
git clone https://github.com/VincentLucifer/AI-Lab2-DeepLearning.git
cd AI-Lab2-DeepLearning
conda create -n ai-lab2 python=3.11 -y
conda activate ai-lab2
python -m pip install notebook ipykernel numpy matplotlib sympy six imageio pillow download
python -m ipykernel install --user --name ai-lab2
jupyter notebook
```

推荐从 `3.gradient.ipynb` 开始，再阅读 NumPy 算子和框架训练实验。

| 实验方向 | 环境与材料 |
| --- | --- |
| 梯度计算 | NumPy、SymPy |
| MindSpore 分类 | 按设备安装 MindSpore；Notebook 下载 MNIST 到 `src/dataset/MNIST_Data/` |
| PyTorch 分类 | 按设备安装 `torch` / `torchvision`，以及 `torchsummary`；使用 CIFAR-10 |
| NumPy 完整训练 | 需准备课程配套 `src` 模块，包括 MNIST 加载器和网络类，并保持相应导入路径 |
| 图像滤波 | 使用课程 `notebook-imgs` 示例图片 |
| LLM 部署 | 按 `1.deepseek.ipynb` 准备 Ollama、模型与 Open WebUI |

框架应选择与 Python、操作系统和计算设备匹配的版本。Notebook 中保留了预处理、梯度对照、训练损失与分类评估的具体代码。

## 技术关注点

这个项目将底层数学与框架实践放在同一条学习路径中：用差分检查导数，用 NumPy 观察张量和卷积，再在框架中组织数据、模型、优化和评估。重点是理解实现之间的对应关系，而不是只调用训练接口。

## 参考与致谢

项目包含课程配套教学材料与实验实现。DeepSeek 部分参考 [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)、[Ollama](https://ollama.com/) 和 [Open WebUI](https://github.com/open-webui/open-webui)。框架、模型及数据集遵循各自使用条款。

## 相关实验

[Lab 1 · Pacman Search](https://github.com/VincentLucifer/AI-Lab1-Pacman) · [Lab 3 · Reinforcement Learning](https://github.com/VincentLucifer/AI-Lab3-ReinforcementLearning)
