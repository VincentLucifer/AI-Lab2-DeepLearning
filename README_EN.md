# Deep Learning Lab · From Gradients to Image Classification

[简体中文](README.md) | [English](README_EN.md)

Experiments centred on **how gradients are computed, how networks learn, and how frameworks organise a complete training workflow**. NumPy exposes the underlying operators; MindSpore and PyTorch provide image-classification experiments. A separate notebook documents local deployment of distilled DeepSeek models.

**Stack:** Python · NumPy · SymPy · MindSpore · PyTorch · Jupyter

## Core topics

### 1. Gradients and backpropagation

- Symbolic differentiation with SymPy.
- Forward and central differences compared with analytic derivatives.
- Forward and reverse chain-rule calculations through example computation graphs.
- Numerical gradient comparisons to inspect parameter gradients from backpropagation.

### 2. Neural-network foundations in NumPy

`util.py` provides Sigmoid, stable Softmax, cross-entropy, SGD, numerical gradients and relative-error utilities. Its convolution section includes:

- Naive convolution and its backward pass;
- `im2col` / `col2im` transformations;
- Matrix-multiplication-based convolution;
- Experiments with stride, padding and output dimensions.

The notebooks also explore grayscale conversion, edge filtering and training workflows for fully connected and convolutional networks.

### 3. MindSpore: MNIST digit recognition

A workflow covering dataset download, normalisation, batching, training, testing and prediction visualisation.

| Setting | Details |
| --- | --- |
| Model | LeNet5: two convolutional layers, average pooling, three fully connected layers |
| Activation | ReLU |
| Loss / optimiser | Cross-Entropy / SGD |
| Batch size | 64 |
| Learning rate / epochs | 0.01 / 15 |
| Observations | Training loss, test accuracy and predicted labels |

### 4. PyTorch: CIFAR-10 classification

Networks are expressed through `nn.Module` and `nn.Sequential` and explored using shared training and validation functions:

- `TwoLayerFC`: a two-layer fully connected network;
- `ThreeLayerConvNet`: two convolutional layers and a fully connected classifier;
- `CustomCNN`: three convolution/BatchNorm/pooling blocks, followed by Dropout and fully connected layers.

The split uses 49,000 training samples, 1,000 validation samples and a separate test set. The custom CNN uses Adam with a 9-epoch training call. Device selection uses CUDA when available and otherwise CPU.

### 5. Local DeepSeek deployment notes

`1.deepseek.ipynb` describes running distilled DeepSeek R1 models with Ollama and adding browser interaction through Open WebUI. This is deployment guidance, organised separately from the image-classification implementations.

## Notebook guide

| File | Topic |
| --- | --- |
| `0.install.ipynb` | Linux, Conda and Jupyter setup |
| `1.deepseek.ipynb` | Ollama / Open WebUI deployment guidance |
| `2.mindspore.ipynb` | MindSpore + MNIST + LeNet5 |
| `3.gradient.ipynb` | Symbolic derivatives, finite differences and the chain rule |
| `4.fc_network.ipynb` | Fully connected networks and gradient checks |
| `5.conv_network.ipynb` | Convolution, filtering and gradient checks |
| `6.pytorch.ipynb` | PyTorch + CIFAR-10 classification |
| `util.py` | NumPy operators and data utilities |
| `train_fcnet.py` / `train_convnet.py` | MNIST training scripts |

Experiment reports at the repository root document the process and analysis.

## Getting started

Use a dedicated Conda environment. These commands prepare Jupyter and the foundational scientific-computing packages:

```bash
git clone https://github.com/VincentLucifer/AI-Lab2-DeepLearning.git
cd AI-Lab2-DeepLearning
conda create -n ai-lab2 python=3.11 -y
conda activate ai-lab2
python -m pip install notebook ipykernel numpy matplotlib sympy six imageio pillow download
python -m ipykernel install --user --name ai-lab2
jupyter notebook
```

Start with `3.gradient.ipynb`, then explore NumPy operators and framework-based training.

| Experiment | Environment and materials |
| --- | --- |
| Differentiation | NumPy and SymPy |
| MindSpore classification | Device-compatible MindSpore; notebook downloads MNIST into `src/dataset/MNIST_Data/` |
| PyTorch classification | Device-compatible `torch` / `torchvision` and `torchsummary`; CIFAR-10 |
| Full NumPy training | Course companion `src` modules, including the MNIST loader and network classes, arranged to match imports |
| Image filtering | Course example images under `notebook-imgs` |
| LLM deployment | Ollama, model downloads and Open WebUI as described in `1.deepseek.ipynb` |

Choose framework releases matching Python, the operating system and compute device. The notebooks contain the preprocessing, gradient comparisons, loss tracking and classification evaluation code.

## Technical focus

The project connects mathematical foundations to framework practice: finite differences inspect derivatives, NumPy exposes tensor and convolution operations, and framework experiments organise data, models, optimisation and evaluation. The emphasis is on understanding how these layers correspond.

## References and acknowledgements

The repository includes course companion materials and experiment implementations. The DeepSeek section references [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1), [Ollama](https://ollama.com/) and [Open WebUI](https://github.com/open-webui/open-webui). Frameworks, models and datasets retain their respective terms.

## Related labs

[Lab 1 · Pacman Search](https://github.com/VincentLucifer/AI-Lab1-Pacman) · [Lab 3 · Reinforcement Learning](https://github.com/VincentLucifer/AI-Lab3-ReinforcementLearning)
