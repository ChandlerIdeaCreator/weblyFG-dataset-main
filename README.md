
# Peer-Learning Pro: 一个用于网络监督细粒度图像识别的优化框架

[![Conference](https://img.shields.io/badge/基于-ICCV%202021-blue)](https://arxiv.org/abs/2108.02399)
[![Framework](https://img.shields.io/badge/框架-PyTorch-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/许可证-MIT-green)](LICENSE)

本项目是ICCV 2021论文 **"Webly Supervised Fine--Grained Recognition: Benchmark Datasets and An Approach"** 中提出的 **Peer-learning (对等学习)** 算法的一个优化和现代化实现。

我们将其命名为 **Peer-learning Pro**，它专为应对现代深度学习挑战而设计，目标是在充满噪声、长尾分布的细粒度数据集上实现顶尖的性能和高效的训练。此框架集成了一系列先进技术，包括 **Swin Transformer**、**ConvNeXt** 等现代骨干网络，**Balanced Softmax (平衡Softmax)** 损失函数，以及专为高端GPU设计的全套性能优化策略。

## 主要特性与增强

本实现远超原始论文的范畴，提供了一个功能强大且高度灵活的训练框架：

🚀 **现代化的网络架构 & 非对称协同训练:**
- **即插即用的骨干网络:** 借助强大的 `timm` 库，可轻松在经典模型 (ResNet, VGG) 和现代架构 (如 **ConvNeXt**, **Swin Transformer**) 之间切换。
- **非对称对等学习:** 原生支持使用两种**不同**的模型架构进行协同训练（例如，一个ConvNeXt搭配一个Swin Transformer），这可以增强模型的多样性，显著提升互相纠错的能力。

🎯 **针对核心挑战的先进损失函数:**
- **集成Balanced Softmax:** 通过在对等学习的“共识集”上应用Balanced Softmax损失，精准地解决了**类别不均衡**问题，极大提升了在长尾数据集上的表现。
- **标签平滑 (Label Smoothing):** 在所有交叉熵损失计算中应用标签平滑，以提高模型的泛化能力和对标签噪声的鲁棒性。

⚡ **顶级的训练效率与性能优化:**
- **自动混合精度训练 (AMP):** 全面支持 `bfloat16` 和 `float16` 格式，能够在现代NVIDIA GPU（如安培/Ada架构的RTX 30/40系列）上**显著加速训练**并**降低显存占用**。
- **AdamW 优化器:** 使用AdamW以获得更好的权重衰减和正则化效果。
- **高级学习率策略:**
    - **余弦退火与预热 (Cosine Annealing with Warmup):** 现代训练流程的标配。
    - **分层学习率衰减 (LLRD):** 对Swin Transformer和ConvNeXt等架构自动启用，以实现最佳微调效果。
    - **判别式学习率:** 对其他架构，为骨干网络和分类头设置不同的学习率。
- **渐进式尺寸调整 (Progressive Resizing):** 训练前期使用较小分辨率以加速收敛，后期切换到更大分辨率进行精调，兼顾速度与精度。
- **强大的数据增强:** 集成了`TrivialAugmentWide`和`RandomErasing`等强力正则化手段，有效对抗过拟合。
- **指数移动平均 (EMA):** 维护模型权重的EMA副本，用于最终评估，通常能带来更稳定和更高的性能。

## 核心算法: Peer-learning Pro

本框架的核心思想忠于原始论文，并进行了关键增强：

1.  **并行训练:** 两个独立的网络 (`net1` 和 `net2`) 被同时训练。
2.  **动态数据划分:** 在每个批次中，样本根据两个网络的预测结果被划分为：
    *   **共识集 (Agreement Set, Gs):** 两个网络预测**相同**的样本。这些被认为是“简单”或大概率干净的样本。
    *   **分歧集 (Disagreement Set, Gd):** 两个网络预测**不同**的样本。这些被认为是“困难”或可能带噪的样本。
3.  **交叉监督与精炼:**
    *   **对于共识集:** 每个网络计算出各自的样本损失。**（增强点）** 此处我们使用 **Balanced Softmax Loss**，使得损失的计算对少数类更敏感。然后，每个网络挑选出对方认为**损失最小**（最简单）的一部分样本进行学习。这能有效防止两个模型互相强化对噪声的错误记忆。
    *   **对于分歧集:** 两个网络都学习全部分歧样本，迫使它们从困难案例中寻找正确的分类边界。

## 快速开始

### 1. 环境配置

- Python 3.8+
- PyTorch 2.0+ (**必须是支持CUDA的GPU版本**)
- NVIDIA GPU (推荐使用安培架构或更新的显卡，如RTX 30/40系列)
- `timm` 库: 如果未安装，启动脚本会自动尝试安装。

```bash
# 克隆本仓库
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 创建并激活conda虚拟环境 (推荐)
conda create -n peer_learning python=3.10
conda activate peer_learning

# 安装PyTorch (请从官网获取与您显卡驱动匹配的命令)
# 访问: https://pytorch.org/get-started/locally/
# 例如，针对CUDA 12.1:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装其他依赖
pip install timm tqdm scikit-learn
```

### 2. 数据准备

本项目假设竞赛场景仅提供训练集，您需要手动将其划分为训练/验证集。

#### 数据集下载

- **训练数据:**
  - **百度网盘:** [https://pan.baidu.com/s/1oYA7nThCwe38fS6dvMtsnA](https://pan.baidu.com/s/1oYA7nThCwe38fS6dvMtsnA) (提取码: `g8pj`)
  - **夸克网盘:** [https://pan.quark.cn/s/efa3db093bee](https://pan.quark.cn/s/efa3db093bee) (提取码: `REbs`)

- **测试数据:**
  - **百度网盘:** [https://pan.baidu.com/s/1dXMHRciDf4v-f-qzF5MF-g](https://pan.baidu.com/s/1dXMHRciDf4v-f-qzF5MF-g) (提取码: `ev3m`)

#### 数据划分步骤

1.  **组织数据:** 下载训练数据后，解压并组织成如下的ImageFolder结构，存放在一个根目录（例如`competition_data`）下：
    ```
    competition_data/
    └── train/
        ├── 000/
        │   ├── image_01.jpg
        │   └── ...
        ├── 001/
        └── ...
    ```

2.  **运行划分脚本:**
    我们使用`split_dataset_final.py`脚本来划分数据。为节省磁盘空间并兼容服务器部署，推荐使用 **'move' (移动/剪切)** 模式。
    **重要提示: 'move'模式会清空原始的`train`文件夹，请务必提前备份！**

    *   **配置脚本:** 打开`split_dataset_final.py`，在文件末尾的配置区修改`COMPETITION_DATA_ROOT`为你自己的数据根目录，并确认`MODE = 'move'`。
    *   **运行划分:**
        ```bash
        python split_dataset_final.py
        ```
        根据提示输入`yes`确认。脚本将生成`train_split_move`和`val_split_move`两个文件夹。

3.  **重命名文件夹:**
    为了与训练脚本兼容，请将生成的文件夹重命名：
    *   `train_split_move` -> `train_split`
    *   `val_split_move` -> `val_split`

### 3. 模型训练

训练过程由高度可配置的`run_training.sh`脚本控制。

1.  **配置 `run_training.sh`:**
    打开脚本，修改文件开头的**“使用配置区”**：
    *   `CUDA_VISIBLE_DEVICES`: 设置您要使用的GPU ID。
    *   `DATA_PATH`: 设置为您的数据集根目录 (例如, `"competition_data"`)。
    *   `NUM_CLASSES`: 设置您数据集的准确类别总数。
    *   `NET1_ARCH` & `NET2_ARCH`: 选择您想使用的模型架构组合。
    *   根据您的硬件情况调整`BATCH_SIZE`、`EPOCHS`和学习率`BASE_LR`。

2.  **开始训练:**
    首先给脚本执行权限，然后运行它：
    ```bash
    chmod +x run_training.sh
    ./run_training.sh
    ```

脚本将自动执行完整的两阶段训练流程，并将每个阶段的最佳模型权重保存在`model/`目录下。

## ReferenceReference




> Zeren Sun, Yazhou Yao, Xiu-Shen Wei, Yongshun Zhang, Fumin Shen, Jianxin Wu, Jian Zhang, and Heng Tao Shen. **"Webly Supervised Fine-Grained Recognition: Benchmark Datasets and An Approach"**. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021.



