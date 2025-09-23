import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
import os
import argparse
import shutil
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture # 需要 scikit-learn

# ==============================================================================
# 辅助类与函数 (无需修改)
# ==============================================================================

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original = super().__getitem__(index)
        path = self.samples[index][0]
        return original[0], original[1], path

class TwoCropTransform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

class SNSCL_Loss(nn.Module):
    def __init__(self, temperature=0.1):
        super(SNSCL_Loss, self).__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
    def forward(self, features):
        N = features.shape[0] // 2
        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        self_mask = torch.eye(2 * N, dtype=torch.bool, device=features.device)
        similarity_matrix.masked_fill_(self_mask, -float('inf'))
        labels1 = torch.arange(N, 2 * N).to(features.device)
        labels2 = torch.arange(N).to(features.device)
        labels = torch.cat([labels1, labels2])
        loss = self.cross_entropy(similarity_matrix, labels)
        return loss

def save_cleaned_dataset(clean_paths, output_dir):
    if os.path.exists(output_dir):
        print(f"正在移除已存在的目录: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for path in tqdm(clean_paths, desc=f"正在保存到 {output_dir}"):
        class_name = os.path.basename(os.path.dirname(path))
        cls_dir = os.path.join(output_dir, class_name)
        os.makedirs(cls_dir, exist_ok=True)
        shutil.copy(path, os.path.join(cls_dir, os.path.basename(path)))
    print(f"清洗后的数据集已成功保存。")

# ==============================================================================
# 核心处理函数 (已更新为支持动态阈值)
# ==============================================================================
def process_dataset(input_path, output_path, num_classes, args):
    """对单个数据集执行完整的清洗流程, 支持固定或动态阈值。"""
    print("\n" + "="*60)
    print(f"开始为以下路径进行SNS-CL清洗: {input_path}")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    # --- 模型设置 ---
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 2048), nn.ReLU(), nn.Linear(2048, 128)
    )
    model = model.to(device)

    dataset_name = os.path.basename(input_path)
    checkpoint_to_load = None
    if dataset_name == 'train_split' and args.train_checkpoint_path:
        checkpoint_to_load = args.train_checkpoint_path
    elif dataset_name == 'val_split' and args.val_checkpoint_path:
        checkpoint_to_load = args.val_checkpoint_path

    # --- 步骤 1: 加载或训练模型 ---
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        print("\n--- 步骤 1: 加载预训练的编码器 ---")
        print(f"从Checkpoint加载模型: {checkpoint_to_load}")
        model.load_state_dict(torch.load(checkpoint_to_load, map_location=device))
    else:
        # (训练部分代码不变)
        print("\n--- 步骤 1: 训练SNS-CL编码器 ---")
        if checkpoint_to_load: print(f"警告: 在'{checkpoint_to_load}'未找到Checkpoint文件。将从头开始训练。")
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)), transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train_dataset = datasets.ImageFolder(root=input_path, transform=TwoCropTransform(train_transform))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        criterion = SNSCL_Loss(temperature=args.temperature).to(device)
        model.train()
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for (images, _) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
                images_cat = torch.cat(images, dim=0).to(device)
                features = model(images_cat)
                loss = criterion(features)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{args.epochs}, 平均损失: {epoch_loss / len(train_loader):.4f}")
        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'snscl_model_{dataset_name}_epoch{args.epochs}.pth')
        print(f"\n正在保存训练好的模型到: {checkpoint_path}")
        torch.save(model.state_dict(), checkpoint_path)

    # --- 步骤 2: 提取特征 (代码不变) ---
    print("\n--- 步骤 2: 使用训练好的编码器提取特征 ---")
    eval_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    full_dataset = ImageFolderWithPaths(root=input_path, transform=eval_transform)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_features, all_labels, all_paths = [], [], []
    with torch.no_grad():
        for (images, labels, paths) in tqdm(full_loader, desc="提取特征中"):
            images = images.to(device)
            features = model(images)
            all_features.append(F.normalize(features, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy()); all_paths.extend(paths)
    all_features = np.vstack(all_features); all_labels = np.array(all_labels)

    # --- 步骤 3: 筛选干净样本 (新增动态/固定选择) ---
    print("\n--- 步骤 3: 筛选干净样本 ---")
    clean_indices = []
    
    # 计算所有样本的相似度分数
    prototypes = np.array([np.mean(all_features[all_labels == i], axis=0) for i in range(num_classes)])
    all_similarities = np.array([np.dot(all_features[i], prototypes[all_labels[i]]) for i in range(len(all_labels))])

    if args.dynamic_threshold:
        print("使用 GMM 动态阈值模式...")
        for i in tqdm(range(num_classes), desc="GMM动态筛选中"):
            class_mask = (all_labels == i)
            if np.sum(class_mask) < 10: # 样本太少，GMM不稳定，回退到固定比例
                if np.sum(class_mask) > 0:
                    class_similarities = all_similarities[class_mask]
                    num_to_keep = int(np.ceil(len(class_similarities) * args.selection_percentile))
                    top_indices_within_class = np.argsort(-class_similarities)[:num_to_keep]
                    original_indices = np.where(class_mask)[0][top_indices_within_class]
                    clean_indices.extend(original_indices)
                continue

            class_similarities = all_similarities[class_mask].reshape(-1, 1)
            
            gmm = GaussianMixture(n_components=2, random_state=0, max_iter=100, reg_covar=1e-5).fit(class_similarities)
            
            # 拥有更高均值（相似度）的那个组件被认为是“干净”的
            clean_component_idx = np.argmax(gmm.means_)
            
            # 计算每个样本属于“干净”组件的概率
            probs = gmm.predict_proba(class_similarities)[:, clean_component_idx]
            
            # 根据概率阈值选择样本
            selected_mask = probs > args.gmm_prob_threshold
            original_indices = np.where(class_mask)[0][selected_mask]
            clean_indices.extend(original_indices)
    else:
        print("使用固定百分比阈值模式...")
        for i in tqdm(range(num_classes), desc="固定比例筛选中"):
            class_mask = (all_labels == i)
            if np.sum(class_mask) == 0: continue
            
            class_similarities = all_similarities[class_mask]
            num_to_keep = int(np.ceil(len(class_similarities) * args.selection_percentile))
            top_indices_within_class = np.argsort(-class_similarities)[:num_to_keep]
            original_indices = np.where(class_mask)[0][top_indices_within_class]
            clean_indices.extend(original_indices)
            
    clean_paths = [all_paths[i] for i in clean_indices]

    # (步骤 4 和 5 代码不变)
    print("\n--- 步骤 4: 保存结果 ---"); save_cleaned_dataset(clean_paths, output_path)
    print("\n--- 清洗总结 ---")
    print(f"数据源: {input_path}")
    print(f"总处理样本数: {len(all_paths)}")
    print(f"筛选出的干净样本数: {len(clean_paths)} ({len(clean_paths)/len(all_paths):.2%})")
    print(f"清洗结果已保存至: {output_path}"); print("------------------\n")

# ==============================================================================
# 主函数 (新增动态阈值相关参数)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='使用SNS-CL和Checkpoint功能进行噪声标签清洗')
    # --- 基本参数 ---
    parser.add_argument('--data_base_path', type=str, required=True, help='数据集的根目录。')
    parser.add_argument('--output_dir', type=str, default='cleaned_dataset_snscl', help='用于保存结果和模型的目录。')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小。')
    
    # --- 训练相关参数 ---
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数 (在没有提供Checkpoint时)。')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='学习率。')
    parser.add_argument('--temperature', type=float, default=0.1, help='对比损失的温度系数。')
    parser.add_argument('--train_checkpoint_path', type=str, default=None, help='(可选) 指向一个已训练好的模型文件，跳过训练。')
    parser.add_argument('--val_checkpoint_path', type=str, default=None, help='(可选) 指向一个已训练好的模型文件，用于验证集。')

    # --- 阈值相关参数 (核心) ---
    parser.add_argument('--dynamic_threshold', action='store_true', help='启用GMM动态阈值模式，而不是固定百分比。')
    parser.add_argument('--selection_percentile', type=float, default=0.7, help='[固定模式] 每个类别保留的样本比例。')
    parser.add_argument('--gmm_prob_threshold', type=float, default=0.8, help='[动态模式] 样本属于“干净”组件的最小概率。')

    args = parser.parse_args()

    # (主函数其余部分代码不变)
    train_input_path = os.path.join(args.data_base_path, 'train_split')
    val_input_path = os.path.join(args.data_base_path, 'val_split')
    train_output_path = os.path.join(args.output_dir, 'cleaned_train')
    val_output_path = os.path.join(args.output_dir, 'cleaned_val')
    if not os.path.exists(train_input_path): print(f"错误: 找不到训练集路径 {train_input_path}。"); return
    num_classes = len(os.listdir(train_input_path)); print(f"发现 {num_classes} 个类别。")

    process_dataset(train_input_path, train_output_path, num_classes, args)
    if os.path.exists(val_input_path):
        auto_checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'snscl_model_train_split_epoch{args.epochs}.pth')
        if args.val_checkpoint_path is None and os.path.exists(auto_checkpoint_path):
             print(f"提示: 自动使用刚训练好的模型处理验证集: {auto_checkpoint_path}")
             args.val_checkpoint_path = auto_checkpoint_path
        elif args.train_checkpoint_path and args.val_checkpoint_path is None:
            print(f"提示: 使用您为训练集指定的模型来处理验证集: {args.train_checkpoint_path}")
            args.val_checkpoint_path = args.train_checkpoint_path
        process_dataset(val_input_path, val_output_path, num_classes, args)
    else:
        print(f"未找到验证集路径 {val_input_path}，跳过处理。")
    print("\n所有处理已完成！")

if __name__ == "__main__":
    main()