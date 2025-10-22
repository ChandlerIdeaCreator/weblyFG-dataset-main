# 文件名: split_dataset.py
import os
import shutil
import random
from tqdm import tqdm

def split_image_folder(original_train_dir, new_base_dir, split_ratio=0.2):
    """
    将ImageFolder格式的数据集划分为训练集和验证集。

    Args:
        original_train_dir (str): 原始的 'train' 文件夹路径。
        new_base_dir (str): 新的数据集根目录，将在此创建 'train_split' 和 'val_split'。
        split_ratio (float): 验证集所占的比例。
    """
    
    # 定义新的训练集和验证集文件夹路径
    new_train_dir = os.path.join(new_base_dir, 'train_split')
    new_val_dir = os.path.join(new_base_dir, 'val_split')
    
    # 检查原始文件夹是否存在
    if not os.path.isdir(original_train_dir):
        print(f"错误: 原始训练文件夹不存在 -> {original_train_dir}")
        return

    # 如果目标文件夹已存在，提示用户，避免覆盖
    if os.path.exists(new_train_dir) or os.path.exists(new_val_dir):
        print("错误: 'train_split' 或 'val_split' 文件夹已存在。")
        print("请先删除它们或指定一个不同的新目录，以避免数据混淆。")
        return
        
    print(f"正在创建新的文件夹: {new_train_dir} 和 {new_val_dir}")
    os.makedirs(new_train_dir, exist_ok=True)
    os.makedirs(new_val_dir, exist_ok=True)

    # 获取所有类别文件夹
    class_dirs = [d for d in os.listdir(original_train_dir) if os.path.isdir(os.path.join(original_train_dir, d))]
    
    if not class_dirs:
        print(f"错误: 在 {original_train_dir} 中没有找到任何类别子文件夹。")
        return

    print(f"找到 {len(class_dirs)} 个类别。开始划分...")

    # 对每个类别文件夹进行操作
    for class_name in tqdm(class_dirs, desc="正在处理类别"):
        original_class_path = os.path.join(original_train_dir, class_name)
        
        # 创建新的类别子文件夹
        new_train_class_path = os.path.join(new_train_dir, class_name)
        new_val_class_path = os.path.join(new_val_dir, class_name)
        os.makedirs(new_train_class_path, exist_ok=True)
        os.makedirs(new_val_class_path, exist_ok=True)
        
        # 获取该类别下的所有图片文件
        images = [f for f in os.listdir(original_class_path) if os.path.isfile(os.path.join(original_class_path, f))]
        
        # 打乱文件列表以保证随机性
        random.shuffle(images)
        
        # 计算划分点
        split_point = int(len(images) * split_ratio)
        
        # 划分验证集和训练集
        val_images = images[:split_point]
        train_images = images[split_point:]

        # 复制文件到新目录
        for img in train_images:
            shutil.copy(os.path.join(original_class_path, img), new_train_class_path)
            
        for img in val_images:
            shutil.copy(os.path.join(original_class_path, img), new_val_class_path)
            
    print("\n数据集划分完成！")
    print(f"新的训练数据位于: {new_train_dir}")
    print(f"新的验证数据位于: {new_val_dir}")


if __name__ == '__main__':
    # --- 使用说明 ---
    # 1. 修改为你原始的、未划分的 'train' 文件夹所在的路径
    original_data_root = "clean_data_fast"
    
    # 2. 指定划分比例（例如0.2代表20%的数据作为验证集）
    validation_ratio = 0.2
    
    # 运行划分
    original_train_path = os.path.join(original_data_root, 'cleaned_all')
    split_image_folder(original_train_path, original_data_root, validation_ratio)