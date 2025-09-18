#!/bin/bash

# --- 请根据您的实际情况修改以下变量 ---

# 1. 比赛数据集的根目录
#    这个目录下必须包含 train_split/ 和 val_split/ 两个子文件夹
DATA_PATH="competition_data" # <--- 修改这里！

# 2. 比赛数据集的总类别数 (手动指定)
#    请根据您对数据集的了解，在这里填入准确的数字。
#    例如，如果类别文件夹是从 '000' 到 '399'，那么总数就是 400。
NUM_CLASSES=400  # <--- 在这里直接告诉程序！

# 3. 您的GPU显存能支持的Batch Size
BATCH_SIZE=16  # <--- 修改这里

# 4. 选择骨干网络
NET_ARCH='bcnn'

# 5. 训练总轮数
TOTAL_EPOCHS=100

# 6. Peer-learning的超参数
DROP_RATE=0.25
T_K=10

# --- 执行训练命令 ---

echo "=============== STARTING STEP 1 ==============="
python main.py \
    --dataset "${DATA_PATH}" \
    --n_classes ${NUM_CLASSES} \
    --net1 ${NET_ARCH} \
    --net2 ${NET_ARCH} \
    --batch_size ${BATCH_SIZE} \
    --epoch ${TOTAL_EPOCHS} \
    --drop_rate ${DROP_RATE} \
    --T_k ${T_K} \
    --step 1 \
    --base_lr 0.001

echo "=============== STARTING STEP 2 ==============="
python main.py \
    --dataset "${DATA_PATH}" \
    --n_classes ${NUM_CLASSES} \
    --net1 ${NET_ARCH} \
    --net2 ${NET_ARCH} \
    --batch_size ${BATCH_SIZE} \
    --epoch ${TOTAL_EPOCHS} \
    --drop_rate ${DROP_RATE} \
    --T_k ${T_K} \
    --step 2 \
    --base_lr 0.0001 \
    --resume

echo "=============== TRAINING FINISHED ==============="

