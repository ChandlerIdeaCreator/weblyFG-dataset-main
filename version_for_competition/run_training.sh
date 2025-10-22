# =============== 模型选择 ===============
NET1_ARCH="${NET1_ARCH:-convnext_base}"               # 保持不变
NET2_ARCH="${NET2_ARCH:-swin_base_patch4_window7_224}" # <-- 唯一的修改！```

---

### 为您准备好的、可以直接运行的完整脚本

下面是修改后的完整脚本，您可以直接复制使用。它现在会启动一个`convnext_base`和一个`swin_base_patch4_window7_224`进行非对称协同训练。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

#############################################
# AIC Peer-Learning Training Launcher (ConvNext + Swin-T 组合)
#############################################

# =============== 基础环境 ===============
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STAMP="$(date +%Y%m%d_%H%M%S)"

# =============== 数据与任务 ===============
DATA_PATH="${DATA_PATH:-competition_data}"
NUM_CLASSES="${NUM_CLASSES:-400}"

if [[ ! -d "${DATA_PATH}/train_split" || ! -d "${DATA_PATH}/val_split" ]]; then
  echo "[ERROR] ${DATA_PATH}/train_split 或 ${DATA_PATH}/val_split 不存在" >&2
  exit 1
fi

# =============== 模型选择 (核心修改点) ===============
NET1_ARCH="${NET1_ARCH:-convnext_base}"
NET2_ARCH="${NET2_ARCH:-swin_base_patch4_window7_224}" # <-- 将bcnn替换为Swin Transformer

# =============== 训练超参 ===============
BATCH_SIZE="${BATCH_SIZE:-16}"
STAGE1_EPOCHS="${STAGE1_EPOCHS:-20}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-88}"
BASE_LR_STAGE1="${BASE_LR_STAGE1:-0.001}"
BASE_LR_STAGE2="${BASE_LR_STAGE2:-5e-5}" # Swin-T可能需要更小的LR，例如5e-5，您可以后续调整

# =============== 算法 & 功能开关 ===============
USE_BALANCED_SOFTMAX="${USE_BALANCED_SOFTMAX:-true}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
USE_LLRD="${USE_LLRD:-auto}"
BB_LR_MULT="${BB_LR_MULT:-0.1}"

USE_TTA="${USE_TTA:-0}"
TTA_SCALES="${TTA_SCALES:-1.0,1.15}"
TTA_HFLIP="${TTA_HFLIP:-1}"

# =============== 性能参数 ===============
NPROC=$(command -v nproc >/dev/null && nproc || echo 8)
DATALOADER_WORKERS="${DATALOADER_WORKERS:-$(( NPROC ))}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
USE_COMPILE="${USE_COMPILE:-1}"

# =============== 依赖检查 ===============
python - <<'PY'
try:
    import timm  # noqa
except Exception:
    import sys
    sys.stderr.write("WARNING: timm 未安装，将自动 pip install -U timm\n")
    sys.exit(1)
PY
if [[ $? -ne 0 ]]; then
  pip install -U timm
fi

# ... (脚本的其余部分完全保持不变) ...

# =============== 环境回显 ===============
echo "==== CONFIG @ ${STAMP} ===="
echo "DATA_PATH=${DATA_PATH}"
echo "NET1_ARCH=${NET1_ARCH}  NET2_ARCH=${NET2_ARCH}"
echo "BATCH_SIZE=${BATCH_SIZE}  EPOCHS: S1=${STAGE1_EPOCHS}, S2=${STAGE2_EPOCHS}"
echo "LR: S1=${BASE_LR_STAGE1}, S2=${BASE_LR_STAGE2}"
echo "BS=${USE_BALANCED_SOFTMAX}  LS=${LABEL_SMOOTHING}"
echo "LLRD=${USE_LLRD}  BB_LR_MULT=${BB_LR_MULT}"
echo "TTA=${USE_TTA} scales=${TTA_SCALES} hflip=${TTA_HFLIP}"
echo "WORKERS=${DATALOADER_WORKERS}  PREFETCH=${PREFETCH_FACTOR}  ACCUM=${GRAD_ACCUM}"
echo "COMPILE=${USE_COMPILE}"
echo "==========================="

# =============== 参数拼装（通用） ===============
EXTRA_ARGS=""
[[ "${USE_BALANCED_SOFTMAX}" == "true" ]] && EXTRA_ARGS+=" --use_balanced_softmax"
EXTRA_ARGS+=" --label_smoothing ${LABEL_SMOOTHING}"
EXTRA_ARGS+=" --workers ${DATALOADER_WORKERS} --prefetch_factor ${PREFETCH_FACTOR} --grad_accum_steps ${GRAD_ACCUM}"

if [[ "${USE_COMPILE}" == "1" ]]; then
  EXTRA_ARGS+=" --compile --compile_mode max-autotune"
else
  EXTRA_ARGS+=" --no_compile"
fi

# =============== Step 1：只训分类头 ===============
echo "=============== STARTING STEP 1: Training Classifier Head ==============="
STEP1_LOG="step1_${NET1_ARCH}_${NET2_ARCH}_${STAMP}.log"
python main.py \
  --dataset "${DATA_PATH}" \
  --n_classes ${NUM_CLASSES} \
  --net1 "${NET1_ARCH}" \
  --net2 "${NET2_ARCH}" \
  --batch_size ${BATCH_SIZE} \
  --epoch ${STAGE1_EPOCHS} \
  --step 1 \
  --base_lr ${BASE_LR_STAGE1} \
  ${EXTRA_ARGS} | tee "${STEP1_LOG}"

MODEL_DIR="model"
MODEL1_PATH="${MODEL_DIR}/net1_step1_${NET1_ARCH}_$( [ ${NUM_CLASSES} -gt 0 ] && echo ${NUM_CLASSES} || echo auto )cls_best_1.pth"
MODEL2_PATH="${MODEL_DIR}/net2_step1_${NET2_ARCH}_$( [ ${NUM_CLASSES} -gt 0 ] && echo ${NUM_CLASSES} || echo auto )cls_best_1.pth"

if [[ ! -f "${MODEL1_PATH}" || ! -f "${MODEL2_PATH}" ]]; then
  echo "[WARN] Step 1 最优权重未找到："
  [[ ! -f "${MODEL1_PATH}" ]] && echo "  - 缺少 ${MODEL1_PATH}"
  [[ ! -f "${MODEL2_PATH}" ]] && echo "  - 缺少 ${MODEL2_PATH}"
  echo "      可能文件名里的类数与 --n_classes 不一致；Step 2 会由 main.py 继续尝试 --resume（含 *_autocls_* 兜底）。"
fi

echo "=============== WAITING 5 SECONDS BEFORE STEP 2 ==============="
sleep 5

# =============== Step 2：全量微调 ===============
echo "=============== STARTING STEP 2: Fine-tuning Whole Network ==============="

_autollrd=0
if [[ "${NET1_ARCH}" == *convnext* || "${NET2_ARCH}" == *swin* || "${NET1_ARCH}" == *swin* || "${NET2_ARCH}" == *convnext* ]]; then
  _autollrd=1
fi

_step2_llrd_flag=""
case "${USE_LLRD}" in
  1|true|TRUE|on|ON)     _step2_llrd_flag="--use_llrd" ;;
  0|false|FALSE|off|OFF) _step2_llrd_flag="--bb_lr_mult ${BB_LR_MULT}" ;;
  auto|AUTO)             [[ ${_autollrd} -eq 1 ]] && _step2_llrd_flag="--use_llrd" || _step2_llrd_flag="--bb_lr_mult ${BB_LR_MULT}" ;;
  *)                     [[ ${_autollrd} -eq 1 ]] && _step2_llrd_flag="--use_llrd" || _step2_llrd_flag="--bb_lr_mult ${BB_LR_MULT}" ;;
esac

STEP2_ARGS="${EXTRA_ARGS} ${_step2_llrd_flag}"

if [[ "${USE_TTA}" == "1" ]]; then
  STEP2_ARGS+=" --use_tta --tta_scales ${TTA_SCALES}"
  [[ "${TTA_HFLIP}" == "1" ]] && STEP2_ARGS+=" --tta_hflip"
fi

STEP2_LOG="step2_${NET1_ARCH}_${NET2_ARCH}_${STAMP}.log"
python main.py \
  --dataset "${DATA_PATH}" \
  --n_classes ${NUM_CLASSES} \
  --net1 "${NET1_ARCH}" \
  --net2 "${NET2_ARCH}" \
  --batch_size ${BATCH_SIZE} \
  --epoch ${STAGE2_EPOCHS} \
  --step 2 \
  --base_lr ${BASE_LR_STAGE2} \
  --resume \
  ${STEP2_ARGS} | tee "${STEP2_LOG}"

echo "=============== TRAINING FINISHED (${STAMP}) ==============="
echo "Logs:"
echo "  - ${STEP1_LOG}"
echo "  - ${STEP2_LOG}"```

### 总结

*   **代码不用动：** 您的所有Python文件（`main.py`, `loss_plm.py`等）都**不需要**任何修改。
*   **脚本只改一行：** 您只需要将`.sh`脚本中`NET2_ARCH`的默认值改为您想要的Swin Transformer模型名称即可。

这样就完全满足了您的要求。这个脚本现在会启动一个**ConvNeXt**和一个**Swin Transformer**作为两个“相互学习”的伙伴，这是一种非常强大且先进的非对称协同训练组合。