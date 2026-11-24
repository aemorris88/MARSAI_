#!/bin/bash
#SBATCH --job-name=rescuellm_qlora
#SBATCH --account=your_group          # <-- replace with your HiPerGator group
#SBATCH --qos=your_qos                # <-- replace with your QOS (e.g. gpu, gpu-a100)
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1                 # 1x A100 80GB
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00               # 12 hours — increase if needed
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aemorris6370@eagle.fgcu.edu

# ── Environment ───────────────────────────────────────────────────
mkdir -p logs
echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# ── Paths — adjust these to your HiPerGator directories ──────────
WORK_DIR=/blue/your_group/your_username/rescuellm_finetune
DATA_DIR=$WORK_DIR/finetune_data
OUTPUT_DIR=$WORK_DIR/checkpoints
HF_CACHE=$WORK_DIR/hf_cache

mkdir -p $OUTPUT_DIR $HF_CACHE

# ── Pull and convert NeMo container to Singularity ───────────────
# Only needs to run once — comment out after first run
SIF_PATH=$WORK_DIR/nemo_nemotron3_nano.sif

if [ ! -f "$SIF_PATH" ]; then
    echo "Pulling NeMo Nemotron 3 Nano container..."
    module load singularity
    singularity pull \
        --docker-login \
        $SIF_PATH \
        docker://nvcr.io/nvidia/nemo:25.11.nemotron_3_nano
fi

# ── Download base model from HuggingFace (first run only) ────────
BASE_MODEL_DIR=$WORK_DIR/base_model

if [ ! -d "$BASE_MODEL_DIR" ]; then
    echo "Downloading Nemotron 3 Nano base model..."
    singularity exec \
        --nv \
        --bind $WORK_DIR:$WORK_DIR \
        --env HF_HOME=$HF_CACHE \
        $SIF_PATH \
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16',
    local_dir='$BASE_MODEL_DIR',
    token=None  # model is public
)
print('Download complete')
"
fi

# ── Convert HuggingFace checkpoint to Megatron format ────────────
MEGATRON_CKPT=$WORK_DIR/megatron_ckpt

if [ ! -d "$MEGATRON_CKPT" ]; then
    echo "Converting HF checkpoint to Megatron format..."
    singularity exec \
        --nv \
        --bind $WORK_DIR:$WORK_DIR \
        $SIF_PATH \
        bash -c "cd /opt/Megatron-Bridge && \
        uv run python examples/models/nemotron/nemotron_3/nano/import_nemotron_3_nano.py \
        --hf-model-path $BASE_MODEL_DIR \
        --output-path $MEGATRON_CKPT"
fi

# ── Run QLoRA Fine-Tuning ─────────────────────────────────────────
echo "Starting QLoRA fine-tuning..."

singularity exec \
    --nv \
    --bind $WORK_DIR:$WORK_DIR \
    $SIF_PATH \
    bash -c "cd /opt/Megatron-Bridge && \
    uv run python -m torch.distributed.run \
    --nproc-per-node=1 \
    examples/models/nemotron/nemotron_3/nano/finetune_nemotron_3_nano.py \
    --peft lora \
    train.global_batch_size=16 \
    train.micro_batch_size=2 \
    train.train_iters=200 \
    scheduler.lr_warmup_iters=20 \
    optim.lr=1e-4 \
    model.peft.lora_rank=64 \
    model.peft.lora_alpha=128 \
    model.peft.target_modules=[linear_qkv,linear_proj,linear_fc1,linear_fc2,in_proj,out_proj] \
    data.train_ds.file_path=$DATA_DIR/train.jsonl \
    data.validation_ds.file_path=$DATA_DIR/val.jsonl \
    checkpoint.pretrained_checkpoint=$MEGATRON_CKPT \
    checkpoint.save_dir=$OUTPUT_DIR/lora_adapter"

echo "Fine-tuning complete: $(date)"

# ── Merge LoRA adapter back into base model ───────────────────────
echo "Merging LoRA adapter into base model..."

MERGED_MEGATRON=$WORK_DIR/merged_megatron_ckpt

singularity exec \
    --nv \
    --bind $WORK_DIR:$WORK_DIR \
    $SIF_PATH \
    bash -c "cd /opt/Megatron-Bridge && \
    uv run python examples/peft/merge_lora.py \
    --hf-model-path $BASE_MODEL_DIR \
    --lora-checkpoint $OUTPUT_DIR/lora_adapter \
    --output-path $MERGED_MEGATRON"

# ── Convert merged model back to HuggingFace format ──────────────
echo "Converting merged model to HuggingFace format..."

HF_OUTPUT=$WORK_DIR/rescuellm_nano_finetuned_hf

singularity exec \
    --nv \
    --bind $WORK_DIR:$WORK_DIR \
    $SIF_PATH \
    bash -c "cd /opt/Megatron-Bridge && \
    uv run python examples/models/nemotron/nemotron_3/nano/export_nemotron_3_nano.py \
    --megatron-checkpoint $MERGED_MEGATRON \
    --output-path $HF_OUTPUT"

echo "All done. Fine-tuned model saved to: $HF_OUTPUT"
echo "Job finished: $(date)"
