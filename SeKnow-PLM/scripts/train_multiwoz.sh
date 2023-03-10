# data augmentation via back-translations and cleaning of training data are not used
# --backtranslations /home/gaosilin/augpt/data/multiwoz-2.1.yaml
# --clean_samples
# --response-loss [ce,unlikelihood]

export WANDB_MODE="dryrun"
num_gpus=1

jbsub -q x86_24h -cores 1x4+1 -mem 64G -require a100 -err "err.log" -out "out.log" python -m torch.distributed.launch --nproc_per_node ${num_gpus} train_multiwoz.py \
        --train-dataset multiwoz-2.1-train \
        --dev-dataset multiwoz-2.1-val \
        --model jkulhanek/augpt-bigdata \
        --response-loss ce \
        --epochs 8 \
        --batch-size 8 \
        --device-shift 0 \
        --fp16 \
        --clean-samples
