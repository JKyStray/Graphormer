CHECKPOINT_PATH="checkpoints/checkpoint-520k.pth"
python train_full_model.py \
    --epochs 10 \
    --lr 1e-6 \
    --weight_decay 0 \
    --max_tokens_per_gpu 1024 \
    --gradient_accumulation 1 \
    --batches_per_epoch 1000 \
    --log_freq 100 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --save_path "checkpoints/checkpoint-520k-evo-test2.pth"