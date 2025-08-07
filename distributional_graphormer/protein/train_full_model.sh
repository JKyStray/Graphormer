#!/bin/bash

# Optimized Full Model Training Runner
# 基于训练结果优化的全模型训练

echo "🚀 优化版全模型训练"
echo "==============================================================" 
echo "🔥 基于训练结果的优化："
echo "   - 增加训练轮数: 10 epochs"
echo "   - 提高学习率: 1.5e-4 "
echo "   - 增加批次数: 1000批次/epoch"
echo "   - 梯度累积: 4步"
echo "   - 改进学习率调度: 余弦退火"
echo "   - 使用基础检查点: checkpoints/checkpoint-520k.pth"
echo ""

# 检查适配器训练后的检查点
# ADAPTER_CHECKPOINT="checkpoints/checkpoint-520k-adapter.pth"
# BASE_CHECKPOINT="checkpoints/checkpoint-520k.pth"
# PREVIOUS_FULL="checkpoints/checkpoint-520k-full-trained.pth"

# if [ -f "$PREVIOUS_FULL" ]; then
#     CHECKPOINT_PATH="$PREVIOUS_FULL"
#     echo "🔄 找到之前的全模型训练结果: $PREVIOUS_FULL"
#     echo "   将基于之前的训练继续优化"
# elif [ -f "$ADAPTER_CHECKPOINT" ]; then
#     CHECKPOINT_PATH="$ADAPTER_CHECKPOINT"
#     echo "✅ 找到适配器训练后的检查点: $ADAPTER_CHECKPOINT"
#     echo "   将基于适配器训练结果进行全模型训练"
# elif [ -f "$BASE_CHECKPOINT" ]; then
#     CHECKPOINT_PATH="$BASE_CHECKPOINT"
#     echo "⚠️ 使用基础检查点: $BASE_CHECKPOINT"
# else
#     echo "❌ 没有找到可用的检查点文件"
#     exit 1
# fi

# 使用基础检查点直接训练
CHECKPOINT_PATH="checkpoints/checkpoint-520k.pth"

echo ""
echo "📋 优化训练配置:"
echo "  - 源检查点: $CHECKPOINT_PATH"
echo "  - 目标检查点: checkpoints/checkpoint-520k-test5.pth"
echo "  - 训练轮数: 5 epoch"
echo "  - 学习率: 1e-4"
echo "  - max_tokens_per_gpu: 1024"
echo "  - 梯度累积: 1步"
echo "  - 每轮批次数: 10000"
echo "  - 有效批次大小: 1 × 1 = 1"
echo "  - 总训练步数: 5 × 10000 ÷ 1 = 50000 权重更新"
echo ""

# 显示GPU信息
if command -v nvidia-smi &> /dev/null; then
    echo "💻 GPU状态:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits | head -1
    echo ""
fi

echo "🚀 开始优化训练..."
echo ""

# 运行优化训练 - 使用正确的训练脚本
python train_full_model.py \
    --epochs 5 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --max_tokens_per_gpu 1024 \
    --gradient_accumulation 1 \
    --batches_per_epoch 10000 \
    --log_freq 1000 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --save_path "checkpoints/checkpoint-520k-test5.pth"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 优化训练成功完成！"
    echo "📁 最终模型保存在: checkpoints/checkpoint-520k-test5.pth"
    
    # 显示所有相关文件
    # echo ""
    # echo "📁 生成的模型文件:"
    # ls -lah checkpoints/checkpoint-520k-full-* 2>/dev/null || echo "   (没有找到相关文件)"
    
    # 比较文件大小
    # if [ -f "checkpoints/checkpoint-520k-full-trained.pth" ] && [ -f "checkpoints/checkpoint-520k-full-optimized.pth" ]; then
    #    echo ""
    #     echo "📊 文件大小对比:"
    #     echo "   原始训练: $(du -h checkpoints/checkpoint-520k-full-trained.pth | cut -f1)"
    #     echo "   优化训练: $(du -h checkpoints/checkpoint-520k-full-optimized.pth | cut -f1)"
    # fi
    
else
    echo ""
    echo "❌ 优化训练失败"
    echo "💡 可能的原因:"
    echo "   1. 学习率过高导致不稳定"
    echo "   2. 批次数过多导致过拟合"
    echo "   3. GPU内存不足"
    echo ""
fi
