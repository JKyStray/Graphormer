#!/bin/bash

# Adaptor-Only Training Runner
# 只训练tFold适配器层

echo "🎯 适配器训练"
echo "==============================================================" 
echo "🔥 适配器训练特点："
echo "   - 只训练tFold适配器层 (tfold_expand, tfold_expand_pair)"
echo "   - 冻结所有其他模型参数"
echo "   - 训练轮数: 5 epochs"
echo "   - 学习率: 1e-3 (适配器训练优化)"
echo "   - 梯度累积: 1步"
echo "   - 期望适配器层在基础检查点中缺失"
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

# 使用基础检查点进行适配器训练
CHECKPOINT_PATH="checkpoints/checkpoint-520k.pth"

echo ""
echo "📋 适配器训练配置:"
echo "  - 源检查点: $CHECKPOINT_PATH (适配器层预期缺失)"
echo "  - 目标检查点: checkpoints/checkpoint-520k-adaptor-test5-centered.pth"
echo "  - 训练轮数: 10 epoch"
echo "  - 学习率: 1e-5 (适配器训练优化)"
echo "  - max_tokens_per_gpu: 1024"
echo "  - 梯度累积: 1步"
echo "  - 每轮批次数: 20000"
echo "  - 有效批次大小: 1 × 1 = 1"
echo "  - 总训练步数: 10 × 20000 ÷ 1 = 200000 权重更新"
echo ""

# 显示GPU信息
if command -v nvidia-smi &> /dev/null; then
    echo "💻 GPU状态:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits | head -1
    echo ""
fi

echo "🚀 开始适配器训练..."
echo ""

# 运行适配器训练 - 只训练tFold适配器层
python train_adaptor_only.py \
    --epochs 10 \
    --lr 1e-3 \
    --weight_decay 0 \
    --max_tokens_per_gpu 1024 \
    --gradient_accumulation 1 \
    --batches_per_epoch 20000 \
    --log_freq 1000 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --save_path "checkpoints/checkpoint-520k-adaptor-test5-centered.pth"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 适配器训练成功完成！"
    echo "📁 最终模型保存在: checkpoints/checkpoint-520k-adaptor-test5-centered.pth"
    
    # 显示相关文件
    # echo ""
    # echo "📁 生成的适配器训练文件:"
    # ls -lah checkpoints/checkpoint-520k-adaptor-* 2>/dev/null || echo "   (没有找到相关文件)"
    
else
    echo ""
    echo "❌ 适配器训练失败"
    echo "💡 可能的原因:"
    echo "   1. 基础检查点文件不存在"
    echo "   2. GPU内存不足"
    echo "   3. 数据加载器问题"
    echo "   4. 适配器层定义问题"
    echo ""
fi
