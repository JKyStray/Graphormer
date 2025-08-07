#!/bin/bash

# Overfitting Full Model Training Runner
# 针对toyset的过拟合训练 - 激进策略

# echo "🔥 过拟合训练模式启动"
# echo "==============================================================" 
# echo "🎯 过拟合策略："
# echo "   - 大幅增加训练轮数: 50 epochs"
# echo "   - 激进学习率: 8e-4 (比之前高5倍)"
# echo "   - 更多批次数: 3000批次/epoch"
# echo "   - 减少梯度累积: 1步 (最频繁更新)"
# echo "   - 更激进学习率调度: 余弦退火 + 重启"
# echo "   - 目标: 在toyset上实现过拟合"
# echo "   - 保存路径: checkpoint-520k-overfit.pth"
# echo ""

# 使用基础检查点直接训练
CHECKPOINT_PATH="checkpoints/checkpoint-520k.pth"

echo ""
echo "📋 训练配置:"
echo "  - 源检查点: $CHECKPOINT_PATH"
echo "  - 目标检查点: checkpoints/checkpoint-520k-large-data-v1.pth"
echo "  - 训练轮数: 50 epochs "
echo "  - 学习率: 8e-4 (激进学习率)"
echo "  - max_tokens_per_gpu: 1024 (保持内存安全)"  
echo "  - 梯度累积: 1步 (最频繁更新)"
echo "  - 每轮批次数: 5000 (大量训练数据)"
echo "  - 有效批次大小: 1 × 1 = 1"
echo "  - 总训练步数: 50 × 5000 ÷ 1 = 250000 权重更新"
echo "  - 日志频率: 每30步记录一次 (频繁监控)"
echo "  - 权重衰减: 1e-7 (极小正则化)"
echo ""

# 显示GPU信息
if command -v nvidia-smi &> /dev/null; then
    echo "💻 GPU状态:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv,noheader,nounits | head -1
    echo ""
fi

echo "🚀 开始训练..."
echo "⚠️  警告: 这是极度过拟合训练，目标是在toyset上达到极低loss"
echo "⚠️  注意: 如果出现内存不足，请降低batches_per_epoch到2000"
echo ""

# 运行训练 - 使用极激进的超参数
python train_full_model.py \
    --epochs 50 \
    --lr 8e-4 \
    --weight_decay 1e-7 \
    --max_tokens_per_gpu 1024 \
    --gradient_accumulation 1 \
    --batches_per_epoch 5000 \
    --log_freq 100 \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --save_path "checkpoints/checkpoint-520k-large-data-v1.pth"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 训练成功完成！"
    echo "📁 模型保存在: checkpoints/checkpoint-520k-large-data-v1.pth"
    echo ""
    echo "📊 训练总结:"
    echo "   ✅ 50个epoch × 5000批次 = 250000个训练批次"
    echo "   ✅ 总计250000次权重更新"
    echo "   ✅ 激进学习率: 8e-4"
    echo "   ✅ 最小梯度累积: 1步"
    echo "   ✅ 极小权重衰减: 1e-7"
    
    # 显示所有相关文件
    echo ""
    echo "📁 生成的模型文件:"
    ls -lah checkpoints/checkpoint-520k-overfit* 2>/dev/null || echo "   (没有找到相关文件)"
    
    # 检查是否有中间检查点
    echo ""
    echo "📁 中间检查点文件:"
    ls -lah checkpoints/checkpoint-520k-overfit_epoch_*.pth 2>/dev/null || echo "   (没有找到中间检查点)"
    
fi