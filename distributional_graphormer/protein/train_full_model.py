#!/usr/bin/env python3
"""
Full Model Training Script
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import time
import argparse
from datetime import datetime
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

# NEW: Import the logger
from common.logger import Logger

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Full Model Training')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (reduced for full model)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (adjusted from overly conservative 5e-6)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--log_freq', type=int, default=10,
                       help='Log every N batches')
    parser.add_argument('--batches_per_epoch', type=int, default=500,
                       help='Number of batches per epoch (to prevent infinite training)')
    
    # 内存优化参数
    parser.add_argument('--max_tokens_per_gpu', type=int, default=1024,
                       help='Ultra-safe memory setting')
    parser.add_argument('--gradient_accumulation', type=int, default=4,
                       help='Gradient accumulation steps (reduced from 8 for better convergence)')
    
    # 路径参数
    parser.add_argument('--checkpoint_path', type=str, 
                       default='checkpoints/checkpoint-520k.pth',
                       help='Path to previous checkpoint')
    parser.add_argument('--save_path', type=str,
                       default='checkpoints/checkpoint-520k-default-trained.pth',
                       help='Path to save full-trained checkpoint')
    
    # NEW: Add log_path argument
    parser.add_argument('--log_path', type=str,
                       default='logs/full_model_training.log',
                       help='Path to save log file')
    
    # NEW: Add console log flag
    parser.add_argument('--console-log', action='store_true', default=True,
                       help='Enable console logging')
    parser.add_argument('--no-console-log', action='store_false', dest='console_log',
                       help='Disable console logging')
    
    return parser.parse_args()

def plot_training_loss(loss_history, batch_history, save_path):
    """
    Create and save a plot showing training loss vs batch number
    
    Args:
        loss_history: List of individual batch loss values
        batch_history: List of corresponding global batch numbers
        save_path: Path to save the plot
    """
    try:
        plt.figure(figsize=(15, 10))
        
        # Create the main loss plot
        plt.subplot(2, 1, 1)
        
        # Plot individual batch losses with transparency
        plt.plot(batch_history, loss_history, 'b-', linewidth=0.5, alpha=0.3, label='Individual Batch Loss')
        
        # Add rolling averages with different window sizes
        if len(loss_history) > 20:
            # Short-term rolling average (last 20 batches)
            window_short = min(20, len(loss_history) // 5)
            rolling_short = []
            for i in range(len(loss_history)):
                start_idx = max(0, i - window_short + 1)
                rolling_short.append(np.mean(loss_history[start_idx:i+1]))
            plt.plot(batch_history, rolling_short, 'r-', linewidth=2, label=f'Rolling Avg ({window_short} batches)')
            
        if len(loss_history) > 50:
            # Long-term rolling average (last 50 batches)
            window_long = min(50, len(loss_history) // 3)
            rolling_long = []
            for i in range(len(loss_history)):
                start_idx = max(0, i - window_long + 1)
                rolling_long.append(np.mean(loss_history[start_idx:i+1]))
            plt.plot(batch_history, rolling_long, 'g-', linewidth=2, label=f'Rolling Avg ({window_long} batches)')
        
        plt.xlabel('Global Batch Number')
        plt.ylabel('Loss')
        plt.title('Training Loss vs Global Batch Number (Cross-Epoch Rolling Averages)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Create a subplot showing recent loss (last 20% of training)
        plt.subplot(2, 1, 2)
        recent_start = max(0, len(loss_history) - len(loss_history) // 5)
        recent_loss = loss_history[recent_start:]
        recent_batches = batch_history[recent_start:]
        
        plt.plot(recent_batches, recent_loss, 'b-', linewidth=0.8, alpha=0.6, label='Recent Batch Loss')
        
        # Add short rolling average for recent data
        if len(recent_loss) > 10:
            window_recent = min(10, len(recent_loss) // 2)
            rolling_recent = []
            for i in range(len(recent_loss)):
                start_idx = max(0, i - window_recent + 1)
                rolling_recent.append(np.mean(recent_loss[start_idx:i+1]))
            plt.plot(recent_batches, rolling_recent, 'r-', linewidth=2, label=f'Rolling Avg ({window_recent} batches)')
            
        plt.xlabel('Global Batch Number')
        plt.ylabel('Loss')
        plt.title('Recent Training Loss (Last 20% of Training)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = str(save_path).replace('.pth', '_loss_plot.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        return None

def save_loss_data(loss_history, batch_history, save_path):
    """
    Save raw loss data to a file for further analysis
    
    Args:
        loss_history: List of individual batch loss values
        batch_history: List of corresponding global batch numbers
        save_path: Path to save the data file
    """
    try:
        import json
        
        data = {
            'batch_numbers': batch_history,
            'loss_values': loss_history,
            'metadata': {
                'total_batches': len(batch_history),
                'min_loss': min(loss_history) if loss_history else None,
                'max_loss': max(loss_history) if loss_history else None,
                'final_loss': loss_history[-1] if loss_history else None,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        data_path = str(save_path).replace('.pth', '_loss_data.json')
        with open(data_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data_path
        
    except Exception as e:
        print(f"Error saving loss data: {e}")
        return None

def setup_memory_config(max_tokens_per_gpu, logger):
    """设置内存配置"""
    from common import config as cfg
    
    original_max_tokens = cfg.max_tokens_per_gpu
    
    # 设置内存配置  
    cfg.max_tokens_per_gpu = max_tokens_per_gpu
    cfg.max_tokens_per_sample = 256
    
    effective_batch_size = cfg.max_tokens_per_gpu // cfg.max_tokens_per_sample
    
    logger.info(f"🔧 内存配置:")
    logger.info(f"   max_tokens_per_gpu: {cfg.max_tokens_per_gpu} (原始: {original_max_tokens})")
    logger.info(f"   max_tokens_per_sample: {cfg.max_tokens_per_sample}")
    logger.info(f"   有效批次大小: {effective_batch_size}")
    
    return effective_batch_size

def setup_full_model_training(model, logger):
    """设置全模型训练 - 不冻结任何层"""
    trainable_params = 0
    
    # 确保所有参数都可训练
    for name, param in model.named_parameters():
        param.requires_grad = True
        trainable_params += param.numel()
        
        # 只显示主要层的信息
        # if any(keyword in name for keyword in ['tfold_expand', 'attention', 'norm', 'linear']):
        #     logger.info(f"✅ Trainable: {name[:50]}... - {param.shape}")
    
    logger.info(f"\n📊 全模型训练参数统计:")
    logger.info(f"   可训练参数: {trainable_params:,}")
    logger.info(f"   训练比例: 100% (全模型训练)")
    
    return trainable_params

def main():
    """主训练函数"""
    args = parse_args()
    
    # NEW: Setup logger
    log_file_path = Path(args.log_path)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger = Logger(log_file_path, console=args.console_log)

    logger.info("🚀 全模型训练")
    logger.info("=" * 80)
    logger.info("🔥 特点:")
    logger.info("   - 训练所有模型参数 (不冻结任何层)")
    logger.info(f"   - 内存设置: {args.max_tokens_per_gpu} tokens")
    logger.info(f"   - 学习率: {args.lr}")
    logger.info(f"   - 梯度累积: {args.gradient_accumulation}步 (平衡内存与收敛)")
    logger.info("   - 有限批次训练: 防止无限训练")
    logger.info(f"   - 训练轮数: {args.epochs} epochs")
    logger.info("")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"设备: {device}")
    
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU内存: {total_memory:.1f} GB")
        torch.cuda.empty_cache()
        logger.info(f"初始GPU使用: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    logger.info("")
    
    try:
        # 1. 设置内存配置
        logger.info("🔧 配置内存设置...")
        effective_batch_size = setup_memory_config(args.max_tokens_per_gpu, logger)
        
        # 2. 加载数据 - 使用FiniteTrainLoader避免无限训练
        logger.info("\n📦 加载数据...")
        from data_provider.train_loader import FiniteTrainLoader
        
        train_loader = FiniteTrainLoader(batches_per_epoch=args.batches_per_epoch)
        logger.info(f"✅ 有限数据加载器创建成功")
        logger.info(f"   批次大小: {effective_batch_size}")
        logger.info(f"   每轮批次数: {args.batches_per_epoch}")
        logger.info(f"   等效批次大小 (累积后): {effective_batch_size * args.gradient_accumulation}")
        
        # 3. 创建模型
        logger.info("\n🏗️ 创建模型...")
        from model.main_model import MainModel
        
        model = MainModel(d_model=768, d_pair=256, n_layer=12, n_heads=32)
        
        # 4. 加载适配器训练后的权重
        logger.info(f"\n📂 加载适配器训练后的检查点...")
        if not Path(args.checkpoint_path).exists():
            logger.info(f"⚠️ 适配器检查点不存在: {args.checkpoint_path}")
            logger.info("使用基础检查点: checkpoints/checkpoint-520k.pth")
            args.checkpoint_path = "checkpoints/checkpoint-520k.pth"
        
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # NEW: Fix for DataParallel prefix issue
        # If keys start with 'module.', it's from a DataParallel-saved model
        if list(state_dict.keys())[0].startswith('module.'):
            logger.info("🔧 Detected 'module.' prefix in checkpoint keys. Removing it.")
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
            state_dict = new_state_dict

        # Load state dict with strict=False to allow for adapter layers
        logger.info("🔧 Loading checkpoint with strict=False to accommodate adapter layers.")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        logger.info(f"  ✅ Checkpoint loaded.")
        if missing_keys:
            logger.info("     - Missing keys (expected for new layers like tfold_adapter):")
            # Print up to 5 missing keys for brevity
            for key in missing_keys[:5]:
                logger.info(f"       - {key}")
            if len(missing_keys) > 5:
                logger.info("       - ... and others")
        if unexpected_keys:
            logger.info("     - Unexpected keys (in checkpoint but not model):")
            for key in unexpected_keys:
                logger.info(f"       - {key}")

        model = model.to(device)
        if torch.cuda.is_available():
            logger.info(f"模型加载后GPU内存: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # 5. 设置全模型训练
        trainable_params = setup_full_model_training(model, logger)
        
        # 6. 优化器 - 为全模型训练优化
        trainable_param_list = [p for p in model.parameters() if p.requires_grad]
        
        # 使用合理的学习率
        optimizer = optim.Adam(
            trainable_param_list,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=1e-8
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=1e-6  # 调整最小学习率
        )
        
        # 7. 训练循环
        logger.info(f"\n🎯 开始全模型训练...")
        logger.info(f"配置: lr={args.lr}, 梯度累积={args.gradient_accumulation}步")
        logger.info(f"每轮批次数: {args.batches_per_epoch}")
        logger.info(f"等效批次大小: {effective_batch_size * args.gradient_accumulation}")
        logger.info("=" * 60)
        
        model.train()
        
        # NEW: Initialize loss tracking
        loss_history = []
        batch_history = []
        global_batch_count = 0  # Track global batch number across all epochs
        
        for epoch in range(args.epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            batch_count = 0
            accumulated_loss = 0.0
            
            logger.info(f"\n📊 Epoch {epoch + 1}/{args.epochs}")
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f"当前学习率: {current_lr:.6e}")
            
            # 使用FiniteTrainLoader，会自动在args.batches_per_epoch后停止
            for batch_idx, batch in enumerate(train_loader):
                try:
                    # 梯度累积循环
                    if batch_count % args.gradient_accumulation == 0:
                        optimizer.zero_grad()
                    
                    # 移到设备
                    for key in batch:
                        if isinstance(batch[key], torch.Tensor):
                            batch[key] = batch[key].to(device, non_blocking=True)
                    
                    # 前向传播
                    output = model(batch, compute_loss=True)
                    loss = output['loss'] / args.gradient_accumulation
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        logger.info(f"⚠️ 跳过无效loss: {loss.item()}")
                        continue
                    
                    # 反向传播
                    loss.backward()
                    accumulated_loss += loss.item()
                    
                    # 每accumulation步更新一次权重
                    if (batch_count + 1) % args.gradient_accumulation == 0:
                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(trainable_param_list, max_norm=1.0)
                        optimizer.step()
                        
                        epoch_loss += accumulated_loss
                        update_count = (batch_count + 1) // args.gradient_accumulation
                        
                        # NEW: Record individual batch loss for plotting (every gradient update)
                        loss_history.append(accumulated_loss)
                        batch_history.append(global_batch_count)
                        global_batch_count += 1
                        
                        # 日志
                        if update_count % args.log_freq == 0:
                            avg_loss = epoch_loss / update_count
                            gpu_mem = torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0
                            
                            log_msg = (f"  Update {update_count:3d} | "
                                       f"Batch Loss: {accumulated_loss:.6f} | "
                                       f"Epoch Avg: {avg_loss:.6f} | "
                                       f"LR: {current_lr:.6e} | "
                                       f"GPU: {gpu_mem:.1f}GB")
                            logger.info(log_msg)
                        
                        accumulated_loss = 0.0
                    
                    batch_count += 1
                    
                    # 更频繁的缓存清理
                    # if batch_count % 20 == 0:
                    #     torch.cuda.empty_cache()
                
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.info(f"⚠️ GPU内存不足，清理缓存...")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                except Exception as e:
                    logger.info(f"❌ 批次失败: {e}")
                    torch.cuda.empty_cache()
                    continue
            
            # Epoch统计
            epoch_time = time.time() - epoch_start_time
            update_count = batch_count // args.gradient_accumulation
            avg_loss = epoch_loss / max(update_count, 1)
            
            logger.info(f"\n📈 Epoch {epoch + 1} 完成:")
            logger.info(f"   平均损失: {avg_loss:.6f}")
            logger.info(f"   处理批次: {batch_count}")
            logger.info(f"   权重更新: {update_count}")
            logger.info(f"   用时: {epoch_time:.1f}s")
            
            # 更新学习率
            scheduler.step()
            
            # 保存中间检查点
            if epoch < args.epochs - 1:
                intermediate_path = args.save_path.replace('.pth', f'_epoch_{epoch+1}.pth')
                checkpoint = {
                    'state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler_state_dict': scheduler.state_dict(),
                    # 'epoch': epoch + 1,
                    # 'loss': avg_loss,
                    # 'metadata': {
                    #     'training_type': 'full_model',
                    #     'timestamp': datetime.now().isoformat(),
                    #     'batches_per_epoch': args.batches_per_epoch,
                    #     'lr': args.lr
                    # }
                }
                torch.save(checkpoint, intermediate_path)
                logger.info(f"💾 中间检查点保存: {intermediate_path}")
            
            # 清理
            # torch.cuda.empty_cache()
        
        # 保存最终模型
        Path(args.save_path).parent.mkdir(exist_ok=True)
        final_checkpoint = {
            'state_dict': model.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'scheduler_state_dict': scheduler.state_dict(),
            # 'epoch': args.epochs,
            # 'loss': avg_loss,
            # 'metadata': {
            #     'training_type': 'full_model_final',
            #     'timestamp': datetime.now().isoformat(),
            #     'total_trainable_params': trainable_params,
            #     'batches_per_epoch': args.batches_per_epoch,
            #     'lr': args.lr,
            #     'note': 'Full model trained with memory settings and finite batches'
            # }
        }
        torch.save(final_checkpoint, args.save_path)
        
        # NEW: Generate and save loss plot
        if loss_history:
            logger.info(f"\n📊 生成损失图...")
            plot_path = plot_training_loss(loss_history, batch_history, args.save_path)
            if plot_path:
                logger.info(f"💾 损失图保存到: {plot_path}")
            else:
                logger.info("⚠️ 损失图生成失败")
        
        # NEW: Save raw loss data
        if loss_history:
            logger.info(f"\n📊 保存原始损失数据...")
            data_path = save_loss_data(loss_history, batch_history, args.save_path)
            if data_path:
                logger.info(f"💾 原始损失数据保存到: {data_path}")
            else:
                logger.info("⚠️ 原始损失数据保存失败")
        
        logger.info(f"\n🎉 全模型训练完成！")
        logger.info(f"📁 最终模型保存到: {args.save_path}")
        logger.info(f"🔢 总训练参数: {trainable_params:,}")
        logger.info(f"📊 训练统计: {args.epochs} epochs × {args.batches_per_epoch} batches")
        
        # NEW: Add loss progression statistics
        if loss_history:
            initial_loss = loss_history[0]
            final_loss = loss_history[-1]
            min_loss = min(loss_history)
            max_loss = max(loss_history)
            loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
            
            logger.info(f"\n📈 损失统计:")
            logger.info(f"   初始损失: {initial_loss:.6f}")
            logger.info(f"   最终损失: {final_loss:.6f}")
            logger.info(f"   最低损失: {min_loss:.6f}")
            logger.info(f"   最高损失: {max_loss:.6f}")
            logger.info(f"   损失降低: {loss_reduction:.2f}%")
            logger.info(f"   记录批次: {len(loss_history)}")
        
        return True
        
    except Exception as e:
        logger.info(f"全模型训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # NEW: Flush the logger to ensure all messages are written
        logger.flush()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    success = main()
    exit(0 if success else 1) 