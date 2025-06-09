import os
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from common import config as cfg
from model.main_model import MainModel
from data_provider.train_loader import TrainLoader

def setup_logging(args):
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'train.log')),
            logging.StreamHandler()
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description='Train Protein Structure Diffusion Model')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                      help='Directory to save checkpoints and logs')
    parser.add_argument('--d_model', type=int, default=768,
                      help='Dimension of the model')
    parser.add_argument('--d_pair', type=int, default=256,
                      help='Dimension of pair features')
    parser.add_argument('--n_layer', type=int, default=12,
                      help='Number of transformer layers')
    parser.add_argument('--n_heads', type=int, default=32,
                      help='Number of attention heads')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=10000,
                      help='Number of warmup steps')
    parser.add_argument('--max_steps', type=int, default=1000000,
                      help='Maximum number of training steps')
    parser.add_argument('--save_steps', type=int, default=10000,
                      help='Save checkpoint every X steps')
    parser.add_argument('--eval_steps', type=int, default=1000,
                      help='Evaluate every X steps')
    parser.add_argument('--resume_from', type=str, default=None,
                      help='Resume training from checkpoint')
    return parser.parse_args()

def get_lr_scheduler(optimizer, warmup_steps, max_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return max(0.0, float(max_steps - step) / float(max(1, max_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def save_checkpoint(model, optimizer, scheduler, step, args):
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{step}.pt')
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, checkpoint_path)
    logging.info(f'Saved checkpoint to {checkpoint_path}')

def load_checkpoint(model, optimizer, scheduler, args):
    checkpoint = torch.load(args.resume_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_step = checkpoint['step']
    logging.info(f'Resumed from checkpoint {args.resume_from} at step {start_step}')
    return start_step

def train(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    setup_logging(args)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'tensorboard'))

    # Initialize model, optimizer, and data loader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MainModel(
        d_model=args.d_model,
        d_pair=args.d_pair,
        n_layer=args.n_layer,
        n_heads=args.n_heads
    ).to(device)
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, args.max_steps)
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume_from is not None:
        start_step = load_checkpoint(model, optimizer, scheduler, args)

    # Initialize data loader
    train_loader = TrainLoader()
    
    # Training loop
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(train_loader, start=start_step):
        if step >= args.max_steps:
            break
            
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        optimizer.zero_grad()
        output = model(batch)
        loss = output['loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Logging
        running_loss += loss.item()
        if step > 0 and step % args.eval_steps == 0:
            avg_loss = running_loss / args.eval_steps
            lr = scheduler.get_last_lr()[0]
            logging.info(f'Step {step}: loss = {avg_loss:.4f}, lr = {lr:.2e}')
            writer.add_scalar('Loss/train', avg_loss, step)
            writer.add_scalar('Learning_rate', lr, step)
            running_loss = 0.0
            
        # Save checkpoint
        if step > 0 and step % args.save_steps == 0:
            save_checkpoint(model, optimizer, scheduler, step, args)
    
    # Save final checkpoint
    save_checkpoint(model, optimizer, scheduler, args.max_steps, args)
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    train(args) 