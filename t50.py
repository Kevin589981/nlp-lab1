# %% [markdown]
# 
# # 处理数据
# 右侧点击Add Input，找到我们的比赛，然后添加
# 
# 从Kaggle输入的train.csv读取数据，随机划分为训练集(95%)和验证集(5%)
# 
# 保存到/kaggle/working/data/samsum目录

# %%
# !pip install rouge-score transformers accelerate

# %%
import pandas as pd
import os
import numpy as np
print("\n" + "=" * 80)
print("准备SAMSum数据集划分")
print("=" * 80)

# 读取原始CSV文件
input_csv = '/kaggle/input/nanogpt-fudannlp-cs-30040/train.csv'
print(f"\n读取数据: {input_csv}")

df = pd.read_csv(input_csv)

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[:100]

total_samples = len(df)
print(f"总样本数: {total_samples}")


# 计算划分点（5%作为验证集）
val_size = int(total_samples * 0.05)
train_size = total_samples - val_size

print(f"训练集样本数: {train_size} ({train_size/total_samples*100:.1f}%)")
print(f"验证集样本数: {val_size} ({val_size/total_samples*100:.1f}%)")

# 划分数据
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

# 创建输出目录
output_dir = '/kaggle/working/data/samsum'
os.makedirs(output_dir, exist_ok=True)

# 保存训练集
train_csv_path = os.path.join(output_dir, 'train.csv')
train_df.to_csv(train_csv_path, index=False)
print(f"\n训练集已保存: {train_csv_path}")

# 保存验证集
val_csv_path = os.path.join(output_dir, 'validation.csv')
val_df.to_csv(val_csv_path, index=False)
print(f"验证集已保存: {val_csv_path}")

print("\n数据集划分完成！")
print("=" * 80)

# %% [markdown]
# # FLAN-T5 模型微调

# %%
import os
import time
import math
import pickle
import csv
from contextlib import nullcontext
from dataclasses import dataclass
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP, DataParallel as DP
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import torch.distributed as dist

# =============================================================================
# 配置参数
# =============================================================================

@dataclass
class Config:
    """FLAN-T5微调配置"""
    
    # 数据集配置
    dataset_path: str = '/kaggle/working/data/samsum'
    dataset: str = 'samsum'
    
    # 模型配置
    model_name: str = 'google/flan-t5-base'  # 使用FLAN-T5-base模型
    max_input_length: int = 512              # 输入序列最大长度
    max_target_length: int = 128             # 目标序列最大长度
    
    # 训练配置
    batch_size: int = 2                      # 每个GPU的批次大小
    gradient_accumulation_steps: int = 16    # 梯度累积步数
    max_epochs: int = 3                      # 训练轮数
    learning_rate: float = 3e-4              # 学习率
    weight_decay: float = 0.01               # 权重衰减
    warmup_steps: int = 100                  # 学习率预热步数
    
    # 优化配置
    dtype: str = 'float16'                   # 只使用float16
    gradient_checkpointing: bool = True      # 开启梯度检查点
    
    # 评估配置
    eval_steps: int = 50                     # 每多少步评估一次
    eval_only: bool = False                  # 是否只评估
    resume: str = None                       # 恢复训练的检查点路径
    
    # I/O配置
    output_dir: str = 'flan-t5-samsum'       # 输出目录
    save_steps: int = 100                    # 保存检查点的步数间隔
    
    # 生成配置
    num_beams: int = 4                       # beam search的beam数量
    do_sample: bool = False                  # 是否使用采样
    temperature: float = 1.0                 # 采样温度
    
    # GPU配置
    local_rank: int = 0                      # 本地GPU rank
    world_size: int = 2                      # 总GPU数量

config = Config()

# =============================================================================
# 多GPU初始化
# =============================================================================

def init_distributed():
    """初始化分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.local_rank = int(os.environ["LOCAL_RANK"])
        config.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(config.local_rank)
        dist.init_process_group(backend="nccl")
        return True
    elif torch.cuda.device_count() > 1:
        # 单机多卡，使用DataParallel
        print(f"使用DataParallel，检测到 {torch.cuda.device_count()} 张GPU")
        return False
    else:
        return False

# 初始化分布式
use_ddp = init_distributed()

# =============================================================================
# 数据集类
# =============================================================================

class SummarizationDataset(Dataset):
    """T5摘要数据集"""
    
    def __init__(self, csv_path, tokenizer, max_input_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # 读取数据
        self.data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append({
                    'dialogue': row['dialogue'],
                    'summary': row['summary']
                })
        
        print(f"加载了 {len(self.data)} 条数据从 {csv_path}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # T5输入格式：summarize: {dialogue}
        input_text = f"summarize: {sample['dialogue']}"
        target_text = sample['summary']
        
        # Tokenize输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize目标
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 准备标签（用于计算loss）
        labels = target_encoding['input_ids'].clone()
        # 将padding token替换为-100，这样在计算loss时会被忽略
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].flatten(),
            'attention_mask': input_encoding['attention_mask'].flatten(),
            'labels': labels.flatten()
        }

# =============================================================================
# 数据加载器
# =============================================================================

def get_dataloader(csv_path, tokenizer, batch_size, shuffle=True):
    """创建数据加载器"""
    dataset = SummarizationDataset(
        csv_path, tokenizer, config.max_input_length, config.max_target_length
    )
    
    # 分布式采样器
    sampler = None
    if use_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=config.world_size, 
            rank=config.local_rank,
            shuffle=shuffle
        )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, dataset

# =============================================================================
# 评估函数
# =============================================================================

@torch.no_grad()
def evaluate_model(model, dataloader, tokenizer, device, max_eval_samples=None):
    """评估模型"""
    model.eval()
    
    all_predictions = []
    all_references = []
    total_loss = 0
    num_batches = 0
    
    eval_samples = 0
    max_samples = max_eval_samples or float('inf')
    
    for batch in tqdm(dataloader, desc="评估中"):
        if eval_samples >= max_samples:
            break
            
        # 移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 计算loss
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
        
        # 生成预测
        with torch.cuda.amp.autocast(dtype=torch.float16):
            predictions = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=config.max_target_length,
                num_beams=config.num_beams,
                do_sample=config.do_sample,
                temperature=config.temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 解码预测和参考
        for i in range(len(predictions)):
            # 解码预测
            pred_text = tokenizer.decode(predictions[i], skip_special_tokens=True)
            all_predictions.append(pred_text)
            
            # 解码参考（需要处理-100标记）
            ref_ids = labels[i].cpu().numpy()
            ref_ids = ref_ids[ref_ids != -100]  # 移除-100标记
            ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
            all_references.append(ref_text)
            
            eval_samples += 1
            if eval_samples >= max_samples:
                break
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    # 计算ROUGE分数
    rouge_scores = None
    try:
        rouge_scores = calculate_rouge_scores(all_references, all_predictions)
    except Exception as e:
        print(f"ROUGE计算失败: {e}")
    
    model.train()
    return avg_loss, rouge_scores, all_predictions, all_references

def calculate_rouge_scores(references, predictions):
    """计算ROUGE分数"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        
        for ref, pred in zip(references, predictions):
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    except ImportError:
        print("警告: rouge-score未安装，跳过ROUGE评估")
        return None

# =============================================================================
# 训练函数
# =============================================================================

def train_model():
    """训练主函数"""
    
    print("=" * 80)
    print("开始FLAN-T5微调训练")
    print("=" * 80)
    
    # 设置设备
    device = torch.device(f'cuda:{config.local_rank}' if torch.cuda.is_available() else 'cpu')
    
    # 加载tokenizer和模型
    print(f"加载模型: {config.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(config.model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16 if config.dtype == 'float16' else torch.float32
    )
    
    # 开启梯度检查点
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    model = model.to(device)
    
    # 多GPU设置
    if use_ddp:
        model = DDP(model, device_ids=[config.local_rank], find_unused_parameters=False)
        print(f"使用DistributedDataParallel，rank={config.local_rank}")
    elif torch.cuda.device_count() > 1:
        model = DP(model)
        print(f"使用DataParallel，GPU数量={torch.cuda.device_count()}")
    
    # 加载数据
    train_dataloader, train_dataset = get_dataloader(
        os.path.join(config.dataset_path, 'train.csv'),
        tokenizer,
        config.batch_size,
        shuffle=True
    )
    
    val_dataloader, val_dataset = get_dataloader(
        os.path.join(config.dataset_path, 'validation.csv'),
        tokenizer,
        config.batch_size,
        shuffle=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 计算总步数
    total_steps = len(train_dataloader) * config.max_epochs // config.gradient_accumulation_steps
    
    # 优化器和调度器
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 混合精度scaler
    scaler = torch.cuda.amp.GradScaler()
    
    # 恢复训练
    start_epoch = 0
    global_step = 0
    best_rouge = 0
    
    if config.resume:
        print(f"从检查点恢复: {config.resume}")
        checkpoint = torch.load(config.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
        best_rouge = checkpoint.get('best_rouge', 0)
        print(f"恢复到epoch {start_epoch}, step {global_step}")
    
    # 如果只评估
    if config.eval_only:
        print("执行纯评估模式...")
        val_loss, rouge_scores, predictions, references = evaluate_model(
            model, val_dataloader, tokenizer, device, max_eval_samples=50
        )
        print(f"验证集损失: {val_loss:.4f}")
        if rouge_scores:
            print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        # 保存评估结果
        os.makedirs(config.output_dir, exist_ok=True)
        results_path = os.path.join(config.output_dir, 'evaluation_results.csv')
        with open(results_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['reference', 'prediction'])
            for ref, pred in zip(references, predictions):
                writer.writerow([ref, pred])
        print(f"评估结果已保存至: {results_path}")
        return
    
    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"开始训练，总共 {config.max_epochs} 轮，{total_steps} 步")
    print(f"有效批次大小: {config.batch_size * config.gradient_accumulation_steps * config.world_size}")
    
    # 训练循环
    model.train()
    
    for epoch in range(start_epoch, config.max_epochs):
        if use_ddp:
            train_dataloader.sampler.set_epoch(epoch)
        
        epoch_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            train_dataloader, 
            desc=f"Epoch {epoch+1}/{config.max_epochs}",
            disable=(use_ddp and config.local_rank != 0)
        )
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # 移动数据到设备
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # 前向传播
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                # 关键修复：如果是多GPU，loss是一个向量，需要求平均值将其变为标量
                if loss.dim() > 0:
                    loss = loss.mean()

                loss = loss / config.gradient_accumulation_steps
            
            # 反向传播
            scaler.scale(loss).backward()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # 梯度累积
            if (step + 1) % config.gradient_accumulation_steps == 0:
                # 梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 优化器步进
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                
                global_step += 1
                
                # 更新进度条
                progress_bar.set_postfix({
                    'loss': f"{loss.item() * config.gradient_accumulation_steps:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
                # 评估和保存
                if global_step % config.eval_steps == 0:
                    print(f"\n在步数 {global_step} 进行评估...")
                    val_loss, rouge_scores, _, _ = evaluate_model(
                        model, val_dataloader, tokenizer, device, max_eval_samples=20
                    )
                    
                    print(f"验证集损失: {val_loss:.4f}")
                    if rouge_scores:
                        current_rouge = rouge_scores['rougeL']  # 使用ROUGE-L作为主要指标
                        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
                        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
                        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
                        
                        # 保存最佳模型
                        if current_rouge > best_rouge:
                            best_rouge = current_rouge
                            save_checkpoint(
                                model, optimizer, scheduler, tokenizer,
                                epoch, global_step, best_rouge,
                                os.path.join(config.output_dir, 'best_model')
                            )
                            print(f"保存最佳模型，ROUGE-L: {best_rouge:.4f}")
                    
                    model.train()  # 重新设置为训练模式
                
                # 定期保存检查点
                if global_step % config.save_steps == 0:
                    save_checkpoint(
                        model, optimizer, scheduler, tokenizer,
                        epoch, global_step, best_rouge,
                        os.path.join(config.output_dir, f'checkpoint-{global_step}')
                    )
                    print(f"保存检查点: checkpoint-{global_step}")
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} 平均损失: {avg_epoch_loss:.4f}")
    
    # 保存最终模型
    save_checkpoint(
        model, optimizer, scheduler, tokenizer,
        config.max_epochs, global_step, best_rouge,
        os.path.join(config.output_dir, 'final_model')
    )
    
    print("训练完成！")
    print(f"最佳ROUGE-L分数: {best_rouge:.4f}")

def save_checkpoint(model, optimizer, scheduler, tokenizer, epoch, global_step, best_rouge, save_path):
    """保存检查点"""
    os.makedirs(save_path, exist_ok=True)
    
    # 获取原始模型（处理DDP/DP包装）
    if hasattr(model, 'module'):
        model_to_save = model.module
    else:
        model_to_save = model
    
    # 保存模型和tokenizer
    model_to_save.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    # 保存训练状态
    checkpoint = {
        'epoch': epoch,
        'global_step': global_step,
        'best_rouge': best_rouge,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'config': config
    }
    
    torch.save(checkpoint, os.path.join(save_path, 'training_state.pt'))

# =============================================================================
# 推理函数
# =============================================================================

def predict_test_set():
    """对测试集进行推理"""
    print("=" * 80)
    print("开始测试集推理")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载最佳模型
    model_path = os.path.join(config.output_dir, 'best_model')
    if not os.path.exists(model_path):
        model_path = os.path.join(config.output_dir, 'final_model')
    
    print(f"从 {model_path} 加载模型")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if config.dtype == 'float16' else torch.float32
    )
    model = model.to(device)
    model.eval()
    
    # 多GPU推理
    if torch.cuda.device_count() > 1:
        model = DP(model)
        print(f"使用 {torch.cuda.device_count()} 张GPU进行推理")
    
    # 读取测试数据
    test_file = '/kaggle/input/nanogpt-fudannlp-cs-30040/test.csv'
    test_data = []
    
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_data.append({
                'id': row['id'],
                'dialogue': row['dialogue']
            })
    
    print(f"加载了 {len(test_data)} 条测试数据")
    
    # 批量推理
    results = []
    batch_size = config.batch_size * 4  # 推理时可以用更大的批次
    
    for i in tqdm(range(0, len(test_data), batch_size), desc="推理中"):
        batch_data = test_data[i:i+batch_size]
        
        # 准备输入
        input_texts = [f"summarize: {item['dialogue']}" for item in batch_data]
        
        # Tokenize
        inputs = tokenizer(
            input_texts,
            max_length=config.max_input_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # 生成摘要
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                outputs = model.generate(
                    **inputs,
                    max_length=config.max_target_length,
                    num_beams=config.num_beams,
                    do_sample=config.do_sample,
                    temperature=config.temperature,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
        
        # 解码结果
        summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # 保存结果
        for j, summary in enumerate(summaries):
            results.append({
                'id': batch_data[j]['id'],
                'summary': summary
            })
    
    # 保存提交文件
    output_file = os.path.join(config.output_dir, 'submission.csv')
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'summary'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"推理完成！结果保存至: {output_file}")
    return output_file

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("FLAN-T5 微调训练脚本")
    print(f"使用模型: {config.model_name}")
    print(f"数据类型: {config.dtype}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    
    if config.eval_only:
        print("评估模式")
    elif config.resume:
        print(f"恢复训练模式: {config.resume}")
    else:
        print("全新训练模式")
    
    # 训练或评估
    train_model()
    
    # 如果不是纯评估模式，进行测试集推理
    if not config.eval_only:
        predict_test_set()

if __name__ == "__main__":
    main()

# %%
# 直接运行训练
main()

# %%
# 如果需要评估已训练的模型，设置eval_only=True
config.eval_only = True
train_model()

# %%
# 进行测试集推理
predict_test_set()