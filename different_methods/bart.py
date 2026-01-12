# %% [markdown]
# 
# # 处理数据
# 右侧点击Add Input，找到我们的比赛，然后添加
# 
# 从Kaggle输入的train.csv读取数据，随机划分为训练集(95%)和验证集(5%)
# 
# 保存到/kaggle/working/data/samsum目录

# %%
import pandas as pd
import os

print("\n" + "=" * 80)
print("准备SAMSum数据集划分")
print("=" * 80)

# 读取原始CSV文件
input_csv = '/kaggle/input/nanogpt-fudannlp-cs-30040/train.csv'
print(f"\n读取数据: {input_csv}")

df = pd.read_csv(input_csv)

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# df = df[:100]  # 仅使用前100条数据进行快速测试

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
# # 配置和数据准备

# %%
import os
import time
import math
import csv
from contextlib import nullcontext
from dataclasses import dataclass

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

# =============================================================================
# 配置参数
# =============================================================================

class Config:
    """配置类：包含所有可调参数"""
    
    # 数据集配置
    dataset_path = '/kaggle/working/data/samsum'
    dataset = 'samsum'
    
    # 模型配置
    model_name = 'facebook/bart-base'
    
    # 训练配置
    init_from = 'pretrained'  # 'pretrained', 'resume', 'scratch'
    resume_from = None  # checkpoint路径，当init_from='resume'时使用
    
    # 批次配置
    batch_size = 16
    gradient_accumulation_steps = 8
    max_source_length = 512  # 输入对话的最大长度
    max_target_length = 128  # 目标摘要的最大长度
    
    # 训练步数
    max_iters = 500
    
    # 优化器配置
    learning_rate = 3e-5
    weight_decay = 0.01
    beta1 = 0.9
    beta2 = 0.999
    grad_clip = 1.0
    
    # 学习率调度
    decay_lr = False
    warmup_iters = 100
    lr_decay_iters = 500
    min_lr = 1e-6
    
    # I/O配置
    out_dir = 'out-bart-summarization'
    eval_interval = 10
    log_interval = 5
    eval_iters = 40
    eval_only = False
    always_save_checkpoint = False
    
    # ROUGE评估配置
    eval_rouge_during_training = True
    rouge_eval_samples = 5
    
    # wandb日志
    wandb_log = False
    wandb_project = 'bart-summarization'
    wandb_run_name = 'bart-base'
    
    # 系统配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile = False
    
    # 生成配置
    num_test_samples = 10
    num_beams = 4
    temperature = 1.0
    top_k = 50
    top_p = 0.95

config = Config()

# =============================================================================
# 数据集类
# =============================================================================

class SummarizationDataset(Dataset):
    """BART摘要任务的Dataset类"""
    
    def __init__(self, csv_path, tokenizer, max_source_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # 读取CSV文件
        self.dialogues = []
        self.summaries = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.dialogues.append(row['dialogue'])
                self.summaries.append(row['summary'])
        
        print(f"  加载了 {len(self.dialogues)} 个样本")
    
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]
        
        # Tokenize输入（对话）
        source = self.tokenizer(
            dialogue,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize目标（摘要）
        target = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        
        # 将padding token设置为-100，这样在计算loss时会被忽略
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_ids,
            'attention_mask': source_mask,
            'labels': labels
        }

# 全局变量：缓存DataLoader
_dataloaders = {'train': None, 'val': None}
_data_iters = {'train': None, 'val': None}

def get_batch(split, tokenizer):
    """获取一个训练批次"""
    global _dataloaders, _data_iters
    
    # 首次调用：创建DataLoader
    if _dataloaders[split] is None:
        csv_file = 'train.csv' if split == 'train' else 'validation.csv'
        csv_path = os.path.join(config.dataset_path, csv_file)
        
        dataset = SummarizationDataset(
            csv_path,
            tokenizer,
            config.max_source_length,
            config.max_target_length
        )
        
        shuffle = (split == 'train')
        
        _dataloaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if 'cuda' in config.device else False,
        )
        _data_iters[split] = iter(_dataloaders[split])
    
    # 获取下一个batch
    try:
        batch = next(_data_iters[split])
    except StopIteration:
        _data_iters[split] = iter(_dataloaders[split])
        batch = next(_data_iters[split])
    
    # 移动到设备
    if 'cuda' in config.device:
        batch = {k: v.to(config.device, non_blocking=True) for k, v in batch.items()}
    else:
        batch = {k: v.to(config.device) for k, v in batch.items()}
    
    return batch

@torch.no_grad()
def estimate_loss(model, ctx, tokenizer):
    """估计训练集和验证集上的损失"""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            batch = get_batch(split, tokenizer)
            with ctx:
                outputs = model(**batch)
                loss = outputs.loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out

def get_lr(iter_num):
    """学习率调度：带预热的余弦衰减"""
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num + 1) / (config.warmup_iters + 1)
    
    if iter_num > config.lr_decay_iters:
        return config.min_lr
    
    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# =============================================================================
# ROUGE评估
# =============================================================================

def calculate_rouge(reference_summary, generated_summary):
    """计算ROUGE分数"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_summary, generated_summary)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    except ImportError:
        return None

@torch.no_grad()
def evaluate_rouge_during_training(model, tokenizer, ctx, num_samples=5):
    """在训练过程中评估ROUGE分数"""
    model.eval()
    
    # 读取验证集
    val_csv = os.path.join(config.dataset_path, 'validation.csv')
    if not os.path.exists(val_csv):
        model.train()
        return None
    
    dialogues = []
    summaries = []
    with open(val_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialogues.append(row['dialogue'])
            summaries.append(row['summary'])
    
    # 随机选择样本
    import random
    indices = random.sample(range(len(dialogues)), min(num_samples, len(dialogues)))
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for idx in indices:
        dialogue = dialogues[idx]
        reference_summary = summaries[idx]
        
        # Tokenize输入
        inputs = tokenizer(
            dialogue,
            max_length=config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(config.device)
        
        # 生成摘要
        with ctx:
            outputs = model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=config.max_target_length,
                num_beams=config.num_beams,
                early_stopping=True
            )
        
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 计算ROUGE
        rouge_scores = calculate_rouge(reference_summary, generated_summary)
        if rouge_scores:
            rouge1_scores.append(rouge_scores['rouge1'])
            rouge2_scores.append(rouge_scores['rouge2'])
            rougeL_scores.append(rouge_scores['rougeL'])
    
    model.train()
    
    if len(rouge1_scores) > 0:
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    return None

# =============================================================================
# 训练函数
# =============================================================================

def train():
    """训练主函数"""
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 创建输出目录
    os.makedirs(config.out_dir, exist_ok=True)
    
    # 设置设备和精度
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype
    )
    
    # 初始化tokenizer
    print(f"\n加载tokenizer: {config.model_name}")
    tokenizer = BartTokenizer.from_pretrained(config.model_name)
    
    # 初始化模型
    print(f"\n模型初始化方式: {config.init_from}")
    
    if config.init_from == 'scratch':
        print("从头开始训练新模型")
        model_config = BartConfig.from_pretrained(config.model_name)
        model = BartForConditionalGeneration(model_config)
        iter_num = 0
        best_val_loss = 1e9
        
    elif config.init_from == 'resume':
        print(f"从checkpoint恢复训练")
        if config.resume_from is None:
            ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
        else:
            ckpt_path = config.resume_from
        
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        model = BartForConditionalGeneration.from_pretrained(config.model_name)
        model.load_state_dict(checkpoint['model'])
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        
    else:  # pretrained
        print(f"从预训练模型加载: {config.model_name}")
        model = BartForConditionalGeneration.from_pretrained(config.model_name)
        iter_num = 0
        best_val_loss = 1e9
    
    model.to(config.device)
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    if config.init_from == 'resume' and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 初始化GradScaler
    scaler = torch.amp.GradScaler('cuda',enabled=(config.dtype == 'float16'))
    
    # 编译模型（可选）
    if config.compile:
        print("编译模型...")
        model = torch.compile(model)
    
    # 训练循环
    print("\n开始训练循环...")
    print(f"总迭代次数: {config.max_iters}")
    print(f"批次大小: {config.batch_size}")
    print(f"梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"有效批次大小: {config.batch_size * config.gradient_accumulation_steps}")
    print("-" * 80)
    
    t0 = time.time()
    local_iter_num = 0
    
    while True:
        # 设置学习率
        lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 评估和保存checkpoint
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, ctx, tokenizer)
            print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # 计算ROUGE分数
            if iter_num > 0 and config.eval_rouge_during_training:
                print("  评估ROUGE分数...")
                rouge_scores = evaluate_rouge_during_training(
                    model, tokenizer, ctx, num_samples=config.rouge_eval_samples
                )
                if rouge_scores:
                    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}, "
                          f"ROUGE-2: {rouge_scores['rouge2']:.4f}, "
                          f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
            
            # 保存checkpoint
            if losses['val'] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(config),
                    }
                    print(f"  保存checkpoint到 {config.out_dir}")
                    torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
        
        if iter_num == 0 and config.eval_only:
            break
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        for micro_step in range(config.gradient_accumulation_steps):
            batch = get_batch('train', tokenizer)
            
            with ctx:
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
        
        # 梯度裁剪
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        
        # 记录日志
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % config.log_interval == 0:
            lossf = loss.item() * config.gradient_accumulation_steps
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
        
        iter_num += 1
        local_iter_num += 1
        
        if iter_num > config.max_iters:
            break
    
    print("\n训练完成！")
    print("=" * 80)

# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("BART 摘要微调".center(80))
    print("=" * 80)
    print("\n当前配置:")
    print(f"  数据集: {config.dataset_path}")
    print(f"  模型: {config.model_name}")
    print(f"  初始化方式: {config.init_from}")
    print(f"  设备: {config.device}")
    print(f"  精度: {config.dtype}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  最大迭代次数: {config.max_iters}")
    print(f"  学习率: {config.learning_rate}")
    
    if not config.eval_only:
        train()
    else:
        print("\neval_only=True，跳过训练")

main()

# %% [markdown]
# # 评估

# %%
!pip install rouge-score

# %%
def load_model():
    """加载训练好的模型"""
    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype
    )
    
    # 加载tokenizer
    tokenizer = BartTokenizer.from_pretrained(config.model_name)
    
    # 加载模型
    print(f"\n从 {config.out_dir} 加载模型...")
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到checkpoint文件 {ckpt_path}")
        print("请先运行训练！")
        return None, None, None
    
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    model = BartForConditionalGeneration.from_pretrained(config.model_name)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(config.device)
    
    if config.compile:
        print("编译模型...")
        model = torch.compile(model)
    
    return model, tokenizer, ctx

def evaluate():
    """评估模型性能"""
    print("\n" + "=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    # 加载模型
    model, tokenizer, ctx = load_model()
    if model is None:
        return
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_file = os.path.join(config.dataset_path, 'validation.csv')
    
    dialogues = []
    summaries = []
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= config.num_test_samples:
                break
            dialogues.append(row['dialogue'])
            summaries.append(row['summary'])
    
    print(f"加载了 {len(dialogues)} 条测试样本")
    
    # 检查ROUGE
    rouge_result = calculate_rouge("test", "test")
    use_rouge = rouge_result is not None
    if not use_rouge:
        print("\n警告: 未安装rouge_score库，将跳过ROUGE评分")
    
    # 评估每个样本
    print("\n" + "-" * 80)
    print("开始生成和评估...")
    print("-" * 80)
    
    all_rouge1_f = []
    all_rouge2_f = []
    all_rougeL_f = []
    
    for idx, (dialogue, reference_summary) in enumerate(zip(dialogues, summaries)):
        print(f"\n[样本 {idx+1}/{len(dialogues)}]")
        print(f"对话: {dialogue[:100]}..." if len(dialogue) > 100 else f"对话: {dialogue}")
        
        # Tokenize输入
        inputs = tokenizer(
            dialogue,
            max_length=config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(config.device)
        
        # 生成摘要
        with torch.no_grad():
            with ctx:
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=config.max_target_length,
                    num_beams=config.num_beams,
                    early_stopping=True
                )
        
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"真实摘要: {reference_summary}")
        print(f"生成摘要: {generated_summary}")
        
        # 计算ROUGE分数
        if use_rouge:
            rouge_scores = calculate_rouge(reference_summary, generated_summary)
            if rouge_scores:
                print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
                print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
                print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
                
                all_rouge1_f.append(rouge_scores['rouge1'])
                all_rouge2_f.append(rouge_scores['rouge2'])
                all_rougeL_f.append(rouge_scores['rougeL'])
    
    # 打印平均分数
    if use_rouge and len(all_rouge1_f) > 0:
        print("\n" + "=" * 80)
        print("平均ROUGE分数:")
        print(f"  ROUGE-1: {np.mean(all_rouge1_f):.4f}")
        print(f"  ROUGE-2: {np.mean(all_rouge2_f):.4f}")
        print(f"  ROUGE-L: {np.mean(all_rougeL_f):.4f}")
        print("=" * 80)

evaluate()

# %%
def predict_test_set():
    """对测试集进行推理"""
    print("\n" + "=" * 80)
    print("开始对测试集进行推理...")
    print("=" * 80)
    
    # 加载模型
    model, tokenizer, ctx = load_model()
    if model is None:
        return
    
    # 读取测试数据
    print("\n加载测试数据...")
    test_file = '/kaggle/input/nanogpt-fudannlp-cs-30040/test.csv'
    
    if not os.path.exists(test_file):
        print(f"错误: 找不到测试文件 {test_file}")
        return
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_data.append({
                'id': row['id'],
                'dialogue': row['dialogue']
            })
    
    print(f"加载了 {len(test_data)} 条测试样本")
    
    # 准备保存结果
    results = []
    
    print("\n" + "-" * 80)
    print("开始生成摘要...")
    print("-" * 80)
    
    start_time = time.time()
    
    for idx, sample in enumerate(tqdm(test_data)):
        sample_id = sample['id']
        dialogue = sample['dialogue']
        
        # Tokenize输入
        inputs = tokenizer(
            dialogue,
            max_length=config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(config.device)
        
        # 生成摘要
        with torch.no_grad():
            with ctx:
                outputs = model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=config.max_target_length,
                    num_beams=config.num_beams,
                    early_stopping=True
                )
        
        generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            'id': sample_id,
            'summary': generated_summary
        })
    
    total_time = time.time() - start_time
    print(f"\n生成完成！总耗时: {total_time/60:.1f}分钟 | 平均: {total_time/len(test_data):.2f}秒/样本")
    
    # 保存结果
    output_file = 'submission.csv'
    output_path = os.path.join(config.out_dir, output_file)
    
    print(f"\n保存结果到 {output_path}")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'summary'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"完成！生成了 {len(results)} 条摘要")
    print("=" * 80)
    
    return output_path

predict_test_set()