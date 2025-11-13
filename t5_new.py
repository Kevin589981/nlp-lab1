# %% [markdown]
# # 处理数据
# 右侧点击Add Input，找到我们的比赛，然后添加
# 从Kaggle输入的train.csv读取数据，随机划分为训练集(95%)和验证集(5%)
# 保存到/kaggle/working/data/samsum目录

# %%
!pip install rouge-score transformers accelerate sentencepiece -q

# %%
import pandas as pd
import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import time
import pickle
from rouge_score import rouge_scorer
import csv

print("\n" + "=" * 80)
print("准备SAMSum数据集划分")
print("=" * 80)

# 读取原始CSV文件
input_csv = '/kaggle/input/nanogpt-fudannlp-cs-30040/train.csv'
print(f"\n读取数据: {input_csv}")
df = pd.read_csv(input_csv)

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[:200]  # 测试200条数据
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
# # 配置类

# %%
class Config:
    """
    配置类：包含所有可调参数
    """
    # 数据集配置
    dataset_path = '/kaggle/working/data/samsum'
    dataset = 'samsum'
    
    # T5特定配置
    model_name = 'google/flan-t5-base'  # 使用flan-t5-base
    max_input_length = 512   # 输入最大长度
    max_target_length = 128  # 目标最大长度
    
    # 训练配置
    init_from = 'scratch'     # 'scratch' 或 'resume'
    batch_size = 2            # 每个GPU的批次大小
    gradient_accumulation_steps = 4  # 梯度累积步数
    max_iters = 50            # 总训练迭代次数
    
    # 优化器配置（修改：降低学习率）
    learning_rate = 5e-5      # 降低学习率
    weight_decay = 0.01       # 减小权重衰减
    grad_clip = 0.5           # 更严格的梯度裁剪
    warmup_steps = 50         # 减少预热步数
    
    # I/O配置
    out_dir = 'out-t5-summarization'  # checkpoint保存目录
    eval_interval = 10        # 每多少步评估一次
    log_interval = 5          # 每多少步打印日志
    eval_iters = 10           # 评估时的迭代次数
    eval_only = False         # 是否只评估不训练
    always_save_checkpoint = True  # 是否每次评估都保存checkpoint
    resume = False            # 是否从checkpoint恢复训练
    
    # ROUGE评估配置
    eval_rouge_during_training = True
    rouge_eval_samples = 5    # 训练时ROUGE评估的样本数
    
    # 系统配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = 'float16'         # 只使用float16
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    # 测试/生成配置
    num_test_samples = 10
    temperature = 0.8
    top_k = 50
    top_p = 0.9
    num_beams = 2             # beam search大小

config = Config()

# %% [markdown]
# # 数据集类

# %%
class SummarizationDataset(Dataset):
    """
    T5摘要任务的Dataset类
    """
    def __init__(self, csv_path, tokenizer, max_input_length=512, max_target_length=128):
        self.data = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # T5的任务前缀
        self.task_prefix = "summarize: "
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        dialogue = row['dialogue']
        summary = row['summary'] if 'summary' in row else ""
        
        # 添加任务前缀
        input_text = self.task_prefix + dialogue
        
        # Tokenize输入
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize目标（如果存在）
        if summary:
            target_encoding = self.tokenizer(
                summary,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            labels = target_encoding.input_ids.squeeze()
            # 将padding token设为-100（用于忽略loss）
            labels[labels == self.tokenizer.pad_token_id] = -100
        else:
            labels = torch.tensor([-100] * self.max_target_length)
        
        return {
            'input_ids': input_encoding.input_ids.squeeze(),
            'attention_mask': input_encoding.attention_mask.squeeze(),
            'labels': labels
        }

# %% [markdown]
# # 训练和评估函数

# %%
def get_dataloader(split, tokenizer):
    """获取DataLoader"""
    csv_path = os.path.join(config.dataset_path, f'{split}.csv')
    dataset = SummarizationDataset(
        csv_path, 
        tokenizer, 
        config.max_input_length, 
        config.max_target_length
    )
    
    shuffle = (split == 'train')
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader

@torch.no_grad()
def estimate_loss(model, tokenizer, device):
    """估计训练集和验证集上的损失"""
    out = {}
    model.eval()
    
    # 判断是否使用了DataParallel
    is_parallel = isinstance(model, nn.DataParallel)
    
    for split in ['train', 'validation']:
        dataloader = get_dataloader(split, tokenizer)
        losses = []
        
        for i, batch in enumerate(dataloader):
            if i >= config.eval_iters:
                break
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 修复autocast语法
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
            # 如果是DataParallel，loss可能是多个GPU的结果，需要取平均
            if is_parallel and loss.dim() > 0:
                loss = loss.mean()
            
            # 检查NaN
            if not torch.isnan(loss) and not torch.isinf(loss):
                losses.append(loss.item())
        
        out[split] = np.mean(losses) if losses else float('inf')
    
    model.train()
    return out


@torch.no_grad()
def evaluate_rouge_during_training(model, tokenizer, device, num_samples=3):
    """训练过程中评估ROUGE分数"""
    model.eval()
    
    # 读取验证集
    val_csv = os.path.join(config.dataset_path, 'validation.csv')
    val_df = pd.read_csv(val_csv)
    
    # 随机选择样本
    indices = np.random.choice(len(val_df), min(num_samples, len(val_df)), replace=False)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    for idx in indices:
        row = val_df.iloc[idx]
        dialogue = row['dialogue']
        reference_summary = row['summary']
        
        # 构建输入
        input_text = "summarize: " + dialogue
        input_ids = tokenizer(
            input_text,
            max_length=config.max_input_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(device)
        
        try:
            # 生成摘要（使用更保守的参数）
            generated_ids = model.generate(
                input_ids,
                max_length=config.max_target_length,
                num_beams=2,
                temperature=1.0,
                do_sample=False,  # 关闭采样，使用贪婪解码
                early_stopping=True
            )
            
            generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # 计算ROUGE分数
            scores = scorer.score(reference_summary, generated_summary)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
        except Exception as e:
            print(f"  生成时出错: {e}")
            continue
    
    model.train()
    
    if rouge1_scores:
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    else:
        return None

# %% [markdown]
# # 训练函数

# %%
def train():
    """训练主函数"""
    print("\n" + "=" * 80)
    print("开始训练 Flan-T5 摘要模型...")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(1337)
    np.random.seed(1337)
    
    # 创建输出目录
    os.makedirs(config.out_dir, exist_ok=True)
    
    # 初始化tokenizer和模型
    print(f"\n加载模型: {config.model_name}")
    tokenizer = T5Tokenizer.from_pretrained(config.model_name)
    
    # 初始化或恢复模型
    checkpoint = None
    if config.resume and not config.eval_only:
        checkpoint_path = os.path.join(config.out_dir, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"从checkpoint恢复: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model = T5ForConditionalGeneration.from_pretrained(config.model_name)
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
            start_iter = checkpoint['iter_num']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        else:
            print("未找到checkpoint，从头开始训练")
            model = T5ForConditionalGeneration.from_pretrained(config.model_name)
            start_iter = 0
            best_val_loss = float('inf')
    else:
        model = T5ForConditionalGeneration.from_pretrained(config.model_name)
        start_iter = 0
        best_val_loss = float('inf')
    
    # 设置设备
    if config.n_gpu > 1:
        print(f"使用 {config.n_gpu} 个GPU进行训练")
        device = torch.device("cuda:0")
        model = model.to(device)
        model = nn.DataParallel(model)
    else:
        device = torch.device(config.device)
        model = model.to(device)
    
    # 打印模型参数数量
    model_for_params = model.module if isinstance(model, nn.DataParallel) else model
    total_params = sum(p.numel() for p in model_for_params.parameters())
    trainable_params = sum(p.numel() for p in model_for_params.parameters() if p.requires_grad)
    print(f"总参数数: {total_params/1e6:.2f}M")
    print(f"可训练参数数: {trainable_params/1e6:.2f}M")
    
    # 如果只是评估，直接返回
    if config.eval_only:
        print("\neval_only=True，只进行评估")
        losses = estimate_loss(model, tokenizer, device)
        print(f"Validation loss: {losses['validation']:.4f}")
        return model, tokenizer
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # 学习率调度器
    train_dataloader = get_dataloader('train', tokenizer)
    total_steps = config.max_iters
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # 混合精度训练
    scaler = GradScaler()
    
    # 如果恢复训练，加载优化器和调度器状态
    if config.resume and checkpoint and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    # 训练循环
    print("\n开始训练循环...")
    print(f"总迭代次数: {config.max_iters}")
    print(f"批次大小: {config.batch_size}")
    print(f"梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"有效批次大小: {config.batch_size * config.gradient_accumulation_steps * config.n_gpu}")
    print("-" * 80)
    
    model.train()
    train_iter = iter(train_dataloader)
    iter_num = start_iter
    accumulated_loss = 0.0
    skip_update = False
    
    while iter_num < config.max_iters:
        t0 = time.time()
        
        # 梯度累积
        for micro_step in range(config.gradient_accumulation_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                batch = next(train_iter)
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 修复autocast语法
            if torch.cuda.is_available():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                    
                    if isinstance(model, nn.DataParallel):
                        if loss.dim() > 0:
                            loss = loss.mean()
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"  警告: 检测到NaN/Inf loss，跳过此批次")
                        skip_update = True
                        break
                        
                    loss = loss / config.gradient_accumulation_steps
            else:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / config.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
        
        # 如果检测到NaN，跳过这次更新
        if skip_update:
            optimizer.zero_grad()
            skip_update = False
            accumulated_loss = 0.0
            continue
        
        # 梯度裁剪
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.module.parameters() if isinstance(model, nn.DataParallel) else model.parameters(),
            config.grad_clip
        )
        
        # 如果梯度有NaN，跳过更新
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"  警告: 检测到NaN/Inf梯度，跳过更新")
            optimizer.zero_grad()
            accumulated_loss = 0.0
            continue
        
        # 优化器步骤
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        # 日志记录
        if iter_num % config.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            lossf = accumulated_loss * config.gradient_accumulation_steps
            if not np.isnan(lossf):
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {scheduler.get_last_lr()[0]:.2e}")
            else:
                print(f"iter {iter_num}: loss NaN detected, time {dt*1000:.2f}ms")
        
        # 评估
        if iter_num % config.eval_interval == 0 and iter_num > 0:
            losses = estimate_loss(model, tokenizer, device)
            if not np.isnan(losses['train']) and not np.isnan(losses['validation']):
                print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['validation']:.4f}")
                
                # ROUGE评估
                if config.eval_rouge_during_training:
                    print("  评估ROUGE分数...")
                    eval_model = model.module if isinstance(model, nn.DataParallel) else model
                    rouge_scores = evaluate_rouge_during_training(
                        eval_model, tokenizer, device, 
                        num_samples=config.rouge_eval_samples
                    )
                    if rouge_scores:
                        print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}, "
                              f"ROUGE-2: {rouge_scores['rouge2']:.4f}, "
                              f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
                
                # 保存checkpoint
                if losses['validation'] < best_val_loss or config.always_save_checkpoint:
                    best_val_loss = losses['validation']
                    
                    model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                    
                    checkpoint = {
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(config)
                    }
                    
                    checkpoint_path = os.path.join(config.out_dir, 'checkpoint.pt')
                    torch.save(checkpoint, checkpoint_path)
                    print(f"  保存checkpoint到 {checkpoint_path}")
                    
                    model_to_save.save_pretrained(os.path.join(config.out_dir, 'model'))
                    tokenizer.save_pretrained(os.path.join(config.out_dir, 'model'))
            else:
                print(f"\nStep {iter_num}: 检测到NaN loss，跳过评估")
        
        accumulated_loss = 0.0
        iter_num += 1
    
    print("\n训练完成！")
    print("=" * 80)
    
    return model, tokenizer

# %% [markdown]
# # 评估函数

# %%
def evaluate():
    """评估模型性能"""
    print("\n" + "=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    # 加载模型
    model_path = os.path.join(config.out_dir, 'model')
    if os.path.exists(model_path):
        print(f"从 {model_path} 加载模型")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    else:
        checkpoint_path = os.path.join(config.out_dir, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"从checkpoint加载模型: {checkpoint_path}")
            tokenizer = T5Tokenizer.from_pretrained(config.model_name)
            model = T5ForConditionalGeneration.from_pretrained(config.model_name)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # 处理可能的DataParallel wrapper
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            print("错误: 未找到训练好的模型")
            return
    
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()
    
    # 加载测试数据
    val_csv = os.path.join(config.dataset_path, 'validation.csv')
    val_df = pd.read_csv(val_csv)
    
    # 限制测试样本数量
    test_samples = min(config.num_test_samples, len(val_df))
    val_df = val_df.head(test_samples)
    
    print(f"评估 {test_samples} 个样本")
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_rouge1 = []
    all_rouge2 = []
    all_rougeL = []
    
    for idx, row in tqdm(val_df.iterrows(), total=len(val_df), desc="生成摘要"):
        dialogue = row['dialogue']
        reference_summary = row['summary']
        
        # 构建输入
        input_text = "summarize: " + dialogue
        input_ids = tokenizer(
            input_text,
            max_length=config.max_input_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(device)
        
        # 生成摘要
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                generated_ids = model.generate(
                    input_ids,
                    max_length=config.max_target_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                    do_sample=True,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    early_stopping=True
                )
        
        generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        # 计算ROUGE分数
        scores = scorer.score(reference_summary, generated_summary)
        all_rouge1.append(scores['rouge1'].fmeasure)
        all_rouge2.append(scores['rouge2'].fmeasure)
        all_rougeL.append(scores['rougeL'].fmeasure)
        
        if idx < 3:  # 打印前3个样本
            print(f"\n样本 {idx+1}:")
            print(f"对话: {dialogue[:200]}...")
            print(f"参考摘要: {reference_summary}")
            print(f"生成摘要: {generated_summary}")
            print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
    
    # 打印平均分数
    print("\n" + "=" * 80)
    print("平均ROUGE分数:")
    print(f"  ROUGE-1: {np.mean(all_rouge1):.4f}")
    print(f"  ROUGE-2: {np.mean(all_rouge2):.4f}")
    print(f"  ROUGE-L: {np.mean(all_rougeL):.4f}")
    print("=" * 80)

def predict_test_set():
    """对测试集进行预测"""
    print("\n" + "=" * 80)
    print("开始对测试集进行预测...")
    print("=" * 80)
    
    # 加载模型
    model_path = os.path.join(config.out_dir, 'model')
    if os.path.exists(model_path):
        print(f"从 {model_path} 加载模型")
        tokenizer = T5Tokenizer.from_pretrained(model_path)
        model = T5ForConditionalGeneration.from_pretrained(model_path)
    else:
        checkpoint_path = os.path.join(config.out_dir, 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"从checkpoint加载模型: {checkpoint_path}")
            tokenizer = T5Tokenizer.from_pretrained(config.model_name)
            model = T5ForConditionalGeneration.from_pretrained(config.model_name)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # 处理可能的DataParallel wrapper
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            model.load_state_dict(new_state_dict)
        else:
            print("错误: 未找到训练好的模型")
            return
    
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()
    
    # 读取测试数据
    test_file = '/kaggle/input/nanogpt-fudannlp-cs-30040/test.csv'
    if not os.path.exists(test_file):
        print(f"错误: 找不到测试文件 {test_file}")
        return
    
    test_df = pd.read_csv(test_file)
    print(f"加载了 {len(test_df)} 条测试样本")
    
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc="生成摘要"):
        sample_id = row['id']
        dialogue = row['dialogue']
        
        # 构建输入
        input_text = "summarize: " + dialogue
        input_ids = tokenizer(
            input_text,
            max_length=config.max_input_length,
            truncation=True,
            return_tensors='pt'
        ).input_ids.to(device)
        
        # 生成摘要
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                generated_ids = model.generate(
                    input_ids,
                    max_length=config.max_target_length,
                    num_beams=config.num_beams,
                    temperature=config.temperature,
                    do_sample=True,
                    top_k=config.top_k,
                    top_p=config.top_p,
                    early_stopping=True
                )
        
        generated_summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        results.append({
            'id': sample_id,
            'summary': generated_summary
        })
    
    # 保存结果
    output_file = os.path.join(config.out_dir, 'submission.csv')
    submission_df = pd.DataFrame(results)
    submission_df.to_csv(output_file, index=False)
    
    print(f"\n预测完成！结果保存到: {output_file}")
    print("=" * 80)
    
    return output_file

# %% [markdown]
# # 主函数

# %%
def main():
    """主函数"""
    print("\n")
    print("=" * 80)
    print("Flan-T5 摘要微调".center(80))
    print("=" * 80)
    print("\n当前配置:")
    print(f"  模型: {config.model_name}")
    print(f"  数据集: {config.dataset_path}")
    print(f"  设备: {config.device}")
    print(f"  GPU数量: {config.n_gpu}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  最大迭代次数: {config.max_iters}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  评估模式: {config.eval_only}")
    print(f"  恢复训练: {config.resume}")
    
    # 训练模型
    model, tokenizer = train()
    
    # 评估模型
    evaluate()
    
    # 对测试集进行预测
    predict_test_set()

if __name__ == "__main__":
    main()