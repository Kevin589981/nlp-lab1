%%writefile train_ddp.py
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

# 检查输入文件是否存在
if not os.path.exists(input_csv):
    print(f"错误：找不到输入文件 {input_csv}，请确保已正确添加数据集。")
    # 如果在Kaggle环境下运行，需要确保数据集已挂载
    # 为了让代码可以运行，这里假设文件存在
    # 如果文件不存在，后续步骤会失败，这里不做更深处理
    pass
else:
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
# # 配置和数据准备 (DDP 改造)

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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig

# =============================================================================
# 配置参数
# =============================================================================

class Config:
    """配置类：包含所有可调参数"""
    
    # DDP/系统配置 (新增/修改)
    ddp_world_size = 1 # 全局进程数
    ddp_rank = 0       # 当前进程的全局排名
    ddp_local_rank = 0 # 当前进程的本地排名 (GPU ID)
    is_master = True   # 是否为主进程 (rank 0)
    device = 'cuda'    # 默认设备
    
    # 数据集配置
    dataset_path = '/kaggle/working/data/samsum'
    dataset = 'samsum'
    
    # 模型配置
    model_name = 'facebook/bart-base'
    label_smoothing_factor = 0.1

    # 训练配置
    init_from = 'pretrained'  # 'pretrained', 'resume', 'scratch'
    resume_from = None  # checkpoint路径，当init_from='resume'时使用
    
    # 批次配置 - 针对2个T4 GPU优化
    batch_size = 64  # **每个 GPU** 的 batch size
    gradient_accumulation_steps = 8  # 梯度累积步数
    max_source_length = 512  # 输入对话的最大长度
    max_target_length = 128  # 目标摘要的最大长度
    
    # 训练步数
    max_iters = 10
    
    # 优化器配置
    learning_rate = 3e-5
    weight_decay = 0.01
    beta1 = 0.9
    beta2 = 0.999
    grad_clip = 1.0
    
    # 学习率调度
    decay_lr = True
    warmup_iters = int(0.1 *max_iters)
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
    rouge_eval_samples = 5 # 评估的样本数量
    
    # wandb日志 (DDP中通常只在主进程进行日志记录)
    wandb_log = False
    wandb_project = 'bart-summarization'
    wandb_run_name = 'bart-base'
    
    # 系统配置
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    compile = True
    
    # 生成配置
    num_test_samples = 10
    num_beams = 8 # 束搜索宽度
    temperature = 1.0
    top_k = 50
    top_p = 0.95

config = Config()

# =============================================================================
# DDP 初始化函数
# =============================================================================

def setup_ddp():
    """初始化 DDP 进程组并配置 DDP 相关的 config 变量"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        config.ddp_rank = int(os.environ['RANK'])
        config.ddp_local_rank = int(os.environ['LOCAL_RANK'])
        config.ddp_world_size = int(os.environ['WORLD_SIZE'])
        
        # 初始化进程组
        dist.init_process_group(backend='nccl', init_method='env://')
        
        # 设置当前进程使用的 GPU 设备
        config.device = f'cuda:{config.ddp_local_rank}'
        torch.cuda.set_device(config.device)
        config.is_master = (config.ddp_rank == 0)
        
        print(f"DDP 初始化成功: Rank {config.ddp_rank}/{config.ddp_world_size}, Device: {config.device}")
        
    else:
        # 单 GPU 或 CPU 模式
        config.ddp_world_size = 1
        config.ddp_rank = 0
        config.ddp_local_rank = 0
        config.is_master = True
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if config.device == 'cuda':
            # 确保单 GPU 模式下，设备是 0
            config.device = 'cuda:0'
            torch.cuda.set_device(config.device)
            
    # 确保主进程能看到完整的配置信息
    if config.is_master:
        print("-" * 80)
        print(f"总有效 Batch Size (跨所有 GPU): {config.batch_size * config.ddp_world_size * config.gradient_accumulation_steps}")
        print("-" * 80)
        

# =============================================================================
# 数据集类
# =============================================================================

class SummarizationDataset(Dataset):
    """BART摘要任务的Dataset类"""
    # ... (与原代码相同，无需 DDP 修改)
    def __init__(self, csv_path, tokenizer, max_source_length, max_target_length):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        self.dialogues = []
        self.summaries = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.dialogues.append(row['dialogue'])
                self.summaries.append(row['summary'])
        
        if config.is_master:
            print(f"  加载了 {len(self.dialogues)} 个样本")
    
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]
        
        source = self.tokenizer(
            dialogue,
            max_length=self.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
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
        
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_ids,
            'attention_mask': source_mask,
            'labels': labels
        }

# 全局变量：缓存DataLoader和 Sampler (DDP 改造: Sampler 需缓存)
_dataloaders = {'train': None, 'val': None}
_data_iters = {'train': None, 'val': None}
_samplers = {'train': None, 'val': None}

def get_batch(split, tokenizer):
    """获取一个训练批次 (DDP 改造: 使用 DistributedSampler)"""
    global _dataloaders, _data_iters, _samplers
    
    # 首次调用：创建 DataLoader 和 Sampler
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
        
        # DDP Sampler
        if config.ddp_world_size > 1:
            sampler = DistributedSampler(
                dataset, 
                num_replicas=config.ddp_world_size, 
                rank=config.ddp_rank, 
                shuffle=shuffle
            )
            # 如果使用 Sampler，DataLoader 中的 shuffle 必须设置为 False
            shuffle = False
            _samplers[split] = sampler
        else:
            sampler = None
        
        num_workers = min(4, os.cpu_count() or 1)
        
        _dataloaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler, # 使用 Sampler
            shuffle=shuffle, # Sampler 存在时为 False
            num_workers=num_workers,
            pin_memory=('cuda' in config.device),
            persistent_workers=(num_workers > 0 and 'cuda' in config.device),
        )
        _data_iters[split] = iter(_dataloaders[split])
    
    # 获取下一个 batch
    try:
        batch = next(_data_iters[split])
    except StopIteration:
        # 如果是训练集，需要更新 Sampler 的 epoch
        if split == 'train' and _samplers['train']:
            # 在 train() 循环中通过 sampler.set_epoch(iter_num) 来处理
            # 这里先重新初始化迭代器，确保能获取新的 batch
            pass 
        _data_iters[split] = iter(_dataloaders[split])
        batch = next(_data_iters[split])
    
    # 移动到设备
    batch = {k: v.to(config.device, non_blocking=True) for k, v in batch.items()}
    
    return batch

@torch.no_grad()
def estimate_loss(model, ctx, tokenizer):
    """估计训练集和验证集上的损失 (DDP 改造: 跨进程平均损失)"""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters, device=config.device)
        for k in range(config.eval_iters):
            batch = get_batch(split, tokenizer)
            with ctx:
                outputs = model(**batch)
                loss = outputs.loss
            
            # loss已经是该进程上的平均loss
            losses[k] = loss.item()
        
        # 计算当前进程上的平均 loss
        local_avg_loss = losses.mean()
        
        # DDP: 跨所有进程求平均
        if config.ddp_world_size > 1:
            # AllReduce 操作：求和后分摊
            dist.all_reduce(local_avg_loss, op=dist.ReduceOp.AVG)
            
        out[split] = local_avg_loss.item()
    
    model.train()
    return out

def get_lr(iter_num):
    """学习率调度：带预热的余弦衰减 (与原代码相同)"""
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num + 1) / (config.warmup_iters + 1)
    
    if iter_num > config.lr_decay_iters:
        return config.min_lr
    
    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

# =============================================================================
# ROUGE评估 (DDP 改造: 仅在主进程进行)
# =============================================================================

def calculate_rouge(reference_summary, generated_summary):
    """计算ROUGE分数 (与原代码相同)"""
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
    """在训练过程中评估ROUGE分数 (DDP 改造: 仅主进程评估)"""
    
    if not config.is_master:
        # 从进程直接返回 None
        return None
    
    # 获取原始模型 (DDP 包装)
    eval_model = model.module if hasattr(model, 'module') else model
    eval_model.eval()
    
    # ... (其余逻辑与原代码相同)
    val_csv = os.path.join(config.dataset_path, 'validation.csv')
    if not os.path.exists(val_csv):
        eval_model.train()
        return None
    
    dialogues = []
    summaries = []
    with open(val_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialogues.append(row['dialogue'])
            summaries.append(row['summary'])
    
    import random
    indices = random.sample(range(len(dialogues)), min(num_samples, len(dialogues)))
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for idx in indices:
        dialogue = dialogues[idx]
        reference_summary = summaries[idx]
        
        # Tokenize输入
        # 注意: inputs.to(config.device) 已经在 DDP setup 中指向了正确的 GPU
        inputs = tokenizer(
            dialogue,
            max_length=config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(config.device)
        
        # 生成摘要
        with ctx:
            # eval_model 是原始模型，没有 DDP 包装
            outputs = eval_model.generate(
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
    
    eval_model.train()
    
    if len(rouge1_scores) > 0:
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    return None

# =============================================================================
# 训练函数 (DDP 改造)
# =============================================================================

def train():
    """训练主函数 (DDP 改造)"""
    
    if config.is_master:
        print("\n" + "=" * 80)
        print("开始训练...")
        print("=" * 80)
    
    # DDP: 设置随机种子
    seed = 1337 + config.ddp_rank
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 主进程创建输出目录
    if config.is_master:
        os.makedirs(config.out_dir, exist_ok=True)
    
    # DDP: 等待所有进程同步，确保主进程完成目录创建
    if config.ddp_world_size > 1:
        dist.barrier()
        
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
    if config.is_master:
        print(f"\n加载tokenizer: {config.model_name}")
    tokenizer = BartTokenizer.from_pretrained(config.model_name)
    
    # 初始化模型
    iter_num = 0
    best_val_loss = 1e9
    
    if config.init_from == 'scratch':
        if config.is_master:
            print("从头开始训练新模型")
        model_config = BartConfig.from_pretrained(config.model_name)
        model = BartForConditionalGeneration(model_config)
        
    elif config.init_from == 'resume':
        if config.is_master:
            print(f"从checkpoint恢复训练")
        if config.resume_from is None:
            ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
        else:
            ckpt_path = config.resume_from
        
        # 注意: 只有主进程加载和保存 checkpoint，所有进程等待加载完成
        if config.is_master and not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

        if config.ddp_world_size > 1:
            dist.barrier() # 等待主进程检查文件
            
        map_location = {'cuda:%d' % 0: 'cuda:%d' % config.ddp_local_rank} if device_type == 'cuda' else 'cpu'
        checkpoint = torch.load(ckpt_path, map_location=map_location)

        model = BartForConditionalGeneration.from_pretrained(config.model_name)
        
        # 处理 DataParallel/DDP 保存的模型
        state_dict = checkpoint['model']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model.load_state_dict(new_state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        
    else:  # pretrained
        if config.is_master:
            print(f"从预训练模型加载: {config.model_name}")
        model = BartForConditionalGeneration.from_pretrained(
            config.model_name,
            label_smoothing_factor=config.label_smoothing_factor
        )
        if config.is_master:
            print(f"  启用标签平滑, factor: {config.label_smoothing_factor}")
    
    model.to(config.device)
    
    # DDP: 使用 DistributedDataParallel 包装模型
    if config.ddp_world_size > 1:
        model = DDP(model, device_ids=[config.ddp_local_rank])
        raw_model = model.module
        if config.is_master:
            print(f"\n使用 DDP 在 {config.ddp_world_size} 个 GPU 上训练")
    else:
        raw_model = model
    
    # 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay
    )
    
    if config.init_from == 'resume' and 'optimizer' in checkpoint:
        # 注意：这里需要在所有进程上加载优化器状态
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # 初始化 GradScaler (只有在 CUDA + fp16/bf16 时启用)
    scaler = torch.amp.GradScaler(enabled=(config.dtype == 'float16'))
    
    # 编译模型（可选）
    if config.compile:
        if config.is_master:
            print("编译模型...")
        # 注意：DDP 和 torch.compile 通常是兼容的
        model = torch.compile(model)
    
    # 训练循环
    if config.is_master:
        print("\n开始训练循环...")
        print(f"总迭代次数: {config.max_iters}")
        print(f"每 GPU 批次大小: {config.batch_size}")
        print(f"GPU 数量: {config.ddp_world_size}")
        print(f"梯度累积步数: {config.gradient_accumulation_steps}")
        print("-" * 80)
    
    t0 = time.time()
    
    while True:
        # DDP: 在训练开始前设置 Sampler 的 epoch
        train_sampler = _samplers.get('train')
        if train_sampler:
            train_sampler.set_epoch(iter_num)
            # DDP: 如果是新的 epoch，需要重置迭代器
            if iter_num > 0 and train_sampler.epoch > 0 and iter_num % len(_dataloaders['train']) == 0:
                global _data_iters
                _data_iters['train'] = iter(_dataloaders['train'])

        # 设置学习率
        lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 评估和保存 checkpoint
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, ctx, tokenizer)
            
            if config.is_master:
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
                
                # 保存checkpoint (仅主进程保存)
                if losses['val'] < best_val_loss or config.always_save_checkpoint:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        checkpoint = {
                            'model': raw_model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'iter_num': iter_num,
                            'best_val_loss': best_val_loss,
                            'config': vars(config),
                        }
                        print(f"  保存checkpoint到 {config.out_dir}")
                        torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
        
        # DDP: 等待所有进程完成评估和主进程的保存操作
        if config.ddp_world_size > 1:
            dist.barrier()

        if iter_num == 0 and config.eval_only:
            break
        
        # 训练步骤
        model.train()
        optimizer.zero_grad()
        
        for micro_step in range(config.gradient_accumulation_steps):
            batch = get_batch('train', tokenizer)
            
            # 使用梯度累积时，只有最后一个 micro_step 进行 backward
            is_last_micro_step = (micro_step == config.gradient_accumulation_steps - 1)
            
            # DDP: 需要设置 no_sync 避免在中间 micro_step 同步梯度
            if config.ddp_world_size > 1 and not is_last_micro_step:
                # DDP 需要一个 context manager 来禁用梯度同步
                # DDP 包装器的 `no_sync()` 方法
                with model.no_sync():
                    with ctx:
                        outputs = model(**batch)
                        loss = outputs.loss
                        # loss 已经是本地 GPU 上的平均 loss (标量)
                        loss = loss / config.gradient_accumulation_steps
                    scaler.scale(loss).backward()
            else:
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
        
        # 记录日志 (仅主进程记录)
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if config.is_master and iter_num % config.log_interval == 0:
            # 这里的 loss 是最后一个 micro_step 的平均 loss 乘以累积步数
            lossf = loss.item() * config.gradient_accumulation_steps 
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, lr {lr:.2e}")
        
        iter_num += 1
        
        if iter_num > config.max_iters:
            break
    
    if config.is_master:
        print("\n训练完成！")
        print("=" * 80)
        
    # DDP: 结束进程组
    if config.ddp_world_size > 1:
        dist.destroy_process_group()


# =============================================================================
# 主函数 (DDP 包装)
# =============================================================================

def main():
    """主函数"""
    
    # 1. 设置 DDP 环境
    setup_ddp()
    
    # 2. 打印配置信息 (仅主进程)
    if config.is_master:
        print("\n")
        print("=" * 80)
        print("BART 摘要微调 (DDP 多GPU)".center(80))
        print("=" * 80)
        print("\n当前配置:")
        print(f"  数据集: {config.dataset_path}")
        print(f"  模型: {config.model_name}")
        print(f"  初始化方式: {config.init_from}")
        print(f"  设备: {config.device}")
        print(f"  精度: {config.dtype}")
        print(f"  GPU 数量: {config.ddp_world_size}")
        print(f"  每 GPU 批次大小: {config.batch_size}")
        print(f"  最大迭代次数: {config.max_iters}")
        print(f"  学习率: {config.learning_rate}")
    
    # 3. 执行训练
    if not config.eval_only:
        train()
    elif config.is_master:
        print("\neval_only=True，跳过训练")

main()

# %% [markdown]
# # 评估 (仅在主进程运行)

# %%
!pip install rouge-score

# %%
# Note: 以下评估和预测函数为了 DDP 兼容性，只在主进程 (config.is_master) 上运行。
# 但为了代码简洁，我们在 DDP 训练结束后，直接在 'main' 进程执行这些单进程任务。
# 重新加载模型时，我们只需要确保配置的 device 是 'cuda:0' 或 'cuda'。

def load_model():
    """加载训练好的模型 (仅在主进程/单 GPU 上运行)"""
    # 确保设备是可用的 CUDA 设备或 CPU
    current_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    device_type = 'cuda' if 'cuda' in current_device else 'cpu'
    ptdtype = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16
    }[config.dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=ptdtype
    )
    
    tokenizer = BartTokenizer.from_pretrained(config.model_name)
    
    print(f"\n从 {config.out_dir} 加载模型到 {current_device}...")
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
    
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到checkpoint文件 {ckpt_path}")
        print("请先运行训练！")
        return None, None, None
    
    # DDP 环境下，主进程加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location=current_device)
    model = BartForConditionalGeneration.from_pretrained(config.model_name)
    
    # 处理可能的 DataParallel/DDP 保存的模型
    state_dict = checkpoint['model']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    model.eval()
    model.to(current_device)
    
    if config.compile:
        print("编译模型...")
        model = torch.compile(model)
    
    # 确保 config.device 在评估时指向当前设备
    config.device = current_device
    
    return model, tokenizer, ctx

def evaluate():
    """评估模型性能 (仅在主进程运行)"""
    if not (config.is_master or config.ddp_world_size == 1):
        return

    print("\n" + "=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    model, tokenizer, ctx = load_model()
    if model is None:
        return
    
    # ... (其余逻辑与原代码相同，使用 config.device)
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
    
    rouge_result = calculate_rouge("test", "test")
    use_rouge = rouge_result is not None
    if not use_rouge:
        print("\n警告: 未安装rouge_score库，将跳过ROUGE评分")
    
    print("\n" + "-" * 80)
    print("开始生成和评估...")
    print("-" * 80)
    
    all_rouge1_f = []
    all_rouge2_f = []
    all_rougeL_f = []
    
    for idx, (dialogue, reference_summary) in enumerate(zip(dialogues, summaries)):
        # ... (推理和评估逻辑与原代码相同)
        print(f"\n[样本 {idx+1}/{len(dialogues)}]")
        print(f"对话: {dialogue[:100]}..." if len(dialogue) > 100 else f"对话: {dialogue}")
        
        inputs = tokenizer(
            dialogue,
            max_length=config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(config.device)
        
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
        
        if use_rouge:
            rouge_scores = calculate_rouge(reference_summary, generated_summary)
            if rouge_scores:
                print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
                print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
                print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
                
                all_rouge1_f.append(rouge_scores['rouge1'])
                all_rouge2_f.append(rouge_scores['rouge2'])
                all_rougeL_f.append(rouge_scores['rougeL'])
    
    if use_rouge and len(all_rouge1_f) > 0:
        print("\n" + "=" * 80)
        print("平均ROUGE分数:")
        print(f"  ROUGE-1: {np.mean(all_rouge1_f):.4f}")
        print(f"  ROUGE-2: {np.mean(all_rouge2_f):.4f}")
        print(f"  ROUGE-L: {np.mean(all_rougeL_f):.4f}")
        print("=" * 80)

if config.is_master or config.ddp_world_size == 1:
    evaluate()

# %%
def predict_test_set():
    """对测试集进行推理 (仅在主进程运行)"""
    if not (config.is_master or config.ddp_world_size == 1):
        return None

    print("\n" + "=" * 80)
    print("开始对测试集进行推理...")
    print("=" * 80)
    
    model, tokenizer, ctx = load_model()
    if model is None:
        return None
    
    # ... (其余逻辑与原代码相同)
    print("\n加载测试数据...")
    test_file = '/kaggle/input/nanogpt-fudannlp-cs-30040/test.csv'
    
    if not os.path.exists(test_file):
        print(f"错误: 找不到测试文件 {test_file}")
        return None
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_data.append({
                'id': row['id'],
                'dialogue': row['dialogue']
            })
    
    print(f"加载了 {len(test_data)} 条测试样本")
    
    results = []
    
    print("\n" + "-" * 80)
    print("开始生成摘要...")
    print("-" * 80)
    
    start_time = time.time()
    
    for idx, sample in enumerate(tqdm(test_data)):
        sample_id = sample['id']
        dialogue = sample['dialogue']
        
        inputs = tokenizer(
            dialogue,
            max_length=config.max_source_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(config.device)
        
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

if config.is_master or config.ddp_world_size == 1:
    predict_test_set()