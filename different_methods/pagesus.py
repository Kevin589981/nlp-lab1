# %% [markdown]
# # 使用PEGASUS-samsum进行文本摘要微调
# 
# 本notebook使用预训练的PEGASUS-samsum模型对对话摘要任务进行微调

# %%
# 安装必要的库
!pip install rouge-score accelerate

# %%
import pandas as pd
import os
import torch
print("\n" + "=" * 80)
print("准备SAMSum数据集划分")
print("=" * 80)

# 读取原始CSV文件
input_csv = '/kaggle/input/nanogpt-fudannlp-cs-30040/train.csv'
print(f"\n读取数据: {input_csv}")

df = pd.read_csv(input_csv)

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 可以根据需要调整数据集大小
# df = df[:1000]  # 仅使用前1000条数据进行快速测试

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
# # 配置参数

# %%
from dataclasses import dataclass
import torch
from typing import Optional

@dataclass
class Config:
    """训练配置"""
    
    # 数据配置
    dataset_path: str = '/kaggle/working/data/samsum'
    test_dataset_path: str = '/kaggle/input/nanogpt-fudannlp-cs-30040/test.csv'
    
    # 模型配置
    model_name: str = "google/pegasus-cnn_dailymail"  # 使用PEGASUS基础模型
    
    # 训练配置
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    max_epochs: int = 3
    warmup_steps: int = 500
    
    # 序列长度配置
    max_input_length: int = 1024
    max_target_length: int = 128
    
    # 生成配置
    num_beams: int = 4
    length_penalty: float = 2.0
    early_stopping: bool = True
    
    # 评估配置
    eval_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    eval_only: bool = False
    resume_from_checkpoint: Optional[str] = None
    
    # 输出配置
    output_dir: str = 'out-pegasus-summarization'
    
    # 设备配置
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp: bool = True  # 使用混合精度训练
    
    # 根据GPU类型选择精度
    def get_dtype(self):
        if not torch.cuda.is_available():
            return torch.float32
        
        # 检查是否支持bfloat16
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            # P100等旧GPU使用float16
            return torch.float16

config = Config()

# %% [markdown]
# # 数据集类

# %%
from torch.utils.data import Dataset, DataLoader
from transformers import PegasusTokenizer
import csv

class SummarizationDataset(Dataset):
    """摘要数据集"""
    
    def __init__(self, csv_path, tokenizer, max_input_length=1024, max_target_length=128):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
        # 读取数据
        self.dialogues = []
        self.summaries = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.dialogues.append(row['dialogue'])
                self.summaries.append(row['summary'])
        
        print(f"加载了 {len(self.dialogues)} 个样本")
    
    def __len__(self):
        return len(self.dialogues)
    
    def __getitem__(self, idx):
        dialogue = self.dialogues[idx]
        summary = self.summaries[idx]
        
        # 对输入进行编码
        inputs = self.tokenizer(
            dialogue,
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 对目标进行编码
        targets = self.tokenizer(
            summary,
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # 将padding token的标签设置为-100（忽略）
        labels = targets['input_ids'].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }

# %% [markdown]
# # 模型和训练

# %%
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from transformers.trainer_callback import TrainerCallback
import numpy as np
from rouge_score import rouge_scorer
import torch
from tqdm import tqdm

def compute_metrics(eval_preds, tokenizer):
    """计算ROUGE分数"""
    predictions, labels = eval_preds
    
    # 解码预测和标签
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 替换标签中的-100为pad token id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 计算ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores),
    }

class RougeCallback(TrainerCallback):
    """训练时显示ROUGE分数的回调"""
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            print(f"\n评估结果 (Step {state.global_step}):")
            print(f"  Loss: {metrics.get('eval_loss', 0):.4f}")
            print(f"  ROUGE-1: {metrics.get('eval_rouge1', 0):.4f}")
            print(f"  ROUGE-2: {metrics.get('eval_rouge2', 0):.4f}")  
            print(f"  ROUGE-L: {metrics.get('eval_rougeL', 0):.4f}")

def train():
    """训练主函数"""
    print("\n" + "=" * 80)
    print("开始训练PEGASUS模型...")
    print("=" * 80)
    
    # 设置设备和精度
    device = torch.device(config.device)
    dtype = config.get_dtype()
    print(f"使用设备: {device}")
    print(f"使用精度: {dtype}")
    
    # 加载tokenizer和模型
    print("\n加载模型和tokenizer...")
    tokenizer = PegasusTokenizer.from_pretrained(config.model_name)
    
    # 根据是否resume来加载模型
    if config.resume_from_checkpoint:
        print(f"从checkpoint恢复: {config.resume_from_checkpoint}")
        model = PegasusForConditionalGeneration.from_pretrained(config.resume_from_checkpoint)
    else:
        model = PegasusForConditionalGeneration.from_pretrained(config.model_name)
    
    # 转换模型精度
    if dtype == torch.bfloat16:
        model = model.to(dtype=torch.bfloat16)
    elif dtype == torch.float16:
        model = model.half()
    
    model = model.to(device)
    
    # 创建数据集
    print("\n准备数据集...")
    train_dataset = SummarizationDataset(
        os.path.join(config.dataset_path, 'train.csv'),
        tokenizer,
        config.max_input_length,
        config.max_target_length
    )
    
    val_dataset = SummarizationDataset(
        os.path.join(config.dataset_path, 'validation.csv'),
        tokenizer,
        config.max_input_length,
        config.max_target_length
    )
    
    # 数据collator（处理padding）
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=config.max_input_length
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.max_epochs if not config.eval_only else 0,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_dir=f'{config.output_dir}/logs',
        logging_steps=config.logging_steps,
        evaluation_strategy="steps" if not config.eval_only else "epoch",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=dtype == torch.float16,
        bf16=dtype == torch.bfloat16,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_smoothing_factor=0.1,
        report_to="none",
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset if not config.eval_only else None,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
        callbacks=[RougeCallback()]
    )
    
    # 训练或评估
    if config.eval_only:
        print("\n仅评估模式...")
        metrics = trainer.evaluate()
        print(f"\n最终评估结果:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
    else:
        print("\n开始训练...")
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        
        # 保存最终模型
        print(f"\n保存模型到 {config.output_dir}/final_model")
        trainer.save_model(f"{config.output_dir}/final_model")
        tokenizer.save_pretrained(f"{config.output_dir}/final_model")
    
    print("\n训练/评估完成！")
    print("=" * 80)
    
    return model, tokenizer

# 执行训练
if not config.eval_only:
    model, tokenizer = train()

# %% [markdown]
# # 评估函数

# %%
def evaluate():
    """详细评估函数"""
    print("\n" + "=" * 80)
    print("开始详细评估...")
    print("=" * 80)
    
    # 加载模型
    print("\n加载训练好的模型...")
    model_path = f"{config.output_dir}/final_model"
    if not os.path.exists(model_path):
        model_path = config.model_name
        print(f"未找到微调模型，使用预训练模型: {model_path}")
    
    tokenizer = PegasusTokenizer.from_pretrained(model_path)
    model = PegasusForConditionalGeneration.from_pretrained(model_path)
    
    # 设置设备和精度（兼容P100的float16验证）
    device = torch.device(config.device)
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
        else:
            model = model.half()
    
    model = model.to(device)
    model.eval()
    
    # 加载验证数据
    val_csv = os.path.join(config.dataset_path, 'validation.csv')
    dialogues = []
    summaries = []
    
    with open(val_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 20:  # 评估前20个样本
                break
            dialogues.append(row['dialogue'])
            summaries.append(row['summary'])
    
    print(f"评估 {len(dialogues)} 个样本")
    
    # 评估
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    all_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
    
    print("\n生成摘要并计算ROUGE分数...")
    for idx, (dialogue, reference) in enumerate(tqdm(zip(dialogues, summaries))):
        # 编码输入
        inputs = tokenizer(
            dialogue,
            max_length=config.max_input_length,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        
        # 生成摘要
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=config.max_target_length,
                num_beams=config.num_beams,
                length_penalty=config.length_penalty,
                early_stopping=config.early_stopping
            )
        
        # 解码
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 计算ROUGE
        scores = scorer.score(reference, generated)
        for metric in ['rouge1', 'rouge2', 'rougeL']:
            all_scores[metric].append(scores[metric].fmeasure)
        
        # 显示前3个例子
        if idx < 3:
            print(f"\n样本 {idx+1}:")
            print(f"对话: {dialogue[:200]}...")
            print(f"参考摘要: {reference}")
            print(f"生成摘要: {generated}")
            print(f"ROUGE-1: {scores['rouge1'].fmeasure:.4f}")
            print(f"ROUGE-2: {scores['rouge2'].fmeasure:.4f}")
            print(f"ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
    
    # 计算平均分数
    print("\n" + "=" * 80)
    print("平均ROUGE分数:")
    for metric in ['rouge1', 'rouge2', 'rougeL']:
        avg_score = np.mean(all_scores[metric])
        print(f"  {metric.upper()}: {avg_score:.4f}")
    print("=" * 80)

# 执行评估
evaluate()

# %% [markdown]
# # 测试集推理

# %%
def predict_test_set():
    """对测试集进行推理"""
    print("\n" + "=" * 80)
    print("开始测试集推理...")
    print("=" * 80)
    
    # 加载模型
    print("\n加载模型...")
    model_path = f"{config.output_dir}/final_model"
    if not os.path.exists(model_path):
        model_path = config.model_name
        print(f"未找到微调模型，使用预训练模型: {model_path}")
    
    tokenizer = PegasusTokenizer.from_pretrained(model_path)
    model = PegasusForConditionalGeneration.from_pretrained(model_path)
    
    # 设置设备和精度
    device = torch.device(config.device)
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            model = model.to(dtype=torch.bfloat16)
        else:
            model = model.half()
    
    model = model.to(device)
    model.eval()
    
    # 读取测试数据
    print("\n加载测试数据...")
    test_data = pd.read_csv(config.test_dataset_path)
    print(f"测试集样本数: {len(test_data)}")
    
    # 生成摘要
    print("\n生成摘要...")
    results = []
    
    with torch.no_grad():
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data)):
            dialogue = row['dialogue']
            
            # 编码输入
            inputs = tokenizer(
                dialogue,
                max_length=config.max_input_length,
                truncation=True,
                return_tensors='pt'
            ).to(device)
            
            # 生成摘要
            outputs = model.generate(
                **inputs,
                max_length=config.max_target_length,
                num_beams=config.num_beams,
                length_penalty=config.length_penalty,
                early_stopping=config.early_stopping
            )
            
            # 解码
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'id': row['id'],
                'summary': summary
            })
    
    # 保存结果
    output_file = f'{config.output_dir}/submission.csv'
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")
    print("=" * 80)

# 执行测试集推理
predict_test_set()