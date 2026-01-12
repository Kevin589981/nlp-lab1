# %% [markdown]
# # 步骤 0: 环境设置
# 首先，安装所有必需的库。`accelerate` 和 `evaluate` 是 Hugging Face 生态系统的重要组成部分，可以简化训练和评估流程。

# %%
!pip install rouge_score

# %% [markdown]
# # 步骤 1: 配置文件
# 我们将所有可配置的参数集中在一个类中，以便于管理和修改。
# 这包括模型名称、数据路径、训练参数等。
# 您可以在这里轻松切换 `eval_only` 和 `resume_from_checkpoint` 模式。

# %%
import torch
from dataclasses import dataclass, field

@dataclass
class TrainingConfig:
    # --- 模型和数据路径配置 ---
    model_checkpoint: str = "google/pegasus-x-base"
    train_file_path: str = "/kaggle/working/data/samsum/train.csv"
    validation_file_path: str = "/kaggle/working/data/samsum/validation.csv"
    test_file_path: str = "/kaggle/input/nanogpt-fudan-cs-30040/test.csv"
    output_dir: str = "/kaggle/working/pegasus-samsum-finetuned"
    submission_file: str = "/kaggle/working/submission.csv"

    # --- 运行模式配置 ---
    # 如果为 True，将只进行评估，不进行训练。需要指定一个有效的 checkpoint 路径
    eval_only: bool = False 
    # 从指定的 checkpoint 继续训练。设为 True 或 checkpoint 路径（例如 "output_dir/checkpoint-500"）
    resume_from_checkpoint: bool = False

    # --- 数据处理配置 ---
    max_input_length: int = 1024  # 输入（对话）的最大长度
    max_target_length: int = 128   # 输出（摘要）的最大长度
    
    # --- 训练参数配置 (Seq2SeqTrainingArguments) ---
    # 评估策略，"epoch" 表示每个 epoch 结束后进行一次评估
    evaluation_strategy: str = "steps"
    eval_steps: int = 100 # 每100步评估一次
    save_steps: int = 100 # 每100步保存一次
    logging_steps: int = 25 # 每25步打印一次日志
    
    # 学习率和优化器
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    
    # 批次大小和梯度累积
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # 有效批次大小 = 2 * 8 = 16
    
    # 训练周期
    num_train_epochs: int = 3
    
    # 混合精度训练 (bf16 适用于 Ampere 架构如 T4, A100; fp16 适用于 P100)
    bf16: bool = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16: bool = not bf16 and torch.cuda.is_available()
    
    # 其他
    save_total_limit: int = 2 # 只保留最新的 2 个 checkpoint
    predict_with_generate: bool = True # 在评估时使用 generate 方法，以计算 ROUGE
    load_best_model_at_end: bool = True # 训练结束后加载最佳模型

    # --- 生成参数配置 (用于评估和推理) ---
    generation_num_beams: int = 4
    generation_max_length: int = 128

# 实例化配置
config = TrainingConfig()

# %% [markdown]
# # 步骤 2: 数据准备
# 这部分代码与您提供的原始代码相同。它读取原始的 `train.csv`，
# 将其打乱并划分为 95% 的训练集和 5% 的验证集，以便后续使用。
# 我们只取前1000条数据用于快速演示。如果要训练完整数据集，请注释掉 `df = df[:1000]` 这一行。
# %%
import pandas as pd
import os

print("\n" + "=" * 80)
print("准备 SAMSum 数据集划分")
print("=" * 80)

# 读取原始CSV文件
input_csv = '/kaggle/input/nanogpt-fudannlp-cs-30040/train.csv'
print(f"\n读取数据: {input_csv}")

df = pd.read_csv(input_csv)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# !!! 注意: 为了快速演示，我们只使用前100个样本。
# !!! 如果要进行完整训练，请注释掉或删除下面这一行。
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
output_dir = os.path.dirname(config.train_file_path)
os.makedirs(output_dir, exist_ok=True)

# 保存训练集和验证集
train_df.to_csv(config.train_file_path, index=False)
print(f"\n训练集已保存: {config.train_file_path}")

val_df.to_csv(config.validation_file_path, index=False)
print(f"验证集已保存: {config.validation_file_path}")

print("\n数据集划分完成！")
print("=" * 80)

# %% [markdown]
# # 步骤 3: 加载数据、模型和 Tokenizer
# 我们使用 `datasets` 库加载 CSV 文件，并加载 PEGASUS 模型及其对应的 Tokenizer。

# %%
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

# 加载数据集
raw_datasets = load_dataset('csv', data_files={'train': config.train_file_path, 'validation': config.validation_file_path})

print("\n数据集结构:")
print(raw_datasets)

# 加载 Tokenizer 和模型
print(f"\n加载预训练模型: {config.model_checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(config.model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(config.model_checkpoint)

# %% [markdown]
# # 步骤 4: 数据预处理
# 定义一个函数来对数据进行分词（tokenize）。
# - `dialogue` 列作为模型的输入。
# - `summary` 列作为标签（label）。
# Tokenizer 会自动处理截断和填充的准备工作。

# %%
def preprocess_function(examples):
    inputs = tokenizer(examples["dialogue"], max_length=config.max_input_length, truncation=True, padding="max_length")
    
    # 将摘要作为标签进行分词
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=config.max_target_length, truncation=True, padding="max_length")

    inputs["labels"] = labels["input_ids"]
    return inputs

# 使用 .map() 方法将预处理函数应用到整个数据集
# batched=True 可以一次性处理多个样本，加快速度
print("\n开始对数据集进行分词...")
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
print("分词完成！")

print("\n处理后的数据集结构:")
print(tokenized_datasets)

# %% [markdown]
# # 步骤 5: 定义评估指标 (ROUGE)
# 我们需要定义一个函数，在评估过程中计算 ROUGE 分数。
# 这个函数会在每个评估步骤被 `Trainer` 自动调用。

# %%
import numpy as np
import evaluate

# 加载 ROUGE 评估指标
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # 解码生成的摘要
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # 将标签中的 -100 替换为 padding token ID，以便解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE 需要在每句话后添加换行符
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]
    
    # 计算 ROUGE 分数
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # 提取关键指标
    result = {key: value * 100 for key, value in result.items()}
    
    # 添加生成文本的平均长度
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

# %% [markdown]
# # 步骤 6: 配置和初始化 Trainer
# - `Seq2SeqTrainingArguments`: 定义所有训练超参数。
# - `DataCollatorForSeq2Seq`: 负责在每个批次中智能地填充（pad）输入和标签。
# - `Seq2SeqTrainer`: 封装了所有训练和评估逻辑的核心类。

# %%
# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=config.output_dir,
    eval_strategy=config.evaluation_strategy,
    eval_steps=config.eval_steps,
    save_steps=config.save_steps,
    logging_steps=config.logging_steps,
    learning_rate=config.learning_rate,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    weight_decay=config.weight_decay,
    save_total_limit=config.save_total_limit,
    num_train_epochs=config.num_train_epochs,
    predict_with_generate=config.predict_with_generate,
    fp16=config.fp16,
    bf16=config.bf16,
    load_best_model_at_end=config.load_best_model_at_end,
    report_to="none",  # 可设置为 "wandb", "tensorboard" 等
)

# 定义数据整理器
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# 初始化 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %% [markdown]
# # 步骤 7: 执行训练或评估
# 根据 `config.eval_only` 的值，执行相应的操作。

# %%
if config.eval_only:
    print("\n" + "="*80)
    print("模式: 只进行评估")
    print("="*80)
    # 确保 resume_from_checkpoint 指向一个有效的模型路径
    if not os.path.isdir(str(config.resume_from_checkpoint)):
         raise ValueError(f"eval_only=True, 但 resume_from_checkpoint ('{config.resume_from_checkpoint}') 不是一个有效的目录。")
    
    print(f"从 checkpoint 加载模型进行评估: {config.resume_from_checkpoint}")
    eval_results = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    print("\n评估结果:")
    print(eval_results)
else:
    print("\n" + "="*80)
    print("模式: 开始训练")
    print("="*80)
    # 如果设置了 resume_from_checkpoint，则从断点继续训练
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    # 保存最终的模型和训练状态
    trainer.save_model()
    trainer.save_state()
    
    print("\n训练完成!")
    metrics = train_result.metrics
    metrics["train_samples"] = len(tokenized_datasets["train"])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

# %% [markdown]
# # 步骤 8: 在测试集上进行推理并生成提交文件
# 训练（或加载）好的最佳模型现在可用于为测试集生成摘要。
# 我们将结果保存为 `submission.csv`。

# %%
from tqdm.auto import tqdm

print("\n" + "="*80)
print("开始在测试集上进行推理...")
print("="*80)

# 加载测试数据
test_df = pd.read_csv(config.test_file_path)
print(f"加载了 {len(test_df)} 条测试样本。")

# 使用 trainer.predict 进行高效推理
# 注意：trainer 内部已经加载了训练过程中的最佳模型（因为 load_best_model_at_end=True）
test_dataset = load_dataset("csv", data_files={"test": config.test_file_path})["test"]

def tokenize_test_data(examples):
    return tokenizer(examples["dialogue"], max_length=config.max_input_length, truncation=True)

tokenized_test_dataset = test_dataset.map(tokenize_test_data, batched=True)

print("开始生成预测...")
predictions = trainer.predict(
    tokenized_test_dataset,
    max_length=config.generation_max_length,
    num_beams=config.generation_num_beams,
)

# 解码预测结果
decoded_summaries = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)

# 清理生成的文本
cleaned_summaries = [s.strip() for s in decoded_summaries]

# 创建提交文件
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'summary': cleaned_summaries
})

submission_df.to_csv(config.submission_file, index=False)

print(f"\n推理完成！提交文件已保存至: {config.submission_file}")
print("\n提交文件预览:")
print(submission_df.head())
print("="*80)