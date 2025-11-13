# %% [markdown]
# # 步骤 0: 环境安装
# 首先，安装运行此项目所需的库。
# 在 Kaggle TPU VM 环境中，torch_xla 通常是预装的。
# transformers, datasets, 和 rouge-score 是我们完成任务所需要的核心库。

# %%
!pip install "transformers==4.35.2" "datasets==2.15.0" "accelerate==0.24.1" "rouge-score==0.1.2" "pandas"

# %% [markdown]
# # 步骤 1: 数据准备
# 这部分代码与您提供的原始代码逻辑保持一致。
# 它会从 Kaggle 输入目录读取 `train.csv`，将其随机划分为 95% 的训练集和 5% 的验证集，
# 并将它们保存到工作目录中，以便后续的模型训练和评估使用。

# %%
import pandas as pd
import os
import argparse

print("\n" + "=" * 80)
print("步骤 1: 准备 SAMSum 数据集划分")
print("=" * 80)

# 定义输入和输出路径
# 注意：请确保在Kaggle的 "Add data" 中添加了相应的数据集
INPUT_DATA_DIR = '/kaggle/input/nanogpt-fudannlp-cs-30040'
OUTPUT_DATA_DIR = '/kaggle/working/data/samsum'

# 创建输出目录
os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)

# 检查原始数据是否存在
train_csv_path_orig = os.path.join(INPUT_DATA_DIR, 'train.csv')
if not os.path.exists(train_csv_path_orig):
    raise FileNotFoundError(f"错误：未在 {INPUT_DATA_DIR} 中找到 train.csv。请确保已将比赛数据集添加到 notebook 中。")

print(f"\n读取原始数据: {train_csv_path_orig}")
df = pd.read_csv(train_csv_path_orig)

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# --- (可选) 快速测试 ---
# 如果需要快速测试，可以取消下面一行的注释，仅使用少量数据
df = df[:200]
# -------------------------

total_samples = len(df)
print(f"总样本数: {total_samples}")

# 计算划分点 (5% 作为验证集)
val_size = int(total_samples * 0.05)
train_size = total_samples - val_size

print(f"训练集样本数: {train_size} ({train_size/total_samples*100:.1f}%)")
print(f"验证集样本数: {val_size} ({val_size/total_samples*100:.1f}%)")

# 划分数据
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

# 保存训练集
train_csv_path = os.path.join(OUTPUT_DATA_DIR, 'train.csv')
train_df.to_csv(train_csv_path, index=False)
print(f"\n训练集已保存: {train_csv_path}")

# 保存验证集
val_csv_path = os.path.join(OUTPUT_DATA_DIR, 'validation.csv')
val_df.to_csv(val_csv_path, index=False)
print(f"验证集已保存: {val_csv_path}")

print("\n数据集划分完成！")
print("=" * 80)


# %% [markdown]
# # 步骤 2: 配置训练参数
# 在这里，我们定义所有与模型、训练、评估相关的超参数。
#
# - **Model Arguments**: 指定要使用的模型，例如 `google/pegasus-base`。
# - **Data Arguments**: 定义数据路径和处理相关的参数，如最大序列长度。
# - **Training Arguments**: 控制训练过程的核心参数，包括学习率、批次大小、保存策略、混合精度等。
#   - `eval_only`: 如果设为 `True`，脚本将只加载模型并进行评估，不会进行训练。
#   - `resume_from_checkpoint`: 如果提供一个检查点路径，将从该断点继续训练。

# %%
import torch

class Args:
    # =========================================================================
    # 核心配置 (Core Configuration)
    # =========================================================================
    model_name_or_path: str = 'google/pegasus-base'
    output_dir: str = '/kaggle/working/pegasus-samsum-finetuned'
    
    # =========================================================================
    # 数据配置 (Data Configuration)
    # =========================================================================
    train_file: str = os.path.join(OUTPUT_DATA_DIR, 'train.csv')
    validation_file: str = os.path.join(OUTPUT_DATA_DIR, 'validation.csv')
    test_file: str = os.path.join(INPUT_DATA_DIR, 'test.csv') # 用于最终提交
    max_source_length: int = 512   # 输入对话的最大长度
    max_target_length: int = 128   # 输出摘要的最大长度
    
    # =========================================================================
    # 训练与评估配置 (Training & Evaluation Configuration)
    # =========================================================================
    # --- 模式控制 ---
    do_train: bool = True
    do_eval: bool = True
    eval_only: bool = False # 设为 True 则只进行评估
    resume_from_checkpoint: str = None # 例如: '/path/to/checkpoint'
    
    # --- 训练超参数 ---
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4 # 可根据显存/内存调整
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8 # 有效批大小 = batch_size * a_steps * num_devices
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    lr_scheduler_type: str = 'linear'
    warmup_steps: int = 50
    
    # --- 混合精度 ---
    # TPU 通常使用 bfloat16。Tesla T4 支持 bfloat16。P100 仅支持 float16。
    # 设置为 'bf16' 或 'fp16'。设为 None 则使用 fp32。
    mixed_precision: str = 'bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'fp16'
    
    # --- 日志与保存 ---
    logging_steps: int = 25
    eval_steps: int = 100 # 每N步在验证集上评估一次
    save_steps: int = 100
    save_total_limit: int = 2 # 最多保留几个检查点
    
    # --- 生成参数 (用于评估和预测) ---
    num_beams: int = 4
    
    # --- 随机种子 ---
    seed: int = 42

# 实例化配置
TRAINING_ARGS = Args()

# 如果 eval_only 为 True，则禁用训练
if TRAINING_ARGS.eval_only:
    TRAINING_ARGS.do_train = False
    # 如果只评估，通常需要指定一个模型断点来加载
    if TRAINING_ARGS.resume_from_checkpoint is None:
        print("警告: 'eval_only' 为 True，但未指定 'resume_from_checkpoint'。将尝试从 'output_dir' 加载模型。")
        TRAINING_ARGS.resume_from_checkpoint = TRAINING_ARGS.output_dir


# %% [markdown]
# # 步骤 3: 模型、分词器和数据处理
# 在这个部分，我们：
# 1.  加载预训练的 PEGASUS 模型和对应的分词器 (Tokenizer)。
# 2.  使用 `datasets` 库加载我们之前划分好的 CSV 文件。
# 3.  定义一个预处理函数，它会将对话和摘要文本转换成模型所需的 `input_ids`, `attention_mask` 和 `labels` 格式。
#     - **重要**: padding token 在 `labels` 中会被替换为 -100，这样它们就不会对损失计算产生影响。
# 4.  应用这个预处理函数到我们的数据集上。

# %%
import sys
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from datasets import load_dataset
import nltk
import numpy as np

# 尝试下载nltk工具包，如果失败则继续
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

print("\n" + "=" * 80)
print("步骤 3: 加载模型、分词器和处理数据")
print("=" * 80)


# --- 1. 加载模型和分词器 ---
print(f"加载模型配置: {TRAINING_ARGS.model_name_or_path}")
config = AutoConfig.from_pretrained(TRAINING_ARGS.model_name_or_path)

print(f"加载分词器: {TRAINING_ARGS.model_name_or_path}")
tokenizer = AutoTokenizer.from_pretrained(TRAINING_ARGS.model_name_or_path)

print(f"加载Seq2Seq模型: {TRAINING_ARGS.model_name_or_path}")
# 根据 resume_from_checkpoint 参数决定是加载预训练模型还是本地断点
model_load_path = TRAINING_ARGS.resume_from_checkpoint if TRAINING_ARGS.resume_from_checkpoint and os.path.exists(TRAINING_ARGS.resume_from_checkpoint) else TRAINING_ARGS.model_name_or_path
print(f"模型加载路径: {model_load_path}")
model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path, config=config)

# --- 2. 加载数据集 ---
data_files = {}
if TRAINING_ARGS.do_train:
    data_files["train"] = TRAINING_ARGS.train_file
if TRAINING_ARGS.do_eval:
    data_files["validation"] = TRAINING_ARGS.validation_file

raw_datasets = load_dataset("csv", data_files=data_files, cache_dir='/kaggle/working/cache')
print(f"\n加载的数据集: {raw_datasets}")


# --- 3. 定义预处理函数 ---
def preprocess_function(examples):
    inputs = examples["dialogue"]
    targets = examples["summary"]
    
    # 对输入进行分词
    model_inputs = tokenizer(
        inputs, 
        max_length=TRAINING_ARGS.max_source_length, 
        truncation=True, 
        padding="max_length"
    )

    # 对目标（摘要）进行分词
    # 使用 `text_target` 参数来为解码器进行分词
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets, 
            max_length=TRAINING_ARGS.max_target_length, 
            truncation=True, 
            padding="max_length"
        )

    # **Loss掩码**: 将label中由padding产生的位置替换为-100，这样损失函数会忽略它们
    labels["input_ids"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --- 4. 应用预处理 ---
if TRAINING_ARGS.do_train:
    train_dataset = raw_datasets["train"]
    train_dataset = train_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="处理训练数据集",
    )
    print(f"\n处理后的训练样本 (一个): {train_dataset[0]}")

if TRAINING_ARGS.do_eval:
    eval_dataset = raw_datasets["validation"]
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="处理验证数据集",
    )
    print(f"处理后的验证样本 (一个): {eval_dataset[0]}")

print("\n数据处理完成!")
print("=" * 80)


# %% [markdown]
# # 步骤 4: 训练与评估
# 这个部分是整个流程的核心。我们执行以下操作：
# 1.  **检测环境**: 判断当前是在 TPU 还是 GPU 环境下运行。Kaggle TPU Notebook 会设置 `TPU_NAME` 环境变量。
# 2.  **设置训练参数**:
#     - **TPU**: 使用 `torch_xla` 要求的特殊配置，如 `tpu_num_cores`。
#     - **GPU**: 使用标准的 `Seq2SeqTrainingArguments`。
# 3.  **定义评估指标**: 创建一个函数 `compute_metrics`，用于在评估过程中计算 ROUGE 分数。这是评估模型摘要质量的关键指标。
# 4.  **初始化 `Trainer`**: `Seq2SeqTrainer` 是 Hugging Face 提供的一个强大的工具，它封装了训练和评估的循环。我们传入模型、参数、数据集和评估函数来初始化它。
# 5.  **开始训练/评估**:
#     - 如果 `do_train` 为 `True`，调用 `trainer.train()`。
#     - 如果 `do_eval` 为 `True`，调用 `trainer.evaluate()`。
# 6.  **保存结果**: 训练完成后，模型、分词器和训练状态会自动保存在 `output_dir` 中。

# %%
from rouge_score import rouge_scorer
import numpy as np

# --- 1. 定义评估指标计算函数 ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # 将 -100 替换回 pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    # 解码预测和标签
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 使用nltk进行分词，为ROUGE计算做准备
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    # 计算 ROUGE 分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    
    result = {
        'rouge1': [], 'rouge2': [], 'rougeL': [], 'rougeLsum': [],
    }
    
    for pred, label in zip(decoded_preds, decoded_labels):
        scores = scorer.score(label, pred)
        result['rouge1'].append(scores['rouge1'].fmeasure)
        result['rouge2'].append(scores['rouge2'].fmeasure)
        result['rougeL'].append(scores['rougeL'].fmeasure)
        result['rougeLsum'].append(scores['rougeLsum'].fmeasure)

    result = {key: np.mean(val) * 100 for key, val in result.items()}
    
    # 添加生成长度的度量
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


# --- 2. 设置训练器 ---
# 检测是否在TPU环境
is_tpu = "TPU_NAME" in os.environ
if is_tpu:
    print("\n检测到 TPU 环境，将使用 torch_xla 进行训练。")
    import torch_xla.core.xla_model as xm
    # TPU 通常有 8 个核心
    tpu_cores = 8
else:
    print("\n未检测到 TPU 环境，将使用 GPU/CPU 进行训练。")

# 设置随机种子
set_seed(TRAINING_ARGS.seed)

# 定义数据整理器
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100 # 确保标签填充符被正确处理
)

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=TRAINING_ARGS.output_dir,
    do_train=TRAINING_ARGS.do_train,
    do_eval=TRAINING_ARGS.do_eval,
    per_device_train_batch_size=TRAINING_ARGS.per_device_train_batch_size,
    per_device_eval_batch_size=TRAINING_ARGS.per_device_eval_batch_size,
    gradient_accumulation_steps=TRAINING_ARGS.gradient_accumulation_steps,
    learning_rate=TRAINING_ARGS.learning_rate,
    weight_decay=TRAINING_ARGS.weight_decay,
    num_train_epochs=TRAINING_ARGS.num_train_epochs,
    lr_scheduler_type=TRAINING_ARGS.lr_scheduler_type,
    warmup_steps=TRAINING_ARGS.warmup_steps,
    
    # 日志和保存
    logging_dir=f"{TRAINING_ARGS.output_dir}/logs",
    logging_strategy="steps",
    logging_steps=TRAINING_ARGS.logging_steps,
    evaluation_strategy="steps",
    eval_steps=TRAINING_ARGS.eval_steps,
    save_strategy="steps",
    save_steps=TRAINING_ARGS.save_steps,
    save_total_limit=TRAINING_ARGS.save_total_limit,
    load_best_model_at_end=True, # 训练结束后加载最佳模型
    
    # 评估相关
    predict_with_generate=True,
    generation_num_beams=TRAINING_ARGS.num_beams,
    
    # 混合精度
    fp16=False if TRAINING_ARGS.mixed_precision == 'bf16' else (TRAINING_ARGS.mixed_precision == 'fp16'),
    bf16=TRAINING_ARGS.mixed_precision == 'bf16',
    
    # TPU 特定参数
    tpu_num_cores=tpu_cores if is_tpu else None,
)

# 初始化 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if TRAINING_ARGS.do_train else None,
    eval_dataset=eval_dataset if TRAINING_ARGS.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 3. 开始训练和评估 ---
print("\n" + "=" * 80)
print("步骤 4: 开始训练与评估")
print("=" * 80)

# 开始训练
if TRAINING_ARGS.do_train:
    print("\n--- 开始训练 ---")
    train_result = trainer.train(resume_from_checkpoint=TRAINING_ARGS.resume_from_checkpoint)
    
    # 保存最终的模型和训练指标
    trainer.save_model() # 保存最终的最佳模型
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("\n训练完成!")
else:
    print("\n跳过训练步骤 ('do_train' is False)。")

# 进行最终评估
if TRAINING_ARGS.do_eval:
    print("\n--- 开始最终评估 ---")
    if not TRAINING_ARGS.do_train:
        # 如果只评估，需要确保模型已经加载
        print(f"从 {model_load_path} 加载模型进行评估。")
    
    metrics = trainer.evaluate(max_length=TRAINING_ARGS.max_target_length, num_beams=TRAINING_ARGS.num_beams, metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print("\n评估完成!")

print("\n模型训练与评估流程结束。")
print(f"最终模型和结果保存在: {TRAINING_ARGS.output_dir}")
print("=" * 80)

# %% [markdown]
# # 步骤 5: 生成提交文件
# 训练和评估完成后，最后一步是使用我们微调好的模型对官方的 `test.csv` 文件进行预测，并生成符合比赛要求的 `submission.csv` 文件。
#
# 流程如下：
# 1.  加载测试数据集。
# 2.  定义一个预处理函数，只对输入的 `dialogue` 进行分词。
# 3.  使用 `trainer.predict()` 方法在测试集上进行推理。这个方法会自动处理所有细节，包括数据加载、模型推理和结果收集。
# 4.  解码模型生成的 token ID，得到最终的摘要文本。
# 5.  将ID和生成的摘要整理成 DataFrame，并保存为 `submission.csv`。

# %%
import pandas as pd
from tqdm.auto import tqdm

print("\n" + "=" * 80)
print("步骤 5: 生成提交文件")
print("=" * 80)

# 1. 加载测试数据
if not os.path.exists(TRAINING_ARGS.test_file):
    print(f"警告: 未找到测试文件 {TRAINING_ARGS.test_file}，跳过提交文件生成。")
else:
    test_df = pd.read_csv(TRAINING_ARGS.test_file)
    print(f"加载了 {len(test_df)} 条测试样本。")

    # 2. 批量推理以提高效率
    predictions = []
    batch_size = TRAINING_ARGS.per_device_eval_batch_size
    
    # 包装tqdm以显示进度条
    for i in tqdm(range(0, len(test_df), batch_size), desc="生成摘要"):
        batch_dialogues = test_df['dialogue'][i:i+batch_size].tolist()
        
        # 对话分词
        inputs = tokenizer(
            batch_dialogues,
            max_length=TRAINING_ARGS.max_source_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(trainer.model.device)
        
        # 使用模型生成摘要
        summary_ids = trainer.model.generate(
            **inputs,
            num_beams=TRAINING_ARGS.num_beams,
            max_length=TRAINING_ARGS.max_target_length,
            early_stopping=True
        )
        
        # 解码
        batch_preds = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        predictions.extend(batch_preds)

    # 3. 创建提交 DataFrame
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'summary': predictions
    })

    # 4. 保存为 CSV
    submission_path = '/kaggle/working/submission.csv'
    submission_df.to_csv(submission_path, index=False)

    print(f"\n提交文件已生成: {submission_path}")
    print("文件内容预览:")
    print(submission_df.head())
    print("=" * 80)