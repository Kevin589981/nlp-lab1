# %% [markdown]
# # Step 1: Install Dependencies
# 首先，安装完成任务所需的库，包括 `transformers`、`datasets` 用于模型和数据处理，`rouge_score` 和 `evaluate` 用于评估。

# %%
# !pip install "transformers[torch]" datasets evaluate rouge_score sacrebleu --quiet
# !pip install accelerate -U --quiet

# %% [markdown]
# # Step 2: Data Preprocessing
# 这部分代码与您提供的原始代码逻辑一致。
# 
# - 从 Kaggle 输入的 `train.csv` 读取数据。
# - 随机划分 95% 的数据作为训练集，5% 作为验证集。
# - 将划分后的数据保存到 `/kaggle/working/data/samsum/` 目录下，以便后续加载。

# %%
import pandas as pd
import os
import argparse
from dataclasses import dataclass, field

# 确保 argparse 在 notebook 环境中能正常工作
import sys
if 'ipykernel' in sys.modules:
    sys.argv = ['']

print("\n" + "=" * 80)
print("准备 SAMSum 数据集划分")
print("=" * 80)

# 读取原始CSV文件
# 注意：请确保已将比赛数据添加到 notebook 的输入目录 /kaggle/input/
input_csv = '/kaggle/input/nanogpt-fudannlp-cs-30040/train.csv'
test_csv_path = '/kaggle/input/nanogpt-fudannlp-cs-30040/test.csv'
print(f"\n读取训练数据: {input_csv}")

df = pd.read_csv(input_csv)

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df = df[:100]

# 计算划分点（5%作为验证集）
total_samples = len(df)
val_size = int(total_samples * 0.05)
train_size = total_samples - val_size

print(f"总样本数: {total_samples}")
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
# # Step 3: Model Fine-tuning with PEGASUS-Base
# 
# 这是重写后的核心部分，使用 `PEGASUS-Base` 模型和 Hugging Face `Trainer` API。
# 
# ## 主要改动和特性：
# 
# 1.  **模型和分词器**:
#     *   加载 `google/pegasus-base` 预训练模型和对应的 `PegasusTokenizer`。
# 
# 2.  **数据加载与处理**:
#     *   使用 `datasets` 库从上一步生成的 CSV 文件中加载数据。
#     *   定义 `preprocess_function` 来对对话（`dialogue`）和摘要（`summary`）进行分词。这是 Seq2Seq 模型的标准做法。
# 
# 3.  **训练框架**:
#     *   采用 `Seq2SeqTrainer` 和 `Seq2SeqTrainingArguments`，这是 Hugging Face 为序列到序列任务（如摘要）设计的标准高级 API。
#     *   **TPU 支持**: `Trainer` API 内部自动处理 PyTorch/XLA，无需手动编写 TPU 特定代码。
#     *   **混合精度**: 通过在 `Seq2SeqTrainingArguments` 中设置 `bf16=True` 来启用 bfloat16 训练。
# 
# 4.  **评估逻辑**:
#     *   定义 `compute_metrics` 函数，该函数在每次评估时被 `Trainer` 调用。
#     *   函数内部对模型生成的 `predictions` 和真实的 `labels` 进行解码，然后使用 `evaluate.load('rouge')` 计算 ROUGE 分数，这与原始评估逻辑的目标完全一致。
# 
# 5.  **参数化**:
#     *   使用 `argparse`（并兼容 notebook）来配置 `eval_only` 和 `resume_from_checkpoint` 参数。你可以通过修改 `parser.parse_args()` 中的参数来控制脚本行为。
# 
# 6.  **模型导出**:
#     *   训练完成后，`Trainer` 会自动将模型权重（`pytorch_model.bin`）、配置文件等保存在指定的 `output_dir` 中。
#     *   这个目录包含了完整的模型，可以被 `transformers` 库在任何支持 PyTorch 的环境（包括 P100 GPU）中加载。
# 
# 7.  **Loss 掩码**:
#     *   在数据预处理和 `DataCollatorForSeq2Seq` 中，用于 padding 的 label token 会被自动设置为 `-100`。PyTorch 的交叉熵损失函数会忽略这些值为 `-100` 的标签，从而实现了正确的 loss 掩码。

# %%
import torch
import numpy as np
import evaluate # Hugging Face's new evaluation library
from datasets import load_dataset
from transformers import (
    PegasusForConditionalGeneration,
    PegasusTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)

# --- Configuration using argparse ---
# We use argparse to handle parameters like eval_only and resume
parser = argparse.ArgumentParser(description="Fine-tune PEGASUS-Base on SAMSum dataset.")
parser.add_argument("--model_name", type=str, default="google/pegasus-base", help="Name of the pretrained model.")
parser.add_argument("--output_dir", type=str, default="/kaggle/working/pegasus-samsum-finetuned", help="Directory to save the model and results.")
parser.add_argument("--train_file", type=str, default=train_csv_path, help="Path to the training CSV file.")
parser.add_argument("--validation_file", type=str, default=val_csv_path, help="Path to the validation CSV file.")
parser.add_argument("--eval_only", action="store_true", help="If set, only run evaluation on the validation set.")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to a checkpoint to resume training from.")

# --- Training Hyperparameters ---
parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per device for training.")
parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per device for evaluation.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of steps for gradient accumulation.")
parser.add_argument("--learning_rate", type=float, default=5.6e-5, help="Learning rate.")
parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps.")
parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
parser.add_argument("--max_input_length", type=int, default=1024, help="Maximum length of input sequences.")
parser.add_argument("--max_target_length", type=int, default=128, help="Maximum length of target (summary) sequences.")
parser.add_argument("--logging_steps", type=int, default=100, help="Log every N steps.")
parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps.")
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps.")

# Parse arguments
# In a script, you would use: args = parser.parse_args()
# For Kaggle notebook compatibility, we manually set the args:
args = parser.parse_args([
    '--num_train_epochs', '1', # Use 1 epoch for a quick demonstration
    '--per_device_train_batch_size', '1', # For TPU v5e-8, you can increase this to 8 or 16
    '--per_device_eval_batch_size', '2',
    '--gradient_accumulation_steps', '8', # Effective batch size = 1 * 8 * 8 (cores) = 64
])
print("Script arguments:", args)


# --- 1. Load Tokenizer and Model ---
print("\n" + "="*80)
print(f"Loading tokenizer and model for '{args.model_name}'...")
# The tokenizer will be used to convert text to numbers (tokens)
tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
# The model is the neural network architecture we will fine-tune
model = PegasusForConditionalGeneration.from_pretrained(args.model_name)
print("Tokenizer and model loaded successfully.")
print("="*80)


# --- 2. Load and Preprocess Dataset ---
print("\n" + "="*80)
print("Loading and preprocessing data...")
# Load data from the CSV files we created earlier
raw_datasets = load_dataset('csv', data_files={'train': args.train_file, 'validation': args.validation_file})

def preprocess_function(examples):
    """Tokenizes the dialogue and summary."""
    # The `text_target` argument is for the labels/summary.
    inputs = tokenizer(examples['dialogue'], max_length=args.max_input_length, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary'], max_length=args.max_target_length, truncation=True, padding="max_length")

    # The DataCollator will handle creating the decoder_input_ids
    # Padded labels are automatically set to -100 to be ignored in loss calculation.
    inputs['labels'] = labels['input_ids']
    return inputs

# Apply the preprocessing function to all splits of the dataset
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
print("Data preprocessing complete.")
print("Sample of tokenized data:", tokenized_datasets["train"][0])
print("="*80)


# --- 3. Define Evaluation Metric (ROUGE) ---
# The logic here is consistent with the original script's goal: calculate ROUGE scores.
rouge = evaluate.load('rouge')

def compute_metrics(eval_pred):
    """Computes ROUGE scores for a batch of predictions."""
    predictions, labels = eval_pred
    
    # Decode predictions and labels back to text
    # The tokenizer.decode function converts token IDs back to strings
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # ROUGE expects a newline after each sentence
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]
    
    # Compute ROUGE scores
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # Extract F-measures
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length to metrics
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}


# --- 4. Configure Training ---
# Data collator pads inputs and labels dynamically for each batch
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Define training arguments
# This object contains all the settings for the training run
training_args = Seq2SeqTrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    
    # Logging, evaluation, and saving settings
    logging_dir=f"{args.output_dir}/logs",
    logging_steps=args.logging_steps,
    evaluation_strategy="steps",
    eval_steps=args.eval_steps,
    save_strategy="steps",
    save_steps=args.save_steps,
    save_total_limit=3, # Only keep the last 3 checkpoints
    
    # Enable bfloat16 for TPU training
    bf16=True, 
    
    # Enable generation for evaluation to calculate ROUGE
    predict_with_generate=True,
    generation_max_length=args.max_target_length,
    
    # Load best model at the end
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    
    # Other settings
    push_to_hub=False,
    report_to="none", # Disable wandb/tensorboard logging for simplicity
)

# Initialize the Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)


# --- 5. Run Training or Evaluation ---
print("\n" + "="*80)
if args.eval_only:
    print("Running evaluation only...")
    # Make sure to specify a checkpoint to evaluate
    if args.resume_from_checkpoint is None:
        raise ValueError("Must provide a checkpoint path via --resume_from_checkpoint for eval_only mode.")
    metrics = trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    print("Evaluation metrics:", metrics)
else:
    print("Starting fine-tuning...")
    # The `resume_from_checkpoint` argument can be a boolean (True) to auto-find the last
    # checkpoint in output_dir, or a string path to a specific checkpoint.
    resume_path = args.resume_from_checkpoint if args.resume_from_checkpoint is not None else False
    train_result = trainer.train(resume_from_checkpoint=resume_path)
    
    # Save training metrics and final model
    trainer.save_model() # This saves the final model, tokenizer, and config
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("Fine-tuning complete.")
    
print("="*80)


# %% [markdown]
# # Step 4: Generate Submission File
# 
# 这一部分使用微调好的模型对官方的 `test.csv` 文件进行推理，并生成 `submission.csv` 文件。
# 
# - 加载 `test.csv`。
# - 遍历每一条对话。
# - 使用 `trainer.predict` 方法高效地生成摘要。
# - 将结果格式化并保存到 CSV 文件中。

# %%
import pandas as pd
from tqdm.auto import tqdm

print("\n" + "=" * 80)
print("Generating summaries for the test set...")

# Load the test data
test_df = pd.read_csv(test_csv_path)
test_dialogues = test_df['dialogue'].tolist()

# Create a prediction dataset
from datasets import Dataset
test_dataset = Dataset.from_dict({"dialogue": test_dialogues})

# Tokenize the test data
def tokenize_test_data(examples):
    return tokenizer(examples['dialogue'], max_length=args.max_input_length, truncation=True, padding="max_length")

tokenized_test_dataset = test_dataset.map(tokenize_test_data, batched=True)

# Generate predictions using the trainer
print("Running prediction on the test set...")
test_predictions = trainer.predict(tokenized_test_dataset)

# Decode the generated summaries
print("Decoding predictions...")
decoded_summaries = tokenizer.batch_decode(test_predictions.predictions, skip_special_tokens=True)

# Clean up the summaries
cleaned_summaries = [s.strip() for s in decoded_summaries]

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'summary': cleaned_summaries
})

# Save the submission file
submission_path = os.path.join(args.output_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)

print(f"\nSubmission file saved to: {submission_path}")
print("Sample of the submission file:")
print(submission_df.head())
print("=" * 80)