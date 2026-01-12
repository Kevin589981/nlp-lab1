# %% [markdown]
# # 使用 FLAN-T5 和 PEFT(LoRA) 高效微调摘要模型
#
# 本脚本使用 Hugging Face `transformers`、`peft` 和 `datasets` 库重构了原始代码。
#
# ## 主要改进:
# 1.  **模型架构**: 从自定义的 GPT-2 (Decoder-Only) 更换为 `google/flan-t5-base` (Encoder-Decoder)，更适合摘要任务。
# 2.  **高效微调**: 集成 PEFT (Parameter-Efficient Fine-Tuning) 中的 LoRA 方法，仅训练模型一小部分参数，大幅降低显存占用和训练时间。
# 3.  **现代化框架**: 全面采用 `transformers.Trainer` API，简化了训练循环、评估、日志记录和模型保存等流程。
# 4.  **数据处理**: 使用 `datasets` 库加载和处理数据，更加高效和规范。
# 5.  **精确的 Loss Masking**: 利用 `DataCollatorForSeq2Seq` 自动处理标签填充（padding），确保损失函数只在有效标签上计算。
# 6.  **配置**: 所有关键参数（包括 `eval_only` 和 `resume_from_checkpoint`）都集中在 `TrainingConfig` 类中，方便管理。
# 7.  **混合精度**: 默认启用 `fp16` (float16) 进行训练，以加速并减少显存。

# %%
# 安装必要的库
!pip install -q evaluate rouge-score peft bitsandbytes

# %%
import pandas as pd
import os
import torch
import numpy as np
import warnings
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate # Hugging Face 的 evaluate 库

# 忽略一些不必要的警告
warnings.filterwarnings("ignore")
os.environ["WANDB_DISABLED"] = "true" # 禁用 wandb

# %% [markdown]
# ### 第1步：数据预处理
# 这部分代码与原始脚本逻辑相同，读取`train.csv`并划分为95%的训练集和5%的验证集。

# %%
print("\n" + "=" * 80)
print("准备SAMSum数据集划分")
print("=" * 80)

# Kaggle环境下的输入路径
input_csv = '/kaggle/input/nanogpt-fudannlp-cs-30040/train.csv'

# 如果输入文件不存在，则创建一个虚拟文件用于本地测试
if not os.path.exists(input_csv):
    print(f"警告: 未找到输入文件 {input_csv}。将创建一个用于演示的虚拟文件。")
    os.makedirs('/kaggle/input/nanogpt-fudannlp-cs-30040', exist_ok=True)
    dummy_data = {
        'id': [f'id_{i}' for i in range(100)],
        'dialogue': ["A: Hi! B: Hello. How are you? A: I'm fine, thanks." for _ in range(100)],
        'summary': ["A and B greeted each other." for _ in range(100)]
    }
    pd.DataFrame(dummy_data).to_csv(input_csv, index=False)


print(f"\n读取数据: {input_csv}")
df = pd.read_csv(input_csv)
total_samples = len(df)
print(f"总样本数: {total_samples}")

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 计算划分点
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

# 保存文件
train_csv_path = os.path.join(output_dir, 'train.csv')
val_csv_path = os.path.join(output_dir, 'validation.csv')
train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)

print(f"\n训练集已保存: {train_csv_path}")
print(f"验证集已保存: {val_csv_path}")
print("\n数据集划分完成！")
print("=" * 80)


# %% [markdown]
# ### 第2步：配置参数
# 将所有训练和模型相关的参数集中在一个配置类中。

# %%
@dataclass
class TrainingConfig:
    # 模型和分词器配置
    model_name_or_path: str = "google/flan-t5-base" # T5-base 约250M参数
    tokenizer_name_or_path: str = "google/flan-t5-base"
    
    # 数据配置
    data_path: str = output_dir
    source_prefix: str = "summarize: " # T5/FLAN-T5 推荐为任务添加前缀
    source_column: str = "dialogue"
    target_column: str = "summary"
    max_source_length: int = 512 # 输入对话的最大长度
    max_target_length: int = 128 # 输出摘要的最大长度
    
    # PEFT (LoRA) 配置
    use_peft: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # 训练控制参数
    output_dir: str = "/kaggle/working/flan-t5-samsum-lora" # 模型权重和结果的输出目录
    eval_only: bool = False # 是否只进行评估而不训练
    resume_from_checkpoint: bool = False # 是否从上一个断点继续训练
    
    # Seq2SeqTrainingArguments 参数
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8 # 有效批次大小 = 4 * 8 = 32
    learning_rate: float = 3e-4 # LoRA 微调通常使用比全量微调稍大的学习率
    num_train_epochs: int = 3
    logging_steps: int = 10 # 每10步记录一次日志
    eval_steps: int = 50 # 每50步评估一次
    save_steps: int = 50 # 每50步保存一次检查点
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    save_total_limit: int = 2 # 只保留最新的2个检查点
    load_best_model_at_end: bool = True # 训练结束后加载最佳模型
    predict_with_generate: bool = True # 在评估时使用generate生成文本
    metric_for_best_model: str = "eval_rougeL" # 使用ROUGE-L作为衡量最佳模型的指标
    fp16: bool = True # 必须使用 float16


config = TrainingConfig()

# %% [markdown]
# ### 第3步：加载模型、分词器和数据

# %%
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name_or_path)

# 加载模型
# `torch_dtype=torch.float16` 确保模型以fp16加载
# `device_map="auto"` 会自动将模型分配到可用的设备上（例如GPU）
model = AutoModelForSeq2SeqLM.from_pretrained(
    config.model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 如果配置使用PEFT (LoRA)
if config.use_peft:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q", "v"] # 在T5的q和v投影矩阵上应用LoRA
    )
    model = get_peft_model(model, peft_config)
    print("\nPEFT (LoRA) 模型已启用:")
    model.print_trainable_parameters()


# 加载数据集
data_files = {
    "train": os.path.join(config.data_path, "train.csv"),
    "validation": os.path.join(config.data_path, "validation.csv"),
}
# 使用 datasets 库加载CSV文件
dataset = load_dataset("csv", data_files=data_files)

print("\n数据集结构:")
print(dataset)
print("\n训练集样本示例:")
print(dataset["train"][0])


# %% [markdown]
# ### 第4步：数据预处理函数和评估指标

# %%
def preprocess_function(examples):
    """数据预处理函数，用于对数据进行分词"""
    inputs = [config.source_prefix + doc for doc in examples[config.source_column]]
    model_inputs = tokenizer(inputs, max_length=config.max_source_length, truncation=True, padding="max_length")

    # 对标签进行分词
    labels = tokenizer(text_target=examples[config.target_column], max_length=config.max_target_length, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 使用 .map() 方法对整个数据集进行预处理
tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
print("\n分词后数据集的列:", tokenized_dataset["train"].column_names)


# 加载 ROUGE 评估指标
rouge = evaluate.load("rouge")

def compute_metrics(eval_pred):
    """计算评估指标的函数"""
    predictions, labels = eval_pred
    # 将模型生成的 token ID 解码为文本
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 将标签中的 -100 替换为 padding token ID，以便解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # ROUGE期望每句摘要后都有一个换行符
    decoded_preds = ["\n".join(pred.strip().split()) for pred in decoded_preds]
    decoded_labels = ["\n".join(label.strip().split()) for label in decoded_labels]

    # 计算 ROUGE 分数
    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
    # 提取主要的 f-measure 分数
    result = {key: value * 100 for key, value in result.items()}
    
    # 添加生成文本的平均长度
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

# %% [markdown]
# ### 第5步：设置 Trainer 并开始训练

# %%
# 定义数据整理器，用于动态填充批次中的数据
# `label_pad_token_id=-100` 是关键，它能确保填充的标签在计算损失时被忽略
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

# 定义训练参数
training_args = Seq2SeqTrainingArguments(
    output_dir=config.output_dir,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    predict_with_generate=config.predict_with_generate,
    fp16=config.fp16,
    learning_rate=config.learning_rate,
    num_train_epochs=config.num_train_epochs,
    logging_strategy="steps",
    logging_steps=config.logging_steps,
    evaluation_strategy=config.evaluation_strategy,
    eval_steps=config.eval_steps,
    save_strategy=config.save_strategy,
    save_steps=config.save_steps,
    save_total_limit=config.save_total_limit,
    load_best_model_at_end=config.load_best_model_at_end,
    metric_for_best_model=config.metric_for_best_model,
    report_to="none", # 禁用 wandb/tensorboard
)

# 初始化 Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    compute_metrics=compute_metrics,
)

# 开始训练
if not config.eval_only:
    print("\n" + "=" * 80)
    print("🚀 开始模型微调...")
    print("=" * 80)
    # 如果 resume_from_checkpoint 为 True，Trainer 会自动查找最新的检查点
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    
    # 保存最终的 LoRA 适配器权重
    final_model_path = os.path.join(config.output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"\n✅ 训练完成！最终模型已保存至: {final_model_path}")
else:
    print("\n" + "=" * 80)
    print("🔍 eval_only=True，跳过训练，直接进行评估。")
    print("=" * 80)


# 评估模型
print("\n" + "=" * 80)
print("📈 开始最终评估...")
print("=" * 80)
eval_results = trainer.evaluate()
print("\n最终评估结果:")
print(eval_results)


# %% [markdown]
# ### 第6步：生成测试集结果并导出

# %%
def predict_on_test_set(config):
    """
    加载微调后的模型，对测试集进行预测，并生成 submission.csv 文件。
    """
    print("\n" + "=" * 80)
    print("📦 开始对测试集进行推理并生成提交文件...")
    print("=" * 80)

    # 加载测试数据
    test_csv_path = '/kaggle/input/nanogpt-fudannlp-cs-30040/test.csv'
    if not os.path.exists(test_csv_path):
        print(f"警告: 未找到测试文件 {test_csv_path}。将创建一个用于演示的虚拟文件。")
        dummy_test_data = {
            'id': [f'test_{i}' for i in range(50)],
            'dialogue': ["A: What's the plan for tonight? B: Let's go to the movies." for _ in range(50)],
        }
        pd.DataFrame(dummy_test_data).to_csv(test_csv_path, index=False)
        
    test_df = pd.read_csv(test_csv_path)
    print(f"加载了 {len(test_df)} 条测试样本。")

    # 加载最佳模型进行推理
    # `trainer.model` 已经在训练结束后加载了最佳模型
    # 如果是 eval_only 模式，需要手动加载
    if config.eval_only:
        from peft import PeftModel
        from transformers import AutoModelForSeq2SeqLM
        
        # 找到最佳检查点
        best_checkpoint_path = trainer.state.best_model_checkpoint
        if best_checkpoint_path is None:
             print("错误：找不到最佳模型检查点，请先运行训练。")
             return

        print(f"从 {best_checkpoint_path} 加载基础模型...")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            config.model_name_or_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("加载 LoRA 适配器...")
        inference_model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
        inference_model = inference_model.merge_and_unload() # 合并权重以便于推理
        inference_model.eval()
    else:
        inference_model = trainer.model
        if config.use_peft:
            # 如果使用的是PEFT模型，最好合并权重以加速推理
            try:
                inference_model = inference_model.merge_and_unload()
            except:
                print("无法自动合并LoRA权重，将使用适配器模式进行推理。")
        inference_model.eval()


    # 准备生成结果
    results = []
    
    from torch.utils.data import DataLoader, Dataset

    class InferenceDataset(Dataset):
        def __init__(self, df, tokenizer, config):
            self.df = df
            self.tokenizer = tokenizer
            self.config = config

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            dialogue = self.df.iloc[idx][self.config.source_column]
            input_text = self.config.source_prefix + dialogue
            return self.tokenizer(input_text, return_tensors="pt", max_length=self.config.max_source_length, truncation=True)
            
    inference_dataset = InferenceDataset(test_df, tokenizer, config)
    # 使用 DataLoader 进行批处理以加速
    data_loader = DataLoader(inference_dataset, batch_size=config.per_device_eval_batch_size)

    print("\n开始生成摘要...")
    for batch in data_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(inference_model.device)
            attention_mask = batch['attention_mask'].to(inference_model.device)
            
            outputs = inference_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=config.max_target_length,
                num_beams=4, # 使用束搜索
                early_stopping=True
            )
            
            # 解码
            summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            # 找到批次对应的ID
            start_index = len(results)
            ids = test_df['id'][start_index : start_index + len(summaries)].tolist()
            
            for sample_id, summary in zip(ids, summaries):
                results.append({'id': sample_id, 'summary': summary.strip()})
        
        if len(results) % 100 == 0:
            print(f"已处理 {len(results)} / {len(test_df)}...")

    # 保存到 submission.csv
    submission_df = pd.DataFrame(results)
    submission_path = os.path.join(config.output_dir, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✅ 推理完成！提交文件已保存至: {submission_path}")
    print("=" * 80)

# 调用预测函数
predict_on_test_set(config)