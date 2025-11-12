# Kaggle 2-GPU 训练设置指南

## 快速开始

### 1. 在Kaggle Notebook中设置

1. 创建新的Kaggle Notebook
2. 在右侧设置中选择 **"GPU T4 x2"** (2个Tesla T4 GPU)
3. 上传 `version12.py` 文件

### 2. 安装依赖

在第一个cell中运行：

```python
!pip install rouge-score
```

### 3. 准备数据

确保你的数据集已经添加到Notebook输入中：
- 训练数据: `/kaggle/input/nanogpt-fudannlp-cs-30040/train.csv`
- 测试数据: `/kaggle/input/nanogpt-fudannlp-cs-30040/test.csv`

### 4. 启动训练

#### 方法A: 使用torchrun（推荐）

```python
!torchrun --nproc_per_node=2 --standalone version12.py
```

#### 方法B: 使用启动脚本

首先创建 `train_ddp.py`，然后运行：

```python
!python train_ddp.py
```

## 代码修改说明

### 关键修改点

1. **Config类添加DDP支持**
```python
class Config:
    # ...
    ddp = True  # 启用DDP
    gradient_accumulation_steps = 8  # 从16改为8（因为有2个GPU）
```

2. **train()函数添加DDP初始化**
```python
def train():
    # DDP设置
    ddp = config.ddp and torch.cuda.device_count() > 1
    if ddp:
        init_process_group(backend=config.backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
```

3. **模型用DDP包装**
```python
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```

4. **只在主进程打印和保存**
```python
if master_process:
    print(f"Step {iter_num}: train loss {losses['train']:.4f}")
    torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
```

## 预期性能

### 训练速度
- **单GPU**: ~2000ms/iteration
- **2 GPU (DDP)**: ~1100ms/iteration
- **加速比**: ~1.8x

### 有效批次大小
```
batch_size × gradient_accumulation_steps × num_gpus
= 16 × 8 × 2
= 256
```

### 内存使用
- 每个GPU: ~14GB (Tesla T4有16GB)
- 两个GPU独立加载模型副本

## 验证DDP是否正常工作

训练开始时应该看到：

```
================================================================================
开始训练...
================================================================================
DDP训练: 2 GPUs

从 out-summarization 加载词表大小: 50257
模型初始化方式: gpt2
从OpenAI GPT-2加载: gpt2
...

开始训练循环...
总迭代次数: 500
批次大小: 16
梯度累积步数: 8
有效批次大小: 256
--------------------------------------------------------------------------------
```

关键指标：
- ✅ 显示 "DDP训练: 2 GPUs"
- ✅ 有效批次大小为 256
- ✅ 梯度累积步数为 8（不是16）

## 监控GPU使用

在另一个terminal或cell中运行：

```python
!watch -n 1 nvidia-smi
```

应该看到两个GPU都在使用，利用率接近100%。

## 常见问题

### Q1: 只看到1个GPU在工作

**原因**: Kaggle设置中没有选择 "GPU T4 x2"

**解决**: 
1. 点击右侧 "Accelerator"
2. 选择 "GPU T4 x2"
3. 重启Notebook

### Q2: "RuntimeError: Address already in use"

**原因**: 端口被占用

**解决**: 指定不同端口
```python
!torchrun --nproc_per_node=2 --master_port=29501 --standalone version12.py
```

### Q3: NCCL通信错误

**解决**: 在代码开始添加环境变量
```python
import os
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_P2P_DISABLE'] = '1'
```

### Q4: 想要单GPU训练

**解决**: 修改Config
```python
class Config:
    ddp = False  # 禁用DDP
    gradient_accumulation_steps = 16  # 恢复为16
```

## 完整的Kaggle Notebook示例

```python
# Cell 1: 安装依赖
!pip install rouge-score

# Cell 2: 检查GPU
import torch
print(f"可用GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Cell 3: 启动训练
!torchrun --nproc_per_node=2 --standalone version12.py

# Cell 4: 查看结果
!ls -lh out-summarization/

# Cell 5: 运行评估（在训练完成后）
# 注意：评估应该在单GPU上运行
import version12
version12.config.ddp = False  # 禁用DDP进行评估
version12.evaluate()

# Cell 6: 生成提交文件
version12.predict_test_set_fast()
```

## 性能优化建议

1. **调整批次大小**: 如果GPU内存充足，可以增加 `batch_size`
2. **使用混合精度**: 已启用 `dtype='float16'`
3. **编译模型**: 已启用 `compile=True` (PyTorch 2.0+)
4. **KV Cache**: 推理时使用 `generate_with_kv_cache` 加速

## 成本考虑

Kaggle免费提供：
- 每周30小时GPU时间
- 2个T4 GPU同时使用计为2倍时间
- 训练500 iterations约需1-2小时（使用2个GPU）

## 下一步

训练完成后：
1. 检查 `out-summarization/ckpt.pt` 是否生成
2. 运行评估查看ROUGE分数
3. 生成测试集预测
4. 下载 `submission.csv` 提交

祝训练顺利！🚀
