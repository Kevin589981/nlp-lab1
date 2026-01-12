# Kaggle 多GPU训练指南 (使用Accelerate)

## 为什么使用Accelerate？

在Kaggle Notebook中，使用 `accelerate` 库比手动配置DDP更简单：
- ✅ 单文件即可，无需额外的启动脚本
- ✅ 自动处理多GPU分布
- ✅ 自动处理混合精度训练
- ✅ 代码更简洁，易于维护

## 快速开始

### 1. Kaggle Notebook设置

1. 创建新的Kaggle Notebook
2. 在右侧设置中选择 **"GPU T4 x2"** (2个Tesla T4 GPU)
3. 添加数据集输入

### 2. 安装依赖

在第一个cell中运行：

```python
!pip install rouge-score accelerate
```

### 3. 上传并运行代码

直接上传 `version12.py` 文件，然后在cell中运行：

```python
# 方法1: 直接运行（推荐）
!python version12.py
```

或者使用accelerate命令：

```python
# 方法2: 使用accelerate launch
!accelerate launch --multi_gpu version12.py
```

就这么简单！Accelerate会自动检测并使用所有可用的GPU。

## 代码说明

### 自动多GPU检测

代码会自动检测GPU数量：

```python
use_accelerate = config.use_accelerate and ACCELERATE_AVAILABLE and torch.cuda.device_count() > 1
```

- 如果有2个GPU → 自动使用多GPU训练
- 如果只有1个GPU → 自动降级为单GPU训练
- 无需修改任何代码！

### 配置参数

在 `Config` 类中：

```python
class Config:
    # 多GPU配置
    use_accelerate = True       # 启用accelerate（自动检测）
    
    # 批次配置
    batch_size = 16             # 每个GPU的批次大小
    gradient_accumulation_steps = 16  # 梯度累积步数
    
    # 有效批次大小 = 16 × 16 × 2 = 512 (使用2个GPU时)
```

### 关键修改点

1. **导入accelerate**
```python
from accelerate import Accelerator
```

2. **初始化accelerator**
```python
accelerator = Accelerator(
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    mixed_precision='fp16'
)
```

3. **准备模型和优化器**
```python
model, optimizer = accelerator.prepare(model, optimizer)
```

4. **训练循环**
```python
with accelerator.accumulate(model):
    logits, loss = model(X, Y)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

## 完整的Kaggle Notebook示例

```python
# ========== Cell 1: 安装依赖 ==========
!pip install rouge-score accelerate

# ========== Cell 2: 检查GPU ==========
import torch
print(f"可用GPU数量: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# ========== Cell 3: 运行训练 ==========
# 直接运行，accelerate会自动处理多GPU
!python version12.py

# ========== Cell 4: 查看训练结果 ==========
!ls -lh out-summarization/
!tail -n 50 out-summarization/train.log  # 如果有日志文件

# ========== Cell 5: 运行评估 ==========
# 评估在单GPU上运行即可
import sys
sys.path.append('.')
import version12

version12.config.use_accelerate = False  # 评估时禁用多GPU
version12.evaluate()

# ========== Cell 6: 生成提交文件 ==========
version12.predict_test_set_fast()

# ========== Cell 7: 下载结果 ==========
from IPython.display import FileLink
FileLink('out-summarization/submission.csv')
```

## 预期性能

### 训练速度对比

| 配置 | 时间/iteration | 加速比 |
|------|---------------|--------|
| 单GPU (T4) | ~2000ms | 1.0x |
| 2 GPU (T4) | ~1100ms | 1.8x |

### 有效批次大小

```
单GPU: 16 × 16 × 1 = 256
双GPU: 16 × 16 × 2 = 512
```

### 训练时间估算

- 500 iterations × 1.1秒 ≈ 9分钟 (使用2个GPU)
- 500 iterations × 2.0秒 ≈ 17分钟 (使用1个GPU)

## 验证多GPU是否工作

训练开始时应该看到：

```
================================================================================
开始训练...
================================================================================
使用Accelerate多GPU训练: 2 GPUs
混合精度: fp16

模型初始化方式: gpt2
...

开始训练循环...
总迭代次数: 500
批次大小: 16
梯度累积步数: 16
GPU数量: 2
有效批次大小: 512
--------------------------------------------------------------------------------
```

关键指标：
- ✅ 显示 "使用Accelerate多GPU训练: 2 GPUs"
- ✅ GPU数量: 2
- ✅ 有效批次大小: 512

## 监控GPU使用

在另一个cell中运行：

```python
!nvidia-smi
```

应该看到两个GPU都在使用，显存占用相似。

## 常见问题

### Q1: 显示"accelerate库未安装"

**解决**: 运行安装命令
```python
!pip install accelerate
```

### Q2: 只使用了1个GPU

**原因**: Kaggle设置中没有选择 "GPU T4 x2"

**解决**: 
1. 点击右侧 "Accelerator"
2. 选择 "GPU T4 x2"（不是单个GPU）
3. 保存并重启Notebook

### Q3: 想要单GPU训练

**解决**: 修改Config
```python
class Config:
    use_accelerate = False  # 禁用accelerate
```

或者在Kaggle设置中选择单个GPU。

### Q4: 内存不足 (OOM)

**解决**: 减小批次大小
```python
class Config:
    batch_size = 8  # 从16减到8
    # 或者
    gradient_accumulation_steps = 8  # 从16减到8
```

### Q5: 训练速度没有提升

**检查**:
1. 确认使用了2个GPU（看训练开始的日志）
2. 运行 `!nvidia-smi` 确认两个GPU都在工作
3. 检查是否有数据加载瓶颈（增加 `num_workers`）

## Accelerate vs 手动DDP

| 特性 | Accelerate | 手动DDP |
|------|-----------|---------|
| 代码复杂度 | 简单 | 复杂 |
| 启动方式 | 直接运行 | 需要torchrun |
| 文件数量 | 单文件 | 需要启动脚本 |
| Kaggle兼容性 | ✅ 完美 | ⚠️ 需要额外配置 |
| 混合精度 | 自动处理 | 手动配置 |
| 梯度累积 | 自动处理 | 手动实现 |

**结论**: 在Kaggle上使用Accelerate更简单、更可靠！

## 高级配置

### 自定义accelerate配置

如果需要更多控制，可以创建配置文件：

```python
# 在notebook cell中
!accelerate config

# 然后按提示选择：
# - 多GPU训练
# - 混合精度: fp16
# - 梯度累积步数: 16
```

### 使用配置文件启动

```python
!accelerate launch --config_file accelerate_config.yaml version12.py
```

## 性能优化建议

1. **批次大小**: 如果GPU内存充足，增加 `batch_size` 到 24 或 32
2. **数据加载**: 增加 `num_workers` 到 4-8
3. **混合精度**: 已自动启用 fp16
4. **梯度检查点**: 对于更大的模型，可以启用梯度检查点节省内存

## 成本考虑

Kaggle免费额度：
- 每周30小时GPU时间
- 2个T4同时使用 = 2倍消耗
- 本训练约需0.3小时（使用2个GPU）
- 可以训练约50次

## 下一步

1. ✅ 确认2个GPU都在工作
2. ✅ 监控训练loss下降
3. ✅ 评估ROUGE分数
4. ✅ 生成测试集预测
5. ✅ 提交结果

祝训练顺利！🚀

## 参考资源

- [Accelerate文档](https://huggingface.co/docs/accelerate)
- [Kaggle GPU文档](https://www.kaggle.com/docs/efficient-gpu-usage)
- [混合精度训练](https://pytorch.org/docs/stable/amp.html)
