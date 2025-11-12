# 快速开始 - Kaggle 2-GPU训练

## 一分钟上手

### 1. Kaggle设置
- 选择 **GPU T4 x2** (2个GPU)
- 添加数据集输入

### 2. 安装依赖
```python
!pip install rouge-score accelerate
```

### 3. 运行训练
```python
!python version12.py
```

就这么简单！代码会自动检测并使用2个GPU。

## 验证多GPU工作

训练开始时应该看到：
```
使用Accelerate多GPU训练: 2 GPUs
GPU数量: 2
有效批次大小: 512
```

## 性能对比

| GPU数量 | 训练时间 | 加速比 |
|---------|---------|--------|
| 1个 T4  | ~17分钟 | 1.0x   |
| 2个 T4  | ~9分钟  | 1.8x   |

## 完整Notebook示例

```python
# Cell 1: 安装
!pip install rouge-score accelerate

# Cell 2: 训练
!python version12.py

# Cell 3: 评估（训练完成后）
import version12
version12.config.use_accelerate = False
version12.evaluate()

# Cell 4: 生成提交
version12.predict_test_set_fast()
```

## 常见问题

**Q: 只看到1个GPU？**
A: 确认Kaggle设置选择了 "GPU T4 x2"

**Q: 想用单GPU？**
A: 设置 `config.use_accelerate = False`

**Q: 内存不足？**
A: 减小 `batch_size` 或 `gradient_accumulation_steps`

## 关键修改

相比原版本，主要改动：

1. **添加accelerate支持**
   - 自动多GPU分布
   - 自动混合精度
   - 无需手动配置

2. **简化启动**
   - 单文件运行
   - 无需torchrun
   - 自动检测GPU

3. **保持兼容**
   - 单GPU自动降级
   - 所有功能不变
   - checkpoint格式相同

## 技术细节

### Accelerate vs DDP

| 特性 | Accelerate | 手动DDP |
|------|-----------|---------|
| 启动 | `python script.py` | `torchrun --nproc_per_node=2 script.py` |
| 文件 | 单文件 | 需要启动脚本 |
| 配置 | 自动 | 手动 |
| Kaggle | ✅ 完美 | ⚠️ 复杂 |

### 代码改动

```python
# 1. 导入
from accelerate import Accelerator

# 2. 初始化
accelerator = Accelerator(
    gradient_accumulation_steps=16,
    mixed_precision='fp16'
)

# 3. 准备
model, optimizer = accelerator.prepare(model, optimizer)

# 4. 训练
with accelerator.accumulate(model):
    loss = model(x, y)
    accelerator.backward(loss)
    optimizer.step()
```

## 下一步

1. ✅ 运行训练
2. ✅ 监控loss
3. ✅ 评估ROUGE
4. ✅ 生成提交
5. ✅ 下载结果

完整文档见 `KAGGLE_ACCELERATE_GUIDE.md`

祝训练顺利！🚀
