# 多GPU并行训练说明

## 修改内容

已将 `version12.py` 修改为支持在Kaggle平台上使用2个Tesla T4 GPU进行分布式数据并行（DDP）训练。

### 主要改动

1. **添加DDP配置**
   - 在 `Config` 类中添加 `ddp = True` 参数
   - 调整 `gradient_accumulation_steps` 从16降到8（因为2个GPU，有效批次大小保持不变）

2. **DDP初始化**
   - 在 `train()` 函数开始时初始化进程组
   - 设置每个进程的设备和随机种子
   - 识别主进程（用于日志和保存checkpoint）

3. **模型包装**
   - 使用 `DistributedDataParallel` 包装模型
   - 正确处理 `raw_model` 引用（用于保存和评估）

4. **日志和保存**
   - 只在主进程（rank 0）打印日志
   - 只在主进程保存checkpoint
   - 添加进程同步点（barrier）

5. **数据准备**
   - 只在主进程准备数据
   - 其他进程等待数据准备完成

## 使用方法

### 方法1：使用启动脚本（推荐）

```bash
python train_ddp.py
```

这个脚本会：
- 自动检测GPU数量
- 如果有多个GPU，使用 `torchrun` 启动DDP训练
- 如果只有1个GPU，直接运行单GPU训练

### 方法2：直接使用torchrun

```bash
torchrun --nproc_per_node=2 --standalone version12.py
```

### 方法3：单GPU训练

如果只想使用单GPU，将 `Config` 中的 `ddp` 设置为 `False`：

```python
class Config:
    # ...
    ddp = False  # 禁用DDP
```

然后直接运行：

```bash
python version12.py
```

## 在Kaggle上运行

### 1. 启用多GPU

在Kaggle Notebook设置中：
- 点击右侧的 "Accelerator"
- 选择 "GPU T4 x2"（2个Tesla T4）

### 2. 安装依赖

```python
!pip install rouge-score
```

### 3. 运行训练

在Notebook cell中：

```python
# 方法1：使用启动脚本
!python train_ddp.py

# 方法2：直接使用torchrun
!torchrun --nproc_per_node=2 --standalone version12.py
```

## 性能提升

使用2个GPU进行DDP训练的预期效果：

- **训练速度**: 约1.8-1.9x加速（接近2x，但有通信开销）
- **有效批次大小**: `batch_size * gradient_accumulation_steps * num_gpus = 16 * 8 * 2 = 256`
- **内存使用**: 每个GPU独立加载模型，内存使用与单GPU相同

## 注意事项

1. **批次大小调整**
   - 原配置: `batch_size=16, gradient_accumulation_steps=16` → 有效批次=256
   - 新配置: `batch_size=16, gradient_accumulation_steps=8, num_gpus=2` → 有效批次=256
   - 保持有效批次大小不变，确保训练稳定性

2. **学习率**
   - 当前配置已经考虑了有效批次大小
   - 如果修改批次大小，可能需要相应调整学习率

3. **随机性**
   - 每个进程使用不同的随机种子（seed + rank）
   - 确保数据增强的多样性

4. **checkpoint保存**
   - 只有主进程（rank 0）保存checkpoint
   - 避免多个进程同时写入同一文件

5. **评估**
   - 评估在主进程上使用原始模型（不是DDP包装的）
   - 确保评估结果的一致性

## 故障排查

### 问题1: "Address already in use"

如果遇到端口占用错误，可以指定不同的端口：

```bash
torchrun --nproc_per_node=2 --master_port=29500 --standalone version12.py
```

### 问题2: NCCL错误

如果遇到NCCL通信错误，可以尝试：

```python
# 在代码开始添加
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_P2P_DISABLE'] = '1'  # 禁用P2P通信
```

### 问题3: 只检测到1个GPU

确保在Kaggle设置中选择了 "GPU T4 x2"，而不是单个GPU。

## 验证DDP是否工作

训练开始时应该看到：

```
DDP训练: 2 GPUs
有效批次大小: 256
```

如果只看到单GPU的日志，说明DDP没有启用。

## 性能监控

训练过程中关注：

- **MFU (Model FLOPs Utilization)**: 应该在10-20%范围内
- **训练速度**: 每个iteration的时间应该比单GPU快约1.8倍
- **GPU利用率**: 使用 `nvidia-smi` 查看两个GPU的利用率应该都接近100%

## 与原版本的兼容性

修改后的代码完全向后兼容：

- 设置 `ddp=False` 即可恢复单GPU训练
- 所有其他功能（评估、生成、ROUGE计算）保持不变
- checkpoint格式完全相同
