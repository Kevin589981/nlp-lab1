好的，我已经按照您的要求，将原始的完整微调代码修改为使用LoRA（Low-Rank Adaptation）进行高效参数微调。

**主要修改点如下：**

1.  **新增LoRA模块**：添加了 `LoRALayer` 类和 `apply_lora_to_linear` 函数，用于实现LoRA的核心逻辑，并通过“猴子补丁”的方式动态地将LoRA功能注入到指定的 `nn.Linear` 层。
2.  **修改模型架构 (`GPT`)**：
    *   在 `GPT` 类的 `__init__` 方法中，增加了应用LoRA的逻辑。它会首先冻结模型的所有原始参数，然后将LoRA层应用到目标模块（如注意力层的 `c_attn` 和 `c_proj`），并只解冻新添加的LoRA参数。
    *   更新了参数数量的统计，现在会明确区分并打印**总参数量**和**可训练参数量**，让您能清晰地看到LoRA带来的效率提升。
3.  **修改优化器配置 (`configure_optimizers`)**：
    *   `configure_optimizers` 方法现在可以智能识别是否启用了LoRA。在LoRA模式下，优化器将只包含需要梯度的参数（即LoRA的参数），并且通常不对这些参数应用权重衰减。
4.  **新增LoRA配置参数 (`Config`)**：
    *   在 `Config` 类中，添加了专门用于控制LoRA微调的超参数，如 `use_lora` (开关)、`lora_rank` (秩)、`lora_alpha` (缩放因子) 和 `lora_target_modules` (目标层)。

您可以直接将以下所有单元格的代码复制并粘贴回您的 `.ipynb` 文件中。单元格的格式和原始内容都已保留。

---

# %% [markdown]
# 
# # 处理数据
# 右侧点击Add Input，找到我们的比赛，然后添加
# 
# 从Kaggle输入的train.csv读取数据，随机划分为训练集(95%)和验证集(5%)
# 
# 保存到/kaggle/working/data/samsum目录

# %%
!pip install rouge-score

# %%

import pandas as pd
import os
import tiktoken
print("\n" + "=" * 80)
print("准备SAMSum数据集划分")
print("=" * 80)

# 读取原始CSV文件
input_csv = '/kaggle/input/nanogpt-fudannlp-cs-30040/train.csv'
print(f"\n读取数据: {input_csv}")

df = pd.read_csv(input_csv)
total_samples = len(df)
print(f"总样本数: {total_samples}")

# 随机打乱数据
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

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
# # 模型架构

# %%

"""
===================================================================================
GPT-2 文本摘要微调 - 完整教学脚本
===================================================================================

本脚本整合了完整的训练和评估流程，适合用于教学和学习。

主要内容：
1. GPT模型定义（完整的Transformer架构）
2. 数据准备和加载
3. 模型训练
4. ROUGE评估
5. 生成和测试

学习建议：
- 初学者：重点关注Config配置部分，了解各参数的作用
- 进阶者：深入理解模型结构、训练循环和数据处理
- 实践者：修改参数进行实验，观察结果变化

===================================================================================
"""

import math
import inspect
from dataclasses import dataclass
import types

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



# %% [markdown]
# ## LoRA (Low-Rank Adaptation)
# 
# 我们将实现LoRA来高效地微调模型。LoRA的核心思想是冻结预训练模型的权重，并在模型的特定层（通常是注意力层）旁边注入可训练的低秩矩阵。
# 
# 1.  **LoRALayer**: 定义了LoRA的核心，包含两个低秩矩阵A和B。
# 2.  **apply_lora_to_linear**: 一个辅助函数，通过“猴子补丁”的方式将LoRA功能动态地添加到现有的`nn.Linear`层，而无需修改原始的模型代码。它会冻结原始权重，并重写`forward`方法。

# %%
class LoRALayer(nn.Module):
    """
    一个简单的LoRA层实现。
    该层被添加到现有的nn.Linear层中。
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(dropout)
        self.scaling = self.alpha / self.rank

        # 初始化权重
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lora_dropout(x)
        update = (x @ self.lora_A @ self.lora_B) * self.scaling
        return update

def apply_lora_to_linear(linear_layer: nn.Linear, rank: int, alpha: float, dropout: float):
    """
    通过猴子补丁（monkey-patching）的方式将LoRA功能添加到nn.Linear层。
    """
    # 1. 冻结原始权重 (虽然外部已全局冻结，但此处明确操作更安全)
    linear_layer.weight.requires_grad = False
    if linear_layer.bias is not None:
        linear_layer.bias.requires_grad = False
        
    # 2. 添加LoRA层作为子模块
    linear_layer.lora_layer = LoRALayer(
        in_features=linear_layer.in_features,
        out_features=linear_layer.out_features,
        rank=rank,
        alpha=alpha,
        dropout=dropout
    )
    
    # 3. 重写forward方法
    def lora_forward(self, x):
        # 原始的线性变换结果
        original_output = F.linear(x, self.weight, self.bias)
        # 加上LoRA的更新
        lora_update = self.lora_layer(x)
        return original_output + lora_update
        
    # 将新的forward方法绑定到linear_layer实例上
    linear_layer.forward = types.MethodType(lora_forward, linear_layer)
    
    return linear_layer

# %% [markdown]
# ## CausalSelfAttentio

# %%
class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # 如果有past_kv，则拼接
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # 在序列维度拼接
            v = torch.cat([past_v, v], dim=2)
        
        # 如果需要cache，保存当前的k, v
        present_kv = (k, v) if use_cache else None
        
        # 更新T为完整的序列长度
        T_full = k.size(2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T_full] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, present_kv



# %% [markdown]
# ## MLP

# %%
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, past_kv=None, use_cache=False):
        attn_out, present_kv = self.attn(self.ln_1(x), past_kv=past_kv, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln_2(x))
        return x, present_kv



# %% [markdown]
# ## GPT

# %%
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    
    # LoRA annd Finetuning config will be added via the main Config class
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_target_modules: list = None


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # LoRA-specific modifications
        if hasattr(config, 'use_lora') and config.use_lora:
            print("Applying LoRA adaptations...")
            # Freeze all parameters in the model
            for param in self.parameters():
                param.requires_grad = False
            
            # Find and adapt linear layers, unfreezing LoRA parameters
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear) and any(target in name for target in config.lora_target_modules):
                    print(f"  - Adapting layer: {name}")
                    apply_lora_to_linear(
                        module, 
                        rank=config.lora_rank, 
                        alpha=config.lora_alpha, 
                        dropout=config.lora_dropout
                    )
                    # Unfreeze the newly added LoRA parameters
                    for param in module.lora_layer.parameters():
                        param.requires_grad = True

        # report number of parameters
        total_params = self.get_num_params()
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"number of total parameters: {total_params/1e6:.2f}M")
        print(f"number of trainable parameters: {trainable_params/1e6:.4f}M ({100 * trainable_params / total_params:.4f}%)")


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, past_key_values=None, use_cache=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        
        # 计算position IDs
        if past_key_values is None:
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)
        else:
            # 如果使用cache，position从past的长度开始
            past_length = past_key_values.size(2) if past_key_values is not None else 0
            pos = torch.arange(past_length, past_length + t, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        # 通过所有block，收集KV cache
        present_key_values = [] if use_cache else None
        for i, block in enumerate(self.transformer.h):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = block(x, past_kv=past_kv, use_cache=use_cache)
            if use_cache:
                present_key_values.append(present_kv)
        
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        if use_cache:
            return logits, loss, present_key_values
        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        
        # Pass LoRA and other override args to the config
        for k, v in override_args.items():
            config_args[k] = v
            print(f"overriding {k} to {v}")

        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param
        # also filter out lora parameters if they exist
        sd_keys = [k for k in sd_keys if 'lora_' not in k]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # 检查是否使用了LoRA
        is_lora = hasattr(self.config, 'use_lora') and self.config.use_lora

        # 收集所有需要计算梯度的参数
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        if is_lora:
            # 如果使用LoRA，只优化LoRA参数，并且通常不使用权重衰减
            print("Using LoRA-specific optimizer: all trainable params with no weight decay.")
            optim_groups = [{'params': list(param_dict.values()), 'weight_decay': 0.0}]
            num_trainable_params = sum(p.numel() for p in param_dict.values())
            print(f"num trainable parameter tensors: {len(param_dict)}, with {num_trainable_params:,} parameters")
        else:
            # 原始的优化器配置逻辑，用于完全微调
            # 将参数分为需要权重衰减和不需要权重衰减的两组
            decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
            nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
            optim_groups = [
                {'params': decay_params, 'weight_decay': weight_decay},
                {'params': nodecay_params, 'weight_decay': 0.0}
            ]
            num_decay_params = sum(p.numel() for p in decay_params)
            num_nodecay_params = sum(p.numel() for p in nodecay_params)
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # 创建AdamW优化器
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 106e9 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        
        Args:
            idx: 输入序列 (b, t)
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: top-k采样
            eos_token_id: 结束token ID，遇到时提前停止（可以是单个ID或ID列表）
        """
        # 将eos_token_id转换为列表
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = set(eos_token_id)
        
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 检查是否生成了结束token
            if eos_token_id is not None:
                # 检查batch中所有序列是否都遇到了结束token
                if idx_next.item() in eos_token_id:
                    break

        return idx
    
    @torch.no_grad()
    def generate_with_kv_cache(self, idx, max_new_tokens, temperature=1.0, top_k=None, eos_token_id=None):
        """
        使用KV cache加速的生成方法
        
        KV cache原理：
        - 在自回归生成中，每一步只需要计算新token的attention
        - 之前token的key和value可以缓存，避免重复计算
        - 这样可以显著加速生成过程（特别是长序列）
        
        Args:
            idx: 输入序列 (b, t)
            max_new_tokens: 最大生成token数
            temperature: 采样温度
            top_k: top-k采样
            eos_token_id: 结束token ID，遇到时提前停止（可以是单个ID或ID列表）
        """
        # 将eos_token_id转换为集合
        if eos_token_id is not None:
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = set(eos_token_id)
        
        # 第一步：处理整个prompt，获取初始的KV cache
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _, past_key_values = self(idx_cond, use_cache=True)
        
        # 对第一个token进行采样
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        # 检查是否立即遇到结束token
        if eos_token_id is not None and idx_next.item() in eos_token_id:
            return idx
        
        # 后续生成步骤：每次只处理一个新token，使用KV cache
        for _ in range(max_new_tokens - 1):
            # 只输入最后一个token，使用past_key_values
            logits, _, past_key_values = self(
                idx[:, [-1]], 
                past_key_values=past_key_values, 
                use_cache=True
            )
            
            # 采样
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 拼接新token
            idx = torch.cat((idx, idx_next), dim=1)
            
            # 提前停止检查
            if eos_token_id is not None and idx_next.item() in eos_token_id:
                break
        
        return idx


# %% [markdown]
# ## 配置参数

# %%
import os
import time
import math
import pickle
import csv
from contextlib import nullcontext

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import Dataset, DataLoader
import tiktoken

# 导入模型定义


# =============================================================================
# 第一部分：配置参数
# 这些参数可以由学生根据需要进行调整
# =============================================================================

class Config:
    """
    配置类：包含所有可调参数
    参数组织方式便于学生理解，参数值与nanoGPT原始配置保持一致
    """
    
    # =========================================================================
    # 数据集配置
    # =========================================================================
    dataset_path = '/kaggle/working/data/samsum'  # 原始数据集路径
    dataset = 'samsum'          # 数据集名称（处理后的数据会保存在data/{dataset}/目录）
    
    # 特殊token定义（用于分隔对话和摘要）
    dialogue_start = "\n\n### DIALOGUE:\n"  # 对话开始标记
    summary_start = "\n\n### SUMMARY:\n"     # 摘要开始标记
    summary_end = "<|endoftext|>"            # 摘要结束标记（GPT-2的EOS token）
    
    # =========================================================================
    # 训练配置（建议学生重点关注这部分）
    # =========================================================================
    # 模型初始化
    init_from = 'gpt2'       # 'scratch'(从头训练) 或 'resume'(继续训练) 或 'gpt2'/'gpt2-xl'(从预训练模型微调)
    
    # 批次配置
    batch_size = 8              # 每个GPU的批次大小（micro-batch size）
    gradient_accumulation_steps = 16  # 梯度累积步数，有效批次 = batch_size * gradient_accumulation_steps
    block_size = 1024           # 上下文窗口大小（最大序列长度）
    
    # 训练步数
    max_iters = 500              # 总训练迭代次数
    
    # 优化器配置（AdamW）
    learning_rate = 3e-4        # 学习率 (LoRA通常使用比完整微调稍大的学习率，例如3e-4)
    weight_decay = 0.0          # 权重衰减系数 (LoRA通常不使用权重衰减)
    beta1 = 0.9                 # Adam的beta1参数
    beta2 = 0.999               # Adam的beta2参数
    grad_clip = 1.0             # 梯度裁剪阈值（0.0表示不裁剪）
    
    # 学习率调度
    decay_lr = True             # 是否使用学习率衰减
    warmup_iters = 100          # 学习率预热步数
    lr_decay_iters = 500      # 学习率衰减的总步数 (通常等于max_iters)
    min_lr = 3e-5               # 最小学习率
    
    # =========================================================================
    # LoRA Fine-tuning Configuration (NEW)
    # =========================================================================
    use_lora = True             # 是否使用LoRA进行微调
    lora_rank = 16              # LoRA的秩 (r)
    lora_alpha = 32.0           # LoRA的alpha缩放因子 (通常是rank的两倍)
    lora_dropout = 0.05         # LoRA层的dropout率
    lora_target_modules = ['c_attn', 'c_proj'] # 要应用LoRA的层 (按名称包含)
    
    # =========================================================================
    # 模型配置（从头训练时需要设置，从预训练模型加载时会被覆盖）
    # =========================================================================
    n_layer = 12                # Transformer层数
    n_head = 12                 # 注意力头数
    n_embd = 768                # 嵌入维度
    dropout = 0.1               # Dropout率（预训练0.0，微调可尝试0.1+）
    bias = False                # LayerNorm和Linear层是否使用bias
    
    # =========================================================================
    # I/O配置
    # =========================================================================
    out_dir = 'out-summarization-lora' # checkpoint保存目录
    eval_interval = 10           # 每多少步评估一次
    log_interval = 5            # 每多少步打印日志
    eval_iters = 40             # 评估时的迭代次数
    eval_only = False           # 是否只评估不训练
    always_save_checkpoint = True  # 是否每次评估都保存checkpoint（False表示只保存最佳模型）
    
    # ROUGE评估配置（训练过程中）
    eval_rouge_during_training = True  # 是否在训练时评估ROUGE分数
    rouge_eval_samples = 5      # 训练时ROUGE评估的样本数（较少避免太慢）
    
    # =========================================================================
    # wandb日志配置（可选）
    # =========================================================================
    wandb_log = False           # 是否启用wandb日志
    wandb_project = 'owt'       # wandb项目名
    wandb_run_name = 'gpt2'     # wandb运行名称
    
    # =========================================================================
    # 系统配置
    # =========================================================================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 训练设备
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
    # dtype = 'float32'
    compile = True#False             # 是否使用PyTorch 2.0编译（需要CUDA Capability >= 7.0，P100不支持）
    backend = 'nccl'            # DDP后端（'nccl'用于GPU，'gloo'用于CPU）
    
    # =========================================================================
    # 测试/生成配置
    # =========================================================================
    num_test_samples = 10       # 测试时评估的样本数量
    max_new_tokens = 100        # 生成时的最大token数
    temperature = 0.7           # 生成温度（1.0=无变化，<1.0=更确定，>1.0=更随机）
    top_k = 50                 # Top-K采样（保留概率最高的K个token）


config = Config()

# %% [markdown]
# ## def prepare_data()

# %%
# =============================================================================
# 第二部分：数据准备
# 读取samsum数据集，格式化为训练格式，并进行tokenization
# =============================================================================

def prepare_data():
    """
    准备摘要数据集
    
    数据格式设计：
    每条训练样本格式为：
    \n\n### DIALOGUE:\n{对话内容}\n\n### SUMMARY:\n{摘要内容}<|endoftext|>
    
    重要：每个样本独立保存，不连接成长序列
    
    这样模型能学习到：
    - 看到 DIALOGUE 标记后，理解后面是对话内容
    - 看到 SUMMARY 标记后，开始生成摘要
    - 看到 <|endoftext|> 表示摘要结束
    """
    print("=" * 80)
    print("准备数据集...")
    print("=" * 80)
    
    # 创建数据目录
    data_dir = os.path.join('data', config.dataset)
    os.makedirs(data_dir, exist_ok=True)
    
    # 初始化tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # 处理训练集和验证集
    for split in ['train', 'validation']:
        print(f"\n处理 {split} 数据集...")
        csv_file = os.path.join(config.dataset_path, f'{split}.csv')
        
        # 读取CSV文件
        dialogues = []
        summaries = []
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                dialogues.append(row['dialogue'])
                summaries.append(row['summary'])
        
        print(f"  读取了 {len(dialogues)} 条数据")
        
        # 格式化并tokenize每条数据，每个样本单独保存
        samples = []  # 存储所有样本的token列表
        valid_count = 0
        skipped_count = 0
        total_tokens = 0
        
        for dialogue, summary in zip(dialogues, summaries):
            # 构建完整的训练样本
            formatted_text = (
                config.dialogue_start + dialogue +
                config.summary_start + summary +
                config.summary_end
            )
            
            # Tokenize
            tokens = enc.encode(formatted_text, allowed_special={config.summary_end})
            
            # 检查长度是否超过block_size
            if len(tokens) <= config.block_size:
                samples.append(tokens)
                valid_count += 1
                total_tokens += len(tokens)
            else:
                # 如果太长，进行截断（保留对话开始和摘要部分）
                # 找到SUMMARY标记的位置
                summary_tokens = enc.encode(config.summary_start, allowed_special={config.summary_end})
                summary_pos = None
                for i in range(len(tokens) - len(summary_tokens)):
                    if tokens[i:i+len(summary_tokens)] == summary_tokens:
                        summary_pos = i
                        break
                
                if summary_pos and (len(tokens) - summary_pos) < config.block_size * 0.3:
                    # 如果能找到摘要位置，且摘要部分不太长，则截断对话部分
                    dialogue_tokens = enc.encode(config.dialogue_start, allowed_special={config.summary_end})
                    available_space = config.block_size - (len(tokens) - summary_pos) - len(dialogue_tokens)
                    
                    if available_space > 0:
                        # 截断对话内容
                        truncated_tokens = (
                            dialogue_tokens +
                            tokens[len(dialogue_tokens):len(dialogue_tokens)+available_space] +
                            tokens[summary_pos:]
                        )
                        samples.append(truncated_tokens)
                        valid_count += 1
                        total_tokens += len(truncated_tokens)
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
        
        print(f"  有效样本数: {valid_count}")
        print(f"  跳过样本数: {skipped_count}")
        print(f"  总token数: {total_tokens:,}")
        print(f"  平均token数: {total_tokens // valid_count if valid_count > 0 else 0}")
        
        # 保存为pickle文件（每个样本单独保存）
        output_file = 'train.pkl' if split == 'train' else 'val.pkl'
        output_path = os.path.join(data_dir, output_file)
        with open(output_path, 'wb') as f:
            pickle.dump(samples, f)
        print(f"  保存到: {output_path}")
    
    # 保存meta信息（词表大小）
    meta = {
        'vocab_size': enc.n_vocab,
        'dialogue_start': config.dialogue_start,
        'summary_start': config.summary_start,
        'summary_end': config.summary_end,
    }
    with open(os.path.join(data_dir, 'meta.pkl'), 'wb') as f:
        pickle.dump(meta, f)
    
    print("\n数据准备完成！")
    print("=" * 80)



# %% [markdown]
# ## class SummarizationDataset

# %%

# =============================================================================
# 第三部分：数据集类和数据加载
# =============================================================================

class SummarizationDataset(Dataset):
    """
    摘要任务的Dataset类
    
    每个样本是一个完整的"对话+摘要"序列，包含：
    - dialogue_start + 对话内容 + summary_start + 摘要内容 + summary_end
    
    这个类负责：
    1. 加载tokenized的样本
    2. Padding/截断到固定长度
    3. 构建输入(x)和目标(y)序列
    """
    
    def __init__(self, data_path, block_size):
        with open(data_path, 'rb') as f:
            self.samples = pickle.load(f)
        self.block_size = block_size
        
        # 为了找到摘要开始的位置，我们需要提前tokenize摘要开始的标记
        enc = tiktoken.get_encoding("gpt2")
        self.summary_token_ids = enc.encode(config.summary_start)
        
        print(f"  加载了 {len(self.samples)} 个样本")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_tokens = self.samples[idx]
        
        # 1. 创建 x 和 y (与之前逻辑相同)
        # x 是输入, y 是目标 (x 向右移动一位)
        x_tokens = sample_tokens[:-1]
        y_tokens = sample_tokens[1:]
        
        # 2. 应用损失掩码到 y 上
        # 找到摘要开始的位置
        summary_start_pos = -1
        # Search for the summary token sequence
        for i in range(len(x_tokens) - len(self.summary_token_ids) + 1):
            if x_tokens[i:i+len(self.summary_token_ids)] == self.summary_token_ids:
                summary_start_pos = i
                break
        
        # 如果找到了摘要标记，将它之前的所有目标 token 设置为 -1
        if summary_start_pos != -1:
            # 我们希望从 "### SUMMARY:\n" 的最后一个 token 开始预测第一个摘要词
            # 所以，掩码应该应用到这个位置之前的所有 token
            mask_end_index = summary_start_pos + len(self.summary_token_ids)
            for i in range(mask_end_index):
                if i < len(y_tokens):
                    y_tokens[i] = -1
        
        # 3. Padding (与之前逻辑相同)
        x_padding_len = self.block_size - len(x_tokens)
        y_padding_len = self.block_size - len(y_tokens)
        
        # 使用 50256 (<|endoftext|>) 作为 padding token for x
        x_padded = x_tokens + * x_padding_len
        y_padded = y_tokens + [-1] * y_padding_len
        
        # 截断以防万一
        x = torch.tensor(x_padded[:self.block_size], dtype=torch.long)
        y = torch.tensor(y_padded[:self.block_size], dtype=torch.long)
        
        return x, y


# 全局变量：缓存DataLoader
_dataloaders = {'train': None, 'val': None}
_data_iters = {'train': None, 'val': None}



# %% [markdown]
# ## getbatch

# %%
def get_batch(split, data_dir):
    """
    获取一个训练批次
    
    使用DataLoader实现，支持：
    1. 自动batch处理
    2. 可选的shuffle（训练集shuffle，验证集不shuffle）
    3. 自动循环迭代（epoch结束后自动重新开始）
    
    参数:
        split: 'train' 或 'val'
        data_dir: 数据目录
    
    返回:
        x: 输入序列 [batch_size, block_size]
        y: 目标序列 [batch_size, block_size]
    """
    global _dataloaders, _data_iters
    
    # 首次调用：创建DataLoader
    if _dataloaders[split] is None:
        data_path = os.path.join(data_dir, f'{split}.pkl')
        dataset = SummarizationDataset(data_path, config.block_size)
        
        # 训练集shuffle，验证集不shuffle
        shuffle = (split == 'train')
        
        _dataloaders[split] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            num_workers=4,  # 主线程加载数据（简单场景足够）
            pin_memory=True if 'cuda' in config.device else False,
        )
        _data_iters[split] = iter(_dataloaders[split])
    
    # 获取下一个batch
    try:
        x, y = next(_data_iters[split])
    except StopIteration:
        # 当前epoch结束，重新开始
        _data_iters[split] = iter(_dataloaders[split])
        x, y = next(_data_iters[split])
    
    # 移动到设备
    if 'cuda' in config.device:
        x = x.to(config.device, non_blocking=True)
        y = y.to(config.device, non_blocking=True)
    else:
        x = x.to(config.device)
        y = y.to(config.device)
    
    return x, y




# %% [markdown]
# ## estimate loss

# %%
@torch.no_grad()
def estimate_loss(model, ctx, data_dir):
    """
    估计训练集和验证集上的损失
    
    通过多次迭代求平均，得到更准确的损失估计
    """
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, data_dir)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out




# %% [markdown]
# ## evaluate rouge

# %%
@torch.no_grad()
def evaluate_rouge_during_training(model, ctx, data_dir, num_samples=3):
    """
    在训练过程中评估ROUGE分数
    
    从验证集中随机选择几个样本，生成摘要并计算ROUGE分数
    这可以帮助我们实时监控模型的摘要质量
    
    参数:
        model: 模型
        ctx: autocast上下文
        data_dir: 数据目录（未使用，保留兼容性）
        num_samples: 评估的样本数量（默认3个，避免评估时间过长）
    
    返回:
        rouge_scores: 包含平均ROUGE分数的字典
    """
    model.eval()
    
    # 初始化tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # 从验证集CSV文件中读取样本
    val_csv = os.path.join(config.dataset_path, 'validation.csv')
    if not os.path.exists(val_csv):
        print("  (跳过ROUGE评估：未找到验证集)")
        model.train()
        return None
    
    # 读取验证集
    dialogues = []
    summaries = []
    with open(val_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialogues.append(row['dialogue'])
            summaries.append(row['summary'])
    
    # 随机选择num_samples个样本
    import random
    indices = random.sample(range(len(dialogues)), min(num_samples, len(dialogues)))
    
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    # 获取<|endoftext|>的token ID作为停止符
    eos_token_id = enc.encode(config.summary_end, allowed_special={config.summary_end})
    
    # 临时保存原始max_new_tokens，训练时使用较少的tokens以加快速度
    original_max_new_tokens = config.max_new_tokens
    config.max_new_tokens = 100  # 训练时用较少的tokens
    
    for idx in tqdm(indices):
        dialogue = dialogues[idx]
        reference_summary = summaries[idx]
        
        # 构建prompt
        prompt = config.dialogue_start + dialogue + config.summary_start
        prompt_tokens = enc.encode(prompt, allowed_special={config.summary_end})
        
        # 如果prompt太长，跳过
        if len(prompt_tokens) > config.block_size - config.max_new_tokens:
            print(f"len(prompt_tokens) > config.block_size - config.max_new_tokens: {len(prompt_tokens)} > {config.block_size - config.max_new_tokens}")
            continue
        
        # 使用KV cache加速生成摘要（但训练时为了速度，使用原始的generate方法）
        # 注意：训练时模型可能还没有完全训练好，所以使用简单的generate方法
        x = torch.tensor(prompt_tokens, dtype=torch.long, device=config.device)[None, ...]
        
        with ctx:
            y = model.generate(
                x,
                max_new_tokens=config.max_new_tokens,
                temperature=0.8,
                top_k=200,
                eos_token_id=eos_token_id
            )
        
        # 解码
        generated_tokens = y.tolist()
        generated_text = enc.decode(generated_tokens)
        
        # 提取摘要（使用公共函数）
        generated_summary = extract_summary(generated_text, prompt, enc)
        
        # 计算ROUGE分数（使用公共函数）
        if generated_summary:  # 确保生成了内容
            rouge_scores = calculate_rouge(reference_summary, generated_summary)
            if rouge_scores:
                rouge1_scores.append(rouge_scores['rouge1'])
                rouge2_scores.append(rouge_scores['rouge2'])
                rougeL_scores.append(rouge_scores['rougeL'])
    
    # 恢复原始max_new_tokens
    config.max_new_tokens = original_max_new_tokens
    
    model.train()
    
    # 返回平均分数
    if len(rouge1_scores) > 0:
        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores)
        }
    else:
        return None




# %% [markdown]
# ## get lr

# %%
def get_lr(iter_num):
    """
    学习率调度：带预热的余弦衰减
    
    1. 前warmup_iters步：线性增加
    2. 之后：余弦衰减到min_lr
    """
    # 线性预热
    if iter_num < config.warmup_iters:
        return config.learning_rate * (iter_num + 1) / (config.warmup_iters + 1)
    
    # 如果超过衰减步数，返回最小学习率
    if iter_num > config.lr_decay_iters:
        return config.min_lr
    
    # 余弦衰减
    decay_ratio = (iter_num - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


def train():
    """训练主函数"""
    print("\n" + "=" * 80)
    print("开始训练...")
    print("=" * 80)
    
    # 设置随机种子
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # 创建输出目录
    os.makedirs(config.out_dir, exist_ok=True)
    
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
    
    # 数据目录
    data_dir = os.path.join('data', config.dataset)
    
    # 加载词表信息
    meta_path = os.path.join(data_dir, 'meta.pkl')
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        meta_vocab_size = meta['vocab_size']
        print(f"从 {meta_path} 加载词表大小: {meta_vocab_size}")
    
    # 初始化模型
    print(f"\n模型初始化方式: {config.init_from}")
    model_args = dict(
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        block_size=config.block_size,
        bias=config.bias,
        vocab_size=None,
        dropout=config.dropout,
        # Pass LoRA config to GPTConfig
        use_lora=config.use_lora,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
    )
    
    if config.init_from == 'scratch':
        # 从头训练
        print("从头开始训练新模型")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size else 50304
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        iter_num = 0
        best_val_loss = 1e9
        
    elif config.init_from == 'resume':
        # 从checkpoint恢复
        print(f"从 {config.out_dir} 恢复训练")
        ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=config.device)
        checkpoint_model_args = checkpoint['model_args']
        
        for k in list(model_args.keys()):
             if k in checkpoint_model_args:
                model_args[k] = checkpoint_model_args[k]

        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        
        model.load_state_dict(state_dict, strict=False) # Use strict=False for LoRA
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        
    elif config.init_from.startswith('gpt2'):
        # 从预训练GPT-2加载
        print(f"从OpenAI GPT-2加载: {config.init_from}")
        override_args = dict(
            dropout=config.dropout,
            use_lora=config.use_lora,
            lora_rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            lora_target_modules=config.lora_target_modules,
        )
        model = GPT.from_pretrained(config.init_from, override_args)
        
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.config, k)
        
        iter_num = 0
        best_val_loss = 1e9
    
    # 调整block_size（如果需要）
    if config.block_size < model.config.block_size:
        model.crop_block_size(config.block_size)
        model_args['block_size'] = config.block_size
    
    model.to(config.device)
    
    # 初始化优化器
    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        device_type
    )
    
    if config.init_from == 'resume' and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    checkpoint = None  # 释放内存
    
    # 编译模型（可选，PyTorch 2.0+）
    if config.compile:
        print("编译模型（首次会比较慢）...")
        unoptimized_model = model
        model = torch.compile(model)
    
    # 初始化GradScaler（用于混合精度训练）
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype == 'float16'))
    
    # 训练循环
    print("\n开始训练循环...")
    print(f"总迭代次数: {config.max_iters}")
    print(f"批次大小: {config.batch_size}")
    print(f"梯度累积步数: {config.gradient_accumulation_steps}")
    print(f"有效批次大小: {config.batch_size * config.gradient_accumulation_steps}")
    print("-" * 80)
    
    X, Y = get_batch('train', data_dir)
    t0 = time.time()
    local_iter_num = 0
    raw_model = model
    running_mfu = -1.0
    
    while True:
        # 设置学习率
        lr = get_lr(iter_num) if config.decay_lr else config.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 评估和保存checkpoint
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, ctx, data_dir)
            print(f"\nStep {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            # 计算ROUGE分数（从第一次评估后开始，避免初始化时模型输出不稳定）
            if iter_num > 0 and config.eval_rouge_during_training:
                print("  评估ROUGE分数...")
                rouge_scores = evaluate_rouge_during_training(
                    model, ctx, data_dir, num_samples=config.rouge_eval_samples
                )
                if rouge_scores:
                    print(f"  ROUGE-1: {rouge_scores['rouge1']:.4f}, "
                          f"ROUGE-2: {rouge_scores['rouge2']:.4f}, "
                          f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
            
            # 保存checkpoint
            if losses['val'] < best_val_loss or config.always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    # For LoRA, we only save the trainable parameters (LoRA weights)
                    model_to_save = raw_model.state_dict()
                    if config.use_lora:
                        lora_weights = {k: v for k, v in model_to_save.items() if 'lora_' in k}
                        model_to_save = lora_weights
                        
                    checkpoint = {
                        'model': model_to_save,
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': vars(config),
                    }
                    print(f"  保存checkpoint到 {config.out_dir}")
                    torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
        
        if iter_num == 0 and config.eval_only:
            break
        
        # 前向-反向传播（带梯度累积）
        for micro_step in range(config.gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / config.gradient_accumulation_steps
            
            # 异步预取下一个batch
            X, Y = get_batch('train', data_dir)
            
            # 反向传播
            scaler.scale(loss).backward()
        
        # 梯度裁剪
        if config.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # 记录日志
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % config.log_interval == 0:
            lossf = loss.item() * config.gradient_accumulation_steps
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(
                    config.batch_size * config.gradient_accumulation_steps, dt
                )
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        iter_num += 1
        local_iter_num += 1
        
        # 终止条件
        if iter_num > config.max_iters:
            break
    
    print("\n训练完成！")
    print("=" * 80)


# %% [markdown]
# ## load_model

# %%
# =============================================================================
# 第四部分：测试和评估
# =============================================================================

def load_model():
    """
    加载训练好的模型
    
    返回:
        model: 加载的模型
        enc: tokenizer
        ctx: autocast上下文
    """
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
    
    # 加载模型
    print(f"\n从 {config.out_dir} 加载模型...")
    ckpt_path = os.path.join(config.out_dir, 'ckpt.pt')

    print(f"path {ckpt_path}")
    if not os.path.exists(ckpt_path):
        print(f"错误: 找不到checkpoint文件 {ckpt_path}")
        print("请先运行训练！")
        return None, None, None
    
    checkpoint = torch.load(ckpt_path, map_location=config.device)
    
    # 加载模型配置，并从预训练模型开始构建
    ckpt_config = checkpoint['config']
    override_args = dict(
        dropout=ckpt_config.get('dropout', 0.0),
        use_lora=ckpt_config.get('use_lora', False),
        lora_rank=ckpt_config.get('lora_rank', 16),
        lora_alpha=ckpt_config.get('lora_alpha', 32),
        lora_dropout=ckpt_config.get('lora_dropout', 0.05),
        lora_target_modules=ckpt_config.get('lora_target_modules', ['c_attn', 'c_proj']),
    )

    # 从 'gpt2' 基础模型开始构建
    model = GPT.from_pretrained('gpt2', override_args)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    # 加载LoRA权重 (或完整权重)
    # 使用 strict=False，因为我们只加载了LoRA的权重，基础模型的权重已经从'gpt2'加载了
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.to(config.device)
    
    if config.compile:
        print("编译模型...")
        model = torch.compile(model)
    
    # 初始化tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    return model, enc, ctx




# %% [markdown]
# ## generate summary

# %%
def generate_summary(model, prompt_tokens, enc, ctx, eos_token_id=None):
    """
    使用KV cache加速生成摘要
    
    参数:
        model: GPT模型
        prompt_tokens: prompt的token列表
        enc: tokenizer
        ctx: autocast上下文
        eos_token_id: 结束token ID，用于提前停止
    
    返回:
        generated_text: 生成的完整文本（包含prompt和摘要）
    """
    # 检查长度，如果太长则截断
    if len(prompt_tokens) > config.block_size - config.max_new_tokens:
        dialogue_start_tokens = enc.encode(config.dialogue_start)
        summary_start_tokens = enc.encode(config.summary_start)
        available_space = config.block_size - config.max_new_tokens - len(dialogue_start_tokens) - len(summary_start_tokens)
        
        if available_space > 0:
            # 找到对话部分的tokens
            # 解码prompt找到对话部分
            prompt_text = enc.decode(prompt_tokens)
            dialogue_start_pos = prompt_text.find(config.dialogue_start)
            summary_start_pos = prompt_text.find(config.summary_start)
            
            if dialogue_start_pos >= 0 and summary_start_pos > dialogue_start_pos:
                dialogue_text = prompt_text[dialogue_start_pos + len(config.dialogue_start):summary_start_pos]
                dialogue_tokens = enc.encode(dialogue_text)
                truncated_dialogue_tokens = dialogue_tokens[:available_space]
                prompt_tokens = dialogue_start_tokens + truncated_dialogue_tokens + summary_start_tokens
            else:
                # 如果找不到标记，直接截断
                prompt_tokens = prompt_tokens[:config.block_size - config.max_new_tokens]
        else:
            # 如果空间不足，只保留必要的标记
            prompt_tokens = dialogue_start_tokens + summary_start_tokens
    
    # 转换为tensor
    x = torch.tensor(prompt_tokens, dtype=torch.long, device=config.device)[None, ...]
    
    # 使用KV cache加速生成摘要
    with torch.no_grad():
        with ctx:
            y = model.generate_with_kv_cache(
                x,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                eos_token_id=eos_token_id
            )
    
    # 解码生成的token
    generated_tokens = y.tolist()
    generated_text = enc.decode(generated_tokens)
    
    return generated_text

def extract_summary(generated_text, prompt_text, enc):
    """
    从生成的文本中提取摘要部分
    
    参数:
        generated_text: 生成的完整文本
        prompt_text: 原始prompt文本
        enc: tokenizer
    
    返回:
        generated_summary: 提取的摘要文本
    """
    # 提取生成的摘要（去除prompt部分）
    if config.summary_start in generated_text:
        generated_summary = generated_text.split(config.summary_start)[-1]
        
        # 去除结束标记
        if config.summary_end in generated_summary:
            generated_summary = generated_summary.split(config.summary_end)
    else:
        # 如果没找到标记，就从prompt长度之后开始提取
        if len(generated_text) > len(prompt_text):
            generated_summary = generated_text[len(prompt_text):]
        else:
            generated_summary = ""
    
    # 清理生成的摘要
    generated_summary = generated_summary.strip()
    
    # 如果生成的摘要为空，使用空字符串
    if not generated_summary:
        generated_summary = ""
    
    return generated_summary


def calculate_rouge(reference_summary, generated_summary):
    """
    计算ROUGE分数
    
    参数:
        reference_summary: 参考摘要
        generated_summary: 生成的摘要
    
    返回:
        dict: 包含rouge1, rouge2, rougeL的字典，如果失败则返回None
    """
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



# %% [markdown]
# ## main

# %%
# =============================================================================
# 第五部分：主函数
# =============================================================================

def main():
    """
    主函数：整合数据准备、训练和评估
    
    执行流程：
    1. 准备数据（如果数据文件不存在）
    2. 训练模型
    3. 评估模型
    """
    print("\n")
    print("=" * 80)
    print("GPT-2 摘要微调教学脚本".center(80))
    print("=" * 80)
    print("\n当前配置:")
    print(f"  数据集: {config.dataset_path}")
    print(f"  模型初始化: {config.init_from}")
    print(f"  设备: {config.device}")
    print(f"  批次大小: {config.batch_size}")
    print(f"  最大迭代次数: {config.max_iters}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  使用LoRA: {config.use_lora}")
    if config.use_lora:
        print(f"    LoRA Rank: {config.lora_rank}")
        print(f"    LoRA Alpha: {config.lora_alpha}")
        print(f"    LoRA Targets: {config.lora_target_modules}")
    
    # 步骤1: 准备数据
    data_dir = os.path.join('data', config.dataset)
    train_pkl = os.path.join(data_dir, 'train.pkl')
    
    print("expect: train_pkl:",train_pkl)
    
    if not os.path.exists(train_pkl):
        print("\n未找到处理后的数据文件，开始准备数据...")
        prepare_data()
    else:
        print("\n找到已处理的数据文件，跳过数据准备步骤")
        print(f"如需重新准备数据，请删除 {data_dir} 目录")
    
    # 步骤2: 训练模型
    if not config.eval_only:
        train()
    else:
        print("\neval_only=True，跳过训练")
    
    

main()

# %%
# # 在你的代码文件底部，或者一个新的 cell 中运行
# from tqdm import tqdm
# import time

# # 假设你的数据集已经准备好了
# data_path = 'data/samsum/train.pkl' 
# block_size = 1024
# dataset = SummarizationDataset(data_path, block_size)

# # 测试前1000个样本的加载时间
# total_time = 0
# num_samples_to_test = 1000

# start_time = time.time()
# for i in tqdm(range(num_samples_to_test)):
#     x, y = dataset[i]
# end_time = time.time()

# avg_time = (end_time - start_time) / num_samples_to_test * 1000  # 转换为毫秒
# print(f"\n平均每个样本的 __getitem__ 耗时: {avg_time:.4f} ms")

# %% [markdown]
# ## 评估

# %%

def evaluate():
    """
    评估模型性能
    
    测试流程：
    1. 加载训练好的模型
    2. 从测试集中读取样本
    3. 给定对话，让模型生成摘要（使用KV cache加速）
    4. 计算生成摘要与真实摘要的ROUGE分数
    """
    print("\n" + "=" * 80)
    print("开始评估...")
    print("=" * 80)
    
    # 加载模型
    model, enc, ctx = load_model()
    if model is None:
        return
    
    # 获取<|endoftext|>的token ID作为停止符
    eos_token_id = enc.encode(config.summary_end, allowed_special={config.summary_end})
    
    # 加载测试数据
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
    
    # 检查是否可以计算ROUGE
    rouge_result = calculate_rouge("test", "test")
    use_rouge = rouge_result is not None
    if not use_rouge:
        print("\n警告: 未安装rouge_score库，将跳过ROUGE评分")
        print("安装命令: pip install rouge-score")
    
    # 评估每个样本
    print("\n" + "-" * 80)
    print("开始生成和评估...")
    print("优化: 使用KV cache + 提前停止")
    print("-" * 80)
    
    all_rouge1_f = []
    all_rouge2_f = []
    all_rougeL_f = []
    
    for idx, (dialogue, reference_summary) in enumerate(zip(dialogues, summaries)):
        print(f"\n[样本 {idx+1}/{len(dialogues)}]")
        print(f"对话: {dialogue[:100]}..." if len(dialogue) > 100 else f"对话: {dialogue}")
        
        # 构建prompt（对话 + 摘要开始标记）
        prompt = config.dialogue_start + dialogue + config.summary_start
        
        # Tokenize prompt
        prompt_tokens = enc.encode(prompt, allowed_special={config.summary_end})
        
        # 检查长度
        if len(prompt_tokens) > config.block_size - config.max_new_tokens:
            print("  警告: prompt太长，进行截断")
        
        # 使用KV cache加速生成摘要
        generated_text = generate_summary(model, prompt_tokens, enc, ctx, eos_token_id=eos_token_id)
        
        # 提取摘要
        generated_summary = extract_summary(generated_text, prompt, enc)
        
        print(f"真实摘要: {reference_summary}")
        print(f"生成摘要: {generated_summary}")
        
        # 计算ROUGE分数
        if use_rouge:
            rouge_scores = calculate_rouge(reference_summary, generated_summary)
            if rouge_scores:
                rouge1_f = rouge_scores['rouge1']
                rouge2_f = rouge_scores['rouge2']
                rougeL_f = rouge_scores['rougeL']
                
                print(f"ROUGE-1: {rouge1_f:.4f}")
                print(f"ROUGE-2: {rouge2_f:.4f}")
                print(f"ROUGE-L: {rougeL_f:.4f}")
                
                all_rouge1_f.append(rouge1_f)
                all_rouge2_f.append(rouge2_f)
                all_rougeL_f.append(rougeL_f)
    
    # 打印平均分数
    if use_rouge and len(all_rouge1_f) > 0:
        print("\n" + "=" * 80)
        print("平均ROUGE分数:")
        print(f"  ROUGE-1: {np.mean(all_rouge1_f):.4f}")
        print(f"  ROUGE-2: {np.mean(all_rouge2_f):.4f}")
        print(f"  ROUGE-L: {np.mean(all_rougeL_f):.4f}")
        print("=" * 80)


def predict_test_set_fast():
    """
    使用KV cache加速的测试集推理
    
    优势：
    1. 使用KV cache，避免重复计算，大幅提升速度
    2. 设置提前停止符，遇到结束token立即停止
    3. 使用我们训练的模型，确保兼容性
    
    流程：
    1. 加载训练好的模型
    2. 读取测试集数据
    3. 使用KV cache逐样本生成摘要
    4. 保存为提交格式
    """
    print("\n" + "=" * 80)
    print("开始对测试集进行KV cache加速推理...")
    print("=" * 80)
    
    # 加载模型
    model, enc, ctx = load_model()
    if model is None:
        return
    
    # 获取<|endoftext|>的token ID作为停止符
    eos_token_id = enc.encode(config.summary_end, allowed_special={config.summary_end})
    
    # 读取测试数据
    print("\n加载测试数据...")
    test_file = '/kaggle/input/nanogpt-fudannlp-cs-30040/test.csv'
    
    if not os.path.exists(test_file):
        print(f"错误: 找不到测试文件 {test_file}")
        return
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_data.append({
                'id': row['id'],
                'dialogue': row['dialogue']
            })
    
    print(f"加载了 {len(test_data)} 条测试样本")
    
    # 准备保存结果
    results = []
    
    print("\n" + "-" * 80)
    print("开始生成摘要...")
    print(f"优化: 使用KV cache + 提前停止 (遇到 '{config.summary_end}' 立即停止)")
    print("-" * 80)
    
    # 对每个样本生成摘要
    import time
    start_time = time.time()
    
    for idx, sample in enumerate(test_data):
        sample_id = sample['id']
        dialogue = sample['dialogue']
        
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            if idx > 0:
                avg_time = elapsed / idx
                remaining = avg_time * (len(test_data) - idx)
                print(f"处理进度: {idx+1}/{len(test_data)} | 平均耗时: {avg_time:.2f}秒/样本 | 预计剩余: {remaining/60:.1f}分钟")
            else:
                print(f"处理进度: {idx+1}/{len(test_data)}")
        
        # 构建prompt（对话 + 摘要开始标记）
        prompt = config.dialogue_start + dialogue + config.summary_start
        
        # Tokenize prompt
        prompt_tokens = enc.encode(prompt, allowed_special={config.summary_end})
        
        # 使用KV cache加速生成摘要
        generated_text = generate_summary(model, prompt_tokens, enc, ctx, eos_token_id=eos_token_id)
        
        # 提取摘要
        generated_summary = extract_summary(generated_text, prompt, enc)
        
        results.append({
            'id': sample_id,
            'summary': generated_summary
        })
    
    total_time = time.time() - start_time
    print(f"\n生成完成！总耗时: {total_time/60:.1f}分钟 | 平均: {total_time/len(test_data):.2f}秒/样本")
    
    # 保存结果到CSV文件
    output_file = 'submission.csv'
    # output_path = os.path.join(config.out_dir, output_file) # Save to out_dir
    output_path = output_file # Save to working directory
    
    print(f"\n保存结果到 {output_path}")
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'summary'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"完成！生成了 {len(results)} 条摘要")
    print("=" * 80)
    
    return output_path


# %% [markdown]
# ## 测试evaluate

# %%
!pip install rouge-score
evaluate()

# %%
predict_test_set_fast()

```