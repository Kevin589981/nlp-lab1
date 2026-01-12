#!/usr/bin/env python
"""
DDP训练启动脚本
在Kaggle上使用2个Tesla T4 GPU进行分布式训练
"""

import os
import subprocess
import torch

def main():
    # 检查GPU数量
    n_gpus = torch.cuda.device_count()
    print(f"检测到 {n_gpus} 个GPU")
    
    if n_gpus < 2:
        print("警告: 只检测到1个GPU，将使用单GPU训练")
        print("直接运行 version12.py 即可")
        import version12
        version12.main()
    else:
        print(f"启动DDP训练，使用 {n_gpus} 个GPU")
        
        # 使用torchrun启动分布式训练
        cmd = [
            "torchrun",
            f"--nproc_per_node={n_gpus}",
            "--standalone",
            "version12.py"
        ]
        
        print(f"执行命令: {' '.join(cmd)}")
        subprocess.run(cmd)

if __name__ == "__main__":
    main()
