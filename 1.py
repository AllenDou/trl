from accelerate import Accelerator
import torch
import torch.nn as nn
import pdb; pdb.set_trace()
# 初始化 Accelerator
accelerator = Accelerator()

# 定义模型
model = nn.Linear(10, 2)

# 使用 Accelerator 进行分布式训练
model = accelerator.prepare(model)  # 可能会被 wrapped in DDP/DataParallel/etc.

# 获取原始模型
unwrapped_model = accelerator.unwrap_model(model)

# 现在可以安全地保存模型
torch.save(unwrapped_model.state_dict(), "model.pth")

