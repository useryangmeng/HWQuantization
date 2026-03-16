import torch
import torch_npu

# 1. 检查NPU是否被识别
print("NPU是否可用:", torch.npu.is_available())
# 2. 检查NPU设备数量
print("NPU设备数:", torch.npu.device_count())
# 3. 测试基础张量操作（核心验证）
try:
    x = torch.tensor([1,2,3]).npu()
    print("NPU张量创建成功:", x)
    print("NPU张量运算:", x + 1)
except Exception as e:
    print("基础NPU操作失败，错误:", e)