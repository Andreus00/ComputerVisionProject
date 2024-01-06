import torch

def getTorchType(string):
    TORCH_TYPES = {
        'float32': torch.float32,
        'float64': torch.float64,
        'float16': torch.float16,
        'int32': torch.int32,
        'int64': torch.int64,
        'int16': torch.int16,
        'int8': torch.int8,
        'uint8': torch.uint8,
        'bool': torch.bool
    }
    return TORCH_TYPES.get(string, torch.float16)