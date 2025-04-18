# Use to verify pytorch is installed properly and a GPU w/ CUDA is available
import torch

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_properties(0))
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
