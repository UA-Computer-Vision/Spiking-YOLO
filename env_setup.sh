# Parameters needed to build torch and torchvision from source
# Specifically for Jetson Orin Nano
export USE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.7"  # Jetson Orin Nano arch
export CUDA_HOME=/usr/local/cuda-12.6
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CUDNN_INCLUDE_DIR=/usr/include
export CUDNN_LIB_DIR=/usr/lib/aarch64-linux-gnu
export CMAKE_PREFIX_PATH="$(dirname $(which python))/../"
export USE_SYSTEM_NCCL=0
export USE_NCCL=0
export USE_MKLDNN=0
export USE_DISTRIBUTED=0
export BUILD_CAFFE2=0
export MAX_JOBS=1