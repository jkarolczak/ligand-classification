ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

##############################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX 8.0"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CUBLAS_WORKSPACE_CONFIG=":4096:8"
ENV PYTHONPATH="/app/src"

RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN pip install sklearn
RUN pip install pandas
RUN pip install torchmetrics>=0.9.0
RUN pip install addict
RUN pip install yapf
RUN pip install neptune-client>=0.14.2

ENV MAX_JOBS=4
RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
                           --install-option="--force_cuda" \
                           --install-option="--blas=openblas"
                           
              
