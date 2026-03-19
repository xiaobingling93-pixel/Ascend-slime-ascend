#!/bin/bash
set -e
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
NPU_DEVICE=${NPU_DEVICE:=A3}

MEGATRON_COMMIT=3714d81d
MindSpeed_COMMIT=fc63de5c
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

echo "1. install SGLang from source"
git clone -b v0.5.8 https://github.com/sgl-project/sglang.git
cd sglang
mv python/pyproject_other.toml python/pyproject.toml
pip install -e python[srt_npu]
git checkout . && git checkout sglang-slime
cd ..

echo "2. install torch & torch_npu & triton_ascend & other basic packages"
pip install torch==2.8.0 torch_npu==2.8.0.post2 torchvision==0.23.0 triton_ascend==3.2.0 transformers==5.0.0

echo "3.install sgl-kernel-npu from release whl"
if [ "$NPU_DEVICE" = "A3" ]; then
    wget --no-check-certificate https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.02.01/sgl-kernel-npu-2026.02.01-torch2.8.0-py311-cann8.5.0-a3-aarch64.zip
fi
if [ "$NPU_DEVICE" = "A2" ]; then
    wget --no-check-certificate https://github.com/sgl-project/sgl-kernel-npu/releases/download/2026.02.01/sgl-kernel-npu-2026.02.01-torch2.8.0-py311-cann8.5.0-910b-aarch64.zip
fi
unzip sgl-kernel-npu*.zip
pip install torch_memory_saver*.whl
pip install sgl_kernel_npu*.whl
pip install deep_ep*.whl
# echo "3. install sgl-kernel-npu form source, detailed readme in https://github.com/sgl-project/sgl-kernel-npu/blob/main/python/deep_ep/README.md"
# git clone https://github.com/sgl-project/sgl-kernel-npu.git
# cd sgl-kernel-npu
# git checkout ba46a30
# sed -i '101s/^/# /' build.sh
# if [ "$NPU_DEVICE" = "A3" ]; then
#     bash build.sh
# fi
# if [ "$NPU_DEVICE" = "A2" ]; then
#     bash build.sh -a deepep2
# fi
# pip install output/torch_memory_saver*.whl
# pip install output/sgl_kernel_npu*.whl
# pip install output/deep_ep*.whl
# cd "$(pip show deep-ep | grep -E '^Location:' | awk '{print $2}')" && ln -s deep_ep/deep_ep_cpp*.so && cd -
# python -c "import deep_ep; print(deep_ep.__path__)"
# cd ..
echo "4.install mbridge & Megatron-Bridge"
git clone https://github.com/ISEEKYAN/mbridge.git 
cd mbridge && git checkout 89eb10887887bc74853f89a4de258c0702932a1c
pip install -e . && cd ..

git clone https://github.com/fzyzcjy/Megatron-Bridge.git
cd Megatron-Bridge && git checkout dev_rl && cd ..
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation

git clone https://github.com/NVIDIA/Megatron-LM.git --recursive && \
cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT} && \
pip install -e . && cd ..

echo "5.install MindSpeed"
git clone https://gitcode.com/Ascend/MindSpeed.git && \
cd MindSpeed/ && git checkout ${MindSpeed_COMMIT} && \
pip install -e . && cd ..

echo "6.install slime-ascend"
cd slime-ascend/ 
pip install -e . 

echo "7.apply npu patches"
cd ../sglang
git am ../slime-ascend/docker/npu_patch/v0.2.2/sglang/*
cd ../Megatron-LM/
git am ../slime-ascend/docker/npu_patch/v0.2.2/megatron/*
cd ../Megatron-Bridge/
git am ../slime-ascend/docker/npu_patch/v0.2.2/megatron-bridge/*
cd ../MindSpeed/
git am ../slime-ascend/docker/npu_patch/v0.2.2/mindspeed/*
cd ../mbridge/
git am ../slime-ascend/docker/npu_patch/v0.2.2/mbridge/*

echo "8. install custom ops, this will be removed in the future after sglang is updated. please refer to https://gitcode.com/cann/cann-recipes-infer/issues/122 if you encounter version check error."
cd ..
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer/ops/ascendc
bash build.sh
./output/CANN-custom_ops-*.run
cd torch_ops_extension
bash build_and_install.sh