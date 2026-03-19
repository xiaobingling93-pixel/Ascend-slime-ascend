# 昇腾（NPU）快速使用

本文档从搭建环境开始，在一小时内带您快速上手 slime-ascend，涵盖 NPU 环境配置、数据准备、训练启动和关键代码解析。

## 硬件支持说明

**slime-ascend** 支持华为昇腾 NPU 硬件平台：

- **Atlas A3/A2 训练系列产品**

**重要说明**：
- 请确保已正确安装 CANN 套件（推荐 CANN 8.5.0 或更高版本）
- 推荐使用 Linux arm64 环境进行部署


## 环境配置

我们提供了一键安装脚本，可快速搭建 NPU 环境：

```bash
conda create -n slime-ascend python=3.11
conda activate slime-ascend
# 进入项目根目录
git clone https://gitcode.com/Ascend/slime-ascend.git
# CANN 安装路径，请根据实际情况修改
export CANN_INSTALL_PATH=/usr/local/Ascend
# NPU 设备类型：A3 
export NPU_DEVICE=A3
# 加载 CANN 环境变量
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh
# 运行安装脚本,如果遇到任何错误,请参照报错信息及脚本注解进行排查
bash scripts/ascend_script/quick_install.sh
```

**安装脚本会自动完成以下工作**：

1. 从源码安装 SGLang 
2. 安装 torch、torch_npu、triton_ascend 等基础依赖包
3. 安装 sgl-kernel-npu（支持 A2 和 A3 两种硬件）
4. 安装 mbridge 和 Megatron-Bridge
5. 安装 Megatron-LM 
6. 安装 MindSpeed 
7. 安装 slime-ascend
8. 应用所有 NPU 相关补丁
9. 安装custom ops


## 训练脚本与参数概览

完成上述准备工作后，即可运行 NPU 训练脚本。我们提供了 GLM-4.7 在 NPU 上的训练脚本示例,参照[glm4.7-30B-A3B.md](../examples/glm4.7-30B-A3B.md)。


## 常见问题排查

### NPU 相关问题

1. **CANN 环境未正确加载**
   - 确保已执行 `source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh`
   - 检查 `ASCEND_TOOLKIT_HOME` 等环境变量是否正确设置

2. **NPU 设备不可见**
   - 检查 `ASCEND_RT_VISIBLE_DEVICES` 环境变量
   - 使用 `npu-smi info` 命令验证 NPU 设备状态

3. **显存不足**
   - 调整 `--sglang-mem-fraction-static` 参数（建议 0.6-0.8）
   - 减少 batch size 或并行度

4. **HCCL 通信错误**
   - 检查 `HCCL_HOST_SOCKET_PORT_RANGE` 和 `HCCL_NPU_SOCKET_PORT_RANGE` 配置
   - 确保端口范围内的端口未被占用

更多常见问题请参考 [Q&A](../zh/get_started/qa.md)
