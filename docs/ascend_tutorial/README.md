# slime-ascend

# 简介

slime 是为 RL scaling 设计的 LLM post‑training 框架, slime-ascend 是针对华为昇腾设备的 slime 适配开发仓。

# 最新消息

---

- [ Mar 19, 2026 ]: 🚀 slime-ascend 已支持 [GLM-4.7-Flash](./examples/glm4.7-30B-A3B.md)、[Qwen3-8B](./examples/qwen3-8B.md) 、[Qwen3-VL-8B](./examples/qwen3-vl-8B.md) 等模型的GRPO算法！


# 版本说明

---
slime-ascend 依赖配套如下表：

<table>
  <tr>
    <th>依赖软件</th>
    <th>版本</th>
  </tr>
  <tr>
    <td>昇腾NPU驱动</td>
    <td rowspan="2">8.5.0</td>
  <tr>
    <td>昇腾NPU固件</td>
  </tr>
  <tr>
    <td>CANN版本</td>
    <td>8.5.0</td>
  </tr>
  <tr>
  </tr>
  <tr>
    <td>Python</td>
    <td>3.11.x</td>
  </tr>
  <tr>
    <td>PyTorch</td>
    <td>2.8.0</td>
  </tr>
  <tr>
    <td>torch_npu</td>
    <td>2.8.0</td>
  </tr>
  <tr>
    <td>triton_ascend</td>
    <td>3.2.0</td>
  </tr>
</table>



# 安装与快速上手

---

快速在昇腾训练设备上运行 slime-ascend 的快速指南，请参考：[快速开始指南](./get_started/quick_start.md)


# 使用指南

---


<table>
  <thead>
    <tr>
      <th>训练算法</th>
      <th>支持模型</th>
      <th>发布状态</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td  rowspan="2"> GRPO</td>
      <td  rowspan="1">
        <a href="./examples/glm4.7-30B-A3B.md">GLM-4.7-Flash</a> <br>
      </td>
      <td> Preview</td>
    </tr>
    </tr>
    <tr>
      <td  rowspan="1">
        <a href="./examples/qwen3-8B.md">Qwen3-8B</a> <br>
        <a href="./examples/qwen3-vl-8B.md">Qwen3-VL-8B</a> <br>
      </td>
      <td> Preview</td>
    </tr>
  </tbody>
</table>

注意："Preview"发布状态表示预览非正式发布版本，"Released"发布状态表示正式发布版本。


#  安全声明

---

[slime-ascend 安全声明](./SECURITYNOTE.md)


# 免责声明

---

## 致slime-ascend用者

1. slime-ascend 提供的模型仅供您用于非商业目的。
2. 对于各模型，slime-ascend 平台仅提示性地向您建议可用于训练的数据集，华为不提供任何数据集，如您使用这些数据集进行训练，请您特别注意应遵守对应数据集的 License，如您因使用数据集而产生侵权纠纷，华为不承担任何责任。
3. 如您在使用 slime-ascend 模型过程中，发现任何问题（包括但不限于功能问题、合规问题），请在 gitcode 提交 issue，我们将及时审视并解决。
4. MindSpeed 功能依赖的 Megatron 等第三方开源软件，均由第三方社区提供和维护，因第三方开源软件导致的问题的修复依赖相关社区的贡献和反馈。您应理解，MindSpeed 仓库不保证对第三方开源软件本身的问题进行修复，也不保证会测试、纠正所有第三方开源软件的漏洞和错误。

## 致数据集所有者

如果您不希望您的数据集在 slime-ascend 中的模型被提及，或希望更新 slime-ascend 中的模型关于您的数据集的描述，请在gitcode 提交 issue，我们将根据您的 issue 要求删除或更新您的数据集描述。衷心感谢您对 slime-ascend 的理解和贡献。

## License声明

slime-ascend 提供的模型，如模型目录下存在 License 的，以该 License 为准。如模型目录下不存在 License 的，以 Apache 2.0 许可证许可，对应许可证文本可查阅 slime-ascend 根目录。

# 致谢

---

slime-ascend 由华为公司的下列部门以及昇腾生态合作伙伴联合贡献 ：

华为公司：

- 计算产品线

感谢来自社区的每一个PR。