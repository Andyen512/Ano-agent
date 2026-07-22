# Benchmark 模型分析

本文基于 [`benchmarks.md`](/data_4/liuyuan/lifebench/benchmarks.md) 中列出的模型名单，结合各模型公开仓库、论文摘要和官方说明，对这些模型的特点、结构、输入输出差异、优势与劣势进行整理，便于横向比较。

## 说明

- `benchmarks.md` 本身只提供模型名称与链接，不包含技术细节，因此本文是补充性分析文档。
- `Qwen2.5VL-7B` 这一行在原始文件中的链接指向 `Qwen3-VL` 仓库，模型名与链接版本口径存在轻微不一致，使用时建议单独核对。
- `Qwen/Qwen3.5-9B` 与前面多数 7B 视频聊天基线不是完全同一代模型，更接近新一代通用多模态基础模型，横向比较时建议单独标注。

## 总体分类

这些模型大致可以分为四类：

1. 早期视频聊天型 baseline
   - `Video-LLaVA-7B`
   - `VideoChat2-7B`
   - `Video-ChatGPT-7B`
   - `MiniGPT4-Video`
2. 强化时序或长视频建模的 Video-LLM
   - `VideoLLaMA2-7B`
   - `Video-CCAM-7B`
   - `mPLUG-Owl3-7B`
3. 新一代统一多模态基础模型
   - `Qwen2.5-VL-7B`
   - `Qwen/Qwen3.5-9B`
   - `VideoLLaMA3-7B`
4. 偏细粒度视频描述与综合理解路线
   - `Tarsier2-7B`

## 总结表

| 模型 | 核心特点 | 典型结构/骨干 | 输入侧区别 | 输出侧区别 | 优势 | 劣势 |
|---|---|---|---|---|---|---|
| Video-LLaVA-7B | 将图像和视频先对齐到统一视觉表征，再接入 LLM | unified visual representation + projector + LLM，强调 alignment before projection | 同时支持 image/video，偏统一视觉表征 | 主要输出自然语言回答与描述 | 结构清晰，复现成本较低，适合作为统一 image/video baseline | 长视频能力较弱，细粒度时序与结构化输出不是强项 |
| VideoChat2-7B | 公开视频理解 benchmark 中较经典的强基线 | 视频编码器 + 对齐模块 + Vicuna/Mistral 类 LLM，依赖较强训练配方 | 以视频为主，也可扩展图像任务 | 主要是开放式问答与描述 | benchmark 覆盖较广，工程成熟，比较适合做对照实验 | 仍是传统 Video-LLM 范式，原生长上下文与结构化输出一般 |
| Video-ChatGPT-7B | 典型早期视频对话模型 | 预训练视觉编码器 + 时空特征适配 + LLM | 视频输入为主 | 文本对话输出 | 历史代表性强，适合做早期基线参考 | 架构相对较早，时序理解和长视频建模偏弱 |
| MiniGPT4-Video | 强调视频帧与文本上下文交错建模 | 延续 MiniGPT 系列，将帧序列映射为可与文本交错的 token | 更强调视频与文本对话上下文融合 | 以自然语言回答为主 | 对多轮对话和上下文连续问答较友好 | 更偏 conversation，时序定位和结构化输出能力有限 |
| VideoLLaMA2-7B | 更重视时空建模，并扩展到音频/音视频理解 | 视频编码器 + LLM，部分版本支持 audio/audio-visual | 支持 video，也支持 audio 或 audio-visual 版本 | 主要为文本回答 | 相比早期模型，多模态范围更广，时序能力更完整 | 仍以拼接式架构为主，长视频与结构化表达不如更新底座模型 |
| Video-CCAM-7B | 用因果交叉注意力掩码增强时序建模和长视频迁移 | 在 projector 或 cross-attention 路径中引入 CCAM 机制 | 训练常基于图像和短视频，目标是泛化到更长视频 | 主要输出文本 | 对时序顺序敏感，长视频迁移能力更强，效率较好 | 更像时序增强型改造，通用多模态能力不如新一代 foundation 模型 |
| Qwen2.5-VL-7B | 新一代通用视觉语言模型，OCR、图表、定位和长视频都强 | dynamic resolution visual encoder + 时空位置编码 + Qwen VL 底座 | image/video 统一输入能力强，可处理较长视频 | 不只文本，还支持 bbox、point、结构化抽取等 | 通用性强，既能问答也能做 grounding、OCR 和结构化输出 | 推理成本通常高于早期 7B 视频聊天模型，和旧基线不完全同层比较 |
| Qwen/Qwen3.5-9B | 更偏原生多模态与 agent 路线，长上下文能力强 | vision encoder + causal LM，采用更现代的长上下文架构 | image/video/text 统一输入，长上下文更强 | 文本为主，也更适合工具调用和统一多模态任务 | 上下文窗口大，综合能力强，适合复杂真实任务 | 与传统 Video-LLM benchmark baseline 不完全同代，公平比较时需要单列 |
| VideoLLaMA3-7B | 更像统一 image/video 基础模型而非单纯 video chat | 基于更现代视觉骨干与 Qwen 系语言底座，联合 image/video 训练 | image/video 统一输入 | 主要是文本回答与描述 | 综合 benchmark 表现更均衡，统一性更好 | 若任务强调结构化输出，其卖点不如 Qwen2.5-VL 明确 |
| mPLUG-Owl3-7B | 重点解决长图像序列和长视频理解 | 面向 long image-sequence understanding 的多模态 LLM | 对长图像序列、长视频、多图串联输入更友好 | 主要为文本输出 | 长序列理解能力较强，适合多图、多帧、长视频任务 | 短视频纯 QA 任务中优势可能不如通用 foundation 模型明显 |
| Tarsier2-7B | 强调细粒度视频描述、时序对齐和综合视频理解 | 基于 Qwen2-VL 类底座，加入大规模 video-text 扩展与偏好优化 | 视频理解和细节描述能力突出 | 文本描述、问答、部分 grounding 更均衡 | 视频描述质量高，对时序细节和复杂场景更敏感 | 若任务是严格格式输出或固定模板回答，需更仔细设计 prompt |

## 输入与输出差异

| 维度 | 早期或传统 Video-LLM | 新一代统一多模态模型 |
|---|---|---|
| 输入形式 | 以 `video -> text` 为主，有些兼容图像 | 更常见 `image + video + text` 统一输入 |
| 时间建模 | 多依赖采样帧与 projector/adapter | 更强调原生时空编码、长序列机制与动态采样 |
| 音频支持 | 多数不强，`VideoLLaMA2` 较突出 | 部分模型统一到更广多模态，但公开能力差异较大 |
| 输出形式 | 主要是自然语言回答、摘要、描述 | 除文本外，常支持 `bbox`、`point`、`JSON`、时间定位等 |
| 长视频能力 | 通常依赖少量采样帧，易丢失信息 | 更重视长上下文和长视频覆盖 |
| 适用场景 | 更适合做 benchmark baseline | 更适合真实应用、复杂任务与结构化理解 |

## 抽帧方式与处理帧数

这一节区分两种口径：

- `模型官方/常见设定`：该模型论文、仓库或默认 processor 中常见的抽帧方式与帧数。
- `当前 lifebench 实际调用`：你这套仓库在 [`lifebench_infer.py`](/data_4/liuyuan/lifebench/evaluation/lifebench_infer.py) 和相关 `compat` 适配器里真正使用的方式。

### 总表

| 模型 | 模型官方/常见设定 | 当前 lifebench 实际调用 | 备注 |
|---|---|---|---|
| Video-LLaVA-7B | 常见为 `uniform` 均匀抽帧，默认 `8` 帧 | 当前适配器调用模型自带 `video_processor`，实际帧数跟 `model.get_video_tower().config.num_frames` 走，仓库训练默认是 `8` 帧 | 当前 runner 没有额外传 `fps/max_frames`，基本沿用模型默认 |
| VideoChat2-7B | 官方配置里常见 `4` 帧 checkpoint，测试可插值到更多帧 | 当前 `compat/videochat2_once.py` 默认 `num_segments=8`，使用均匀分段抽帧 `get_index(len(vr), num_segments)`，即均匀抽 `8` 帧 | 这是“本地实际 8 帧”与“模型原始 checkpoint 4 帧”差异最明显的一个 |
| Video-ChatGPT-7B | `load_video(..., num_frm=100)`，按全视频均匀分段取中间帧，最多 `100` 帧 | 当前改为全视频均匀抽样，最多 `32` 帧，即 `load_video(..., num_frm=min(args.max_frames, 32))` | 从旧的 `100` 帧改成受控 `32` 帧预算，以提高跨模型公平性 |
| MiniGPT4-Video | 常见做法是按全视频长度等间隔采样，Mistral 版本常取 `90` 帧，其他版本常取 `45` 帧 | 当前改为全视频等间隔抽帧，最多 `32` 帧；通过 `sampling_interval = total_num_frames / max_frames` 实现 | 从旧的 `90/45` 帧口径统一收敛到 `32` 帧预算 |
| VideoLLaMA2-7B | 模型名就是 `VideoLLaMA2.1-7B-16F`，常见设定是均匀抽 `16` 帧 | 当前 `processor["video"]` 内部调用 `process_video(..., num_frames=num_frames)`，实际也是均匀抽 `16` 帧 | 这是最标准、最清晰的一类固定帧模型 |
| Video-CCAM-7B | 官方教程常见 `uniform` 抽 `32` 帧，也支持按 `fps` 抽帧 | 当前调用 `load_decord(..., sample_type="uniform", num_frames=min(args.max_frames, 32))`，默认就是均匀抽 `32` 帧 | 本地没有启用它支持的 `fps` 模式 |
| Qwen2.5-VL-7B | 更偏 `dynamic FPS sampling`，不是固定帧数模型 | 当前改为受控动态采样：`fps = min(1.0, 32 / 视频秒数)`，目标是全视频覆盖且总体控制在约 `32` 帧预算内 | 这是典型“按采样率驱动，而非固定 N 帧”的模型 |
| Qwen3.5-9B | 官方 API 示例默认 `fps=2` 且 `do_sample_frames=True`，也支持直接指定 `num_frames` | 当前 local runner 里统一改成 `processor_kwargs={\"num_frames\": min(args.max_frames, 32)}`，即最多 `32` 帧 | 本地调用改为固定 `32` 帧上限，以便和其他模型更公平对比 |
| VideoLLaMA3-7B | processor 默认配置为 `fps=1`、`max_frames=128`；短视频时按 `fps` 采样，长视频或超限时退到均匀采样 | 当前改为 `fps = min(1.0, 32 / 视频秒数)` 且 `max_frames <= 32`，保证全视频覆盖和受控帧预算 | 不再像之前那样被裁到 `4` 帧 |
| mPLUG-Owl3-7B | 更偏长序列图像/视频理解，常见实现允许较灵活的视频帧序列 | 当前 `load_video_frames_simple(..., max_num_frames=min(args.max_frames, 32))`：先大致按 `1 FPS` 取候选帧，再均匀压缩到最多 `32` 帧 | 本地实现仍是简化采样器，但已和公平预算对齐 |
| Tarsier2-7B | 默认配置 `n_frames=16`、`max_n_frames=256`，采样策略为 `video_sampler_version: v1`，即均匀抽帧且保留首尾 | 当前直接读取 [`configs/tarser2_default_config.yaml`](/data_4/liuyuan/lifebench/code/Tarsier2-7B/configs/tarser2_default_config.yaml)，因此实际就是默认 `16` 帧、最多允许动态扩到 `256` 的那套配置 | 在你当前 quick-start 推理里，可近似视为均匀抽 `16` 帧 |

### 当前 lifebench 实际推理口径

如果只看你现在这套仓库真实跑分时的输入处理方式，可以近似总结为：

| 模型 | 当前 lifebench 抽帧方式 | 当前 lifebench 处理帧数 |
|---|---|---|
| Video-LLaVA-7B | 模型内置 video processor，通常均匀抽帧 | 约 `8` 帧 |
| VideoChat2-7B | 均匀分段抽帧 | `8` 帧 |
| Video-ChatGPT-7B | 全视频均匀分段取中间帧 | 最多 `32` 帧 |
| MiniGPT4-Video | 全视频等间隔抽帧 | 最多 `32` 帧 |
| VideoLLaMA2-7B | 均匀抽帧 | `16` 帧 |
| Video-CCAM-7B | 均匀抽帧 | `32` 帧 |
| Qwen2.5-VL-7B | 按 `fps = min(1.0, 32 / 视频秒数)` 动态抽样 | 约 `32` 帧预算 |
| Qwen3.5-9B | 直接限制 `num_frames` | 最多 `32` 帧 |
| VideoLLaMA3-7B | 按 `fps = min(1.0, 32 / 视频秒数)` 动态抽样，并限制上限 | 最多 `32` 帧 |
| mPLUG-Owl3-7B | 先近似按 `1 FPS` 取候选，再均匀压缩 | 最多 `32` 帧 |
| Tarsier2-7B | 默认均匀抽帧 | `16` 帧 |

### 特别需要注意的差异

- `VideoChat2-7B`：模型 checkpoint 常见是 `4` 帧，但你本地适配器实际喂的是 `8` 帧。
- `VideoLLaMA3-7B`：模型默认 processor 能到 `128` 帧，但当前 benchmark 已统一改成受控 `32` 帧预算，更强调公平对比而不是单模型极限。
- `Qwen2.5-VL-7B`：不是固定 `N` 帧范式，更像“按 FPS 和 token/pixel 预算动态处理”。
- `Qwen3.5-9B`：官方 API 示例常见 `fps=2`，但你本地这版代码为了公平性改成了 `num_frames<=32` 的 fixed-cap 路线。
- `mPLUG-Owl3-7B`：你本地实现仍是简化采样，并不一定等于官方推荐设置，但现在帧预算已统一到 `32`。

## 分组结论

### 1. 早期视频聊天型 baseline

代表模型包括 `Video-LLaVA-7B`、`VideoChat2-7B`、`Video-ChatGPT-7B`、`MiniGPT4-Video`。这类模型大多采用“视频编码器 + 投影器/桥接层 + 7B 语言模型”的典型结构，优势是设计清晰、社区使用广、适合做历史基线；不足是对长视频、细粒度时序、结构化输出支持通常较弱。

### 2. 强化时序或长视频建模

代表模型包括 `VideoLLaMA2-7B`、`Video-CCAM-7B`、`mPLUG-Owl3-7B`。这类模型的改进重点从“是否能看视频”转为“如何更好地建模时间、长序列和多模态融合”。它们通常比早期 baseline 更重视帧顺序、时序依赖与长视频迁移，但在通用多模态输出能力上未必超过更新的 foundation 模型。

### 3. 新一代统一多模态基础模型

代表模型包括 `Qwen2.5-VL-7B`、`Qwen/Qwen3.5-9B`、`VideoLLaMA3-7B`。这类模型输入更统一，常同时支持图像、视频和文本，并具备更强的 OCR、定位、结构化输出和长上下文能力。它们更接近“真实应用底座模型”，而不只是“视频问答模型”。

### 4. 偏细粒度视频描述路线

`Tarsier2-7B` 的特色并不只是视频问答分数，而是更注重细粒度视频描述、时序对齐和偏好优化。这种路线通常更适合需要详细视频理解、复杂场景描述和更自然回答风格的任务。

## 使用建议

如果目的是做 benchmark 基线对比，建议优先保留以下三类代表：

- 早期 baseline：`Video-ChatGPT-7B`、`VideoChat2-7B`、`Video-LLaVA-7B`
- 时序增强类：`VideoLLaMA2-7B`、`Video-CCAM-7B`、`mPLUG-Owl3-7B`
- 新一代统一多模态：`Qwen2.5-VL-7B`、`VideoLLaMA3-7B`

如果目的是做真实应用选型，建议额外关注以下维度：

- 是否支持长视频
- 是否支持结构化输出，例如 `JSON`、`bbox`、时间戳
- 是否支持音频或音视频联合理解
- 是否擅长细粒度视频描述
- 是否具备较强 OCR、图表理解和 grounding 能力

## 建议补充到 benchmark 表中的字段

为了让 `benchmarks.md` 更适合后续实验与报告，建议额外补三列：

- 是否支持音频
- 是否支持结构化输出，例如 `JSON`、`bbox`、`timestamp`
- 是否偏长视频建模

## 参考链接

- Video-LLaVA: https://github.com/PKU-YuanGroup/Video-LLaVA
- VideoChat2 / Ask-Anything: https://github.com/OpenGVLab/Ask-Anything
- Video-ChatGPT: https://github.com/mbzuai-oryx/Video-ChatGPT
- MiniGPT4-Video: https://github.com/Vision-CAIR/MiniGPT4-video
- VideoLLaMA2: https://github.com/DAMO-NLP-SG/VideoLLaMA2
- Video-CCAM: https://github.com/QQ-MM/Video-CCAM
- Qwen2.5-VL: https://qwenlm.github.io/blog/qwen2.5-vl/
- Qwen3.5-9B: https://modelscope.cn/models/Qwen/Qwen3.5-9B
- VideoLLaMA3: https://github.com/DAMO-NLP-SG/VideoLLaMA3
- mPLUG-Owl3: https://github.com/X-PLUG/mPLUG-Owl
- Tarsier2: https://github.com/bytedance/tarsier
