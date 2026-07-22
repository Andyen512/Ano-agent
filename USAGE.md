# LifeBench 使用说明

## 1. 项目目标

这个目录用于做一组视频大模型在居家安全视频上的统一推理与评测。整体流程是：

1. 准备测试视频和结构化提示词
2. 用统一入口对不同模型做单视频推理
3. 把模型原始回答整理成统一 JSON 结果
4. 用统一评测脚本把预测结果和 Ground Truth 对齐打分

核心目录：

- 数据目录：[data](/data_4/liuyuan/lifebench/data)
- 模型目录：[models](/data_4/liuyuan/lifebench/models)
- 单模型推理入口：[lifebench_infer.py](/data_4/liuyuan/lifebench/evaluation/lifebench_infer.py)
- 批量 benchmark 流水线：[benchmark_autopipeline.py](/data_4/liuyuan/lifebench/evaluation/benchmark_autopipeline.py)
- 评测入口：[evaluate_outputs.py](/data_4/liuyuan/lifebench/evaluation/evaluate_outputs.py)

## 2. 数据处理流程

### 2.1 输入数据

当前 benchmark 测试视频在：

- [A01.mp4](/data_4/liuyuan/lifebench/data/A01.mp4)
- [A02.mp4](/data_4/liuyuan/lifebench/data/A02.mp4)
- [A03.mp4](/data_4/liuyuan/lifebench/data/A03.mp4)
- [A04.mp4](/data_4/liuyuan/lifebench/data/A04.mp4)

结构化推理提示词在：

- [infer_prompt_structured.txt](/data_4/liuyuan/lifebench/data/infer_prompt_structured.txt)

Ground Truth 在：

- [gen_prompt.txt](/data_4/liuyuan/lifebench/data/gen_prompt.txt)

### 2.2 推理阶段的数据流

统一推理入口 [lifebench_infer.py](/data_4/liuyuan/lifebench/evaluation/lifebench_infer.py) 会做这些事：

1. 根据 `--backend` 和 `--model-id` 找到本地模型目录
2. 按不同模型后端调用对应 runner
3. 读取视频并按各模型自己的方式抽帧/处理视觉输入
4. 把提示词送进模型，得到原始文本输出 `response`
5. 尝试从输出里直接提取 JSON；如果模型没严格返回 JSON，就做一层规则化补救
6. 最终写出统一格式的结果 JSON

统一输出 JSON 里包含：

- `backend`
- `model_id`
- `video_path`
- `prompt`
- `response`
- `parsed_response`
- `generation`
- `elapsed_seconds`

### 2.3 评测阶段的数据流

评测脚本 [evaluate_outputs.py](/data_4/liuyuan/lifebench/evaluation/evaluate_outputs.py) 会：

1. 从 [gen_prompt.txt](/data_4/liuyuan/lifebench/data/gen_prompt.txt) 读取 Ground Truth
2. 在 [common.py](/data_4/liuyuan/lifebench/evaluation/common.py) 中把每行文本解析成结构化标签
3. 从推理输出目录里收集每个模型的预测 JSON
4. 对每条预测做 `structured_prediction()`，把模型回答统一抽成：
   - `has_risk`
   - `safe_segment_desc`
   - `risk_description`
   - `solution`
   - `time_spans`
5. 计算风险存在、时间 IoU、无风险描述、风险描述、解决方案等指标
6. 汇总为每模型平均分

Ground Truth 解析相关函数：

- [parse_ground_truth_line](/data_4/liuyuan/lifebench/evaluation/common.py#L347)
- [load_ground_truth](/data_4/liuyuan/lifebench/evaluation/common.py#L396)

预测结构化相关函数：

- [structured_prediction](/data_4/liuyuan/lifebench/evaluation/common.py#L493)

## 3. 当前模型状态

### 3.1 已接通并进入评测的模型

- `LanguageBind/Video-LLaVA-7B`
- `OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf`
- `MBZUAI/Video-ChatGPT-7B`
- `Vision-CAIR/MiniGPT4-Video`
- `DAMO-NLP-SG/VideoLLaMA2.1-7B-16F`
- `JaronTHU/Video-CCAM-7B-v1.2`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `Qwen/Qwen3.5-9B`
- `DAMO-NLP-SG/VideoLLaMA3-7B`
- `mPLUG/mPLUG-Owl3-7B-241101`
- `omni-research/Tarsier2-7b-0115`

### 3.2 最终结果表格

自动生成的最终汇总表在：

- [FINAL_RESULTS.md](/data_4/liuyuan/lifebench/FINAL_RESULTS.md)
- [final_model_results.csv](/data_4/liuyuan/lifebench/outputs/evaluation/latest/final_model_results.csv)

### 3.3 兼容适配说明

当前有几条模型通过本地兼容脚本接入：

- [videollava_once.py](/data_4/liuyuan/lifebench/compat/videollava_once.py)
- [videochat2_once.py](/data_4/liuyuan/lifebench/compat/videochat2_once.py)
- [minigpt4_video_once.py](/data_4/liuyuan/lifebench/compat/minigpt4_video_once.py)

`Qwen2.5-VL` 还依赖本地隔离的新版 Transformers shim：

- [/data_4/liuyuan/lifebench/.vendor/qwen251_shim](/data_4/liuyuan/lifebench/.vendor/qwen251_shim)
- [/data_4/liuyuan/lifebench/.vendor/minigpt4_transformers](/data_4/liuyuan/lifebench/.vendor/minigpt4_transformers)
- [/data_4/liuyuan/lifebench/.vendor/tarsier_flashattn](/data_4/liuyuan/lifebench/.vendor/tarsier_flashattn)

`Qwen3.5-9B` 需要更新的 Transformers。当前这套框架已经在本地接入并验证了 `transformers 5.4.0` 的 shim，建议单独装到：

- [/data_4/liuyuan/lifebench/.vendor/qwen35_shim](/data_4/liuyuan/lifebench/.vendor/qwen35_shim)

## 4. 单模型推理方法

### 4.1 激活环境

```bash
. /data_4/liuyuan/lifebench/activate_lifebench_vlm.sh
```

### 4.2 单条视频推理示例

示例：Qwen2.5-VL

```bash
python /data_4/liuyuan/lifebench/evaluation/lifebench_infer.py \
  --backend qwen25vl \
  --model-id Qwen/Qwen2.5-VL-7B-Instruct \
  --video-path /data_4/liuyuan/lifebench/data/A01.mp4 \
  --prompt-file /data_4/liuyuan/lifebench/data/infer_prompt_structured.txt \
  --max-new-tokens 96 \
  --output-json /data_4/liuyuan/lifebench/outputs/manual_qwen_a01.json
```

示例：Qwen3.5-9B

```bash
python /data_4/liuyuan/lifebench/evaluation/lifebench_infer.py \
  --backend qwen35vl \
  --model-id Qwen/Qwen3.5-9B \
  --model-path /data_4/liuyuan/lifebench/models/Qwen/Qwen3.5-9B \
  --video-path /data_4/liuyuan/lifebench/code/Ask-Anything/example/yoga.mp4 \
  --prompt "Describe the video briefly in one sentence." \
  --max-new-tokens 96 \
  --max-frames 64 \
  --output-json /data_4/liuyuan/lifebench/outputs/manual_qwen35_yoga.json
```

常用后端名：

- `videollava`
- `videochat2`
- `videollama2`
- `videoccam`
- `qwen25vl`
- `qwen35vl`
- `videollama3`
- `mplugowl3`
- `videochatgpt`
- `minigpt4video`
- `commercial`

示例：闭源商用模型（通过 [llm.py](/data_4/liuyuan/lifebench/llm.py) 的 OpenAI-compatible 配置调用）

```bash
python /data_4/liuyuan/lifebench/evaluation/lifebench_infer.py \
  --backend commercial \
  --model-id gpt-5.4 \
  --video-path /data_4/liuyuan/lifebench/data/A01.mp4 \
  --prompt-file /data_4/liuyuan/lifebench/data/infer_prompt_structured.txt \
  --max-new-tokens 96 \
  --max-frames 12 \
  --output-json /data_4/liuyuan/lifebench/outputs/manual_commercial_a01.json
```

这个 backend 不依赖本地模型权重，而是会对视频做均匀抽帧，然后把采样帧按时间顺序送给商用多模态模型。

### 4.3 结果文件位置

手工指定的输出文件由 `--output-json` 决定。

如果走 benchmark 流水线，结果会写到：

- `/data_4/liuyuan/lifebench/outputs/benchmark_inference/<model_name>/<run_id>/`

如果你想在自定义数据集批跑里加入商用模型，可以显式传：

```bash
python /data_4/liuyuan/lifebench/evaluation/benchmark_custom_dataset.py \
  --videos-dir /path/to/videos \
  --prompt-file /data_4/liuyuan/lifebench/data/infer_prompt_structured.txt \
  --predictions-root /path/to/preds \
  --models commercial:gpt-5.4
```

这里的 `commercial:<model_id>` 会走同一个 `commercial` backend，产出的 JSON 可以直接进入现有评测脚本。

## 5. 批量 benchmark 推理方法

跑一轮当前可用模型：

```bash
python /data_4/liuyuan/lifebench/evaluation/benchmark_autopipeline.py --once --max-new-tokens 96
```

如果你想在 `caochunshui` 开始跑终端程序或 GPU 程序时自动让路，可以改用这个包装脚本：

```bash
bash /data_4/liuyuan/lifebench/run_benchmark_autopipeline_when_caochunshui_idle.sh --once --max-new-tokens 96
```

它的行为是：

1. 默认监控 `caochunshui`
2. 如果检测到他名下出现了非 shell 的终端进程，或者 GPU 计算进程，就终止你当前启动的 benchmark 子进程
3. 等他重新空闲至少 30 秒后，再自动重启同一条 benchmark 命令

检测细节：

- 默认忽略 `bash`、`tmux`、`systemd --user` 这类会话常驻进程
- 默认的 `any` 模式只把“终端里正在跑的非 shell 程序”或 “GPU 计算进程”当作忙
- 如果你想包别的命令，也可以直接调用
  [supervise_command_on_user_busy.py](/data_4/liuyuan/lifebench/supervise_command_on_user_busy.py)
  并在命令前加 `--`

这个脚本会：

1. 检查每个 benchmark 模型是否下载完整
2. 检查额外依赖是否齐全
3. 对可运行模型逐个跑 `A01` 到 `A04`
4. 自动调用评测脚本
5. 更新状态文件和最终汇总

主要输出位置：

- 模型状态目录：
  [/outputs/benchmark_inference/status](/data_4/liuyuan/lifebench/outputs/benchmark_inference/status)
- 模型日志目录：
  [/outputs/benchmark_inference/logs](/data_4/liuyuan/lifebench/outputs/benchmark_inference/logs)
- 模型预测目录：
  [/outputs/benchmark_inference](/data_4/liuyuan/lifebench/outputs/benchmark_inference)

## 5.1 真实居家数据初筛

如果你想直接用 `Qwen/Qwen3.5-9B` 对 [data/public_data](/data_4/liuyuan/lifebench/data/public_data) 里的真实居家视频做第一轮风险/异常初筛，可以使用：

```bash
. /data_4/liuyuan/lifebench/activate_lifebench_vlm.sh

python /data_4/liuyuan/lifebench/public_data_prescreen.py \
  --backend qwen35vl \
  --model-id Qwen/Qwen3.5-9B \
  --model-path /data_4/liuyuan/lifebench/models/Qwen/Qwen3.5-9B
```

默认 prompt 在：

- [public_data_prescreen_prompt.txt](/data_4/liuyuan/lifebench/data/public_data_prescreen_prompt.txt)

你后续可以直接改这个 prompt 文件再重跑。

默认输出目录会是：

- `/data_4/liuyuan/lifebench/outputs/public_data_prescreen/<run_id>/`

里面主要包括：

- `manifests/tasks.json`：本轮待跑视频清单
- `predictions/.../*.json`：每个视频的原始模型结果
- `batch_summary.json`：批量推理过程汇总
- `screening_report.json`：聚合后的初筛总表
- `screening_report.csv`：便于人工筛看的 CSV

只想先验证链路，不想全量跑时，可以先做一条 smoke test：

```bash
. /data_4/liuyuan/lifebench/activate_lifebench_vlm.sh

python /data_4/liuyuan/lifebench/public_data_prescreen.py \
  --backend qwen35vl \
  --model-id Qwen/Qwen3.5-9B \
  --model-path /data_4/liuyuan/lifebench/models/Qwen/Qwen3.5-9B \
  --datasets UR_Fall_Dataset \
  --limit 1
```

如果你只想生成 manifest 和输出目录结构，不实际调用模型，可以加：

```bash
--dry-run
```

## 6. 评测方法

### 6.1 只跑评测

```bash
python /data_4/liuyuan/lifebench/evaluation/evaluate_outputs.py \
  --predictions-root /data_4/liuyuan/lifebench/outputs/benchmark_inference \
  --output-dir /data_4/liuyuan/lifebench/outputs/evaluation/latest
```

### 6.2 评测输出

评测输出目录：

- [/outputs/evaluation/latest](/data_4/liuyuan/lifebench/outputs/evaluation/latest)

最重要的结果文件：

- 汇总 JSON：
  [summary.json](/data_4/liuyuan/lifebench/outputs/evaluation/latest/summary.json)
- 每条预测的结构化结果：
  [structured_predictions.json](/data_4/liuyuan/lifebench/outputs/evaluation/latest/structured_predictions.json)
- 每视频分数表：
  [summary.csv](/data_4/liuyuan/lifebench/outputs/evaluation/latest/summary.csv)

### 6.3 当前评测逻辑

默认会优先尝试 LLM Judge。

如果 Judge 不可用，会自动回退到启发式规则匹配。

主入口：

- [main](/data_4/liuyuan/lifebench/evaluation/evaluate_outputs.py#L264)
- [evaluate_prediction](/data_4/liuyuan/lifebench/evaluation/evaluate_outputs.py#L102)
- [summarize_by_model](/data_4/liuyuan/lifebench/evaluation/evaluate_outputs.py#L195)

## 7. 完整示例

下面改用这次 `generated_videos_20260329T130129Z` 评测里的真实样本 `R000001_abnormal` 做完整示例。这一节统一以当前仓库里真实存在的产物为准：

- 当前 GT：
  [gen_prompt_generated_videos_v2.txt](/data_4/liuyuan/lifebench/data/generated_videos/gen_prompt_generated_videos_v2.txt)
- 当前统一推理 prompt：
  [infer_prompt_structured.txt](/data_4/liuyuan/lifebench/data/infer_prompt_structured.txt)
- 当前真实推理输出：
  [R000001_abnormal.json](/data_4/liuyuan/lifebench/outputs/generated_videos_20260329T130129Z/benchmark_inference/commercial__gpt_5_4/20260329T130129Z/R000001_abnormal.json)
- 当前单条评测缓存：
  [R000001_abnormal.json](/data_4/liuyuan/lifebench/outputs/generated_videos_20260329T130129Z/evaluation/latest/per_prediction_results/gpt-5.4/R000001_abnormal.json)

### 7.1 输入

这个样本对应的视频文件在：

- [R000001_abnormal.mp4](/data_4/liuyuan/lifebench/data/generated_videos/家中客厅/儿童/跌倒失稳/儿童在沙发上蹦跳时滑倒摔下/abnormal/R000001_abnormal.mp4)

这次批跑的输入组合是：

- 视频目录：
  [generated_videos](/data_4/liuyuan/lifebench/data/generated_videos)
- 推理 prompt：
  [infer_prompt_structured.txt](/data_4/liuyuan/lifebench/data/infer_prompt_structured.txt)
- GT：
  [gen_prompt_generated_videos_v2.txt](/data_4/liuyuan/lifebench/data/generated_videos/gen_prompt_generated_videos_v2.txt)

其中 `R000001_abnormal` 在 GT 里的原始 TSV 行是：

```text
R000001_abnormal	家中客厅/儿童/跌倒失稳/儿童在沙发上蹦跳时滑倒摔下	abnormal	[Normal Behavior][0-2s] An East Asian boy is energetically jumping up and down on a beige fabric sofa in the living room. [Abnormal Sign][2-5s] On one high jump, his foot lands on the edge of a loose, smooth cushion, causing it to slide out from under him. He loses his balance, arms flailing. [Abnormal Result][5-8s] He slips sideways off the sofa, falling onto the hardwood floor with a thud. He lands on his side and starts crying, sitting on the floor rubbing his arm. A toy is knocked off the sofa beside him.	Solution for the person: Comfort the child and check for injuries. Do not move him if a serious injury is suspected; call for medical help if needed. If he is okay, help him stand up away from the sofa. Solution for the hazard source: Immediately stop the jumping activity. Secure all sofa cushions and remove any objects from the floor around the sofa. Follow-up prevention solution: Consider placing a soft, non-slip rug in front of the sofa to cushion potential falls. Consistently supervise and redirect high-energy play to safer areas like a carpeted floor or play mat.
```

当前这轮推理统一使用
[infer_prompt_structured.txt](/data_4/liuyuan/lifebench/data/infer_prompt_structured.txt)。
它要求模型先判断 `risk_status`，再按状态输出不同 JSON schema。对这条视频，模型最终应该走 `abnormal` schema，也就是只输出：

```json
{
  "risk_status": "abnormal",
  "risk_type": "",
  "risk_description": "",
  "time_spans": [[start_second, end_second]],
  "solution": ""
}
```

### 7.2 GT 结构化提取

当前 GT 的提取主入口是
[parse_ground_truth_line](/data_4/liuyuan/lifebench/evaluation/common.py#L1022)。
对这条 `generated_videos` 的 TSV，会分派到
[parse_ground_truth_tsv_line](/data_4/liuyuan/lifebench/evaluation/common.py#L1098)。
逻辑可以概括成下面几步：

1. 先把一行 TSV 拆成五段：
   - `video_id = "R000001_abnormal"`
   - `path_text = "家中客厅/儿童/跌倒失稳/儿童在沙发上蹦跳时滑倒摔下"`
   - `explicit_status_hint = "abnormal"`
   - `timeline_text = "[Normal Behavior][0-2s] ... [Abnormal Sign][2-5s] ... [Abnormal Result][5-8s] ..."`
   - `solution_text = "Solution for the person: ... Solution for the hazard source: ... Follow-up prevention solution: ..."`
2. 再把 `path_text` 按 `/` 切成固定字段：
   - `location = "家中客厅"`
   - `subject = "儿童"`
   - `annotated_risk_type = "跌倒失稳"`
   - `scenario = "儿童在沙发上蹦跳时滑倒摔下"`
3. `timeline_text` 会被 `parse_timeline()` 解析成三个带标签片段：
   - `[Normal Behavior][0-2s]`：沙发上正常蹦跳
   - `[Abnormal Sign][2-5s]`：踩到松动坐垫边缘并失衡
   - `[Abnormal Result][5-8s]`：从沙发侧滑摔到木地板上
4. 因为这条时间轴已经显式带了标签，`split_safe_and_risk_segments()` 会直接按标签分段：
   - `safe_segments = [0-2s]`
   - `risk_segments = [2-5s, 5-8s]`
   这里不需要再靠关键词去猜风险起点。
5. 风险状态直接采用第三列给出的 `abnormal`。风险类型则会先把路径里的 `跌倒失稳` 规范化成评测内部使用的枚举 `fall_instability`。
6. solution 解析会识别英文 section 标题：
   - `Solution for the person`
   - `Solution for the hazard source`
   - `Follow-up prevention solution`
   然后在 `summary_gt["solution"]` 里统一改写成当前预测 schema 使用的三段中文前缀：
   - `对人的解决方案：`
   - `对危险源的解决方案：`
   - `整体的解决方案：`
7. 最后再经过 [align_summary_to_risk_status](/data_4/liuyuan/lifebench/evaluation/common.py#L536) 对齐到 `abnormal` schema：
   - 保留 `risk_type`、`risk_description`、`solution`、`time_spans`
   - 清空 `video_description` 和 `safe_segment_desc`

### 7.3 R000001_abnormal 结构化结果

按上面的规则，这条样本在真正评测时，会以
[ground_truth_structured.json](/data_4/liuyuan/lifebench/outputs/generated_videos_20260329T130129Z/evaluation/latest/ground_truth_structured.json)
里的这段结构参与打分。下面直接贴实际内容：

```json
{
  "summary_gt": {
    "risk_status": "abnormal",
    "has_risk": true,
    "risk_type": "fall_instability",
    "video_description": "",
    "safe_segment_desc": "",
    "risk_description": "On one high jump, his foot lands on the edge of a loose, smooth cushion, causing it to slide out from under him. He loses his balance, arms flailing.；He slips sideways off the sofa, falling onto the hardwood floor with a thud. He lands on his side and starts crying, sitting on the floor rubbing his arm. A toy is knocked off the sofa beside him.",
    "solution": "对人的解决方案：Comfort the child and check for injuries. Do not move him if a serious injury is suspected; call for medical help if needed. If he is okay, help him stand up away from the sofa. 对危险源的解决方案：Immediately stop the jumping activity. Secure all sofa cushions and remove any objects from the floor around the sofa. 整体的解决方案：Consider placing a soft, non-slip rug in front of the sofa to cushion potential falls. Consistently supervise and redirect high-energy play to safer areas like a carpeted floor or play mat.",
    "time_spans": [
      [2.0, 8.0]
    ],
    "solution_sections": {
      "person_solution": "Comfort the child and check for injuries. Do not move him if a serious injury is suspected; call for medical help if needed. If he is okay, help him stand up away from the sofa.",
      "hazard_solution": "Immediately stop the jumping activity. Secure all sofa cushions and remove any objects from the floor around the sofa.",
      "prevention_solution": "Consider placing a soft, non-slip rug in front of the sofa to cushion potential falls. Consistently supervise and redirect high-energy play to safer areas like a carpeted floor or play mat."
    },
    "time_interval": [2.0, 8.0]
  }
}
```

### 7.4 模型推理输出

下面直接看 `gpt-5.4` 对该视频的当前真实输出。文件在：

- [R000001_abnormal.json](/data_4/liuyuan/lifebench/outputs/generated_videos_20260329T130129Z/benchmark_inference/commercial__gpt_5_4/20260329T130129Z/R000001_abnormal.json)

其 `parsed_response` 如下：

```json
{
  "risk_status": "abnormal",
  "risk_type": "跌倒失稳",
  "risk_description": "孩子在客厅沙发上跳下时，前方地面有毛绒玩具，落地后被玩具绊到并失去平衡，随后跪坐摔倒在木地板上，之后坐在地上哭泣，已发生实际跌倒异常。",
  "solution": "对人的解决方案：立即安抚并检查孩子四肢及头部是否受伤，必要时及时就医。\n对危险源的解决方案：清理落地区域的毛绒玩具，避免在沙发前方放置绊脚物。\n整体的解决方案：减少孩子在沙发上跳跃玩耍，并保持客厅活动区域整洁通畅。",
  "time_spans": [
    [1.0, 3.0]
  ]
}
```

### 7.5 最终得分是怎么算的

这条样本的单条评测结果保存在：

- [R000001_abnormal.json](/data_4/liuyuan/lifebench/outputs/generated_videos_20260329T130129Z/evaluation/latest/per_prediction_results/gpt-5.4/R000001_abnormal.json)
- [per_video_scores.json](/data_4/liuyuan/lifebench/outputs/generated_videos_20260329T130129Z/evaluation/latest/per_video_scores.json)

这次评测的 `judge_mode` 是 `mixed_openai`。含义不是“所有文本项都走 LLM judge”，而是：

- `risk_status_accuracy`：规则精确匹配，`abnormal == abnormal`，所以是 `1.0`
- `risk_type_score`：先把预测里的 `跌倒失稳` 规范化成 `fall_instability`，再做规则精确匹配；这条匹配成功，所以是 `5.0`
- `time_iou`：直接比较 GT 的 `[[2, 8]]` 和预测的 `[[1, 3]]`
- `video_description_score`：这条 GT 是 `abnormal`，按 schema 不该出现 `video_description`；模型确实没输出，所以这里不是文本相似度打分，而是字段缺失检查，记 `5.0`
- `risk_description_score`：交给 OpenAI judge 打分。这条只拿到 `2.0`，因为结果描述里把真实致因“踩到松动坐垫边缘滑脱失衡”说成了“被毛绒玩具绊倒”
- `solution_score`：把 `solution` 拆成三段后分别评，再取平均。当前三项子分数是：
  `person_solution_score = 4.0`
  `hazard_solution_score = 2.0`
  `overall_solution_score = 2.0`
  所以 `solution_score = (4.0 + 2.0 + 2.0) / 3 = 2.667`

这条样本的实际结果是：

- `risk_status_accuracy = 1.0`
- `risk_type_accuracy = 1.0`
- `risk_type_score = 5.0`
- `time_iou = 0.1429`
- `video_description_score = 5.0`
- `risk_description_score = 2.0`
- `solution_score = 2.667`
- `overall_score = 65.811`

最后再把这六项合成总分。当前实现里的权重是：

```text
normalized =
  0.15 * risk_status_accuracy
  + 0.15 * (risk_type_score / 5.0)
  + 0.15 * time_iou
  + 0.15 * (video_description_score / 5.0)
  + 0.20 * (risk_description_score / 5.0)
  + 0.20 * (solution_score / 5.0)

overall_score = normalized * 100
```

把这条样本的真实分数代进去就是：

```text
normalized =
  0.15 * 1.0
  + 0.15 * (5.0 / 5.0)
  + 0.15 * 0.1429
  + 0.15 * (5.0 / 5.0)
  + 0.20 * (2.0 / 5.0)
  + 0.20 * (2.667 / 5.0)
  = 0.658115

overall_score = 65.811
```

也就是说：

- `risk_status_accuracy` 和 `time_iou` 是 `0-1` 指标
- `risk_type_score`、`video_description_score`、`risk_description_score`、`solution_score` 是 `0-5` 指标
- 其中 `risk_type_score` 和某些 schema 检查项仍然是规则打分，不是 LLM 打分
- 文本类分数会先除以 `5`，再和 `risk_status_accuracy`、`time_iou` 一起按权重汇总成 `0-100` 的最终分数

这条样本最终只有 `65.811`，主要原因也很直接：

- 风险状态和风险类型都判断对了，所以前两项拿满
- `time_spans` 只标了 `1-3s`，而 GT 是 `2-8s`，只有很小重叠，所以 `time_iou` 很低
- `video_description_score = 5.0` 不是“描述得特别好”，而是因为在 `abnormal` schema 下正确省略了这个字段
- `risk_description` 抓到了“沙发跌落、木地板、哭泣”的结果，但把关键致因说错了，所以只有 `2.0`
- `solution` 里的“对人处理”还可以，但“危险源”和“整体预防”都没切中真实问题里的松动坐垫与活动区域防滑，所以整体只到 `2.667`

## 8. 当前建议

目前 10 个模型都已经完成统一推理和评测，当前阶段的重点不再是“补模型文件”，而是“优化效果与稳定性”。

当前最值得继续优化的是 `VideoLLaMA3`，因为它虽然已经能跑，但输出质量明显弱于其他模型。可以优先调这些项：

- 提示词
- 抽帧策略
- 解码参数
- 视觉 token 上限

其次可以关注：

- `MiniGPT4-Video`
  当前能跑通，但质量偏弱，推理链路也较脆弱，建议继续做输出清洗和生成参数调优。
- `Tarsier2`
  当前链路依赖本地 `flash_attn` 和本地 remote-code 兼容，建议后续做更稳的正式集成，减少对目录结构的特殊要求。

如果只看最终结果，建议直接查看：

- [FINAL_RESULTS.md](/data_4/liuyuan/lifebench/FINAL_RESULTS.md)
- [summary.json](/data_4/liuyuan/lifebench/outputs/evaluation/latest/summary.json)
