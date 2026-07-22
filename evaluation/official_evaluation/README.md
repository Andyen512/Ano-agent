# LifeBench 评测方案总览

## 一、评测流程

```
GT (real_gt.txt) ──> parse_ground_truth_line() ──> GroundTruthEntry[]
                              │
预测文件 (*.json) ──> discover_prediction_files() ──> StructuredPrediction[]
                              │
                              ▼
            evaluate_prediction(gt, pred)
                  │            │
                  ▼            ▼
           heuristic 或    LLM Judge (OpenAI-compatible)
          关键字匹配         qwen2.5:7b 打分 0-5
                  │            │
                  └─────┬──────┘
                        ▼
        5 个维度 25+ 项指标 → per_video_scores.json
                        │
                        ▼
            aggregate → summary.json → FINAL_RESULTS.md
```

## 二、环境配置

- **评测范围**: real_videos（监控/YouTube/SmartHome 场景共 2854 条 GT）
- **模型数**: 10 个 VLM benchmark 模型
- **预测条数**: 28478 条
- **Judge 模式**: `mixed_openai`（优先 LLM，失败 fallback 到 heuristic）
- **Judge 模型**: `qwen2.5:7b`（Ollama 部署）
- **Worker 数**: 4
- **输出目录**: `outputs/evaluation/real_videos/`

## 三、数据格式

### 3.1 Ground Truth (`data/annotation/real_gt.txt`)

TSV 格式，按 risk_status 分三类：

| 类型 | 格式 |
|---|---|
| **normal** | `<video_id>\t<path>\tnormal\t<视频描述>` |
| **abnormal** | `<video_id>\t<path>\tabnormal\t<描述含时间戳>\t<解决方案>` |
| **risk_only** | `<video_id>\t<path>\trisk_only\t<描述含时间戳>\t<解决方案>` |

**新增可选列**（放在 solution 之后）：

| 列 | 格式 | 示例 |
|---|---|---|
| `risk_sources` | 逗号分隔 | `刀具,灶台明火,宠物狗` |
| `abnormal_actions` | 逗号分隔 | `摔倒,撞击,攀爬` |
| `consequences` | 逗号分隔 | `抓伤,咬伤,设施破坏` |
| `causal_chain` | 管道分隔 `source:action:consequence` | `宠物狗:追逐:咬伤\|湿滑地面:滑倒:摔伤` |

### 3.2 预测文件 (`outputs/real_video_inference/<model>/<timestamp>/<video_id>.json`)

```json
{
  "model_id": "Qwen/Qwen3.5-9B",
  "response": "{\"risk_status\": \"normal\", \"video_description\": \"...\"}",
  "parsed_response": { "risk_status": "normal", "video_description": "..." },
  ...
}
```

## 四、评测指标

共 4 个维度 18 项指标。

### Perception — 异常感知能力

| 指标 | 范围 | 评分方式 | 现有/新增 | 说明 |
|---|---|---|---|---|
| `risk_status_accuracy` | 0/1 | heuristic | 现有 | 判断 normal / risk_only / abnormal 是否正确 |
| `risk_type_accuracy` | 0/1 | heuristic | 现有 | 风险类别标签是否完全匹配 |
| `risk_source_recognition_score` | 0-5 | LLM + heuristic | **新增** | 是否识别刀、火、动物、车辆、湿滑地面等风险源。对比 GT 与预测的 `risk_sources` 列表 |
| `abnormal_action_recognition_score` | 0-5 | LLM + heuristic | **新增** | 是否识别摔倒、撞击、攀爬、割伤等异常动作。对比 GT 与预测的 `abnormal_actions` 列表 |
| `affected_object_recognition_score` | 0-5 | LLM + heuristic | **新增** | 是否识别老人、小孩、宠物、设备等受影响对象。对比 GT 与预测的 `affected_objects` 列表 |

### Cognition — 异常认知/推理能力

| 指标 | 范围 | 评分方式 | 现有/新增 | 说明 |
|---|---|---|---|---|
| `risk_description_score` | 0-5 | heuristic + LLM | 现有 | 模型对风险含义的描述是否正确 |
| `consequence_understanding_score` | 0-5 | LLM + heuristic fallback | **新增** | 是否理解抓伤、咬伤、设施破坏、病菌传播等后果。需 GT `consequences` 列 |
| `factual_boundary_error_score` | 0-5 | LLM | **新增** | 是否把"可能发生的风险"说成"已经发生的事实"。反向计分，越高越好 |
| `causal_chain_understanding_score` | 0-5 | LLM + heuristic fallback | **新增** | 是否理解"风险源 → 异常行为 → 后果"的因果链。需 GT `causal_chain` 列 |

### Temporal Grounding — 时间定位能力

| 指标 | 范围 | 评分方式 | 现有/新增 | 说明 |
|---|---|---|---|---|
| `time_iou` | 0-1 | heuristic IoU | 现有 | 预测异常区间与 GT 区间的 IoU |
| `start_time_error` | 秒 | heuristic | **新增** | 开始时间绝对误差 |
| `end_time_error` | 秒 | heuristic | **新增** | 结束时间绝对误差 |
| `duration_error` | 秒 | heuristic | **新增** | 持续时间绝对误差 |
| `start_position` | early/middle/late | heuristic | **新增** | 异常发生在视频的哪个阶段 |

### Intervention Planning — 解决方案/干预规划能力

| 指标 | 范围 | 评分方式 | 现有/新增 | 说明 |
|---|---|---|---|---|
| `person_solution_score` | 0-5 | LLM + heuristic | 现有 | 对人方案：是否保护人员、检查伤情、及时求助 |
| `hazard_solution_score` | 0-5 | LLM + heuristic | 现有 | 对隐患方案：是否处理风险源、移除危险物 |
| `overall_solution_score` | 0-5 | LLM + heuristic | 现有 | 整体预防方案：是否提出环境层面的预防措施 |
| `solution_score` | 0-5 | — | 现有 | 上述三者的平均值 |

## 五、加权总分

```
perception    = 0.20 × risk_status + 0.20 × norm(risk_type) + 0.20 × norm(risk_source) 
                + 0.20 × norm(abnormal_action) + 0.20 × norm(affected_object)
cognition     = 0.25 × norm(risk_desc) + 0.25 × norm(consequence) 
                + 0.25 × norm(factual_boundary) + 0.25 × norm(causal_chain)
temporal      = time_iou  (0-1)
intervention  = norm(solution_score)

overall = 0.30 × perception + 0.25 × cognition + 0.20 × temporal + 0.25 × intervention
```

其中 `norm(v) = min(max(v / 5.0, 0.0), 1.0)`。

## 六、Judge 模式

- **heuristic**: 关键字组匹配（预定义风险类型/动作/对象关键词组 + 分组打分），不需要 LLM
- **LLM judge (openai)**: 调用 OpenAI-compatible API（Ollama），返回 `{"score": 0-5, "reason": "..."}`
- **mixed**: 优先 LLM judge，调用失败则 fallback 到 heuristic

需要 LLM 判断的指标：
- `risk_source_recognition_score`（fallback 到 heuristic 列表匹配）
- `abnormal_action_recognition_score`（fallback 到 heuristic 列表匹配）
- `affected_object_recognition_score`（fallback 到 heuristic 列表匹配）
- `consequence_understanding_score`（fallback 到 heuristic）
- `causal_chain_understanding_score`（fallback 到 heuristic）
- `factual_boundary_error_score`

## 七、数据样例详解

### 场景: 正常厨房操作 (normal)

**GT 条目**:
```
video_id: Single_user_simple_event/data/p101/Cook.Usestove/c4s0/00101_c4s0
risk_status: normal
description: [Safe Action] 视频中可见一名身穿白色T恤和灰色裤子的人站在厨房内，
             面向炉灶。厨房配有白色台面、灰色橱柜……
risk_sources: (空 — normal 类型不需要)
abnormal_actions: (空)
```

**Qwen3.5-9B 预测**:
```json
{"risk_status": "normal", "video_description": "一名男子在厨房灶台前站立，背对镜头
 进行烹饪操作，期间保持正常站姿，未出现跌倒、碰撞或接触危险源等异常事件。"}
```

**评测结果**:
- ✅ Perception: `risk_status_accuracy=1.0`
- ✅ Cognition: `risk_description_score=5.0`（normal 要求 risk_description 为空）
- ✅ Summarization: `video_description_score=5.0`（内容一致）
- ✅ Temporal: `time_iou=1.0`（无异常区间）
- ✅ Intervention: `solution_score=5.0`（要求 solution 为空，预测也为空）
- 🔢 Overall: `100.0`

### 场景: 异常摔倒 (abnormal)

**GT 条目**:
```
video_id: youtube2/Wyze_videos/1dlH2pHZzcU
risk_status: abnormal
time_spans: [13, 24]
description: [Abnormal Result][13,24] 年轻男性在客厅光脚行走时突然摔倒...
risk_sources: 光滑地板,玩具射击道具
abnormal_actions: 行走,摔倒
consequences: 头部撞击,手腕扭伤,膝盖擦伤
causal_chain: 光滑地板:行走打滑:摔倒受伤
```

**评测逻辑**:
- `risk_status_accuracy`: 预测是否为 `abnormal`
- `risk_source_recognition`: 是否提到"光滑地板"、"玩具道具"
- `abnormal_action_recognition`: 是否提到"摔倒"、"行走"等
- `consequence_understanding`: 是否提到"头部撞击/手腕扭伤/膝盖擦伤"
- `causal_chain`: 是否理解"光滑地板 → 打滑 → 摔倒"的因果链
- `factual_boundary`: 是否有把风险说成事实（如 GT 是 risk_only 时）
- `time_iou`: 预测区间与 [13,24] 的重叠度
- `start_time_error / end_time_error / duration_error`: 时间定位精度
- `solution_score`: 对人/对隐患/整体方案三个维度的评分
- `overall`: 五大维度的加权汇总

## 八、当前结果摘要（real_videos）

| 排名 | 模型 | 总分 | 状态 |
|---|---|---|---:|
| 1 | MiniGPT4-Video | 65.07 | completed |
| 2 | Qwen3.5-9B | 58.77 | running |
| 3 | Video-LLaVA-7B | 56.93 | completed |
| 4 | Qwen2.5-VL-7B | 54.96 | completed |
| 5 | VideoLLaMA3-7B | 54.19 | completed |
| 6 | Tarsier2-7b | 53.94 | completed |
| 7 | mPLUG-Owl3-7B | 52.51 | failed |
| 8 | Video-ChatGPT-7B | 47.99 | failed |
| 9 | VideoChat2-7B | 46.35 | completed |
| 10 | VideoLLaMA2.1-7B | 41.60 | completed |

> 注：以上总分基于现有指标，切换到 5 分类体系后总分将重新计算。
