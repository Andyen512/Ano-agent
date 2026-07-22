# LifeBench Real-Videos 评测规则详解

## 算例数据

以下全文以一个真实视频的 GT 和模型预测（Qwen2.5-VL-7B）为例，在每个指标的评分规则中嵌入算例。

**Video ID**: `web_h264/1岁女童摔倒对着监控喊爸爸...`

### GT

```
risk_status: abnormal
risk_type: "fall/instability"
risk_sources:        ["台阶边缘及高处跌落风险", "自身身体健康状况（如失去平衡）"]
abnormal_actions:    ["孩子倒在台阶边缘"]
affected_objects:    ["儿童"]
risk_description:    视频中的孩子倒在台阶边缘。
consequence_understanding:
  - 孩子有进一步跌落的危险
  - 孩子有从高台跌落的风险
causal_chain:
  link0: source="台阶边缘"  action="摔倒"      consequence="进一步跌落"
  link1: source="高台"      action="失去平衡"   consequence="从高台跌落"
time_spans: [[0.0, 18.0]]
person_solution:      照护者应确保儿童在台阶边缘处有人看护...
hazard_solution:      应通过加装护栏或栏杆来提高安全性...
prevention_solution:  照护者应尽量避免让孩子独自前往庭院的台阶附近...
```

### 模型预测

```
risk_status: abnormal
risk_type: 跌倒失稳
risk_sources: ["地面"]
abnormal_actions: ["摔倒", "坐倒在地"]
affected_objects: ["儿童"]
risk_description: 孩子在台阶上摔倒，身体失去平衡，
                  可能是因为台阶边缘不平或地面湿滑导致。
consequence_understanding: 儿童可能因摔倒导致头部撞击台阶或四肢扭伤。
causal_chain (原始字符串): "台阶边缘不平或地面湿滑 → 孩子摔倒 → 可能导致头部撞击或四肢扭伤。"
causal_chain (解析为 link):
  link0: source="台阶边缘不平或地面湿滑"
  link1: action="孩子摔倒"
  link2: consequence="可能导致头部撞击或四肢扭伤。"
time_spans: [[0, 7]]
solution:
  对人的解决方案：立即扶起孩子并检查是否受伤。
  对危险源的解决方案：立刻擦干地面水渍并移走翻倒水杯。
  整体的解决方案：在易湿滑区域加防滑措施，并养成及时清理液体的习惯。
```

---

## 一、评测维度总览

| 维度 | 权重 | 指标数 | 评分方式 |
|---|---|---|---|
| Perception | 0.25 | 5 | 字符串匹配 / schema 检查（非Normal）+ LLM Judge（风险源/行为/对象） |
| Cognition | 0.25 | 4 | LLM Judge |
| Temporal | 0.25 | 1(+3) | IoU 计算 |
| Planning | 0.25 | 4 | LLM Judge |

每个指标 0-5 分（或 0/1），压缩到 0-1 区间后加权求和得到 0-100 总分。

---

## 二、数据来源

### GT

来自 `evaluation/official_evaluation/real_videos/` 下 4 个 consolidated JSON 文件：

| 文件 | 维度 | 关键字段 |
|---|---|---|
| `perception/perception_gt.json` | 感知 | `risk_status`, `risk_type`, `risk_sources[]`, `abnormal_actions[]`, `affected_objects[]` |
| `cognition/cognition_gt.json` | 认知 | `risk_description`, `consequence_understanding[]`, `causal_chain[]` (source/action/consequence) |
| `grounding/grounding_gt.json` | 定位 | `time_spans[]`, `start_time`, `end_time`, `duration` |
| `planning/planning_gt.json` | 规划 | `person_solution`, `hazard_solution`, `prevention_solution` |

按 `video_id` 跨维度合并为 `RealVideoGT` 对象。`risk_status` 三分法：`normal` / `risk_only` / `abnormal`。

### Predictions

来自 `prediction/real_videos/<model>/<timestamp>/*.json`。预测文件名 stem 与 GT `video_id` 的 basename 匹配。

---

## 三、Perception（感知能力）— 权重 0.25

> **Normal 视频**: 仅评估 `risk_status_accuracy`，即 `perception = norm(risk_status_accuracy)`。

### 3.1 risk_status_accuracy（权重 0.25）

> **0/1 二值**

**规则：** 将 GT 和预测的 `risk_status` 分别标准化（别名映射到标准值 `normal` / `risk_only` / `abnormal`），精确匹配。

**算例：**

```
GT:    "abnormal"  → abnormal
Pred:  "abnormal"  → abnormal
匹配:  abnormal == abnormal  →  1.0
```

---

### 3.2 risk_type_accuracy（权重 0.25）

> **0/1 二值**

**规则：** 先将 GT 和预测的 `risk_type` 分别标准化（中文→英文映射、别名→标准值）。若 GT 的 risk_type 为空，直接给 1.0。否则标准化后精确匹配 → 1.0，不匹配 → 0.0

**算例：**

```
GT:    "fall/instability"  → 标准化 → "fall_instability"
Pred:  "跌倒失稳"            → 标准化 → "fall_instability"
匹配:  "fall_instability" == "fall_instability"  →  1.0
```

> `normalize_risk_type_cn` 先将中文标签 "跌倒失稳" 映射为 "fall_instability"，再与 GT 标准化后的 "fall_instability" 精确匹配。

---

### 3.3 risk_source_recognition_score（权重 0.20）

> **0-5 分**

**规则：** 通过 LLM Judge 进行 API 语义打分。

**LLM Judge 调用格式：**

```
System: 你是一个视频安全评测专家。根据给定的 Ground Truth 和模型输出，
        仅对指定维度进行打分。
        返回 JSON 格式 {"score": 0-5, "reason": "简短的中文解释"}。
        不要输出任何额外内容。

User:   {
          "dimension": "risk_source_recognition",
          "rubric": "对比 GT 的 risk_sources 列表与模型预测的 risk_sources 列表。允许同义词或改写（如\"灶台\"与\"厨房电器\"）。5 分表示所有风险源都被正确识别，0 分表示完全未识别。",
          "ground_truth": {
            "gt_risk_sources": ["台阶边缘及高处跌落风险", "自身身体健康状况（如失去平衡）"]
          },
          "prediction": {
            "pred_risk_sources": ["地面"],
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT risk_sources:        ["台阶边缘及高处跌落风险", "自身身体健康状况（如失去平衡）"]
Pred risk_sources:      ["地面"]
LLM Judge: "地面" 与 GT 风险源完全不相关 → score 0
```

---

### 3.4 abnormal_action_recognition_score（权重 0.20）

> **0-5 分**

**规则：** 与 3.3 相同 — 通过 LLM Judge 进行 API 语义打分。

**LLM Judge 调用格式：**

```
User:   {
          "dimension": "abnormal_action_recognition",
          "rubric": "对比 GT 的 abnormal_actions 列表与模型预测的 abnormal_actions 列表。允许同义词或改写（如\"摔倒\"与\"失去平衡后倒下\"）。5 分表示所有异常行为都被正确识别，0 分表示完全未识别。",
          "ground_truth": {
            "gt_abnormal_actions": ["孩子倒在台阶边缘"]
          },
          "prediction": {
            "pred_abnormal_actions": ["摔倒", "坐倒在地"],
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT abnormal_actions:      ["孩子倒在台阶边缘"]
Pred abnormal_actions:    ["摔倒", "坐倒在地"]
LLM Judge: "孩子在台阶上摔倒" 与 "孩子倒在台阶边缘" 语义相近 → score 3~4
```

---

### 3.5 affected_object_recognition_score（权重 0.10）

> **0-5 分**

**规则：** 与 3.3 相同 — 通过 LLM Judge 进行 API 语义打分。

**LLM Judge 调用格式：**

```
User:   {
          "dimension": "affected_object_recognition",
          "rubric": "对比 GT 的 affected_objects 列表与模型预测的 affected_objects 列表。允许同义词或改写（如\"儿童\"与\"小孩\"）。5 分表示所有受影响对象都被正确识别，0 分表示完全未识别。",
          "ground_truth": {
            "gt_affected_objects": ["儿童"]
          },
          "prediction": {
            "pred_affected_objects": ["儿童"],
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT affected_objects:       ["儿童"]
Pred affected_objects:     ["儿童"]
Heuristic: 列表直接匹配 → 1/1 = 5.0
LLM Judge: "儿童" == "儿童"，完全一致 → score 5
```

---

### Perception 维度分

```
perception = 0.20 × norm(5.0)  + 0.20 × norm(5.0)  + 0.20 × norm(0.0)  + 0.20 × norm(0.0)  + 0.20 × norm(5.0)
           = 0.20 + 0.20 + 0.0 + 0.0 + 0.20
           = 0.60  →  60.00 (百分制)
```

| 指标 | 权重 | 算例得分 |
|---|---|---|
| risk_status_accuracy | 0.20 | 5.0 |
| risk_type_accuracy | 0.20 | 5.0 |
| risk_source_recognition | 0.20 | 0.0 |
| abnormal_action_recognition | 0.20 | 0.0 |
| affected_object_recognition | 0.20 | 5.0 |
| **Perception** | — | **60.00** |

---

## 四、Cognition（认知能力）— 权重 0.25

Normal 和非 Normal 视频均通过 LLM Judge 进行 API 语义打分。

### 4.1 risk_description_score（权重 0.25）

> **0-5 分**

**规则：**

- **Normal / 非 Normal 统一**: 通过 LLM Judge 对比 GT risk_description 与预测文本，评估是否准确描述视频内容。Normal 视频的 GT risk_description 描述正常活动，预测不应包含风险内容；非 Normal 视频 GT risk_description 描述异常事件。

**LLM Judge 调用格式：**

**Normal 视频** (`dimension: "risk_description_normal")`:

```
User:   {
          "dimension": "risk_description_normal",
          "rubric": "评估模型是否正确报告了正常活动（无异常），以及模型对视频中活动描述的准确度。关注视频内容是否与 GT 描述一致、是否遗漏或误判。模型不应声称存在风险。如果没有声称存在风险且活动描述准确则打 5 分，如果错误地声称存在风险或活动描述严重偏差则打 0 分。",
          "ground_truth": {
            "gt_risk_description": "一个婴儿正在床上玩耍。"
          },
          "prediction": {
            "video_description": "视频中一名婴儿坐在床上，周围环境看起来像是家庭卧室。婴儿穿着黄色衣服，表情活泼，似乎在玩耍或尝试站立。婴儿周围有枕头和毯子，床上铺着带有图案的床单。",
            "raw_response": "{...}"
          }
        }
```

**非 Normal 视频** (`dimension: "risk_description")`:

```
User:   {
          "dimension": "risk_description",
          "rubric": "评估风险段落的描述是否准确，重点关注风险类型、动作链条和场景一致性。",
          "ground_truth": {
            "risk_status": "abnormal",
            "gt_risk_description": "视频中的孩子倒在台阶边缘。",
            "gt_risk_type": "fall/instability"
          },
          "prediction": {
            "risk_description": "孩子在台阶上摔倒，身体失去平衡...",
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT risk_description:   "视频中的孩子倒在台阶边缘。"
Pred risk_description: "孩子在台阶上摔倒，身体失去平衡，
                        可能是因为台阶边缘不平或地面湿滑导致。"

  (LLM Judge 模式下会给出语义级评分，不受字面匹配限制)
```

---

### 4.2 consequence_understanding_score（权重 0.25）

> **0-5 分**

**规则：** LLM Judge 评估模型是否理解异常事件可能导致的后果

**LLM Judge 调用格式：**

```
User:   {
          "dimension": "consequence_understanding",
          "rubric": "对比 GT 的后果列表与模型预测的后果列表。允许同义描述。5 分表示所有后果都被正确识别，0 分表示完全未识别。",
          "ground_truth": {
            "gt_consequences": ["孩子有进一步跌落的危险", "孩子有从高台跌落的风险"],
            "gt_risk_description": "视频中的孩子倒在台阶边缘。"
          },
          "prediction": {
            "pred_consequences": ["儿童可能因摔倒导致头部撞击台阶或四肢扭伤。"],
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT consequences:       ["孩子有进一步跌落的危险", "孩子有从高台跌落的风险"]
Pred consequences:     ["儿童可能因摔倒导致头部撞击台阶或四肢扭伤。"]

LLM Judge: 预测后果 ≠ GT 标注后果 → score 1~2
```

---

### 4.3 factual_boundary_error_score（权重 0.25）

> **0-5 分，反向计分**（分越高 = 越少错误）

**规则：** 仅 LLM Judge 评估，分两种情况：

- **abnormal 视频**: 模型应准确判定已发生的风险事件，不应使用"可能"、推测等模糊表述 → 5.0；若用推测语气 → 0.0
- **risk_only 视频**: 模型不应将潜在风险征兆描述为已发生的事实。完全不混淆 → 5.0，严重混淆 → 0.0

**LLM Judge 调用格式：**

```
User:   {
          "dimension": "factual_boundary_error",
          "rubric": "根据 gt_risk_status，评估模型是否混淆了事实与推测：若为 abnormal，模型应准确判定已发生的风险事件，不应使用"可能"、推测等模糊表述，正确判定 → 5 分，使用推测表述 → 0 分；若为 risk_only，模型不应将潜在风险征兆描述为已发生的事实，准确区分 → 5 分，混淆事实与推测 → 0 分。",
          "ground_truth": {
            "gt_risk_status": "abnormal",
            "gt_summary": "视频中的孩子倒在台阶边缘。"
          },
          "prediction": {
            "pred_text": "孩子在台阶上摔倒，身体失去平衡...",
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT risk_status: abnormal
→ 事件已发生，不存在"把可能说成事实"的混淆 → score 5
```

---

### 4.4 causal_chain_understanding_score（权重 0.25）

> **0-5 分，按 component 粒度计分**

**规则：** 仅 LLM Judge 评估，判断模型是否理解"风险源 → 异常行为 → 后果"的因果链。5 分表示完整因果链正确，0 分表示完全没有捕捉到因果关系。

**LLM Judge 调用格式：**

```
User:   {
          "dimension": "causal_chain_understanding",
          "rubric": "评估模型是否理解因果链：风险源 → 异常行为 → 后果。5 分表示完整因果链描述正确，0 分表示没有捕捉到因果关系。",
          "ground_truth": {
            "gt_causal_chain": [
              {"source": "台阶边缘", "action": "摔倒", "consequence": "进一步跌落"},
              {"source": "高台", "action": "失去平衡", "consequence": "从高台跌落"}
            ],
            "gt_risk_description": "视频中的孩子倒在台阶边缘。"
          },
          "prediction": {
            "pred_causal_chain": [
              {"source": "台阶边缘不平或地面湿滑"},
              {"action": "孩子摔倒"},
              {"consequence": "可能导致头部撞击或四肢扭伤。"}
            ],
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT causal_chain:
  link0: source="台阶边缘"  action="摔倒"      consequence="进一步跌落"
  link1: source="高台"      action="失去平衡"   consequence="从高台跌落"

Pred causal_chain:
  link0: source="台阶边缘不平或地面湿滑"
  link1: action="孩子摔倒"
  link2: consequence="可能导致头部撞击或四肢扭伤。"

LLM Judge: 预测提到了"台阶边缘"和"摔倒"，但没提到"进一步跌落"和"高台" → score 2~3
```

---

### Cognition 维度分

```
cognition = 0.25 × norm(0.0) + 0.25 × norm(0.0) + 0.25 × norm(5.0) + 0.25 × norm(2.5)
          = 0.0 + 0.0 + 0.25 + 0.125
          = 0.375  →  37.50 (百分制)
```

| 指标 | 算例得分 |
|---|---|
| risk_description_score | 0.0 |
| consequence_understanding | 0.0 |
| factual_boundary_error | 5.0 |
| causal_chain_understanding | 2.5 |
| **Cognition** | **37.50** |

---

## 五、Temporal Grounding（时序定位）— 权重 0.25

### 5.1 time_iou（权重 1.0，即维度分直接用此值）

**规则：**

- **Normal 视频**: 预测有 time_spans → 0.0（不应有时间区间），无 → 1.0
- **非 Normal 视频**: 标准 IoU 计算。对 GT 和预测的 time_spans 分别做区间合并，计算交集/并集长度比

```
IoU = intersection_length / union_length
```

- GT 无 time_spans 但预测有 → 0.0
- GT 无 time_spans 且预测无 → 1.0

**算例：**

```
GT time_spans:   [[0.0, 18.0]]  →  合并后: [0.0, 18.0]  长度=18.0
Pred time_spans: [[0, 7]]       →  合并后: [0, 7]       长度=7.0

intersection = min(18,7) - max(0,0) = 7.0 - 0 = 7.0
union = max(18,7) - min(0,0) = 18.0 - 0 = 18.0

IoU = 7.0 / 18.0 = 0.3889
```

```
temporal = 0.3889  →  38.89 (百分制)
```

### 5.2 辅助诊断指标（参与输出但不计入维度分）

| 指标 | 算例 |
|---|---|
| `start_time_error` | abs(0.0 - 0) = **0.0** |
| `end_time_error` | abs(18.0 - 7) = **11.0** |
| `duration_error` | abs(18.0 - 7) = **11.0** |

---

## 六、Planning（决策规划）— 权重 0.25

通过 LLM Judge 评估。预测的 `solution` 文本按标题分段提取。

### 6.1 person_solution_score（权重 1/3）

> **0-5 分**

**规则：** LLM Judge 评估对人方案是否优先保护人身安全、是否具体可操作。

**LLM Judge 调用格式：**

```
User:   {
          "dimension": "solution_person",
          "rubric": "评估对人的解决方案是否将人员安全放在首位、是否可操作、是否符合风险后果。",
          "ground_truth": {
            "solution_section": "照护者应确保儿童在台阶边缘处有人看护...",
            "risk_type": "fall/instability",
            "risk_description": "视频中的孩子倒在台阶边缘。"
          },
          "prediction": {
            "solution_section": "立即扶起孩子并检查是否受伤。",
            "full_solution": "对人的解决方案：立即扶起孩子...\n对危险源的解决方案：...",
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT:  "照护者应确保儿童在台阶边缘处有人看护，防止跌落。"
Pred: "立即扶起孩子并检查是否受伤。"

LLM Judge: GT 讲预防、Pred 讲急救，方案不完全一致 → score 2
```

### 6.2 hazard_solution_score（权重 1/3）

> **0-5 分**

**规则：** LLM Judge 评估对危险源的方案是否能切断或消除危险、是否具体可执行。

**LLM Judge 调用格式：**

```
User:   {
          "dimension": "solution_hazard",
          "rubric": "评估对危险源的解决方案是否能切断或消除危险、是否具体且可执行。",
          "ground_truth": {
            "solution_section": "应通过加装护栏或栏杆来提高安全性，平台表面采用防滑材料。",
            "risk_type": "fall/instability",
            "risk_description": "视频中的孩子倒在台阶边缘。"
          },
          "prediction": {
            "solution_section": "立刻擦干地面水渍并移走翻倒水杯。",
            "full_solution": "...",
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT:  "加装护栏、采用防滑材料"（台阶场景）
Pred: "擦干水渍、移走水杯"（地面水渍场景）

LLM Judge: 场景完全不同 → score 0
```

### 6.3 overall_solution_score（权重 1/3）

> **0-5 分**

**规则：** LLM Judge 评估整体预防方案是否包含复盘和预防措施、是否符合家庭场景常识。

**LLM Judge 调用格式：**

```
User:   {
          "dimension": "solution_overall",
          "rubric": "评估整体/预防方案是否包含复盘和预防措施、是否符合家庭场景的常识。",
          "ground_truth": {
            "solution_section": "照护者应尽量避免让孩子独自前往庭院的台阶附近，防止再次摔倒后无人搀扶。",
            "risk_type": "fall/instability",
            "risk_description": "视频中的孩子倒在台阶边缘。"
          },
          "prediction": {
            "solution_section": "在易湿滑区域加防滑措施，并养成及时清理液体的习惯。",
            "full_solution": "...",
            "raw_response": "{...}"
          }
        }
```

**算例：**

```
GT:  "避免孩子独自前往台阶附近"
Pred: "加防滑措施、清理液体"

LLM Judge: GT 讲看护，Pred 讲环境改造，方向不同 → score 2
```

### 6.4 Planning 维度分

```
solution_score = (person_score + hazard_score + overall_score) / 3
planning       = norm(solution_score)
```

---

## 七、总分计算

```python
def norm(v, scale=5.0):
    return clamp(v / scale, 0, 1)
```

**非 Normal 视频：**

```
perception  = 0.20 × norm(rs_acc) + 0.20 × norm(rt_acc)
            + 0.20 × norm(risk_source) + 0.20 × norm(abnormal_action)
            + 0.20 × norm(affected_object)

cognition   = 0.25 × norm(risk_desc) + 0.25 × norm(consequence)
            + 0.25 × norm(factual_boundary) + 0.25 × norm(causal_chain)

temporal    = time_iou

planning    = norm(solution_score)

overall     = (0.25 × perception + 0.25 × cognition
             + 0.25 × temporal + 0.25 × planning) × 100
```

**Normal 视频：**

```
perception  = norm(risk_status_accuracy)
cognition   = norm(risk_description_score)

overall     = (0.5 × perception + 0.5 × cognition) × 100
```

### 算例总分

```
perception = 0.20 × norm(5×1.0) + 0.20 × norm(5×1.0) + 0.20 × norm(0.0) + 0.20 × norm(0.0) + 0.20 × norm(5.0)
           = 0.600

cognition  = 0.25 × 0.0 + 0.25 × 0.0 + 0.25 × 1.0 + 0.25 × 0.5
           = 0.375

temporal   = 0.389

planning   = 0.000

overall    = (0.25 × 0.600 + 0.25 × 0.375 + 0.25 × 0.389 + 0.25 × 0.000) × 100
            = (0.150 + 0.094 + 0.097 + 0.000) × 100
            = 34.10
```

| 维度 | 百分制 | 关键原因 |
|---|---|---|
| Perception | 60.00 | risk_status + risk_type + affected_obj 正确 |
| Cognition | 37.50 | factual_boundary 满分 + causal_chain 3/6 匹配 |
| Temporal | 38.89 | 区间部分重叠 (预测 0-7s vs GT 0-18s) |
| Planning | 0.00 | 方案描述的是不同场景（地面水渍 vs 台阶边缘） |
| **Overall** | **34.10** | |

---

## 八、LLM Judge 调用

所有需要语义评估的指标均通过 LLM Judge（OpenAI-compatible API）打分，不使用 heuristic 回退。

需要 Judge 的指标：

- `risk_source_recognition_score` — GT "台阶边缘及高处跌落风险" vs 预测隐含的风险源识别
- `abnormal_action_recognition_score` — GT "孩子倒在台阶边缘" vs 预测 "孩子在台阶上摔倒"
- `affected_object_recognition_score` — GT "儿童" vs 预测 "儿童"（同义词字面不同时需 Judge）
- `risk_description_score` — GT "孩子倒在台阶边缘" vs Pred "孩子在台阶上摔倒" → 语义相近可由 Judge 给出较高分
- `consequence_understanding_score` — GT "进一步跌落" vs Pred "头部撞击或四肢扭伤" → 语义相近但字面不匹配
- `factual_boundary_error_score` — 判断是否混淆 risk_only 与 abnormal
- `causal_chain_understanding_score` — GT 的因果链语义匹配
- `person_solution_score` / `hazard_solution_score` / `overall_solution_score` — 方案语义是否合理
