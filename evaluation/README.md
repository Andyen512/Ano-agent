# LifeBench 评测模块

对 `outputs/predictions/` 下的模型推理结果，按 ground truth 做自动评分。
支持本地多 GPU ollama cluster 作为 judge，可横向扩展。

## 目录

- `evaluate_outputs.py` — 主评测入口，CLI 入口 `run_benchmark_evaluation.sh`
- `render_final_results.py` — 把 `summary.json` 渲染成 `final_model_results.csv` + `FINAL_RESULTS.md`
- `common.py` — 评分逻辑、GT 解析、judge 客户端
- `scripts/start_ollama_cluster.sh` — 启动 N 个 ollama 实例分别绑 N 张 GPU
- `scripts/run_evaluation.sh` — 一键评测，自动检测 endpoint 并拼好命令

## 快速开始

### 1. 启动 ollama cluster（8 卡，qwen2.5:7b）

```bash
bash scripts/start_ollama_cluster.sh start qwen2.5:7b 8
```

启动后会在 11434–11441 起 8 个 ollama，分别绑 GPU 0–7，自动 `ollama pull` 模型。
查看状态 / 停止：

```bash
bash scripts/start_ollama_cluster.sh status
bash scripts/start_ollama_cluster.sh stop
```

### 2. 跑评测

```bash
bash scripts/run_evaluation.sh
```

默认会用 8 个 endpoint、`--workers 8`、要求 10 个 benchmark 模型齐套。参数透传给 `run_benchmark_evaluation.sh`：

```bash
# 只评测单个模型
bash scripts/run_evaluation.sh --model-id Qwen/Qwen3.5-9B

# 用更小规模
EVAL_NUM_GPUS=4 bash scripts/run_evaluation.sh

# 单卡单 ollama
EVAL_NUM_GPUS=1 bash scripts/run_evaluation.sh
```

跑完后会自动写 `summary.json`、`per_video_scores.json`、`FINAL_RESULTS.md`、`final_model_results.csv` 到 `--output-dir`（默认 `outputs/evaluation/generated_videos/`）。

### 3. 重新渲染 FINAL_RESULTS.md

如果改过 `summary.json` 但没改评测主逻辑，可以单独跑：

```bash
python3 evaluation/render_final_results.py \
    --summary-json outputs/evaluation/generated_videos/summary.json
```

## 关键环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `EVAL_JUDGE_MODEL` | `qwen2.5:7b` | judge 模型名（要 ollama 已 pull） |
| `EVAL_JUDGE_API_KEY` | `ollama` | ollama 不校验 key，但 OpenAI 协议需要非空 |
| `EVAL_JUDGE_TIMEOUT_SECONDS` | `180` | 单次 judge 请求超时 |
| `EVAL_NUM_GPUS` | `8` | 并发 ollama 实例数（= endpoint 数） |
| `EVAL_BASE_PORT` | `11434` | ollama 起始端口 |
| `EVAL_HOST` | `127.0.0.1` | ollama host |
| `EVAL_WORKERS` | `=EVAL_NUM_GPUS` | 评测线程数。建议 ≤ endpoint 数 × 每实例并行数 |
| `EVAL_GT_PATH` | `data/annotation/gen_prompt_generated_all.txt` | ground truth 路径 |
| `EVAL_PREDICTIONS_ROOT` | `outputs/predictions` | 预测结果根目录 |
| `EVAL_OUTPUT_DIR` | `outputs/evaluation/generated_videos` | 评测输出目录 |
| `EVAL_REQUIRE_COMPLETE` | `1` | 是否要求所有 expected model 都有结果 |
| `EVAL_EXPECTED_MODELS` | （10 个 benchmark 模型） | 显式列出的模型 id，空格分隔 |

## 单 endpoint 模式（兼容老用法）

不用 ollama cluster、只用单个 endpoint 时，直接用 `run_benchmark_evaluation.sh`：

```bash
EVAL_JUDGE_BASE_URL=http://localhost:11434/v1 \
EVAL_JUDGE_API_KEY=ollama \
EVAL_JUDGE_MODEL=qwen2.5:7b \
bash evaluation/run_benchmark_evaluation.sh \
  --judge-mode openai \
  --ground-truth data/annotation/gen_prompt_generated_all.txt \
  --predictions-root outputs/predictions \
  --output-dir outputs/evaluation/generated_videos \
  --require-complete-videos \
  --expected-models \
    LanguageBind/Video-LLaVA-7B \
    OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf \
    MBZUAI/Video-ChatGPT-7B \
    Vision-CAIR/MiniGPT4-Video \
    DAMO-NLP-SG/VideoLLaMA2.1-7B-16F \
    Qwen/Qwen2.5-VL-7B-Instruct \
    Qwen/Qwen3.5-9B \
    DAMO-NLP-SG/VideoLLaMA3-7B \
    mPLUG/mPLUG-Owl3-7B-241101 \
    omni-research/Tarsier2-7b-0115
```

也可以 `--judge-base-url http://...`（单数）。`--judge-base-urls` 优先级更高。

## 多 endpoint 负载分发

`--judge-base-urls` 接受任意数量的 URL，脚本会构造 `MultiEndpointJudge`，
按 thread-local round-robin 调度，遇到 endpoint 失败自动转移。
每个 endpoint 仍可设 `OLLAMA_NUM_PARALLEL` 让单实例内部进一步并发。

典型配置：

| GPU 数 | `--judge-base-urls` | 端点形式 | 期望加速 |
|---|---|---|---|
| 1 | 1 个 URL | 单 ollama，串行 | 1× |
| 8（多实例） | 8 个 URL | 8 个 ollama 进程，各绑 1 卡 | ~8× |
| 8（单实例） | 1 个 URL | 1 个 ollama 进程跨 8 卡（设 `OLLAMA_NUM_PARALLEL=16`） | ~6-8× |

## 缓存机制

评测结果缓存到 `<output-dir>/per_prediction_results/<model_dir>/<video_id>.json`。
缓存 key = `prediction_hash + ground_truth_hash + judge_signature`。
**任何**一项变化（包括 `--judge-base-urls` 改变 endpoint 集合）都会让旧 cache 失效，全部重跑。
所以切换 endpoint 不用手动清缓存。

`--resume`（默认 `False`）开启后会复用 cache，`--resume` 关闭会强制重跑（但不会清旧 cache，新结果会覆盖同路径文件）。

## 评测指标

| 字段 | 含义 | judge 模式 |
|---|---|---|
| `risk_status_accuracy` | risk_status 预测是否正确 | heuristic 精确匹配 |
| `risk_type_accuracy` | risk_type 标签是否匹配 | heuristic 精确匹配 |
| `risk_type_score` | risk_type 5 分制 | 优先 judge，否则 heuristic |
| `time_iou` | time_spans 与 GT 的 IoU | heuristic |
| `video_description_score` | 视频描述 5 分制 | normal 走 judge，abnormal/risk_only 不评 |
| `risk_description_score` | 风险描述 5 分制 | abnormal/risk_only 走 judge，normal 不评 |
| `solution_score` | 整体方案 5 分制（person/hazard/overall 平均） | 见下 |
| `person_solution_score` | 对人方案 5 分制 | abnormal/risk_only 走 judge |
| `hazard_solution_score` | 对危险源方案 5 分制 | abnormal/risk_only 走 judge |
| `overall_solution_score` | 整体方案 5 分制 | abnormal/risk_only 走 judge |
| `overall_score` | 加权总分 0–100 | 6 个分量的加权 |

权重见 `common.py:weighted_overall_score`。

## 常见问题

**Q: judge 调用失败，报 "Your request was blocked" / 403？**
A: endpoint 端被封或 API key 失效。检查 `EVAL_JUDGE_BASE_URL` / `EVAL_JUDGE_API_KEY` 拼写，或换 endpoint。

**Q: 跑得慢怎么办？**
A: 按上面表格调 `EVAL_NUM_GPUS` 启动多实例 ollama；或设 `OLLAMA_NUM_PARALLEL` 让单实例内部并发；或换更小模型（如 `qwen2.5:3b`）。

**Q: cache 怎么强制重跑？**
A: 加 `--resume` 时 cache hit 会跳过；不加（默认）会强制重算（但旧 cache 文件不删，新结果会覆盖）。要彻底清空：`rm -rf <output-dir>/per_prediction_results`。

**Q: 评测中途能中断吗？**
A: 可以，下次跑同一命令会基于已有 cache（`--resume` 默认开）继续。`state.json` 跟踪进度。

**Q: 怎么跑 real_videos 评测？**
A: 改 `EVAL_GT_PATH`、`EVAL_PREDICTIONS_ROOT`、`EVAL_OUTPUT_DIR` 即可，例如：
```bash
EVAL_GT_PATH=data/annotation/gen_prompt_real.txt \
EVAL_PREDICTIONS_ROOT=outputs/real_video_predictions \
EVAL_OUTPUT_DIR=outputs/evaluation/real_videos \
bash scripts/run_evaluation.sh
```
（具体路径以仓库实际为准。）

**Q: `--require-complete-videos` 报 0 个齐套视频？**
A: `BENCHMARK_MODELS` 包含所有 benchmark 模型（含没有预测结果的），需要 `--expected-models` 显式列实际有结果的模型子集。
