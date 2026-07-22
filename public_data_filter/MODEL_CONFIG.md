# public_data_filter Model Config

## Files

- `multi_view_risk_review.py`
  Single-video 5-model, 5-perspective risk review entry.
- `batch_multi_view_risk_review.py`
  Batch runner over all videos under `data/public_data`.
- `persistent_batch_multi_view_risk_review.py`
  Batch runner with long-lived per-model workers. Each worker loads one model once and continuously consumes videos.
- `run_persistent_resume.sh`
  One-command launcher for the persistent runner. It resumes the newest batch output directory by default.
- `rule.md`
  The five review perspectives and their prompts.
- `convert_public_data_images_to_videos.py`
  Convert image sequences into videos for datasets that originally lack video files.

## Model Pool

The current fixed model pool in `multi_view_risk_review.py` is:

1. `Qwen/Qwen3.5-9B` -> `qwen35vl`
2. `omni-research/Tarsier2-7b-0115` -> `tarsier2`
3. `mPLUG/mPLUG-Owl3-7B-241101` -> `mplugowl3`
4. `DAMO-NLP-SG/VideoLLaMA3-7B` -> `videollama3`
5. `OpenGVLab/InternVL3_5-8B` -> `internvl35`

Each video is reviewed once from each of the five perspectives in `rule.md`. The final decision is majority vote.

## Optional Replacement Model

`openbmb/MiniCPM-V-4_5` is available as backend `minicpmv45`.

It is intentionally not part of the default five-model pool while the current full run is in progress. Use it later to rerun or repair the Tarsier2-backed Conservative Safety Perspective without changing the existing resume behavior.

## Default Inference Args

`multi_view_risk_review.py` currently forwards these defaults to `lifebench_infer.py`:

- `temperature = 0.0`
- `top_p = 0.9`
- `max_new_tokens = 256`
- `fps = 1.0`
- `max_frames = 16`
- `merge_size = 2`
- `device_map = auto`

In practice, because `temperature = 0.0`, the generation path is deterministic and `top_p` is mostly inactive.

## Frame Sampling Policy

Current effective sampling policy for the five models:

- `Qwen3.5-9B`
  Uses `num_frames = min(max_frames, 32)`.
  Under current defaults, this is `16` frames.

- `InternVL3.5-8B`
  Uses uniform full-video sampling with `num_segments = min(max_frames, 32)`.
  Under current defaults, this is `16` frames.

- `mPLUG-Owl3-7B`
  Uses `load_video_frames_simple(..., max_num_frames=min(max_frames, 32))`.
  Under current defaults, this is at most `16` frames.

- `VideoLLaMA3-7B`
  Uses frame budget `min(max_frames, 32)` with dynamic FPS:
  `effective_fps = min(1.0, frame_budget / video_duration_seconds)`.
  Under current defaults, this is a `16`-frame budget.

- `Tarsier2-7B`
  This repo locally overrides Tarsier2 only inside `lifebench_infer.py`.
  Effective runtime override:
  - `n_frames = min(max_frames, 32)`
  - `max_n_frames = min(max_frames, 32)`
  - `use_multi_images_for_video = false`

  So under current defaults, Tarsier2 now also uses true `16`-frame input and no longer expands each sampled frame into duplicate images.

## Tarsier2 Override Note

The original Tarsier2 config file under:

- `code/Tarsier2-7B/configs/tarser2_default_config.yaml`

still contains:

- `n_frames: 16`
- `use_multi_images_for_video: true`

But the unified runner override in `lifebench_infer.py` takes precedence for this pipeline, so the original file is not modified.

## Batch Runner

`batch_multi_view_risk_review.py` is the full-data traversal runner.

It:

- scans all videos under `data/public_data`
- assigns one worker per GPU id from `--gpus`
- binds each worker through `CUDA_VISIBLE_DEVICES=<gpu_id>`
- runs `multi_view_risk_review.py` once per video
- supports `--resume`
- writes batch-level JSON and CSV summaries

Recommended launch:

```bash
/data_4/liuyuan/anaconda3/envs/lifebench-vlm/bin/python \
/data_4/liuyuan/lifebench/public_data_filter/batch_multi_view_risk_review.py \
  --gpus 0,1,2,3,4,5,6,7 \
  --seed 0 \
  --resume \
  --log-every 10
```

## Persistent Batch Runner

For the full public-data pass, prefer `persistent_batch_multi_view_risk_review.py`.

It avoids reloading every model for every video. With 8 GPUs and 5 models, the default `--worker-plan auto` allocates:

- `qwen35vl`: 2 workers
- `mplugowl3`: 2 workers
- `tarsier2`: 2 workers
- `internvl35`: 1 worker
- `videollama3`: 1 worker

This matches the observed slow-model bottleneck better than a plain one-model-per-GPU layout.

Recommended launch:

```bash
cd /data_4/liuyuan/lifebench

public_data_filter/run_persistent_resume.sh
```

To continue a specific interrupted run in-place:

```bash
cd /data_4/liuyuan/lifebench

public_data_filter/run_persistent_resume.sh \
  /data_4/liuyuan/lifebench/public_data_filter/batch_outputs/20260424T064251Z
```

Optional faster-but-less-complete mode:

```bash
  --early-stop-majority
```

With early stopping, a video can finish once a 3-of-5 majority is already determined.

## Current Video Count

Current count under `data/public_data`:

- total videos: `21948`

By dataset:

- `Multiple_users_compound_event`: `2242`
- `Single_user_simple_event`: `2289`
- `SmartHome-Bench`: `1010`
- `UR_Fall_Dataset`: `100`
- `labimag`: `192`
- `toyota_smarthome_mp4`: `16115`

By extension:

- `.mp4`: `21756`
- `.avi`: `192`

## Prompt Source

The review prompts are stored in:

- `rule.md`

The prompt and required model output labels are English. New `review_summary.json` files include:

- `prompt_language: "en"`
- `output_schema_version: "risk_full_en_v2"`

Resume mode only treats English `risk_full_en_v2` summaries as complete, so older outputs will be rerun instead of silently mixed into the current run.

The persistent runner retries once with a shorter strict-format prompt if a model does not emit a parseable `Risk presence` field. This is mainly for models that understand the video but drift into free-form description instead of the requested schema.

Each agent now emits:

- `Risk presence`
- `Level 1 scene`
- `Level 2 subject`
- `Level 3 risk type`
- `Risk description`
- `Risk time interval`
- `Solution for person`
- `Solution for hazard source`
- `Solution to prevent recurrence`
- `Normal video description`

`Risk presence` must be `Yes` or `No`.

If `Risk presence` is `Yes`, the model outputs Level 1/2/3 keywords, a risk description, risk-localization time interval, and three solution fields. If `Risk presence` is `No`, the model outputs Level 1/2 keywords and a normal-video description.

Preferred taxonomy keywords are:

- Level 1 scene: `dining room`, `kitchen`, `study`, `balcony`, `living room`, `bathroom`, `yard`
- Level 2 subject: `child`, `older adult`, `young adult`, `middle-aged adult`
- Level 3 risk type: `fall/instability`, `heat/fire source`, `collision/crush injury`, `sharp-object danger`, `electrical safety`, `poisoning/accidental ingestion`, `interpersonal conflict`

If none of the preferred keywords fits the video, the model may output a concise custom keyword. Batch summaries include top aggregated risk keywords from the agents that voted `Yes`, and top normal-scene/subject keywords from the agents that voted `No`.

The five perspectives are:

1. Daily Behavior Baseline Perspective
2. Conservative Safety Perspective
3. Permissive Behavior Tolerance Perspective
4. Vulnerable-Subject-First Perspective
5. Consequence-Severity-First Perspective

## Output Layout

Single-video review output:

- `public_data_filter/outputs/<run_id>/...`

Batch review output:

- `public_data_filter/batch_outputs/<run_id>/...`

Important batch artifacts:

- `batch_manifest.json`
- `batch_review_summary.json`
- `batch_review_summary.csv`
- `reviews/<dataset>/.../<video_stem>/review_summary.json`

## Strict Chinese Schema Extraction

After the five-agent review is complete, run a text language model over the
existing `review_summary.json` files to normalize the final output into the
Chinese template:

```bash
cd /data_4/liuyuan/lifebench

public_data_filter/run_strict_chinese_schema_extraction.sh
```

The launcher waits for `models/Qwen__Qwen3-8B-Base` to finish downloading, loads the
model once, and writes resumable outputs under:

- `public_data_filter/batch_outputs/persistent_full/strict_chinese_schema.jsonl`
- `public_data_filter/batch_outputs/persistent_full/strict_chinese_schema.json`
- `public_data_filter/batch_outputs/persistent_full/strict_chinese_schema.csv`

The normalized schema is:

- `风险`: `有` or `无`
- `Level 1 场景`: `餐厅`, `厨房`, `书房`, `阳台`, `客厅`, `卫生间`, `庭院`
- `Level 2 主体`: `儿童`, `老年人`, `年轻人`, `中年人`
- `Level 3 风险类型`: `跌倒失稳`, `高温火源`, `碰撞砸伤`, `锐器危险`, `用电安全`, `中毒误食`, `人际冲突`; no-risk videos use `无`
- `正常视频描述或风险描述`
- `风险定位`: integer second intervals such as `[1,3]`; no-risk videos use `无`; if source outputs contain no time evidence, use `未知`
- `解决方案`: `对人`, `对危险源`, `防止危险复发`

Useful options:

```bash
# Smoke test without loading the language model.
public_data_filter/run_strict_chinese_schema_extraction.sh --no-llm --limit 10

# Resume a stopped full extraction.
public_data_filter/run_strict_chinese_schema_extraction.sh --resume

# Fail instead of falling back when the language model output is invalid.
public_data_filter/run_strict_chinese_schema_extraction.sh --strict-llm
```
