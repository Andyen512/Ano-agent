# Official Model Scripts

These entrypoints run inference and evaluation for the eight models retained in
`outputs/evaluation/real_videos/FINAL_RESULTS.md`. They use one model call per
video with `data/infer_prompt_structured.txt`, then split the complete JSON into
the fields consumed by the existing evaluator. They preserve the existing output
layout under `data/public_data_release/prediction/real_videos` and
`outputs/evaluation/real_videos`.

Supported entrypoints:

```text
run_qwen25vl_8gpu_infer_eval.sh
run_qwen35_8gpu_infer_eval.sh
run_internvl35_8gpu_infer_eval.sh
run_tarsier2_8gpu_infer_eval.sh
run_videollama2_8gpu_infer_eval.sh
run_videollama3_8gpu_infer_eval.sh
run_videochat2_8gpu_infer_eval.sh
run_videochatgpt_8gpu_infer_eval.sh
```

`MiniGPT4-Video` is intentionally excluded. The wrappers delegate to the
existing model-specific runners and force `single` inference mode; no existing
runner is moved or removed.

Runtime environments:

```text
Qwen2.5-VL, Tarsier2, VideoLLaMA2, VideoLLaMA3, VideoChat2, Video-ChatGPT: 3dpose
Qwen3.5, InternVL3.5: lifebench-qwen35
Qwen3 judge servers: 3dpose
```

The paths can still be overridden with the existing `LIFEBENCH_*_PYTHON`
environment variables.

For the generated-video release, run the batch entrypoint below. It reuses the
same eight model wrappers, writes predictions under
`data/public_data_release/prediction/generated_videos`, evaluates against the
generated GT tree, and renders the five-dimension table under
`outputs/evaluation/generated_videos/FINAL_RESULTS.md`:

```bash
bash evaluation/scripts/run_all_generated_videos_infer_eval.sh
```

Before inference/evaluation, the entrypoint compares the generated GT schema
with `annotations/real_videos` and writes the report to
`outputs/evaluation/generated_videos/gt_schema_comparison.json`.

The generated run is resumable. Override the input/output roots with
`LIFEBENCH_GENERATED_VIDEOS_DIR`, `LIFEBENCH_GENERATED_PREDICTIONS_ROOT`,
`LIFEBENCH_GENERATED_GT_DIR`, and `LIFEBENCH_GENERATED_EVAL_OUTPUT_ROOT`.
