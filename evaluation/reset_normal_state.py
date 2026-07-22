#!/usr/bin/env python3
"""重置所有判为 normal 的视频的 auto_infer_state，以便用新 prompt 重新推理"""
import json
from pathlib import Path
from collections import defaultdict

PRED_DIR = Path("/home/caiqingyuan/code/lifebench/data/public_data_release/prediction/generated_videos")
STATE_DIR = PRED_DIR / ".auto_infer_state"

def extract_risk_status(resp_str: str) -> str:
    """从 response JSON 字符串中提取 risk_status"""
    if resp_str in ("{}", ""):
        return ""
    try:
        resp = json.loads(resp_str)
    except json.JSONDecodeError:
        return ""
    rs = resp.get("risk_status", "")
    if not rs and "parsed_response" in resp:
        pr = resp.get("parsed_response", {})
        if isinstance(pr, dict):
            rs = pr.get("risk_status", "")
    return rs


def main():
    # 1. 扫描所有输出 JSON，找到每个模型的 normal 预测的 video_keys
    model_normal_vkeys = defaultdict(set)

    for model_dir in sorted(PRED_DIR.iterdir()):
        if not model_dir.is_dir() or model_dir.name == ".auto_infer_state":
            continue
        for fp in model_dir.rglob("*.json"):
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            vk = data.get("video_key", "")
            if not vk:
                continue
            rs = extract_risk_status(data.get("response", "{}"))
            if rs == "normal":
                model_normal_vkeys[model_dir.name].add(vk)

    # 打印统计
    total_resets = 0
    total_dimensions = 0
    for model in sorted(model_normal_vkeys):
        cnt = len(model_normal_vkeys[model])
        total_resets += cnt
        total_dimensions += cnt * 4
        print(f"{model}: {cnt} 个视频需要重置")

    print(f"\n共 {total_resets} 个视频, {total_dimensions} 个维度需要重置")
    print(f"涉及 {len(model_normal_vkeys)} 个模型\n")

    # 2. 更新 state 文件
    for model, normal_vkeys in sorted(model_normal_vkeys.items()):
        # state key 格式与模型 key 对应
        # 模型目录名 → state file 名
        # mplug_owl3_7b → mplug-owl3-7b.json
        # videollama2_7b → videollama2-7b.json
        # video_chatgpt_7b → video-chatgpt-7b.json
        # video_llava_7b → video-llava-7b.json
        # videollama3_7b → videollama3-7b.json
        # qwen2.5vl_7b → qwen2.5vl-7b.json
        # qwen3.5_9b → qwen3.5-9b.json
        # minigpt4_video → minigpt4-video.json
        state_key = model.replace("_", "-").replace("2.5vl", "2.5vl")
        # 特殊处理
        if model == "qwen2.5vl_7b":
            state_key = "qwen2.5vl-7b"
        elif model == "qwen3.5_9b":
            state_key = "qwen3.5-9b"
        elif model == "minigpt4_video":
            state_key = "minigpt4-video"
        elif model == "mplug_owl3_7b":
            state_key = "mplug-owl3-7b"
        elif model == "video_chatgpt_7b":
            state_key = "video-chatgpt-7b"
        elif model == "video_llava_7b":
            state_key = "video-llava-7b"
        elif model == "videollama2_7b":
            state_key = "videollama2-7b"
        elif model == "videollama3_7b":
            state_key = "videollama3-7b"

        state_path = STATE_DIR / f"{state_key}.json"
        if not state_path.exists():
            print(f"  [WARN] state 文件不存在: {state_path}")
            continue

        state = json.loads(state_path.read_text(encoding="utf-8"))
        videos = state.get("videos", {})

        reset_count = 0
        dim_count = 0
        for vk in normal_vkeys:
            if vk in videos:
                dims = videos[vk].get("dimensions", {})
                for dim in list(dims.keys()):
                    dims[dim] = {}
                    dim_count += 1
                videos[vk]["dimensions"] = {}
                reset_count += 1

        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  [{model}] 已重置 {reset_count} 个视频, {dim_count} 个维度")

    print("\n完成！")


if __name__ == "__main__":
    main()
