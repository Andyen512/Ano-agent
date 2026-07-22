#!/bin/bash
# ============================================================================
# gen_all_gt.sh - 生成全部 4 个评测维度的 GT JSON 文件
#
# 用法:
#   ./gen_all_gt.sh                          # 处理全部视频
#   ./gen_all_gt.sh web_h264                 # 只处理 path 以 web_h264/ 开头的视频
#   ./gen_all_gt.sh web_h264 gpt-5.4         # 指定子集 + 模型
#   ./gen_all_gt.sh "" gpt-5.4               # 全部视频 + 指定模型
#
# 流程:
#   1. 过滤 GT（可选）
#   2. grounding / planning / perception（并行）
#   3. cognition（依赖 perception）
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GT_RELEASE="/home/caiqingyuan/code/lifebench/data/annotation/real_gt_expanded_release.txt"
API_HOST="www.lingganyaapi.com"
API_KEY="sk-ke2RsoBH9l60EuXpKFLmCWTCuhT08B8jII9UkkHnaxWWWnRf"
MODEL="${2:-gpt-5.6-sol}"
FILTER_PREFIX="${1:-}"

PERCEPTION_OUT="${SCRIPT_DIR}/perception/perception_gt.json"
COGNITION_OUT="${SCRIPT_DIR}/cognition/cognition_gt.json"
GROUNDING_OUT="${SCRIPT_DIR}/grounding/grounding_gt.json"
PLANNING_OUT="${SCRIPT_DIR}/planning/planning_gt.json"

echo "============================================"
echo " LifeBench GT Generation Pipeline"
echo "============================================"
echo "GT source:  ${GT_RELEASE}"
echo "Model:      ${MODEL}"
echo "API host:   ${API_HOST}"

if [ -n "$FILTER_PREFIX" ]; then
    FILTERED_GT="/tmp/lifebench_filtered_gt_$$.txt"
    grep "^${FILTER_PREFIX}" "$GT_RELEASE" | grep -v "^${FILTER_PREFIX}_segments/" > "$FILTERED_GT" || {
        echo "ERROR: No entries matching '${FILTER_PREFIX}'"
        exit 1
    }
    GT_FILE="$FILTERED_GT"
    echo "Filter:     path starts with '${FILTER_PREFIX}'"
else
    GT_FILE="$GT_RELEASE"
    echo "Filter:     none (all entries)"
fi
echo "Total lines: $(wc -l < "$GT_FILE")"
echo "============================================"
echo ""

mkdir -p "${SCRIPT_DIR}/perception" "${SCRIPT_DIR}/cognition" \
         "${SCRIPT_DIR}/grounding" "${SCRIPT_DIR}/planning" \

# 描述已在 real_gt_expanded_release.txt 中预扩充，直接使用

# ==================== Python config patcher ====================
patch_and_run() {
    local step_name=$1
    local step_dir=$2
    local output_json=$3
    shift 3
    local extra_replacements=("$@")

    local src_py="${step_dir}/gen_${step_name}_gt.py"
    local tmp_py="/tmp/gen_${step_name}_gt_$$.py"

    if [ ! -f "$src_py" ]; then
        echo "WARN: ${src_py} not found, skipping ${step_name}"
        return 0
    fi

    echo ">>> [$(date '+%H:%M:%S')] Starting: ${step_name}"

    python3 - "$src_py" "$tmp_py" "$GT_FILE" "$MODEL" "$output_json" "${extra_replacements[@]}" << 'PYEOF'
import sys, re

src = sys.argv[1]
dst = sys.argv[2]
gt_file = sys.argv[3]
model = sys.argv[4]
output_json = sys.argv[5]
extra = sys.argv[6:]

with open(src, 'r') as f:
    content = f.read()

content = re.sub(
    r'GT_TSV\s*=\s*Path\(["\']([^"\']+)["\']\)',
    f'GT_TSV = Path("{gt_file}")',
    content
)

content = re.sub(
    r'MODEL\s*=\s*["\'][^"\']+["\']',
    f'MODEL = "{model}"',
    content
)

content = re.sub(
    r'OUTPUT_JSON\s*=\s*Path\(["\'][^"\']+["\']\)',
    f'OUTPUT_JSON = Path("{output_json}")',
    content
)

if 'PERCEPTION_JSON' in content and len(extra) >= 1:
    content = re.sub(
        r'PERCEPTION_JSON\s*=\s*Path\(["\'][^"\']+["\']\)',
        f'PERCEPTION_JSON = Path("{extra[0]}")',
        content
    )

if 'COGNITION_JSON' in content:
    cog_path = extra[1] if len(extra) >= 2 else (extra[0] if len(extra) >= 1 else output_json)
    content = re.sub(
        r'COGNITION_JSON\s*=\s*Path\(["\'][^"\']+["\']\)',
        f'COGNITION_JSON = Path("{cog_path}")',
        content
    )

with open(dst, 'w') as f:
    f.write(content)
print(f"Patched config written to {dst}")
PYEOF

    python3 "$tmp_py"
    local rc=$?
    rm -f "$tmp_py"

    if [ $rc -ne 0 ]; then
        echo "ERROR: ${step_name} failed with exit code ${rc}"
        exit $rc
    fi
    echo "<<< [$(date '+%H:%M:%S')] Done: ${step_name} -> ${output_json}"
    echo ""
}
# ==================================================================

# ==================== 辅助函数：跳过已存在的条目 ====================
run_with_skip() {
    local step_name=$1
    local step_dir=$2
    local output_json=$3
    shift 3
    local extra=("$@")

    # 检查已有结果
    local existing_ids=""
    if [ -f "$output_json" ]; then
        existing_ids=$(python3 -c "
import json
data = json.load(open('${output_json}'))
if isinstance(data, list):
    print('\n'.join(item['video_id'] for item in data))
" 2>/dev/null || echo "")
    fi

    if [ -n "$existing_ids" ]; then
        local existing_count=$(echo "$existing_ids" | wc -l)
        # 过滤 GT_FILE：排除已有 video_id
        local filtered_gt="/tmp/gen_filtered_${step_name}_$$.txt"
        python3 - "$GT_FILE" "$filtered_gt" "$output_json" << 'PYEOF'
import sys, json, os
gt_file = sys.argv[1]
out_file = sys.argv[2]
existing = set()
if os.path.exists(sys.argv[3]) and os.path.getsize(sys.argv[3]) > 0:
    try:
        existing = {item['video_id'] for item in json.load(open(sys.argv[3]))}
    except:
        pass
with open(gt_file) as f:
    lines = f.readlines()
new_lines = [l for l in lines if l.split('\t')[0] not in existing]
open(out_file, 'w').writelines(new_lines)
print(f"已有 {len(existing)} 条, 新增 {len(new_lines)}/{len(lines)}")
PYEOF
        local new_count=$(wc -l < "$filtered_gt")
        if [ "$new_count" -eq 0 ] || [ "$new_count" -le 1 ]; then
            echo "<<< [$(date '+%H:%M:%S')] ${step_name}: 全部已存在，跳过"
            rm -f "$filtered_gt"
            return 0
        fi
        # 用过滤后的 GT 跑
        local old_gt="$GT_FILE"
        GT_FILE="$filtered_gt"
        patch_and_run "$step_name" "$step_dir" "/tmp/gen_partial_${step_name}_$$.json" "${extra[@]}"
        # 合并
        python3 - "$output_json" "/tmp/gen_partial_${step_name}_$$.json" "$output_json" << 'PYEOF'
import json, sys, os
old_path = sys.argv[1]
new_path = sys.argv[2]
out_path = sys.argv[3]
old = []
if os.path.exists(old_path) and os.path.getsize(old_path) > 0:
    try: old = json.load(open(old_path))
    except: pass
new_data = json.load(open(new_path))
old_by_id = {item['video_id']: item for item in old}
for item in new_data:
    old_by_id[item['video_id']] = item
merged = list(old_by_id.values())
json.dump(merged, open(out_path, 'w'), ensure_ascii=False, indent=2)
print(f"合并完成: {len(old)} + {len(new_data)} = {len(merged)}")
PYEOF
        rm -f "$filtered_gt" "/tmp/gen_partial_${step_name}_$$.json"
        GT_FILE="$old_gt"
    else
        patch_and_run "$step_name" "$step_dir" "$output_json" "${extra[@]}"
    fi
}
# ==================================================================

# ==================== GT Generation Steps ====================

# Step 1: Grounding (无 LLM，独立)
run_with_skip "grounding" "${SCRIPT_DIR}/grounding" "$GROUNDING_OUT"

# Step 2: Planning (无 LLM，独立)
run_with_skip "planning" "${SCRIPT_DIR}/planning" "$PLANNING_OUT"

# Step 3: Perception (LLM，独立)
run_with_skip "perception" "${SCRIPT_DIR}/perception" "$PERCEPTION_OUT"

# Step 4: Cognition (LLM，依赖 perception)
run_with_skip "cognition" "${SCRIPT_DIR}/cognition" "$COGNITION_OUT" "$PERCEPTION_OUT"

# ==================== 摘要 ====================
echo "============================================"
echo " All GT generation complete!"
echo "============================================"
for name in grounding planning perception cognition; do
    json_path="${SCRIPT_DIR}/${name}/${name}_gt.json"
    if [ -f "$json_path" ]; then
        count=$(python3 -c "import json; d=json.load(open('${json_path}')); print(len(d) if isinstance(d,list) else len(d))")
        echo "  ${name}: ${json_path} (${count} entries)"
    else
        echo "  ${name}: MISSING"
    fi
done

rm -f "/tmp/lifebench_filtered_gt_$$.txt"
