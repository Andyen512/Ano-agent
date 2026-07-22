#!/bin/bash
# Convert mpeg4 videos to h264 for browser playback
# Skips files already converted (checks if h264 codec)

LOG_FILE="/data_4/liuyuan/lifebench/convert_log.txt"
CONVERTED=0
SKIPPED=0
FAILED=0

for dataset in "Single_user_simple_event" "Multiple_users_compound_event"; do
    echo "Processing dataset: $dataset" | tee -a "$LOG_FILE"
    for f in $(find /data_4/liuyuan/lifebench/data/public_data/$dataset -name "*.mp4" -type f); do
        # Check current codec
        codec=$(ffprobe -v quiet -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 "$f" 2>/dev/null)
        
        if [ "$codec" = "h264" ]; then
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        # Convert to h264 in-place
        tmp_file="${f}.h264tmp.mp4"
        if ffmpeg -y -i "$f" -c:v libx264 -preset fast -crf 23 -c:a aac -b:a 128k "$tmp_file" -loglevel error 2>/dev/null; then
            mv "$tmp_file" "$f"
            CONVERTED=$((CONVERTED + 1))
            echo "Converted: $f" | tee -a "$LOG_FILE"
        else
            rm -f "$tmp_file"
            FAILED=$((FAILED + 1))
            echo "FAILED: $f" | tee -a "$LOG_FILE"
        fi
        
        # Progress report every 50 files
        total=$((CONVERTED + SKIPPED + FAILED))
        if [ $((total % 50)) -eq 0 ]; then
            echo "Progress: $total files processed ($CONVERTED converted, $SKIPPED skipped, $FAILED failed)" | tee -a "$LOG_FILE"
        fi
    done
done

echo "Done! Converted: $CONVERTED, Skipped: $SKIPPED, Failed: $FAILED" | tee -a "$LOG_FILE"
