#!/bin/bash
# run-env.sh - 启动与本地环境一致的Docker容器

# docker stop ccman-box3 2>/dev/null
# docker rm ccman-box3 2>/dev/null

# 生成唯一的容器名
CONTAINER_NAME="ccman-$(date +%s)-$RANDOM"

echo "启动容器: $CONTAINER_NAME"


docker run -it --gpus all --name "$CONTAINER_NAME" \
  -v /data_4/liuyuan:/data_4/liuyuan \
  -v "$HOME:$HOME" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -u "$(id -u):$(id -g)" \
  -e "HOME=$HOME" \
  -e "USER=$(whoami)" \
  -e "LOGNAME=$(whoami)" \
  -e "SHELL=$SHELL" \
  -w /data_4/liuyuan \
  -e "LANG=$LANG" \
  -e "LC_ALL=$LC_ALL" \
  ccman-with-codex:latest bash


# # 1. 启动容器
# ./run-ccman.sh
# # 在容器内：I have no name!@xxxxxxxx:/data_4/liuyuan$ 

# # 2. 在容器内输入 exit
# exit

# # 3. 在主机上查看容器状态
# docker ps -a --filter "name=ccman" --format "table {{.Names}}\t{{.Status}}"
# # 输出示例：
# # NAMES             STATUS
# # ccman-XXXXXXXX    Exited (0) 2 minutes ago