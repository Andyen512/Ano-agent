#!/bin/bash
# run-env-host.sh - 使用 host 网络启动容器，便于直接复用宿主机本地代理

CONTAINER_NAME="ccman-$(date +%s)-$RANDOM"

echo "启动容器: $CONTAINER_NAME"

docker run -it --gpus all --name "$CONTAINER_NAME" \
  --network host \
  -v /data_4/liuyuan:/data_4/liuyuan \
  -v "$HOME:$HOME" \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /run/user/$(id -u)/bus:/run/user/$(id -u)/bus \
  --device /dev/dri:/dev/dri \
  -e DISPLAY \
  -u "$(id -u):$(id -g)" \
  -e "HOME=$HOME" \
  -e "USER=$(whoami)" \
  -e "LOGNAME=$(whoami)" \
  -e "SHELL=$SHELL" \
  -w /data_4/liuyuan \
  -e "LANG=$LANG" \
  -e "LC_ALL=$LC_ALL" \
  ccman-with-codex:latest bash