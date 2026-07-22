#!/usr/bin/env sh

PROXY_HOST_RESOLVED="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT_RESOLVED="${PROXY_PORT:-9871}"
PROXY_URL="http://${PROXY_HOST_RESOLVED}:${PROXY_PORT_RESOLVED}"

export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export ALL_PROXY="${PROXY_URL}"
export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export all_proxy="${PROXY_URL}"
