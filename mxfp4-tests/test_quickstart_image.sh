#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")/.." && pwd)
eval "$(sed -n '/^ensure_quickstart_image() {/,/^}/p' "$SCRIPT_DIR/docker-quickstart.sh")"
RT=(fake_runtime)
say() { :; }
calls=()
present=0
build_result=0
fake_runtime() {
  calls+=("$*")
  case "$1" in
    image) return "$present" ;;
    build) return "$build_result" ;;
  esac
}

IMAGE=radiance117-integration:test
present=1
ensure_quickstart_image
[[ ${calls[*]} == *"build -f $SCRIPT_DIR/Dockerfile.ggz14.top -t $IMAGE $SCRIPT_DIR"* ]]
calls=()
present=0
ensure_quickstart_image
[[ ${calls[*]} != *build* ]]
calls=()
IMAGE=example/custom:latest
present=1
ensure_quickstart_image
[[ ${#calls[@]} == 0 ]]
IMAGE=radiance117-integration:test
build_result=1
if ensure_quickstart_image; then
  echo 'failed build must return an error' >&2
  exit 1
fi
echo 'quickstart image checks passed'
