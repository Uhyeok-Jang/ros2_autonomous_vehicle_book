#!/bin/bash
# Change only model weights; use the workspace's existing driving launcher.
set -euo pipefail

task_workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
case "${1-}" in
    v3) task_model_name=v3_good_corner4_issue ;;
    v3.1) task_model_name=v3.1_not_good_corner4_good ;;
    v3.2) task_model_name=v3.2_finetuned_corner4_with_v3 ;;
    -h|--help)
        printf 'Usage: %s {v3|v3.1|v3.2} [ROS launch arguments...]\n' "$0"
        exit 0 ;;
    *)
        printf 'Choose a model: %s {v3|v3.1|v3.2}\n' "$0" >&2
        exit 2 ;;
esac
shift
for task_arg in "$@"; do
    if [[ "$task_arg" == yolo_model:=* ]]; then
        printf 'Use the version argument to select a model; yolo_model override is ambiguous.\n' >&2
        exit 2
    fi
done
task_model_path="$task_workspace/model/$task_model_name.pt"
if [[ ! -f "$task_model_path" ]]; then
    printf 'Model file missing: %s\n' "$task_model_path" >&2
    exit 1
fi
if [[ -x "$task_workspace/run_driving_real.sh" ]]; then
    task_runner="$task_workspace/run_driving_real.sh"
elif [[ -x "$task_workspace/run_driving_sim.sh" ]]; then
    task_runner="$task_workspace/run_driving_sim.sh"
else
    printf 'Driving launcher missing in %s\n' "$task_workspace" >&2
    exit 1
fi
printf 'Model: %s\nPath: %s\n' "$task_model_name" "$task_model_path" >&2
exec "$task_runner" "yolo_model:=$task_model_path" "$@"
