#!/usr/bin/env bash
# Start the FR3 arm-only ROS 2 + MoveIt stack without the stock Franka gripper.

set -eo pipefail

MODE="${1:---fake}"
if [[ "$MODE" != "--fake" && "$MODE" != "--real" ]]; then
  echo "usage: $0 [--fake|--real]" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_IP="${FRANKA_ROBOT_IP:-172.16.0.2}"
source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/third_party/franka_ros2_ws/install_ros/setup.bash"
set -u

USE_FAKE=true
if [[ "$MODE" == "--real" ]]; then
  USE_FAKE=false
  if [[ "$(uname -r)" != *realtime* ]]; then
    echo "refusing real FR3 control on non-realtime kernel: $(uname -r)" >&2
    exit 3
  fi
fi

CONFIG_SHARE="$(ros2 pkg prefix --share franka_fr3_moveit_config)"
CONTROLLERS_YAML="$CONFIG_SHARE/config/fr3_ros_controllers.yaml"
BRINGUP_PID=""
MOVE_GROUP_PID=""

cleanup() {
  trap - INT TERM EXIT
  stop_group "$MOVE_GROUP_PID"
  stop_group "$BRINGUP_PID"
  wait "$MOVE_GROUP_PID" 2>/dev/null || true
  wait "$BRINGUP_PID" 2>/dev/null || true
}

stop_group() {
  local group_id="$1"
  if [[ -z "$group_id" ]] || ! kill -0 -- "-$group_id" 2>/dev/null; then
    return
  fi
  kill -INT -- "-$group_id" 2>/dev/null || true
  for _ in $(seq 1 25); do
    if ! kill -0 -- "-$group_id" 2>/dev/null; then
      return
    fi
    sleep 0.2
  done
  kill -TERM -- "-$group_id" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

setsid ros2 launch franka_bringup franka.launch.py \
  arm_id:=fr3 \
  robot_ip:="$ROBOT_IP" \
  load_gripper:=false \
  use_fake_hardware:="$USE_FAKE" \
  fake_sensor_commands:=false \
  controllers_yaml:="$CONTROLLERS_YAML" &
BRINGUP_PID=$!

CONTROLLER_MANAGER_READY=false
for _ in $(seq 1 75); do
  if ros2 service list 2>/dev/null | grep -qx /controller_manager/list_controllers; then
    CONTROLLER_MANAGER_READY=true
    break
  fi
  sleep 0.2
done
if [[ "$CONTROLLER_MANAGER_READY" != true ]]; then
  echo "controller_manager did not become ready" >&2
  exit 4
fi

ros2 run controller_manager spawner fr3_arm_controller

setsid ros2 launch franka_fr3_moveit_config move_group.launch.py \
  robot_ip:="$ROBOT_IP" \
  load_gripper:=false \
  use_fake_hardware:="$USE_FAKE" \
  fake_sensor_commands:=false \
  namespace:=/ &
MOVE_GROUP_PID=$!

wait "$MOVE_GROUP_PID"
