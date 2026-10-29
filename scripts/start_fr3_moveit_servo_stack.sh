#!/usr/bin/env bash
# Start the FR3 arm-only MoveIt stack plus one MoveIt Servo node.
# This script never publishes a Cartesian or joint motion command.

set -eo pipefail

MODE="--fake"
SERVO_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --fake|--real) MODE="$arg" ;;
    --servo-only) SERVO_ONLY=true ;;
    -h|--help)
      echo "usage: $0 [--fake|--real] [--servo-only]"
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_IP="${FRANKA_ROBOT_IP:-172.16.0.2}"
source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/third_party/franka_ros2_ws/install_ros/setup.bash"
set -u

if [[ "$MODE" == "--real" && "$(uname -r)" != *realtime* ]]; then
  echo "refusing real FR3 stack on non-realtime kernel: $(uname -r)" >&2
  exit 3
fi

/usr/bin/python3 "$REPO_ROOT/scripts/preflight_fr3_moveit_servo.py" \
  --config "$REPO_ROOT/configs/fr3_moveit_servo.yaml" \
  $([[ "$MODE" == "--real" ]] && echo --require-realtime)

STACK_PID=""
SERVO_PID=""

stop_group() {
  local group_id="$1"
  if [[ -n "$group_id" ]] && kill -0 -- "-$group_id" 2>/dev/null; then
    kill -INT -- "-$group_id" 2>/dev/null || true
  fi
}

cleanup() {
  trap - INT TERM EXIT
  stop_group "$SERVO_PID"
  stop_group "$STACK_PID"
  [[ -z "$SERVO_PID" ]] || wait "$SERVO_PID" 2>/dev/null || true
  [[ -z "$STACK_PID" ]] || wait "$STACK_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

if [[ "$SERVO_ONLY" == false ]]; then
  if ros2 service list 2>/dev/null | grep -qx /controller_manager/list_controllers; then
    echo "a controller_manager is already running; use --servo-only" >&2
    exit 4
  fi
  setsid "$REPO_ROOT/scripts/start_fr3_moveit_arm_only.sh" "$MODE" &
  STACK_PID=$!
fi

STACK_READY=false
for _ in $(seq 1 100); do
  if ros2 node list 2>/dev/null | grep -qx /move_group && \
     ros2 control list_controllers 2>/dev/null \
       | sed -E $'s/\x1B\\[[0-9;]*[mK]//g' \
       | grep -Eq '^fr3_arm_controller .* active$'; then
    STACK_READY=true
    break
  fi
  sleep 0.2
done
if [[ "$STACK_READY" != true ]]; then
  echo "MoveIt/controller stack did not become ready" >&2
  exit 5
fi

SERVO_ARGS=(
  "$REPO_ROOT/scripts/launch_fr3_moveit_servo.py"
  --config "$REPO_ROOT/configs/fr3_moveit_servo.yaml"
  --robot-ip "$ROBOT_IP"
)
if [[ "$MODE" == "--fake" ]]; then
  SERVO_ARGS+=(--use-fake-hardware)
fi
setsid /usr/bin/python3 "${SERVO_ARGS[@]}" &
SERVO_PID=$!

SERVO_READY=false
for _ in $(seq 1 100); do
  if ros2 node list 2>/dev/null | grep -qx /servo_node && \
     ros2 topic info /servo_node/delta_twist_cmds 2>/dev/null | grep -Eq 'Subscription count: [1-9]'; then
    SERVO_READY=true
    break
  fi
  sleep 0.2
done
if [[ "$SERVO_READY" != true ]]; then
  echo "MoveIt Servo did not become ready" >&2
  exit 6
fi

PREFLIGHT_ARGS=(
  --config "$REPO_ROOT/configs/fr3_moveit_servo.yaml"
  --runtime
  --configure-twist
)
if [[ "$MODE" == "--real" ]]; then
  PREFLIGHT_ARGS+=(--require-realtime)
fi
/usr/bin/python3 "$REPO_ROOT/scripts/preflight_fr3_moveit_servo.py" "${PREFLIGHT_ARGS[@]}"

echo "FR3 MoveIt + Servo is ready; no motion command has been sent."
echo "input:  /servo_node/delta_twist_cmds (TwistStamped, speed units)"
echo "output: /fr3_arm_controller/joint_trajectory (JointTrajectory)"

wait "$SERVO_PID"
