#!/usr/bin/env bash
# Run one repository ROS integration tool in the already-running ROS 2 stack.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/third_party/franka_ros2_ws/install_ros/setup.bash"
set -u
exec /usr/bin/python3 "$@"
