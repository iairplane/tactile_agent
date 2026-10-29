# FR3 MoveIt Servo boundary

The test-tube live adapter needs a resident Cartesian velocity path.  The
existing `move_group` process plans complete collision-checked trajectories, but
it does not accept the 50 Hz bounded corrections used by the closed-loop task.
MoveIt Servo fills that boundary.

## Install blocker

The host currently does not contain `moveit_servo`.  Installation is intentionally
not performed by this repository:

```bash
sudo apt install ros-jazzy-moveit-servo
```

The configuration was matched statically to the host's Jazzy candidate,
MoveIt Servo `2.12.4`.  It has not yet been runtime-qualified on this FR3.

## One process boundary

From a stopped ROS graph, start the complete fake or real stack with one command:

```bash
cd /home/vitai/lzr/tactile_agent
scripts/start_fr3_moveit_servo_stack.sh --fake
scripts/start_fr3_moveit_servo_stack.sh --real
```

If `move_group`, `controller_manager`, and `fr3_arm_controller` are already
running, attach only Servo:

```bash
scripts/start_fr3_moveit_servo_stack.sh --real --servo-only
```

The launcher starts `/servo_node`, selects `TWIST` through
`/servo_node/switch_command_type`, and verifies the graph.  It sends no Twist or
trajectory command.  The live task owns `/servo_node/delta_twist_cmds` after
startup.

## Command and collision semantics

- Input is `geometry_msgs/TwistStamped` on
  `/servo_node/delta_twist_cmds`, expressed in speed units.
- Servo emits `trajectory_msgs/JointTrajectory` on
  `/fr3_arm_controller/joint_trajectory`.  The active
  `joint_trajectory_controller` subscribes to that topic.
- Coarse `move_group` actions and Servo output are two command sources for the
  same controller.  They must be mutually exclusive: finish/cancel a coarse
  action, publish a zero Twist, then enter Servo; stop Servo before another
  coarse action.
- Servo checks self/scene collision and singularity proximity.  It consumes the
  primary `move_group` scene from `/monitored_planning_scene`.
- The loaded model ends at `fr3_link8`.  The Hand-E, adapter, fingers and the
  190 mm canonical TCP offset are absent.  Collision checking is therefore
  **arm-only** and does not certify contact motion or clearance around the
  physical gripper.
- Servo rotates the model tip (`fr3_link8`).  The task controller closes the
  loop on its measured canonical TCP, but a large rotational correction would
  still make the offset TCP trace an unmodelled arc.  Keep Servo rotations
  bounded until a measured Hand-E URDF/SRDF/TCP is installed.

MoveIt Servo also requires an explicit command-type selection before it accepts
Twist commands.  Both the launcher and the live adapter confirm that selection;
a subscriber existing on the topic alone is not sufficient evidence.

## Read-only checks

Static preflight (currently expected to report `ros_package_missing:moveit_servo`):

```bash
source /opt/ros/jazzy/setup.bash
source third_party/franka_ros2_ws/install_ros/setup.bash
/usr/bin/python3 scripts/preflight_fr3_moveit_servo.py
```

After startup, inspect the live graph and select TWIST without sending motion:

```bash
/usr/bin/python3 scripts/preflight_fr3_moveit_servo.py \
  --runtime --configure-twist --require-realtime
```

Runtime qualification still needs a fake-hardware zero/nonzero/zero Twist smoke
test, observed Servo status, and a real low-speed free-space test before this
path can be labelled qualified.
