from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_preflight():
    path = ROOT / "scripts" / "preflight_fr3_moveit_servo.py"
    spec = importlib.util.spec_from_file_location("preflight_fr3_moveit_servo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fr3_servo_config_matches_live_adapter_topics() -> None:
    module = _load_preflight()
    config, blockers = module.load_and_validate(
        ROOT / "configs" / "fr3_moveit_servo.yaml"
    )
    assert blockers == []
    assert config["move_group_name"] == "fr3_arm"
    assert config["cartesian_command_in_topic"] == "~/delta_twist_cmds"
    assert config["command_out_topic"] == "/fr3_arm_controller/joint_trajectory"
    assert config["check_collisions"] is True
    assert config["is_primary_planning_scene_monitor"] is False
    assert config["publish_joint_positions"] is True
    assert config["publish_joint_velocities"] is False


def test_launcher_preserves_arm_only_boundary() -> None:
    source = (ROOT / "scripts" / "launch_fr3_moveit_servo.py").read_text(
        encoding="utf-8"
    )
    assert '" hand:=false"' in source
    assert 'name="servo_node"' in source
    assert '{"moveit_servo": servo_params}' in source


def test_runtime_graph_waits_for_topic_endpoints_after_nodes_appear() -> None:
    module = _load_preflight()

    class FakeNode:
        ready = False

        def get_node_names_and_namespaces(self):
            return [("servo_node", "/"), ("move_group", "/")]

        def count_subscribers(self, topic):
            return int(self.ready)

        def count_publishers(self, topic):
            return int(self.ready)

    class FakeClient:
        def __init__(self, node):
            self.node = node

        def service_is_ready(self):
            return self.node.ready

    class FakeRclpy:
        spins = 0

        @classmethod
        def spin_once(cls, node, *, timeout_sec):
            cls.spins += 1
            if cls.spins == 2:
                node.ready = True

    node = FakeNode()
    blockers = module._wait_for_runtime_graph(
        node,
        rclpy_module=FakeRclpy,
        pause_client=FakeClient(node),
        switch_client=FakeClient(node),
        timeout_s=0.1,
    )

    assert FakeRclpy.spins == 2
    assert blockers == []


def test_runtime_graph_timeout_reports_missing_endpoint() -> None:
    module = _load_preflight()

    class FakeNode:
        def get_node_names_and_namespaces(self):
            return [("servo_node", "/"), ("move_group", "/")]

        def count_subscribers(self, topic):
            return 0 if topic.endswith("joint_trajectory") else 1

        def count_publishers(self, topic):
            return 1

    class FakeClient:
        @staticmethod
        def service_is_ready():
            return True

    class FakeRclpy:
        @staticmethod
        def spin_once(node, *, timeout_sec):
            return None

    blockers = module._wait_for_runtime_graph(
        FakeNode(),
        rclpy_module=FakeRclpy,
        pause_client=FakeClient(),
        switch_client=None,
        timeout_s=0.0,
    )

    assert blockers == ["trajectory_controller_subscription_missing"]


def test_demo07_alone_disables_motion_guards_and_receipt_records_it() -> None:
    for field in ("disable_pose_delta_guard", "disable_joint_sweep_guards"):
        enabled = []
        for path in (ROOT / "configs").glob("test_tube_closed_loop*.json"):
            config = json.loads(path.read_text(encoding="utf-8"))
            moveit = config.get("live_adapter", {}).get("robot", {}).get("moveit", {})
            if moveit.get(field) is True:
                enabled.append(path.name)
        assert enabled == ["test_tube_closed_loop.demo07.json"]

    source = (ROOT / "scripts" / "moveit_pose_goal.py").read_text(encoding="utf-8")
    assert '"--disable-pose-delta-guard"' in source
    assert '"pose_delta_guard_enabled": not args.disable_pose_delta_guard' in source
    assert (
        '"pose_delta_guard_disabled_by_operator_config": ('
        in source
    )
    assert '"--disable-joint-sweep-guards"' in source
    assert (
        '"joint_sweep_guards_enabled": not args.disable_joint_sweep_guards'
        in source
    )
    assert (
        '"joint_sweep_guards_disabled_by_operator_config": ('
        in source
    )
