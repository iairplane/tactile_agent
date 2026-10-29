from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


def _load_stack():
    path = Path(__file__).resolve().parents[1] / "scripts/run_test_tube_live_stack.py"
    spec = importlib.util.spec_from_file_location("loopback_stack", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("hostname", ["127.0.0.1", "localhost"])
def test_loopback_health_bypasses_broken_environment_proxy(hostname, monkeypatch):
    module = _load_stack()
    requests = []

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            body = json.dumps({"ready": True, "backend": {
                "name": "f2g_pose", "pose_only": True,
            }}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.setenv(name, "http://127.0.0.1:1")
    monkeypatch.setenv("no_proxy", "deliberately-invalid.example")
    monkeypatch.setenv("NO_PROXY", "deliberately-invalid.example")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Health)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = module._service_health(
            f"http://{hostname}:{server.server_port}", backend_name="f2g_pose",
            label="pose", require_pose_only=True, timeout_s=1,
        )
        assert result == (True, "ready")
        assert requests == ["/health"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_resident_launch_overrides_proxy_bypass_after_setup_scripts(tmp_path, monkeypatch):
    module = _load_stack()
    root = tmp_path / "fake repository"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    # The generated command launches this harmless fixture, never the live runner.
    (scripts / "run_test_tube_closed_loop.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'upper': os.environ.get('NO_PROXY'), "
        "'lower': os.environ.get('no_proxy'), 'setup': os.environ.get('LOOPBACK_TEST_SETUP'), "
        "'argv': sys.argv[1:]}))\n"
    )
    monkeypatch.setattr(module, "REPO_ROOT", root)
    setup = tmp_path / "hostile proxy setup.bash"
    setup.write_text(
        "export NO_PROXY=bad-upper.example\n"
        "export no_proxy=bad-lower.example\n"
        "export LOOPBACK_TEST_SETUP=loaded\n"
    )
    task = {"live_adapter": {"robot": {"moveit": {"setup_scripts": [str(setup)]}}}}
    task_config = tmp_path / "task config.json"
    output = tmp_path / "task output"
    argv = module._task_command(
        task=task, task_config=task_config, task_output=output,
        accept_arm_only_collision_model=True,
    )
    env = {**os.environ, "NO_PROXY": "parent-bad.example", "no_proxy": "parent-bad.example"}
    result = subprocess.run(argv, env=env, text=True, capture_output=True, timeout=10, check=True)
    child = json.loads(result.stdout.splitlines()[-1])
    assert child["setup"] == "loaded"
    assert child["upper"] == child["lower"] == "127.0.0.1,localhost,::1"
    assert child["argv"][child["argv"].index("--config") + 1] == str(task_config)
    assert child["argv"][child["argv"].index("--output-dir") + 1] == str(output)
    assert "--accept-arm-only-collision-model" in child["argv"]
