#!/usr/bin/env python3
"""Allow Ray ServeController to be pinned off the small QS2 head node.

QS2 RayJobs often give the head node only ~10 GiB memory. Relax uses Ray Serve
for its core services, and upstream Ray pins ServeController to the head node.
For long Relax runs this can trigger Ray memory protection before any GPU actor
starts. This patch keeps upstream behavior by default, but when
RAY_SERVE_CONTROLLER_NODE_RESOURCE is set, it lets the controller use that node
resource instead.
"""

from __future__ import annotations

from pathlib import Path


DEFAULT_IMPL = Path("/usr/local/lib/python3.12/dist-packages/ray/serve/_private/default_impl.py")
CONTROLLER = Path("/usr/local/lib/python3.12/dist-packages/ray/serve/_private/controller.py")
CONSTANTS = Path("/usr/local/lib/python3.12/dist-packages/ray/serve/_private/constants.py")


def replace_once(path: Path, old: str, new: str) -> bool:
    text = path.read_text()
    if new in text:
        return False
    if old not in text:
        raise RuntimeError(f"Expected patch anchor not found in {path}")
    path.write_text(text.replace(old, new))
    return True


def main() -> None:
    changed = []

    if "import os\n" not in DEFAULT_IMPL.read_text().split("import ray\n", 1)[0]:
        replace_once(DEFAULT_IMPL, "import asyncio\n", "import asyncio\nimport os\n")
        changed.append(str(DEFAULT_IMPL))

    if "controller_node_resource = os.environ.get(\"RAY_SERVE_CONTROLLER_NODE_RESOURCE\")" not in DEFAULT_IMPL.read_text():
        replace_once(
            DEFAULT_IMPL,
            """def get_controller_impl():
    from ray.serve._private.controller import ServeController

    controller_impl = ray.remote(
""",
            """def get_controller_impl():
    from ray.serve._private.controller import ServeController

    controller_node_resource = os.environ.get("RAY_SERVE_CONTROLLER_NODE_RESOURCE")
    controller_resources = (
        {controller_node_resource: 0.001}
        if controller_node_resource
        else {HEAD_NODE_RESOURCE_NAME: 0.001}
    )

    controller_impl = ray.remote(
""",
        )
        changed.append(str(DEFAULT_IMPL))

    replace_once(
        DEFAULT_IMPL,
        "        resources={HEAD_NODE_RESOURCE_NAME: 0.001},\n",
        "        resources=controller_resources,\n",
    )

    replace_once(
        CONTROLLER,
        """        assert (
            self._controller_node_id == get_head_node_id()
        ), "Controller must be on the head node."
""",
        """        if not os.environ.get("RAY_SERVE_CONTROLLER_NODE_RESOURCE"):
            assert (
                self._controller_node_id == get_head_node_id()
            ), "Controller must be on the head node."
""",
    )

    replace_once(
        CONSTANTS,
        "HTTP_PROXY_TIMEOUT = 60\n",
        'HTTP_PROXY_TIMEOUT = get_env_int_positive("RAY_SERVE_HTTP_PROXY_TIMEOUT_S", 60)\n',
    )

    print("Ray Serve controller pin patch ready")
    if changed:
        print("changed:", ", ".join(changed))


if __name__ == "__main__":
    main()
