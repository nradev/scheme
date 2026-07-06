"""Integration test for send_comm and read_comms tools against a live Docker sandbox.

Tests that messages survive round-trip through the sandbox filesystem, with emphasis
on messages containing shell-special characters that broke the original {content!r}
quoting approach.

Run with:
    pytest tests/test_comm_tools.py -m integration -s
"""
from __future__ import annotations

import pytest
from inspect_ai import Task
from inspect_ai import eval as run_eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import Generate, TaskState, solver
from inspect_ai.util import SandboxEnvironmentSpec

from scheme.environment.tools import create_read_comms_tool, create_send_comm_tool
from scheme.image_utils.get_sandbox_config import get_docker_config_content

pytestmark = pytest.mark.integration

COMMON_CHANNEL_FILE = "comms/comms.txt"
COMMON_CHANNEL_PERMS = {
    "read": {"agent_0": [COMMON_CHANNEL_FILE], "agent_1": [COMMON_CHANNEL_FILE]},
    "write": {"agent_0": [COMMON_CHANNEL_FILE], "agent_1": [COMMON_CHANNEL_FILE]},
}

# Peer-to-peer: filename contains brackets, exercising the shlex.quote(file_path) fix.
P2P_FILE = "comms/comms_[agent_0]_[agent_1].txt"
P2P_PERMS = {
    "read": {"agent_0": [P2P_FILE], "agent_1": [P2P_FILE]},
    "write": {"agent_0": [P2P_FILE], "agent_1": [P2P_FILE]},
}

# Messages designed to exercise edge cases in shell quoting.
# The 5th entry is exactly what agent_0 sent in the failing trajectory.
MESSAGES_FROM_0 = [
    "simple message from agent_0",
    "message with 'single quotes'",
    'message with "double quotes"',
    "both ' single and \"double\" quotes in one message",
    "output token is ' blue'. We need (\"On a clear summer day\")",
    "backticks: `done()` tool call",
]
MESSAGES_FROM_1 = [
    "reply from agent_1",
    "agent_1 also has 'quotes' and \"double quotes\"",
]


def _make_roundtrip_solver(comm_permissions):
    @solver
    def roundtrip_solver():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            send_0 = create_send_comm_tool("agent_0", comm_permissions)
            send_1 = create_send_comm_tool("agent_1", comm_permissions)
            read_0 = create_read_comms_tool("agent_0", comm_permissions)
            read_1 = create_read_comms_tool("agent_1", comm_permissions)

            for msg in MESSAGES_FROM_0:
                result = await send_0(msg)
                assert "Successfully sent" in result, f"send_comm (agent_0) failed for {msg!r}: {result}"

            for msg in MESSAGES_FROM_1:
                result = await send_1(msg)
                assert "Successfully sent" in result, f"send_comm (agent_1) failed for {msg!r}: {result}"

            all_messages = MESSAGES_FROM_0 + MESSAGES_FROM_1

            for reader_id, read_fn in [("agent_0", read_0), ("agent_1", read_1)]:
                contents = await read_fn()
                missing = [m for m in all_messages if m not in contents]
                assert not missing, (
                    f"{reader_id} missing messages:\n"
                    + "\n".join(f"  {m!r}" for m in missing)
                    + f"\n\nActual contents:\n{contents}"
                )

            state.metadata["test_passed"] = True
            return state

        return solve

    return roundtrip_solver()


@pytest.fixture
def compose_file(tmp_path):
    content = get_docker_config_content("ubuntu:22.04", allow_internet=False, working_dir="/workspace")
    compose = tmp_path / "compose.yaml"
    compose.write_text(content)
    return str(compose)


def _run_task(solver_instance, compose_file, tmp_path):
    sandbox_spec = SandboxEnvironmentSpec(type="docker", config=compose_file)
    sample = Sample(input="test comm tools", id="comm-tools-test", sandbox=sandbox_spec)
    task = Task(dataset=MemoryDataset([sample]), solver=solver_instance)
    logs = run_eval(
        tasks=[task],
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path / "logs"),
    )
    assert logs[0].status == "success", f"Eval failed: {logs[0].error}"
    assert logs[0].samples[0].metadata.get("test_passed") is True


def test_comm_tools_common_channel(compose_file, tmp_path):
    """Both agents write and read a shared channel; all messages including shell-special chars survive."""
    _run_task(_make_roundtrip_solver(COMMON_CHANNEL_PERMS), compose_file, tmp_path)


def test_comm_tools_peer_to_peer_channel(compose_file, tmp_path):
    """Peer-to-peer channel with brackets in the filename (exercises shlex.quote on file_path)."""
    _run_task(_make_roundtrip_solver(P2P_PERMS), compose_file, tmp_path)