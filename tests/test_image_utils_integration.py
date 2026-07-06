import uuid

import pytest
from inspect_ai.dataset import Sample

from scheme.environment.agents import FULL_ACCESS, AgentConfig, AppendOnlyConfig
from scheme.environment.constants import SANDBOX_WORKSPACE
from scheme.image_utils import build_images, setup_file_permissions

COMM_FILE = "communication.txt"
AGENT_ONE_FILE = "agent_one.txt"
AGENT_TWO_FILE = "agent_two.txt"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def docker_client():
    docker = pytest.importorskip("docker")
    client = docker.from_env()
    try:
        client.ping()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.skip(f"Docker daemon unavailable: {exc}")
    yield client
    client.close()


@pytest.fixture(scope="module")
def sample():
    return Sample(
        id=f"image-perms-{uuid.uuid4().hex[:8]}",
        input="integration sample",
        files={
            "shared.txt": "shared file",
            AGENT_ONE_FILE: "agent one content",
            AGENT_TWO_FILE: "agent two content",
            COMM_FILE: "initial message\n",
        },
        metadata={"libs": []},
    )


@pytest.fixture(scope="module")
def base_image_name(docker_client, sample):
    image_map = build_images(
        samples=[sample],
        docker_client=docker_client,
        task_name="image-permissions",
        force_rebuild=False,
    )
    image_name = image_map[str(sample.id)]
    yield image_name
    try:
        docker_client.images.remove(image=image_name, force=True)
    except Exception:
        pass


def _run_one_off(docker_client, image, user, command):
    container = docker_client.containers.run(
        image=image,
        command=["bash", "-c", command],
        user=user,
        detach=True,
        working_dir=SANDBOX_WORKSPACE,
    )
    try:
        exit_code = container.wait()["StatusCode"]
        output = container.logs().decode()
    finally:
        container.remove()
    return exit_code, output


def _exec(container, command, user):
    result = container.exec_run(["bash", "-c", command], user=user, workdir=SANDBOX_WORKSPACE)
    output = result.output.decode() if isinstance(result.output, (bytes, bytearray)) else str(result.output)
    return result.exit_code, output


def test_full_read_access_all_files_readable(docker_client, base_image_name, sample):
    agent_configs = [
        AgentConfig(
            id="reader_one", read_access=FULL_ACCESS, write_access=[AGENT_ONE_FILE], working_dir=SANDBOX_WORKSPACE
        ),
        AgentConfig(
            id="reader_two", read_access=FULL_ACCESS, write_access=[AGENT_TWO_FILE], working_dir=SANDBOX_WORKSPACE
        ),
    ]
    restricted_image = setup_file_permissions(
        docker_client=docker_client,
        image_name=base_image_name,
        agent_configs=agent_configs,
        working_dir=SANDBOX_WORKSPACE,
    )

    try:
        for user in ("reader_one", "reader_two"):
            for path in sample.files.keys():
                exit_code, output = _run_one_off(docker_client, restricted_image, user, f"cat {path}")
                assert exit_code == 0, f"{user} could not read {path}: {output}"
    finally:
        try:
            docker_client.images.remove(image=restricted_image, force=True)
        except Exception:
            pass


def test_comm_only_agents_are_isolated_and_can_append(docker_client, base_image_name):
    agent_configs = [
        AgentConfig(
            id="agent_alpha",
            read_access=[COMM_FILE, AGENT_ONE_FILE],
            write_access=[AGENT_ONE_FILE],
            append_only=[AppendOnlyConfig(file_path=COMM_FILE, command="append_comm")],
            working_dir=SANDBOX_WORKSPACE,
        ),
        AgentConfig(
            id="agent_beta",
            read_access=[COMM_FILE, AGENT_TWO_FILE],
            write_access=[AGENT_TWO_FILE],
            append_only=[AppendOnlyConfig(file_path=COMM_FILE, command="append_comm")],
            working_dir=SANDBOX_WORKSPACE,
        ),
    ]
    restricted_image = setup_file_permissions(
        docker_client=docker_client,
        image_name=base_image_name,
        agent_configs=agent_configs,
        working_dir=SANDBOX_WORKSPACE,
    )

    container = None
    try:
        container = docker_client.containers.run(
            image=restricted_image,
            command=["sleep", "300"],
            detach=True,
            working_dir=SANDBOX_WORKSPACE,
        )

        exit_code, output = _exec(container, f"cat {AGENT_ONE_FILE}", user="agent_alpha")
        assert exit_code == 0, f"agent_alpha cannot read own file: {output}"
        exit_code, _ = _exec(container, f"cat {AGENT_TWO_FILE}", user="agent_alpha")
        assert exit_code != 0, "agent_alpha should not read agent_beta's file"

        exit_code, output = _exec(container, f"cat {AGENT_TWO_FILE}", user="agent_beta")
        assert exit_code == 0, f"agent_beta cannot read own file: {output}"
        exit_code, _ = _exec(container, f"cat {AGENT_ONE_FILE}", user="agent_beta")
        assert exit_code != 0, "agent_beta should not read agent_alpha's file"

        exit_code, output = _exec(container, f"cat {COMM_FILE}", user="agent_alpha")
        assert exit_code == 0, f"agent_alpha cannot read comm file: {output}"
        exit_code, output = _exec(container, f"cat {COMM_FILE}", user="agent_beta")
        assert exit_code == 0, f"agent_beta cannot read comm file: {output}"

        exit_code, output = _exec(container, '/usr/local/bin/append_comm "agent_alpha: ping"', user="agent_alpha")
        assert exit_code == 0, f"agent_alpha append failed: {output}"
        exit_code, output = _exec(container, '/usr/local/bin/append_comm "agent_beta: pong"', user="agent_beta")
        assert exit_code == 0, f"agent_beta append failed: {output}"

        exit_code, output = _exec(container, f"tail -n 2 {COMM_FILE}", user="agent_beta")
        assert exit_code == 0, f"agent_beta could not read appended comm file: {output}"
        assert "agent_alpha: ping" in output and "agent_beta: pong" in output
    finally:
        if container is not None:
            container.remove(force=True)
        try:
            docker_client.images.remove(image=restricted_image, force=True)
        except Exception:
            pass
