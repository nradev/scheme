"""Generate sandbox configuration files for multi-agent evaluation."""

import logging
from pathlib import Path
from textwrap import dedent
from typing import Literal

from platformdirs import user_cache_dir

logger = logging.getLogger(__name__)

# Cache directory for compose files
COMPOSE_FILES_DIR = Path(user_cache_dir("inspect_multiagent_eval")) / "compose_files"


def get_sandbox_config_file(
    sample_id: str,
    image_name: str,
    sandbox_type: Literal["docker", "k8s"] = "docker",
    allow_internet: bool = False,
    sandbox_config_template_file: str | None = None,
    working_dir: str = "/workspace",
) -> str:
    """
    Generate and save a sandbox configuration file for the given sample.

    Args:
        sample_id: Unique identifier for the sample
        image_name: Docker image name to use
        sandbox_type: Type of sandbox ("docker" or "k8s")
        allow_internet: Whether to allow internet access
        sandbox_config_template_file: Optional custom template file path
        working_dir: Working directory in the container

    Returns:
        Path to the generated configuration file
    """
    COMPOSE_FILES_DIR.mkdir(parents=True, exist_ok=True)

    # If a custom sandbox config template file is provided, use it
    if sandbox_config_template_file:
        template_path = Path(sandbox_config_template_file)
        if not template_path.exists():
            raise FileNotFoundError(f"Sandbox config template file not found: {sandbox_config_template_file}")

        template_content = template_path.read_text()
        content = template_content.replace("{{IMAGE_NAME}}", image_name)
        config_file = COMPOSE_FILES_DIR / f"{sample_id}-custom.yaml"
    elif sandbox_type == "docker":
        config_file = COMPOSE_FILES_DIR / f"{sample_id}-compose.yaml"
        content = get_docker_config_content(image_name, allow_internet, working_dir)
    elif sandbox_type == "k8s":
        config_file = COMPOSE_FILES_DIR / f"{sample_id}-values.yaml"
        content = get_k8s_config_content(image_name, allow_internet, working_dir)
    else:
        raise ValueError(f"Unknown sandbox type: {sandbox_type}")

    # Ensure parent directory exists (sample_id may contain subdirectories)
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with config_file.open(mode="w") as f:
        f.write(content)

    logger.debug(f"Created sandbox config file: {config_file}")
    return str(config_file)


def get_docker_config_content(
    image_name: str,
    allow_internet: bool = False,
    working_dir: str = "/workspace",
) -> str:
    """
    Generate Docker Compose configuration content.

    Args:
        image_name: Docker image name to use
        allow_internet: Whether to allow internet access
        working_dir: Working directory in the container

    Returns:
        Docker Compose YAML content
    """
    content = dedent(f"""
        services:
            default:
                image: {image_name}
                command: "sleep infinity"
                working_dir: {working_dir}
                {"network_mode: none" if not allow_internet else ""}
                deploy:
                    resources:
                        limits:
                            cpus: '1'
                            memory: 2G
        """)
    return content


def get_k8s_config_content(
    image_name: str,
    allow_internet: bool = False,
    working_dir: str = "/workspace",
) -> str:
    """
    Generate Kubernetes configuration content.

    Args:
        image_name: Docker image name to use
        allow_internet: Whether to allow internet access
        working_dir: Working directory in the container

    Returns:
        Kubernetes values YAML content
    """
    content = dedent(f"""
        services:
            default:
                image: {image_name}
                command: ["tail", "-f", "/dev/null"]
                workingDir: {working_dir}
        {'allowDomains: ["*"]' if allow_internet else ""}
        """)
    return content
