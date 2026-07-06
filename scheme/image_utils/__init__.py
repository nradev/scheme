"""Image utilities for multi-agent evaluation.

This module provides utilities for building, caching, and configuring Docker images
for multi-agent evaluation tasks, similar to the SWE-bench implementation.
"""

from .build_images import (
    build_dockerfile_content,
    build_images,
    compute_task_hash,
    get_image_name_for_sample,
    image_exists,
)
from .get_sandbox_config import (
    get_docker_config_content,
    get_k8s_config_content,
    get_sandbox_config_file,
)
from .setup_permissions import setup_file_permissions

__all__ = [
    "image_exists",
    "build_images",
    "compute_task_hash",
    "get_image_name_for_sample",
    "build_dockerfile_content",
    "get_sandbox_config_file",
    "get_docker_config_content",
    "get_k8s_config_content",
    "setup_file_permissions",
]
