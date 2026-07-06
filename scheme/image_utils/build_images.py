"""Build and cache Docker images for multi-agent evaluation tasks."""

from __future__ import annotations

import hashlib

# Build the image
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict

from docker.client import DockerClient
from docker.errors import ImageNotFound
from inspect_ai.dataset import Sample
from loguru import logger

from scheme.constants import LOG_DIR

# Import apply_patch utility
from scheme.image_utils.apply_patch import process_patch


def image_exists(docker_client, image_name: str) -> bool:
    """
    Check image existence without listing all local images, which is slow
    on hosts with many cached layers.
    """
    try:
        docker_client.images.get(image_name)
        return True
    except ImageNotFound:
        return False


def save_build_logs(image_name: str, log_entries) -> Path | None:
    """
    Persist docker build logs to disk.

    Returns the log file path when at least one entry was written, otherwise None.
    """
    if log_entries is None:
        return None

    log_dir = LOG_DIR / "docker_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().astimezone().strftime("%Y-%m-%dT%H-%M-%S%z")
    # Use the same time format as inspect to group runs by date
    if len(timestamp) >= 5:
        timestamp = timestamp[:-2] + "-" + timestamp[-2:]
    log_file = log_dir / f"{timestamp}_{image_name.replace(':', '_').replace('/', '_')}.log"

    wrote_any = False
    with open(log_file, "w") as f:
        for log_entry in log_entries:
            wrote_any = True
            if isinstance(log_entry, dict):
                if "stream" in log_entry:
                    f.write(log_entry["stream"])
                elif "status" in log_entry:
                    f.write(f"{log_entry['status']}\n")
                elif "errorDetail" in log_entry:
                    f.write(f"{log_entry['errorDetail']}\n")
                else:
                    f.write(f"{log_entry}\n")
            else:
                f.write(f"{log_entry}\n")

    if not wrote_any:
        log_file.unlink(missing_ok=True)
        return None

    return log_file


def apply_sample_patches(sample: Sample, tmpdir: Path) -> None:
    """Apply patches from sample metadata to files in the build context.

    This function applies patches to revert implementations back to NotImplementedError,
    requiring agents to re-implement them in the evaluation sandbox.

    Args:
        sample: The Sample object that may contain patches in metadata
        tmpdir: The temporary directory containing files to be patched
    """
    # Check if sample has patches defined
    if not hasattr(sample, "metadata") or not sample.metadata:
        return

    patches = sample.metadata.get("patches")
    if not patches:
        return

    # Support both list of patches and single patch string
    if isinstance(patches, str):
        patches_list = [patches]
    elif isinstance(patches, list):
        patches_list = patches
    else:
        logger.warning(f"Invalid patches type: {type(patches)}, expected str or list")
        return

    logger.info(f"Applying {len(patches_list)} patch(es) to build context...")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmpdir)

        def open_file(path: str) -> str:
            with open(path, "rt") as f:
                return f.read()

        def write_file(path: str, content: str) -> None:
            with open(path, "wt") as f:
                f.write(content)

        def remove_file(path: str) -> None:
            os.remove(path)

        # Apply each patch in sequence
        for i, patch in enumerate(patches_list):
            logger.info(f"Applying patch {i}/{len(patches_list)}...")
            result = process_patch(patch, open_file, write_file, remove_file)
            logger.info(f"Patch {i} applied successfully: {result}")

        logger.info("All patches applied successfully")
    except Exception as e:
        logger.error(f"Failed to apply patches: {e}")
        raise
    finally:
        os.chdir(old_cwd)


def compute_task_hash(sample: Sample) -> str:
    """
    Compute a hash of the task files to determine if image needs rebuilding.

    Args:
        sample: Sample containing files mapping

    Returns:
        SHA256 hash of all file contents
    """
    # Sort files by path to ensure consistent hashing
    sorted_files = sorted(sample.files.items()) if sample.files else []

    hasher = hashlib.sha256()
    for workspace_path, task_path in sorted_files:
        # Add file path to hash
        hasher.update(workspace_path.encode())

        # Read and add file content to hash
        try:
            file_path = Path(task_path)
            if file_path.exists() and file_path.is_file():
                hasher.update(file_path.read_bytes())
        except Exception as e:
            logger.warning(f"Could not read file {task_path} for hashing: {e}")

    return hasher.hexdigest()[:16]  # Use first 16 chars for shorter image names


def get_image_name_for_sample(sample: Sample, task_name: str = "task") -> str:
    """
    Generate a consistent Docker image name for a sample.

    Args:
        sample: Sample to generate image name for
        task_name: Name of the task (default: "task")

    Returns:
        Docker image name with tag
    """
    # Use sample ID if available, otherwise use task hash
    if sample.id:
        image_id = str(sample.id).replace("/", "_").replace(":", "_")
    else:
        task_hash = compute_task_hash(sample)
        image_id = f"{task_name}_{task_hash}"

    return f"multiagent.eval.{image_id}:latest"


def collect_all_requirement_files(paths_to_copy) -> str:
    # Detect and install from requirements.txt files
    requirements_install_cmd = ""
    # Find all requirements.txt files in the build context
    requirements_files = []
    for file_path in paths_to_copy:
        if file_path.endswith("requirements.txt"):
            requirements_files.append(file_path)

    if requirements_files:
        logger.info(f"Found {len(requirements_files)} requirements.txt file(s): {requirements_files}")
        installs = "\n".join(
            [f"RUN pip install --no-cache-dir -r /workspace/{req_file}" for req_file in requirements_files]
        )
        requirements_install_cmd = f"""
# Install dependencies from requirements.txt files
{installs}
"""
    return requirements_install_cmd


def build_dockerfile_content(
    sample: Sample,
    base_image: str = "python:3.11-slim",
    file_paths: list[str] | None = None,
    python_packages: list[str] | None = None,
    build_context_path: Path | None = None,
) -> str:
    """
    Create Dockerfile content for the evaluation environment.

    Args:
        sample: Sample containing task metadata
        base_image: Base Docker image to use
        file_paths: List of file paths to copy (if None, uses sample.files.keys())
        python_packages: List of Python package names to install (if None, uses default packages)
        build_context_path: Path to the build context directory (tmpdir) to check if items are files or directories

    Returns:
        Dockerfile content as string
    """
    # Use default packages if none provided
    python_packages = python_packages or []
    python_packages += [
        "pytest",
        "pytest-asyncio",
        "pytest-cov",
        "pydantic",  # Required for apply_patch tool
    ]
    python_packages = list(set(python_packages))

    # Filter out standard library modules that shouldn't be pip installed
    # These are built-in to Python and will cause pip install to fail
    # Use sys.stdlib_module_names (Python 3.10+) for accurate stdlib detection
    stdlib_modules = getattr(sys, "stdlib_module_names", set())

    # Filter packages to only include pip-installable ones
    # Extract package name from version specifiers like "package>=1.0.0"
    filtered_packages = []
    removed_packages = []
    for pkg in python_packages:
        # Extract base package name (before any version specifier)
        pkg_name = (
            pkg.split("==")[0]
            .split(">=")[0]
            .split("<=")[0]
            .split("~=")[0]
            .split(">")[0]
            .split("<")[0]
            .split("!=")[0]
            .strip()
        )
        if pkg_name not in stdlib_modules:
            filtered_packages.append(pkg)
        else:
            removed_packages.append(pkg)

    if removed_packages:
        logger.info(f"Filtered out standard library modules from pip install: {removed_packages}")

    python_packages = filtered_packages

    # Build COPY commands for sample files
    copy_commands = ""
    paths_to_copy = file_paths if file_paths is not None else (sample.files.keys() if sample.files else [])

    if paths_to_copy:
        # Group files by top-level item to minimize layers while avoiding
        # copying unwanted files (Dockerfile, apply_patch.py)
        # Extract unique top-level items from file paths

        top_level_items = set()
        for file_path in paths_to_copy:
            parts = Path(file_path).parts
            if parts:
                top_level_items.add(parts[0])

        # Generate COPY commands, checking if each item is a file or directory
        copy_commands = "\n# Copy sample files into workspace\n"
        for item in sorted(top_level_items):
            if build_context_path:
                item_path = build_context_path / item
                if item_path.is_dir():
                    copy_commands += f"COPY {item}/ /workspace/{item}/\n"
                else:
                    copy_commands += f"COPY {item} /workspace/{item}\n"
            else:
                # Fallback: assume it's a directory (shouldn't happen in practice)
                copy_commands += f"COPY {item}/ /workspace/{item}/\n"

    # Build pip install command
    pip_install_cmd = ""
    if python_packages:
        pip_packages_str = " \\\n    ".join(python_packages)
        pip_install_cmd = f"""
# Install Python packages
RUN pip install --no-cache-dir \\
    {pip_packages_str}
"""

    requirements_install_cmd = ""
    # TODO this should be controlled by the sample definition
    if sample.id.split("/")[0] not in ["inspect_eval"]:
        requirements_install_cmd = collect_all_requirement_files(paths_to_copy)

    # Detect and install local packages with pyproject.toml
    editable_install_cmd = ""
    if build_context_path and paths_to_copy:
        # Find all pyproject.toml files in the build context
        pyproject_dirs = []
        for file_path in paths_to_copy:
            if "pyproject.toml" in file_path:
                # Get the directory containing pyproject.toml
                pkg_dir = str(Path(file_path).parent)
                if pkg_dir == ".":
                    pkg_dir = Path(file_path).parts[0] if Path(file_path).parts else "."
                else:
                    # Get the top-level directory in workspace
                    pkg_dir = Path(file_path).parts[0]
                if pkg_dir not in pyproject_dirs and pkg_dir != ".":
                    pyproject_dirs.append(pkg_dir)

        if pyproject_dirs:
            installs_cmds = []
            for d in pyproject_dirs:
                env_prefix = ""
                if d == "inspect_ai":
                    # setuptools_scm needs a version when .git metadata is absent
                    env_prefix = "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_INSPECT_AI=0.0.0 "
                extras = (sample.metadata or {}).get("editable_extras", {}).get(d, "")
                extras_spec = f"[{extras}]" if extras else ""
                installs_cmds.append(f"RUN {env_prefix}pip install --no-cache-dir -e /workspace/{d}{extras_spec}")
            installs = "\n".join(installs_cmds)
            editable_install_cmd = f"""
# Install local packages in editable mode
{installs}
"""

    # Extra build steps from sample metadata
    extra_build_cmd = ""
    docker_build_steps = sample.metadata.get("docker_build_steps") if sample.metadata else None
    if docker_build_steps:
        steps = "\n".join(f"RUN {cmd}" for cmd in docker_build_steps)
        extra_build_cmd = f"""
# Extra build steps
{steps}
"""

    return f"""FROM {base_image}

# Install required packages
RUN apt-get update && apt-get install -y \\
    git \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*
{pip_install_cmd}

# Copy applypatch tool from https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/apply_patch.py
# Because GPT-5 loves it and just can't live without it
COPY apply_patch.py /usr/local/bin/applypatch
RUN chmod +x /usr/local/bin/applypatch && ln -s /usr/local/bin/applypatch /usr/local/bin/apply_patch

# Create workspace directory
WORKDIR /workspace
{copy_commands}{requirements_install_cmd}{editable_install_cmd}{extra_build_cmd}
# Set up non-root user (optional but recommended)
RUN useradd -m -s /bin/bash agent && \\
    chown -R agent:agent /workspace

USER agent

# Keep container running
CMD ["tail", "-f", "/dev/null"]
"""


def build_images(
    samples: list[Sample],
    docker_client: DockerClient | None = None,
    task_name: str = "task",
    force_rebuild: bool = False,
    base_image: str = "python:3.11-slim",
) -> Dict[str, str]:
    """
    Build Docker images for evaluation samples with caching.

    This function checks if images already exist locally and only builds
    them if they don't exist or if force_rebuild is True.

    Args:
        samples: List of samples to build images for
        docker_client: Docker client instance (will create one if None)
        task_name: Name of the task for image naming
        force_rebuild: If True, rebuild images even if they exist
        base_image: Base Docker image to use

    Returns:
        Dictionary mapping sample ID to Docker image name
    """
    logger.info(f"Building images for {len(samples)} samples...")
    if docker_client is None:
        docker_client = DockerClient.from_env()

    id_to_image_map = {}
    images_to_build = []

    # Determine which images need to be built
    for sample in samples:
        image_name = get_image_name_for_sample(sample, task_name)
        sample_id = str(sample.id) if sample.id else compute_task_hash(sample)
        id_to_image_map[sample_id] = image_name

        if force_rebuild or not image_exists(docker_client, image_name):
            images_to_build.append((sample, image_name))
            logger.info(f"Adding {image_name} for sample {sample_id} to build queue.")
        else:
            logger.info(f"Using cached image {image_name} for sample {sample_id}.")

    # Build images that don't exist
    if images_to_build:
        for sample, image_name in images_to_build:
            logger.info(f"Building image {image_name}...")

            with tempfile.TemporaryDirectory() as tmpdir:
                # Copy sample files directly from source to tmpdir (handles both text and binary)
                file_paths = []
                if hasattr(sample, "_source_files"):
                    # If we have source file paths, copy them directly
                    for workspace_path, source_path in sample._source_files.items():
                        dest_path = Path(tmpdir) / workspace_path
                        dest_path.parent.mkdir(parents=True, exist_ok=True)

                        # Copy file (preserves binary data correctly)
                        shutil.copy2(source_path, dest_path)
                        file_paths.append(workspace_path)
                        logger.debug(f"Copied file {workspace_path} to build context")
                elif sample.files:
                    # Fallback: use sample.files dict
                    for file_path, content in sample.files.items():
                        full_path = Path(tmpdir) / file_path
                        full_path.parent.mkdir(parents=True, exist_ok=True)

                        if content:
                            full_path.write_text(content)
                            file_paths.append(file_path)
                            logger.debug(f"Wrote file {file_path} ({len(content)} bytes) to build context")
                        else:
                            logger.warning(f"File {file_path} has empty content!")

                # Copy apply_patch.py into build context
                apply_patch_source = Path(__file__).parent / "apply_patch.py"
                apply_patch_dest = Path(tmpdir) / "apply_patch.py"
                if apply_patch_source.exists():
                    shutil.copy2(apply_patch_source, apply_patch_dest)

                # Apply patches from sample metadata (if any)
                apply_sample_patches(sample, Path(tmpdir))

                libs = sample.metadata.get("libs")

                # Map import names to pip install names for packages with mismatches
                import_to_pip = {
                    "sklearn": "scikit-learn",
                    "skimage": "scikit-image",
                    "cv2": "opencv-python-headless",  # headless version for servers
                    "PIL": "Pillow",
                    "yaml": "PyYAML",
                    "bs4": "beautifulsoup4",
                    "dateutil": "python-dateutil",
                }

                if libs:
                    libs = [import_to_pip.get(pkg, pkg) for pkg in libs]

                # Create Dockerfile content with the list of files we actually copied
                dockerfile_content = build_dockerfile_content(
                    sample, base_image, file_paths, python_packages=libs, build_context_path=Path(tmpdir)
                )

                dockerfile_path = os.path.join(tmpdir, "Dockerfile")
                with open(dockerfile_path, "w") as f:
                    f.write(dockerfile_content)

                try:
                    logger.info(f"Building image {image_name}...")
                    logger.debug(f"Dockerfile content:\n{dockerfile_content}")
                    _, build_logs = docker_client.images.build(
                        path=tmpdir,
                        tag=image_name,
                        rm=True,
                        nocache=force_rebuild,
                    )
                    log_path = save_build_logs(image_name, build_logs)
                    if log_path:
                        logger.info(f"Build logs saved to {log_path}")
                    logger.info(f"Successfully built image {image_name}")
                except Exception as e:
                    log_path = save_build_logs(image_name, getattr(e, "build_log", None))
                    logger.error(f"Failed to build image {image_name}: {e}")
                    if log_path:
                        logger.error(f"Build logs saved to {log_path}")
                    raise
    else:
        logger.info("All images already cached, no builds necessary")

    return id_to_image_map
