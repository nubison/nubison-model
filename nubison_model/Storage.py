"""DVC utilities for handling large model files."""

import hashlib
import json
import logging
import subprocess
from os import cpu_count, getenv, path, walk
from typing import Dict, List, Optional

import yaml
from dvc.repo import Repo

logger = logging.getLogger(__name__)

ENV_VAR_DVC_ENABLED = "DVC_ENABLED"
ENV_VAR_DVC_REMOTE_URL = "DVC_REMOTE_URL"
ENV_VAR_DVC_SIZE_THRESHOLD = "DVC_SIZE_THRESHOLD"
ENV_VAR_DVC_JOBS = "DVC_JOBS"
ENV_VAR_DVC_FILE_EXTENSIONS = "DVC_FILE_EXTENSIONS"
ENV_VAR_AWS_ENDPOINT_URL = "AWS_ENDPOINT_URL"

DEFAULT_SIZE_THRESHOLD = 100 * 1024 * 1024

WEIGHT_FILE_EXTENSIONS = {
    ".pt",
    ".pth",
    ".bin",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".onnx",
    ".safetensors",
    ".ckpt",
    ".pb",
    ".weights",
    ".model",
}

DVC_FILES_TAG_KEY = "dvc_files"
DVC_DEFAULT_REMOTE_NAME = "storage"


class DVCError(Exception):
    """Base exception for DVC operations."""

    pass


class DVCPushError(DVCError):
    """Exception raised when DVC push fails."""

    pass


class DVCPullError(DVCError):
    """Exception raised when DVC pull fails."""

    pass


class PathTraversalError(DVCError):
    """Exception raised when path traversal attack is detected."""

    pass


def _ensure_dvc_initialized() -> bool:
    """
    Ensure DVC is initialized in the current directory.

    Uses --no-scm option to initialize without Git integration.

    Returns:
        True if DVC is already initialized or initialization succeeded.
        False if initialization failed.
    """
    if path.isdir(".dvc"):
        logger.debug("DVC already initialized")
        return True

    logger.info("DVC: Initializing repository (--no-scm)")
    result = subprocess.run(
        ["dvc", "init", "--no-scm"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"DVC init failed: {result.stderr}")
        return False

    logger.info("DVC: Repository initialized successfully")
    return True


def _run_dvc_command(
    args: List[str], error_msg: str, raise_on_error: bool = True
) -> subprocess.CompletedProcess:
    """Run a DVC command and handle errors."""
    result = subprocess.run(args, capture_output=True, text=True)
    if raise_on_error and result.returncode != 0:
        raise DVCPushError(f"{error_msg}: {result.stderr}")
    return result


def _set_remote_option(remote_name: str, key: str, value: str) -> None:
    """Set a DVC remote configuration option."""
    _run_dvc_command(
        ["dvc", "remote", "modify", remote_name, key, value],
        f"Failed to set DVC remote {key}",
    )


def _ensure_dvc_remote_configured(remote_name: str = DVC_DEFAULT_REMOTE_NAME) -> None:
    """Ensure DVC remote is configured from environment variable."""
    remote_url = getenv(ENV_VAR_DVC_REMOTE_URL)
    if not remote_url:
        raise DVCPushError(
            f"{ENV_VAR_DVC_REMOTE_URL} environment variable must be set."
        )

    logger.info(f"DVC: Configuring remote '{remote_name}' -> {remote_url}")
    result = _run_dvc_command(
        ["dvc", "remote", "add", "-d", remote_name, remote_url],
        f"Failed to add DVC remote '{remote_name}'",
        raise_on_error=False,
    )

    if result.returncode != 0:
        if "already exists" in result.stderr:
            _set_remote_option(remote_name, "url", remote_url)
            logger.info(f"DVC: Remote '{remote_name}' updated")
        else:
            raise DVCPushError(
                f"Failed to add DVC remote '{remote_name}': {result.stderr}"
            )
    else:
        logger.info(f"DVC: Remote '{remote_name}' configured successfully")

    endpoint_url = getenv(ENV_VAR_AWS_ENDPOINT_URL)
    if endpoint_url:
        logger.info(f"DVC: Setting endpointurl -> {endpoint_url}")
        _set_remote_option(remote_name, "endpointurl", endpoint_url)

    jobs = getenv(ENV_VAR_DVC_JOBS)
    if jobs:
        logger.info(f"DVC: Setting jobs -> {jobs}")
        _set_remote_option(remote_name, "jobs", jobs)


def ensure_dvc_ready() -> None:
    """Ensure DVC is initialized and remote is configured."""
    if not _ensure_dvc_initialized():
        raise DVCPushError("Failed to initialize DVC repository")

    _ensure_dvc_remote_configured()


def validate_safe_path(base_dir: str, file_path: str) -> str:
    """Validate that file_path doesn't escape base_dir (path traversal prevention)."""
    abs_base = path.realpath(base_dir)
    full_path = path.join(base_dir, file_path)
    abs_path = path.realpath(full_path)

    if not (abs_path.startswith(abs_base + path.sep) or abs_path == abs_base):
        raise PathTraversalError(
            f"Path traversal detected: '{file_path}' escapes base directory"
        )

    return abs_path


def is_dvc_enabled() -> bool:
    """Check if DVC is enabled via environment variable."""
    return getenv(ENV_VAR_DVC_ENABLED, "").lower() in ("true", "1", "yes")


def get_size_threshold() -> int:
    """Get the file size threshold for DVC (in bytes)."""
    threshold_str = getenv(ENV_VAR_DVC_SIZE_THRESHOLD, "")
    if threshold_str:
        try:
            return int(threshold_str)
        except ValueError:
            logger.warning(
                f"Invalid DVC_SIZE_THRESHOLD value: {threshold_str}, using default"
            )
    return DEFAULT_SIZE_THRESHOLD


def get_weight_extensions() -> set:
    """Get weight file extensions (default + custom from env)."""
    extensions = set(WEIGHT_FILE_EXTENSIONS)
    custom = getenv(ENV_VAR_DVC_FILE_EXTENSIONS, "")
    if custom:
        for ext in custom.split(","):
            ext = ext.strip().lower()
            if ext:
                if not ext.startswith("."):
                    ext = "." + ext
                extensions.add(ext)
    return extensions


def is_weight_file(filepath: str) -> bool:
    """Check if a file is a weight file based on its extension."""
    _, ext = path.splitext(filepath.lower())
    return ext in get_weight_extensions()


def should_use_dvc(filepath: str, size_threshold: Optional[int] = None) -> bool:
    """Check if a file should be handled by DVC (weight file AND exceeds size threshold)."""
    if not is_weight_file(filepath):
        return False

    if size_threshold is None:
        size_threshold = get_size_threshold()

    try:
        file_size = path.getsize(filepath)
        return file_size >= size_threshold
    except OSError:
        return False


def iter_artifact_files(artifact_dirs: str):
    """Iterate over all files in artifact directories, yielding (full_path, base_dir)."""
    for entry in artifact_dirs.split(","):
        entry = entry.strip()
        if not entry or not path.exists(entry):
            continue
        if path.isfile(entry):
            yield entry, path.dirname(entry) or "."
        elif path.isdir(entry):
            for root, _, files in walk(entry):
                for f in files:
                    yield path.join(root, f), entry


def find_weight_files(
    artifact_dirs: str, size_threshold: Optional[int] = None
) -> List[str]:
    """Find weight files exceeding size threshold in artifact directories."""
    if size_threshold is None:
        size_threshold = get_size_threshold()

    return [
        full_path
        for full_path, _ in iter_artifact_files(artifact_dirs)
        if should_use_dvc(full_path, size_threshold)
    ]


def parse_dvc_file(dvc_file_path: str) -> Optional[str]:
    """Parse a .dvc file and extract the md5 hash."""
    if not path.exists(dvc_file_path):
        return None

    try:
        with open(dvc_file_path, "r") as f:
            dvc_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse DVC file {dvc_file_path}: {e}")
        return None

    if not dvc_data or "outs" not in dvc_data:
        return None

    outs = dvc_data["outs"]
    if not outs or len(outs) == 0:
        return None

    return outs[0].get("md5")


def push_to_dvc(file_paths: List[str], fail_fast: bool = True) -> Dict[str, str]:
    """Add files to DVC and push to remote. Returns dict mapping paths to md5 hashes."""
    ensure_dvc_ready()

    try:
        repo = Repo(".")
    except Exception as e:
        raise DVCPushError(f"Failed to open DVC repository: {e}")

    try:
        repo.add(file_paths)
        logger.info(f"DVC: Added {len(file_paths)} file(s)")
    except Exception as e:
        raise DVCPushError(f"dvc add failed: {e}")

    dvc_info = {}
    failed_files = []

    for file_path in file_paths:
        dvc_file = f"{file_path}.dvc"
        md5 = parse_dvc_file(dvc_file)

        if md5:
            dvc_info[file_path] = md5
            logger.info(f"DVC: {file_path} (md5: {md5[:8]}...)")
        else:
            error_msg = f"Failed to parse .dvc file for {file_path}"
            if fail_fast:
                raise DVCPushError(error_msg)
            logger.error(error_msg)
            failed_files.append(file_path)

    if failed_files:
        raise DVCPushError(f"DVC add failed for files: {failed_files}")

    if dvc_info:
        try:
            dvc_files = [f"{fp}.dvc" for fp in dvc_info.keys()]
            repo.push(dvc_files)
            logger.info("DVC: Push completed successfully")
        except Exception as e:
            raise DVCPushError(f"dvc push failed: {e}")

    return dvc_info


def pull_from_dvc(
    dvc_files: Dict[str, str],
    local_base_dir: str = ".",
    show_progress: bool = True,
) -> None:
    """Download files from DVC remote using dvc pull command.

    Expects .dvc files to already exist in local_base_dir (from MLflow artifacts).
    """
    if not getenv(ENV_VAR_DVC_REMOTE_URL):
        raise DVCPullError(
            f"{ENV_VAR_DVC_REMOTE_URL} environment variable must be set."
        )

    ensure_dvc_ready()

    jobs = getenv(ENV_VAR_DVC_JOBS) or str(cpu_count() or 4)
    dvc_file_paths = []

    for file_path in dvc_files.keys():
        local_path = validate_safe_path(local_base_dir, file_path)
        dvc_file_path = f"{local_path}.dvc"

        if not path.exists(dvc_file_path):
            raise DVCPullError(
                f"DVC file not found: {dvc_file_path}. "
                "Ensure model was registered with DVC-tracked files."
            )

        dvc_file_paths.append(dvc_file_path)

    if show_progress:
        logger.info(
            f"DVC: Pulling {len(dvc_file_paths)} file(s) with {jobs} workers..."
        )

    result = subprocess.run(
        ["dvc", "pull", "-j", jobs] + dvc_file_paths,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise DVCPullError(f"dvc pull failed: {result.stderr}")

    if show_progress:
        logger.info("DVC: Pull completed successfully")


def serialize_dvc_info(dvc_info: Dict[str, str]) -> str:
    """Serialize DVC info to JSON string for MLflow tag."""
    return json.dumps(dvc_info)


def deserialize_dvc_info(dvc_info_json: str) -> Dict[str, str]:
    """Deserialize DVC info from MLflow tag JSON string."""
    return json.loads(dvc_info_json)


def get_dvc_cache_key(model_uri: str, dvc_info: Dict[str, str]) -> str:
    """Generate a unique cache key for DVC restoration status."""
    content = f"{model_uri}:{json.dumps(dvc_info, sort_keys=True)}"
    return hashlib.md5(content.encode()).hexdigest()
