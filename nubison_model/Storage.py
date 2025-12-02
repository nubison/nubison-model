"""DVC utilities for handling large model files."""

import hashlib
import json
import logging
import subprocess
import time
from functools import wraps
from os import getenv, makedirs, path, walk
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import yaml

# Setup logging
logger = logging.getLogger(__name__)

# Type variable for generic retry function
T = TypeVar("T")

# Environment variables
ENV_VAR_DVC_ENABLED = "DVC_ENABLED"
ENV_VAR_DVC_REMOTE_URL = "DVC_REMOTE_URL"
ENV_VAR_DVC_SIZE_THRESHOLD = "DVC_SIZE_THRESHOLD"
ENV_VAR_AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
ENV_VAR_AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
ENV_VAR_AWS_ENDPOINT_URL = "AWS_ENDPOINT_URL"  # Optional: for MinIO/S3-compatible storage

# Default file size threshold: 100MB (files smaller than this stay in MLflow)
DEFAULT_SIZE_THRESHOLD = 100 * 1024 * 1024

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_RETRY_BACKOFF = 2.0  # exponential backoff multiplier

# Weight file extensions that should be handled by DVC
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

# MLflow tag key for DVC files
DVC_FILES_TAG_KEY = "dvc_files"

# DVC default remote name
DVC_DEFAULT_REMOTE_NAME = "storage"


# Exception classes
class DVCError(Exception):
    """Base exception for DVC operations."""

    pass


class DVCPushError(DVCError):
    """Exception raised when DVC push fails."""

    pass


class DVCPullError(DVCError):
    """Exception raised when DVC pull fails."""

    pass


class ChecksumMismatchError(DVCError):
    """Exception raised when file checksum doesn't match expected value."""

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
    # Check if .dvc directory exists
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


def _validate_aws_credentials(
    remote_url: str, error_class: type = DVCPushError
) -> None:
    """
    Validate AWS credentials are set when using S3 remote.

    Args:
        remote_url: DVC remote URL
        error_class: Exception class to raise on validation failure

    Raises:
        error_class: If S3 URL but AWS credentials are missing
    """
    if not remote_url.startswith("s3://"):
        return

    missing_vars = []
    if not getenv(ENV_VAR_AWS_ACCESS_KEY_ID):
        missing_vars.append(ENV_VAR_AWS_ACCESS_KEY_ID)
    if not getenv(ENV_VAR_AWS_SECRET_ACCESS_KEY):
        missing_vars.append(ENV_VAR_AWS_SECRET_ACCESS_KEY)

    if missing_vars:
        raise error_class(
            f"AWS credentials required for S3 remote. "
            f"Missing environment variables: {', '.join(missing_vars)}"
        )


def _run_dvc_command(
    args: List[str], error_msg: str, raise_on_error: bool = True
) -> subprocess.CompletedProcess:
    """
    Run a DVC command and handle errors.

    Args:
        args: Command arguments list
        error_msg: Error message prefix for DVCPushError
        raise_on_error: If True, raise DVCPushError on failure

    Returns:
        CompletedProcess result

    Raises:
        DVCPushError: If command fails and raise_on_error is True
    """
    result = subprocess.run(args, capture_output=True, text=True)
    if raise_on_error and result.returncode != 0:
        raise DVCPushError(f"{error_msg}: {result.stderr}")
    return result


def _set_remote_option(remote_name: str, key: str, value: str) -> None:
    """
    Set a DVC remote configuration option.

    Args:
        remote_name: Name of the remote
        key: Configuration key (e.g., "url", "endpointurl")
        value: Configuration value
    """
    _run_dvc_command(
        ["dvc", "remote", "modify", remote_name, key, value],
        f"Failed to set DVC remote {key}",
    )


def _ensure_dvc_remote_configured(remote_name: str = DVC_DEFAULT_REMOTE_NAME) -> None:
    """
    Ensure DVC remote is configured from environment variable.

    Args:
        remote_name: Name for the DVC remote (default: "storage")

    Raises:
        DVCPushError: If DVC_REMOTE_URL is not set or AWS credentials missing for S3
    """
    remote_url = getenv(ENV_VAR_DVC_REMOTE_URL)
    if not remote_url:
        raise DVCPushError(
            f"{ENV_VAR_DVC_REMOTE_URL} environment variable must be set."
        )

    # Validate AWS credentials for S3 remote
    _validate_aws_credentials(remote_url)

    # Try to add new remote
    logger.info(f"DVC: Configuring remote '{remote_name}' -> {remote_url}")
    result = _run_dvc_command(
        ["dvc", "remote", "add", "-d", remote_name, remote_url],
        f"Failed to add DVC remote '{remote_name}'",
        raise_on_error=False,
    )

    if result.returncode != 0:
        if "already exists" in result.stderr:
            # Update existing remote URL
            _set_remote_option(remote_name, "url", remote_url)
            logger.info(f"DVC: Remote '{remote_name}' updated")
        else:
            raise DVCPushError(
                f"Failed to add DVC remote '{remote_name}': {result.stderr}"
            )
    else:
        logger.info(f"DVC: Remote '{remote_name}' configured successfully")

    # Configure endpoint URL for MinIO/S3-compatible storage (optional)
    endpoint_url = getenv(ENV_VAR_AWS_ENDPOINT_URL)
    if endpoint_url:
        logger.info(f"DVC: Setting endpointurl -> {endpoint_url}")
        _set_remote_option(remote_name, "endpointurl", endpoint_url)


def ensure_dvc_ready() -> None:
    """
    Ensure DVC is initialized and remote is configured.

    This is called automatically by push_to_dvc().

    Raises:
        DVCPushError: If DVC initialization or remote configuration fails
    """
    if not _ensure_dvc_initialized():
        raise DVCPushError("Failed to initialize DVC repository")

    _ensure_dvc_remote_configured()


def validate_safe_path(base_dir: str, file_path: str) -> str:
    """
    Validate that file_path doesn't escape base_dir (path traversal prevention).

    Args:
        base_dir: Base directory that file_path should be contained within
        file_path: Relative file path to validate

    Returns:
        Validated absolute path

    Raises:
        PathTraversalError: If path traversal is detected
    """
    abs_base = path.realpath(base_dir)
    full_path = path.join(base_dir, file_path)
    abs_path = path.realpath(full_path)

    # Ensure the resolved path is within the base directory
    if not (abs_path.startswith(abs_base + path.sep) or abs_path == abs_base):
        raise PathTraversalError(
            f"Path traversal detected: '{file_path}' escapes base directory"
        )

    return abs_path


def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_RETRY_DELAY,
    backoff_multiplier: float = DEFAULT_RETRY_BACKOFF,
    exceptions: Tuple[type, ...] = (Exception,),
):
    """
    Decorator that retries a function with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Tuple of exception types to catch and retry

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay *= backoff_multiplier
                    else:
                        logger.error(
                            f"All {max_retries + 1} attempts failed for {func.__name__}: {e}"
                        )

            raise last_exception  # type: ignore

        return wrapper

    return decorator


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


def is_weight_file(filepath: str) -> bool:
    """Check if a file is a weight file based on its extension."""
    _, ext = path.splitext(filepath.lower())
    return ext in WEIGHT_FILE_EXTENSIONS


def should_use_dvc(filepath: str, size_threshold: Optional[int] = None) -> bool:
    """
    Check if a file should be handled by DVC.

    Args:
        filepath: Path to the file
        size_threshold: Minimum file size in bytes (default: from env or 100MB)

    Returns:
        True if file should use DVC (is weight file AND exceeds size threshold)
    """
    if not is_weight_file(filepath):
        return False

    if size_threshold is None:
        size_threshold = get_size_threshold()

    try:
        file_size = path.getsize(filepath)
        return file_size >= size_threshold
    except OSError:
        return False


def find_weight_files(
    artifact_dirs: str, size_threshold: Optional[int] = None
) -> List[str]:
    """
    Find all weight files in the given artifact directories that exceed size threshold.

    Args:
        artifact_dirs: Comma-separated list of directories to scan
        size_threshold: Minimum file size in bytes (default: from env or 100MB)

    Returns:
        List of weight file paths that should be handled by DVC
    """
    if size_threshold is None:
        size_threshold = get_size_threshold()

    weight_files = []

    for dir_path in artifact_dirs.split(","):
        dir_path = dir_path.strip()
        if not dir_path or not path.exists(dir_path):
            continue

        if path.isfile(dir_path):
            if should_use_dvc(dir_path, size_threshold):
                weight_files.append(dir_path)
        else:
            for root, _, files in walk(dir_path):
                for file in files:
                    full_path = path.join(root, file)
                    if should_use_dvc(full_path, size_threshold):
                        weight_files.append(full_path)

    return weight_files


def parse_dvc_file(dvc_file_path: str) -> Optional[str]:
    """
    Parse a .dvc file and extract the md5 hash.

    Args:
        dvc_file_path: Path to the .dvc file

    Returns:
        md5 hash string or None if not found
    """
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


def calculate_file_md5(file_path: str) -> str:
    """
    Calculate MD5 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        MD5 hash as hexadecimal string
    """
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def verify_file_checksum(file_path: str, expected_md5: str) -> bool:
    """
    Verify file integrity by comparing MD5 checksums.

    Args:
        file_path: Path to the file to verify
        expected_md5: Expected MD5 hash

    Returns:
        True if checksum matches, False otherwise
    """
    actual_md5 = calculate_file_md5(file_path)
    return actual_md5 == expected_md5


def push_to_dvc(file_paths: List[str], fail_fast: bool = True) -> Dict[str, str]:
    """
    Add files to DVC and push to remote.

    Automatically initializes DVC and configures remote if not already done.
    Uses dvc.repo.Repo Python API for add and push operations.

    Args:
        file_paths: List of file paths to add to DVC
        fail_fast: If True, raise exception on first failure. If False, continue and report all failures.

    Returns:
        Dictionary mapping file paths to their md5 hashes

    Raises:
        DVCPushError: If any DVC operation fails (when fail_fast=True)
    """
    from dvc.repo import Repo

    # Ensure DVC is initialized and remote is configured (raises DVCPushError on failure)
    ensure_dvc_ready()

    dvc_info = {}
    failed_files = []

    try:
        repo = Repo(".")
    except Exception as e:
        raise DVCPushError(f"Failed to open DVC repository: {e}")

    for file_path in file_paths:
        try:
            # Use DVC Python API for add
            repo.add(file_path)
            logger.info(f"DVC: Added {file_path}")

            # Parse the generated .dvc file
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

        except DVCPushError:
            raise
        except Exception as e:
            error_msg = f"dvc add failed for {file_path}: {e}"
            if fail_fast:
                raise DVCPushError(error_msg)
            logger.error(error_msg)
            failed_files.append(file_path)

    # Push all files to remote using DVC Python API
    if dvc_info:
        try:
            dvc_files = [f"{fp}.dvc" for fp in dvc_info.keys()]
            repo.push(dvc_files)
            logger.info("DVC: Push completed successfully")
        except Exception as e:
            error_msg = f"dvc push failed: {e}"
            if fail_fast:
                raise DVCPushError(error_msg)
            logger.error(error_msg)

    if failed_files:
        raise DVCPushError(f"DVC push failed for files: {failed_files}")

    return dvc_info


def get_dvc_remote_path(remote_url: str, md5: str) -> str:
    """
    Get the remote storage path for a file based on its md5 hash.

    DVC stores files as: {remote}/files/md5/{hash[:2]}/{hash[2:]}

    Args:
        remote_url: Base URL of DVC remote storage
        md5: md5 hash of the file

    Returns:
        Full remote path to the file
    """
    return f"{remote_url.rstrip('/')}/files/md5/{md5[:2]}/{md5[2:]}"


def pull_from_dvc(
    dvc_files: Dict[str, str],
    local_base_dir: str = ".",
    verify_checksum: bool = True,
    show_progress: bool = True,
) -> None:
    """
    Download files from DVC remote storage using md5 hashes.

    Args:
        dvc_files: Dictionary mapping file paths to md5 hashes
        local_base_dir: Base directory for local file paths
        verify_checksum: If True, verify MD5 checksum after download
        show_progress: If True, show download progress

    Raises:
        DVCPullError: If download fails or required environment variables are missing
        ChecksumMismatchError: If checksum verification fails
    """
    remote_url = getenv(ENV_VAR_DVC_REMOTE_URL)
    if not remote_url:
        raise DVCPullError(
            f"{ENV_VAR_DVC_REMOTE_URL} environment variable must be set. "
        )

    # Validate AWS credentials for S3 remote
    _validate_aws_credentials(remote_url, DVCPullError)

    total_files = len(dvc_files)

    for idx, (file_path, md5) in enumerate(dvc_files.items(), 1):
        # Validate path to prevent path traversal attacks
        local_path = validate_safe_path(local_base_dir, file_path)
        remote_path = get_dvc_remote_path(remote_url, md5)

        # Create parent directory if needed
        makedirs(path.dirname(local_path) or ".", exist_ok=True)

        if show_progress:
            logger.info(f"DVC: Downloading [{idx}/{total_files}] {file_path}...")

        # Download based on remote type
        _download_from_remote(remote_path, local_path, show_progress=show_progress)

        # Verify checksum
        if verify_checksum:
            if not verify_file_checksum(local_path, md5):
                raise ChecksumMismatchError(
                    f"Checksum mismatch for {file_path}: expected {md5}"
                )
            logger.debug(f"DVC: Checksum verified for {file_path}")


def _get_downloader(remote_path: str):
    """
    Get appropriate downloader function based on URL scheme.

    Args:
        remote_path: Remote URL of the file

    Returns:
        Tuple of (downloader_function, supports_progress)
    """
    schemes = [
        ("s3://", _download_from_s3, True),
        ("http://", _download_from_http, True),
        ("https://", _download_from_http, True),
    ]
    for scheme, downloader, supports_progress in schemes:
        if remote_path.startswith(scheme):
            return downloader, supports_progress
    return _download_with_dvc, False


def _download_from_remote(
    remote_path: str, local_path: str, show_progress: bool = True
) -> None:
    """
    Download a file from remote storage.

    Supports S3 and HTTP(S) URLs with fallback to DVC.

    Args:
        remote_path: Remote URL of the file
        local_path: Local path to save the file
        show_progress: If True, show download progress
    """
    downloader, supports_progress = _get_downloader(remote_path)
    if supports_progress:
        downloader(remote_path, local_path, show_progress)
    else:
        downloader(remote_path, local_path)


@retry_with_backoff(
    max_retries=DEFAULT_MAX_RETRIES,
    exceptions=(Exception,),
)
def _download_from_s3(
    remote_path: str, local_path: str, show_progress: bool = True
) -> None:
    """Download file from S3 with retry logic and multipart download.

    Supports custom endpoint URL for MinIO/S3-compatible storage via AWS_ENDPOINT_URL env var.
    Uses multipart download for large files to improve download speed.
    """
    # Get optional endpoint URL for MinIO/S3-compatible storage
    endpoint_url = getenv(ENV_VAR_AWS_ENDPOINT_URL)

    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.exceptions import ClientError

        # Parse S3 URL: s3://bucket/key
        path_parts = remote_path[5:].split("/", 1)
        bucket = path_parts[0]
        key = path_parts[1] if len(path_parts) > 1 else ""

        # Create S3 client with optional endpoint URL
        if endpoint_url:
            s3 = boto3.client("s3", endpoint_url=endpoint_url)
        else:
            s3 = boto3.client("s3")

        file_size = 0
        if show_progress:
            # Get file size for progress
            try:
                response = s3.head_object(Bucket=bucket, Key=key)
                file_size = response.get("ContentLength", 0)
                logger.info(f"  Size: {file_size / (1024*1024):.1f} MB")
            except ClientError:
                pass

        # Configure multipart download for faster transfers
        # multipart_threshold: Use multipart for files larger than 8MB
        # max_concurrency: Number of parallel download threads
        # multipart_chunksize: Size of each chunk
        transfer_config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,  # 8MB
            max_concurrency=10,
            multipart_chunksize=8 * 1024 * 1024,  # 8MB chunks
        )

        s3.download_file(bucket, key, local_path, Config=transfer_config)

    except ImportError:
        # Fallback to AWS CLI
        logger.debug("boto3 not available, using AWS CLI")
        aws_command = ["aws", "s3", "cp", remote_path, local_path]
        if endpoint_url:
            aws_command.extend(["--endpoint-url", endpoint_url])
        result = subprocess.run(
            aws_command,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise DVCPullError(f"AWS CLI download failed: {result.stderr}")


@retry_with_backoff(
    max_retries=DEFAULT_MAX_RETRIES,
    exceptions=(Exception,),
)
def _download_from_http(
    remote_path: str, local_path: str, show_progress: bool = True
) -> None:
    """Download file from HTTP(S) URL with retry logic and progress."""
    import urllib.request

    # Get file size if possible
    if show_progress:
        try:
            with urllib.request.urlopen(remote_path) as response:
                file_size = int(response.headers.get("Content-Length", 0))
                if file_size:
                    logger.info(f"  Size: {file_size / (1024*1024):.1f} MB")
        except Exception:
            pass

    urllib.request.urlretrieve(remote_path, local_path)


@retry_with_backoff(
    max_retries=DEFAULT_MAX_RETRIES,
    exceptions=(subprocess.CalledProcessError,),
)
def _download_with_dvc(remote_path: str, local_path: str) -> None:
    """Fallback: use DVC to download with retry logic."""
    result = subprocess.run(
        ["dvc", "get-url", remote_path, local_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise DVCPullError(f"DVC get-url failed: {result.stderr}")


def serialize_dvc_info(dvc_info: Dict[str, str]) -> str:
    """Serialize DVC info to JSON string for MLflow tag."""
    return json.dumps(dvc_info)


def deserialize_dvc_info(dvc_info_json: str) -> Dict[str, str]:
    """Deserialize DVC info from MLflow tag JSON string."""
    return json.loads(dvc_info_json)


def get_dvc_cache_key(model_uri: str, dvc_info: Dict[str, str]) -> str:
    """
    Generate a unique cache key for DVC restoration status.

    This key changes when either the model URI or DVC file info changes,
    ensuring that different model versions get their own DVC files restored.

    Args:
        model_uri: MLflow model URI
        dvc_info: Dictionary of DVC file info

    Returns:
        MD5 hash string to use as cache key
    """
    content = f"{model_uri}:{json.dumps(dvc_info, sort_keys=True)}"
    return hashlib.md5(content.encode()).hexdigest()
