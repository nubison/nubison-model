"""DVC utilities for handling large model files."""

import hashlib
import json
import logging
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from os import cpu_count, getenv, makedirs, path, walk
from typing import Callable, Dict, List, Optional, Tuple, TypeVar

import yaml
from dvc.repo import Repo

logger = logging.getLogger(__name__)

T = TypeVar("T")

ENV_VAR_DVC_ENABLED = "DVC_ENABLED"
ENV_VAR_DVC_REMOTE_URL = "DVC_REMOTE_URL"
ENV_VAR_DVC_SIZE_THRESHOLD = "DVC_SIZE_THRESHOLD"
ENV_VAR_DVC_JOBS = "DVC_JOBS"
ENV_VAR_DVC_FILE_EXTENSIONS = "DVC_FILE_EXTENSIONS"
ENV_VAR_AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
ENV_VAR_AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
ENV_VAR_AWS_ENDPOINT_URL = "AWS_ENDPOINT_URL"

DEFAULT_SIZE_THRESHOLD = 100 * 1024 * 1024
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 1.0
DEFAULT_RETRY_BACKOFF = 2.0

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
        raise DVCPushError(f"{ENV_VAR_DVC_REMOTE_URL} environment variable must be set.")

    _validate_aws_credentials(remote_url)

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
            raise DVCPushError(f"Failed to add DVC remote '{remote_name}': {result.stderr}")
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


def retry_with_backoff(
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_RETRY_DELAY,
    backoff_multiplier: float = DEFAULT_RETRY_BACKOFF,
    exceptions: Tuple[type, ...] = (Exception,),
):
    """Decorator that retries a function with exponential backoff."""

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


def calculate_file_md5(file_path: str) -> str:
    """Calculate MD5 hash of a file."""
    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)
    return md5_hash.hexdigest()


def verify_file_checksum(file_path: str, expected_md5: str) -> bool:
    """Verify file integrity by comparing MD5 checksums."""
    actual_md5 = calculate_file_md5(file_path)
    return actual_md5 == expected_md5


def push_to_dvc(file_paths: List[str], fail_fast: bool = True) -> Dict[str, str]:
    """Add files to DVC and push to remote. Returns dict mapping paths to md5 hashes."""
    ensure_dvc_ready()

    try:
        repo = Repo(".")
    except Exception as e:
        raise DVCPushError(f"Failed to open DVC repository: {e}")

    # Batch add all files at once for better performance
    try:
        repo.add(file_paths)
        logger.info(f"DVC: Added {len(file_paths)} file(s)")
    except Exception as e:
        raise DVCPushError(f"dvc add failed: {e}")

    # Parse .dvc files to extract md5 hashes
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

    # Push all files to remote
    if dvc_info:
        try:
            dvc_files = [f"{fp}.dvc" for fp in dvc_info.keys()]
            repo.push(dvc_files)
            logger.info("DVC: Push completed successfully")
        except Exception as e:
            raise DVCPushError(f"dvc push failed: {e}")

    return dvc_info


def get_dvc_remote_path(remote_url: str, md5: str) -> str:
    """Get DVC remote path: {remote}/files/md5/{hash[:2]}/{hash[2:]}"""
    return f"{remote_url.rstrip('/')}/files/md5/{md5[:2]}/{md5[2:]}"


def pull_from_dvc(
    dvc_files: Dict[str, str],
    local_base_dir: str = ".",
    verify_checksum: bool = True,
    show_progress: bool = True,
) -> None:
    """Download files from DVC remote storage using md5 hashes with parallel downloads."""
    remote_url = getenv(ENV_VAR_DVC_REMOTE_URL)
    if not remote_url:
        raise DVCPullError(f"{ENV_VAR_DVC_REMOTE_URL} environment variable must be set.")

    _validate_aws_credentials(remote_url, DVCPullError)

    total_files = len(dvc_files)
    workers = int(getenv(ENV_VAR_DVC_JOBS) or max(1, (cpu_count() or 4)))

    def download_single(item: Tuple[str, str]) -> str:
        file_path, md5 = item
        local_path = validate_safe_path(local_base_dir, file_path)
        remote_path = get_dvc_remote_path(remote_url, md5)
        makedirs(path.dirname(local_path) or ".", exist_ok=True)
        _download_from_remote(remote_path, local_path, show_progress=False)
        if verify_checksum and not verify_file_checksum(local_path, md5):
            raise ChecksumMismatchError(f"Checksum mismatch for {file_path}: expected {md5}")
        return file_path

    if show_progress:
        logger.info(f"DVC: Downloading {total_files} file(s) with {workers} workers...")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_single, item): item[0] for item in dvc_files.items()}
        for future in as_completed(futures):
            file_path = futures[future]
            future.result()  # Raises exception if download failed
            if show_progress:
                logger.info(f"DVC: Downloaded {file_path}")


def _get_downloader(remote_path: str):
    """Get appropriate downloader function based on URL scheme."""
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
    """Download a file from remote storage (S3, HTTP, or DVC fallback)."""
    downloader, supports_progress = _get_downloader(remote_path)
    if supports_progress:
        downloader(remote_path, local_path, show_progress)
    else:
        downloader(remote_path, local_path)


def _parse_s3_url(url: str) -> Tuple[str, str]:
    """Parse S3 URL into bucket and key."""
    parts = url[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def _download_s3_with_cli(remote_path: str, local_path: str, endpoint_url: Optional[str]) -> None:
    """Download from S3 using AWS CLI."""
    cmd = ["aws", "s3", "cp", remote_path, local_path]
    if endpoint_url:
        cmd.extend(["--endpoint-url", endpoint_url])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DVCPullError(f"AWS CLI download failed: {result.stderr}")


@retry_with_backoff(max_retries=DEFAULT_MAX_RETRIES, exceptions=(Exception,))
def _download_from_s3(
    remote_path: str, local_path: str, show_progress: bool = True
) -> None:
    """Download file from S3 with multipart download."""
    endpoint_url = getenv(ENV_VAR_AWS_ENDPOINT_URL)

    try:
        import boto3
        from boto3.s3.transfer import TransferConfig

        bucket, key = _parse_s3_url(remote_path)
        s3 = boto3.client("s3", endpoint_url=endpoint_url) if endpoint_url else boto3.client("s3")

        if show_progress:
            try:
                size = s3.head_object(Bucket=bucket, Key=key).get("ContentLength", 0)
                logger.info(f"  Size: {size / (1024*1024):.1f} MB")
            except Exception:
                pass

        config = TransferConfig(
            multipart_threshold=8 * 1024 * 1024,
            max_concurrency=10,
            multipart_chunksize=8 * 1024 * 1024,
        )
        s3.download_file(bucket, key, local_path, Config=config)

    except ImportError:
        logger.debug("boto3 not available, using AWS CLI")
        _download_s3_with_cli(remote_path, local_path, endpoint_url)


@retry_with_backoff(max_retries=DEFAULT_MAX_RETRIES, exceptions=(Exception,))
def _download_from_http(
    remote_path: str, local_path: str, show_progress: bool = True
) -> None:
    """Download file from HTTP(S) URL with retry logic."""
    if show_progress:
        try:
            with urllib.request.urlopen(remote_path) as response:
                file_size = int(response.headers.get("Content-Length", 0))
                if file_size:
                    logger.info(f"  Size: {file_size / (1024*1024):.1f} MB")
        except Exception:
            pass

    urllib.request.urlretrieve(remote_path, local_path)


@retry_with_backoff(max_retries=DEFAULT_MAX_RETRIES, exceptions=(subprocess.CalledProcessError,))
def _download_with_dvc(remote_path: str, local_path: str) -> None:
    """Fallback: use DVC to download."""
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
    """Generate a unique cache key for DVC restoration status."""
    content = f"{model_uri}:{json.dumps(dvc_info, sort_keys=True)}"
    return hashlib.md5(content.encode()).hexdigest()
