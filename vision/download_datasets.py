"""Download the public playing-card datasets / assets and convert them to the canonical layout.

Two stages, both idempotent:

1. **fetch**  -> ``Paths.raw/<key>/`` (untouched download)

   ==============  ===================================================================
   ``source``      how it is fetched
   ==============  ===================================================================
   huggingface     ``huggingface_hub.snapshot_download(repo_type="dataset")``; archives
                   inside the snapshot are extracted next to themselves
   roboflow        ``roboflow.Roboflow(api_key).workspace(ws).project(p).version(v)
                   .download("yolov8")`` (``v`` = ``spec.version`` or the newest one);
                   key from ``--roboflow-key`` or ``$ROBOFLOW_API_KEY``
   kaggle          ``kaggle`` python API (``dataset_download_files(unzip=True)``) with a
                   fallback to the ``kaggle`` CLI; credentials via ``$KAGGLE_USERNAME`` +
                   ``$KAGGLE_KEY``, ``$KAGGLE_API_TOKEN`` or ``~/.kaggle/kaggle.json``
   url             plain HTTPS download with retries (DTD tar.gz), extracted; every
                   ``images/**/*.jpg`` is linked into ``Paths.assets_backgrounds``
   github_raw      the 52 public-domain card PNGs + jokers + back from
                   raw.githubusercontent.com into ``Paths.assets_cards/<CANONICAL>.png``
                   (``10C.png``, ``AS.png``, ``JOKER_RED.png``, ``JOKER_BLACK.png``, ``BACK.png``)
   ==============  ===================================================================

2. **convert** -> ``Paths.datasets/<key>/`` in the canonical layout of ``vision/labels.py``
   (``images/{train,val[,test]}``, ``labels/...``, ``data.yaml``, ``manifest.json``,
   optional ``views/obb``).  Class names are remapped onto the canonical vocabulary of
   ``vision/cards.py`` with ``cards.map_dataset_names``; boxes of unknown classes are
   dropped and listed in the manifest.  JackFurby's pickle annotations (4 exact card
   corners per card) become whole-card AABB labels plus an OBB view.

Missing credentials and blocked / unreachable hosts never abort the run: the dataset is
reported in the final summary table (with the env var or URL to fix) and the remaining
datasets are still processed.

Usage
-----
::

    python -m vision.download_datasets --dry-run --datasets all      # plan only, no network
    python -m vision.download_datasets                                # group "free"
    python -m vision.download_datasets --datasets augstartups,pcc3 --roboflow-key $KEY --preview 8
    python -m vision.download_datasets --datasets jackfurby --convert-only --force
    python -m vision.download_datasets --datasets cardart,dtd --root /data/rummy_vision

Exit codes: ``0`` everything requested succeeded (datasets skipped for missing
credentials are tolerated as long as something else succeeded), ``2`` when any requested
dataset failed (network error, blocked host, conversion error) or when *every* requested
dataset was skipped for missing credentials, ``1`` for a usage error.

The plan (``--dry-run``) and the final summary table go to stdout; everything else is
logged (``--verbose`` for DEBUG).

Public helpers used by the other vision scripts / tests
--------------------------------------------------------
``fetch_card_art(dest_dir, force=False) -> list[Path]``
``convert_yolo(spec, raw_dir, out_dir, ...)``, ``convert_jackfurby(spec, raw_dir, out_dir, ...)``
``detect_box_semantics(root)``, ``write_previews(root, n)``, ``resolve_dataset_keys(text)``
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

if __package__ in (None, ""):  # allow `python vision/download_datasets.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vision import cards, labels  # noqa: E402
from vision.config import DATASET_GROUPS, DATASETS, REPO_ROOT, DatasetSpec, Paths  # noqa: E402

log = logging.getLogger("vision.download_datasets")

# --------------------------------------------------------------------------- constants
CARD_ART_BASE_URL = "https://raw.githubusercontent.com/hayeah/playing-cards-assets/master/png/"
#: extra card-art files: output stem -> upstream file name
CARD_ART_EXTRAS: Dict[str, str] = {"JOKER_RED": "red_joker.png", "JOKER_BLACK": "black_joker.png", "BACK": "back.png"}
JACKFURBY_CLASSES_FILE = REPO_ROOT / "vision" / "configs" / "jackfurby_card_classes.txt"

#: hosts contacted per source (used for the plan and for blocked-host diagnostics)
SOURCE_HOSTS: Dict[str, str] = {
    "huggingface": "huggingface.co",
    "roboflow": "api.roboflow.com",
    "kaggle": "www.kaggle.com",
    "github_raw": "raw.githubusercontent.com",
}
SPLIT_ALIASES: Dict[str, Tuple[str, ...]] = {
    "train": ("train", "training"),
    "val": ("val", "valid", "validation"),
    "test": ("test", "testing"),
}
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
IN_PROGRESS_SUFFIXES = (".part", ".incomplete")   # partial downloads: ours (``download_file``) / huggingface_hub
FETCHED_MARKER = ".fetched_ok"                    # written into raw/<key>/ once a fetch completed
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

EXIT_OK, EXIT_USAGE, EXIT_FAILED = 0, 1, 2

_BLOCKED_HINTS = (
    "403", "407", "forbidden", "tunnel connection failed", "proxyerror", "proxy error", "connect tunnel",
    "name or service not known", "nodename nor servname", "temporary failure in name resolution",
    "connection refused", "network is unreachable", "timed out", "max retries exceeded", "connection reset",
    "remote end closed", "eof occurred", "unreachable",
)


class DownloadError(RuntimeError):
    """A network fetch failed after retries."""


class CredentialsError(RuntimeError):
    """A source needs credentials that are not configured (message names the env var)."""


class ConversionError(RuntimeError):
    """The raw download could not be converted to the canonical layout."""


# --------------------------------------------------------------------------- options / results
@dataclass
class Options:
    """Run-wide options shared by the fetchers and converters (mirrors the CLI flags)."""
    paths: Paths = field(default_factory=Paths)
    force: bool = False
    copy: bool = False
    roboflow_key: Optional[str] = None
    retries: int = 3
    timeout: float = 60.0
    preview: int = 0
    convert_only: bool = False
    skip_convert: bool = False
    val_ratio: float = 0.1        # carved out of train when a YOLO export has no val split


@dataclass
class DatasetResult:
    """Outcome of one dataset for the summary table / exit code."""
    key: str
    source: str
    fetch: str = "pending"        # ok | cached | skipped | no-credentials | failed
    convert: str = "pending"      # ok | up-to-date | skipped | n/a | failed
    message: str = ""
    raw_dir: Optional[Path] = None
    out_dir: Optional[Path] = None
    images: int = 0
    boxes: int = 0
    semantics: str = ""           # "<declared>/<detected>" for detection datasets

    @property
    def failed(self) -> bool:
        return self.fetch == "failed" or self.convert == "failed"

    @property
    def succeeded(self) -> bool:
        return not self.failed and self.fetch != "no-credentials"


@dataclass
class PlanEntry:
    """What ``--dry-run`` prints for one dataset."""
    spec: DatasetSpec
    raw_dir: Path
    target: Path
    host: str
    env_vars: Tuple[str, ...]
    creds_ok: bool
    creds_note: str
    raw_present: bool
    converted: bool


# --------------------------------------------------------------------------- dataset selection
def resolve_dataset_keys(text: str) -> List[str]:
    """Expand ``all|free|detection|assets|corner|card|key1,key2`` into ordered unique keys.

    Raises ``ValueError`` naming the unknown token(s).
    """
    keys: List[str] = []
    unknown: List[str] = []
    for tok in re.split(r"[,\s]+", (text or "").strip()):
        if not tok:
            continue
        if tok in DATASET_GROUPS:
            keys.extend(DATASET_GROUPS[tok])
        elif tok in DATASETS:
            keys.append(tok)
        else:
            unknown.append(tok)
    if unknown:
        raise ValueError(
            f"unknown dataset(s) {unknown}; choose from groups {sorted(DATASET_GROUPS)} or keys {sorted(DATASETS)}"
        )
    seen: set = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


# --------------------------------------------------------------------------- credentials
def roboflow_api_key(cli_key: Optional[str] = None) -> Optional[str]:
    return (cli_key or os.environ.get("ROBOFLOW_API_KEY") or "").strip() or None


def kaggle_config_dir() -> Path:
    return Path(os.environ.get("KAGGLE_CONFIG_DIR") or Path.home() / ".kaggle")


def kaggle_credentials_hint() -> str:
    return ("set KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN), or put kaggle.json in "
            f"{kaggle_config_dir()} (https://www.kaggle.com/settings -> API)")


def kaggle_credentials() -> Tuple[bool, str]:
    """``(present, how)``: env pair, ``KAGGLE_API_TOKEN`` or the ``~/.kaggle`` files."""
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True, "KAGGLE_USERNAME/KAGGLE_KEY env"
    if os.environ.get("KAGGLE_API_TOKEN"):
        return True, "KAGGLE_API_TOKEN env"
    kdir = kaggle_config_dir()
    for name in ("kaggle.json", "access_token"):
        if (kdir / name).is_file():
            return True, str(kdir / name)
    return False, kaggle_credentials_hint()


def credentials_status(spec: DatasetSpec, roboflow_key: Optional[str] = None) -> Tuple[bool, str]:
    """Whether the credentials a source needs are configured, plus a human note."""
    if spec.source == "roboflow":
        if roboflow_api_key(roboflow_key):
            return True, "ROBOFLOW_API_KEY present"
        return False, "set ROBOFLOW_API_KEY or pass --roboflow-key (https://app.roboflow.com -> Settings -> API)"
    if spec.source == "kaggle":
        return kaggle_credentials()
    return True, "no credentials needed"


def source_host(spec: DatasetSpec) -> str:
    if spec.source == "url":
        return urlparse(spec.ref).netloc
    return SOURCE_HOSTS.get(spec.source, spec.source)


# --------------------------------------------------------------------------- low level network
def _urlopen(url: str, timeout: float):
    """Thin wrapper around ``urllib.request.urlopen`` (patched in tests)."""
    req = urllib.request.Request(url, headers={"User-Agent": "rummy-vision-downloader/1.0"})
    return urllib.request.urlopen(req, timeout=timeout)


def download_file(url: str, dest: Path, retries: int = 3, timeout: float = 60.0, backoff: float = 1.5,
                  chunk_size: int = 1 << 20, progress: bool = True) -> Path:
    """Download ``url`` to ``dest`` atomically (``.part`` then rename) with retries.

    Raises ``DownloadError`` after ``retries`` failed attempts.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    _unlink_quiet(part)   # leftover of an interrupted run; never resumed, always re-fetched
    try:
        return _download_attempts(url, dest, part, retries, timeout, backoff, chunk_size, progress)
    finally:
        _unlink_quiet(part)   # nothing partial survives, not even after Ctrl-C / SIGTERM


def _unlink_quiet(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _download_attempts(url: str, dest: Path, part: Path, retries: int, timeout: float, backoff: float,
                       chunk_size: int, progress: bool) -> Path:
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max(1, retries) + 1):
        try:
            with _urlopen(url, timeout) as resp:
                status = getattr(resp, "status", 200)
                if status and int(status) >= 400:
                    raise DownloadError(f"HTTP {status} for {url}")
                total = 0
                try:
                    total = int(resp.headers.get("Content-Length") or 0)
                except (AttributeError, TypeError, ValueError):
                    total = 0
                bar = None
                if progress and total > 4 * chunk_size:
                    try:
                        from tqdm import tqdm
                        bar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name, leave=False)
                    except ImportError:  # pragma: no cover
                        bar = None
                with open(part, "wb") as f:
                    while True:
                        buf = resp.read(chunk_size)
                        if not buf:
                            break
                        f.write(buf)
                        if bar is not None:
                            bar.update(len(buf))
                if bar is not None:
                    bar.close()
            os.replace(part, dest)
            log.debug("downloaded %s -> %s (%d bytes)", url, dest, dest.stat().st_size)
            return dest
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, DownloadError, ValueError) as exc:
            last_exc = exc
            _unlink_quiet(part)
            if attempt < retries:
                wait = backoff ** attempt
                log.warning("download attempt %d/%d failed for %s (%s); retrying in %.1fs", attempt, retries, url, exc, wait)
                time.sleep(wait)
    raise DownloadError(f"giving up on {url} after {retries} attempt(s): {last_exc}") from last_exc


def looks_blocked(exc: BaseException) -> bool:
    """Heuristic: does the exception chain look like a proxy denial / unreachable host?"""
    seen = 0
    cur: Optional[BaseException] = exc
    while cur is not None and seen < 8:
        text = f"{type(cur).__name__}: {cur}".lower()
        if any(h in text for h in _BLOCKED_HINTS):
            return True
        cur = cur.__cause__ or cur.__context__
        seen += 1
    return False


def describe_fetch_error(spec: DatasetSpec, exc: BaseException, raw_dir: Path) -> str:
    short = f"{type(exc).__name__}: {str(exc).strip().splitlines()[0] if str(exc).strip() else ''}"[:200]
    if looks_blocked(exc):
        return (f"host {source_host(spec)} unreachable or blocked by the network/proxy policy ({short}); "
                f"allow the host or place a manual download of {spec.ref} into {raw_dir}")
    return f"fetch failed ({short}); ref={spec.ref}"


# --------------------------------------------------------------------------- archives
def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in ARCHIVE_SUFFIXES)


def archive_target_dir(archive: Path) -> Path:
    name = archive.name
    for s in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(s):
            name = name[: -len(s)]
            break
    return archive.parent / name


def _safe_member(target: Path, name: str) -> bool:
    """True if extracting ``name`` stays inside ``target`` (no traversal / absolute paths)."""
    if not name or name.startswith(("/", "\\")) or ".." in Path(name).parts:
        return False
    try:
        (target / name).resolve().relative_to(target.resolve())
    except ValueError:
        return False
    return True


def extract_archive(archive: Path, target: Optional[Path] = None, force: bool = False) -> Path:
    """Extract one zip / tar archive next to itself (``<archive-without-suffix>/``), skipping unsafe members."""
    target = target or archive_target_dir(archive)
    marker = target / ".extracted_ok"
    if marker.exists() and not force:
        log.debug("already extracted: %s", archive)
        return target
    target.mkdir(parents=True, exist_ok=True)
    log.info("extracting %s -> %s", archive.name, target)
    skipped = 0
    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not _safe_member(target, info.filename):
                    skipped += 1
                    continue
                zf.extract(info, target)
    else:
        with tarfile.open(archive) as tf:
            members = []
            for m in tf.getmembers():
                if not (m.isfile() or m.isdir()) or not _safe_member(target, m.name):
                    skipped += 1
                    continue
                members.append(m)
            tf.extractall(target, members=members)
    if skipped:
        log.warning("skipped %d unsafe/special member(s) in %s", skipped, archive.name)
    marker.write_text("ok\n")
    return target


def extract_archives(directory: Path, force: bool = False, max_passes: int = 3) -> List[Path]:
    """Extract every archive below ``directory`` (recursively, incl. archives inside archives)."""
    directory = Path(directory)
    done: List[Path] = []
    handled: set = set()
    for _ in range(max_passes):
        archives = sorted(p for p in directory.rglob("*") if p.is_file() and is_archive(p) and p not in handled
                          and not any(part.startswith(".") for part in p.relative_to(directory).parts))
        if not archives:
            break
        for a in archives:
            handled.add(a)
            if (archive_target_dir(a) / ".extracted_ok").exists() and not force:
                log.debug("already extracted: %s", a)
                continue
            done.append(extract_archive(a, force=force))
    return done


def _in_progress(path: Path) -> bool:
    return path.name.lower().endswith(IN_PROGRESS_SUFFIXES)


def raw_has_content(raw_dir: Path) -> bool:
    """A raw dir holds data when it has at least one non-hidden, completed (non ``.part``/``.incomplete``) file."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        return False
    for p in raw_dir.rglob("*"):
        if p.is_file() and not _in_progress(p) and not any(part.startswith(".") for part in p.relative_to(raw_dir).parts):
            return True
    return False


def raw_in_progress(raw_dir: Path) -> bool:
    """True if an interrupted download left partial files anywhere below ``raw_dir`` (hidden dirs included,
    e.g. huggingface_hub's ``.cache/huggingface/download/*.incomplete``)."""
    raw_dir = Path(raw_dir)
    return raw_dir.is_dir() and any(p.is_file() and _in_progress(p) for p in raw_dir.rglob("*"))


def raw_is_fetched(raw_dir: Path) -> bool:
    """Whether ``raw_dir`` holds a *complete* fetch, i.e. is safe to reuse without re-downloading.

    True with the marker ``mark_fetched`` leaves after a successful fetch, or - for data placed there by
    hand - when it holds completed files and no partial download.  An interrupted fetch (partial files
    present, or nothing usable) is fetched again; the clients resume/skip what already landed.
    """
    raw_dir = Path(raw_dir)
    if (raw_dir / FETCHED_MARKER).is_file():
        return True
    return raw_has_content(raw_dir) and not raw_in_progress(raw_dir)


def mark_fetched(raw_dir: Path) -> Path:
    marker = Path(raw_dir) / FETCHED_MARKER
    marker.write_text("ok\n")
    return marker


# --------------------------------------------------------------------------- fetchers
def fetch_huggingface(spec: DatasetSpec, raw_dir: Path, opts: Options) -> Path:
    from huggingface_hub import snapshot_download  # lazy: network client

    raw_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    log.info("huggingface: snapshot_download(%s, repo_type=dataset) -> %s", spec.ref, raw_dir)
    snapshot_download(repo_id=spec.ref, repo_type="dataset", local_dir=str(raw_dir), token=token,
                      force_download=opts.force, max_workers=4)
    extract_archives(raw_dir, force=opts.force)
    return raw_dir


def latest_roboflow_version(project: Any) -> int:
    versions = []
    for v in project.versions() or []:
        raw = getattr(v, "version", None) or getattr(v, "id", "")
        try:
            versions.append(int(str(raw).rsplit("/", 1)[-1]))
        except (TypeError, ValueError):
            continue
    if not versions:
        raise DownloadError("project has no versions")
    return max(versions)


def fetch_roboflow(spec: DatasetSpec, raw_dir: Path, opts: Options) -> Path:
    key = roboflow_api_key(opts.roboflow_key)
    if not key:
        raise CredentialsError(credentials_status(spec)[1])
    from roboflow import Roboflow  # lazy: network client

    if "/" not in spec.ref:
        raise ConversionError(f"roboflow ref must be workspace/project, got {spec.ref!r}")
    ws, proj = spec.ref.split("/", 1)
    raw_dir.parent.mkdir(parents=True, exist_ok=True)   # roboflow creates the location itself
    # Version.download() silently returns without downloading when ``location`` exists and
    # ``overwrite`` is False, so overwrite unless the dir already holds a complete fetch.
    overwrite = bool(opts.force) or not raw_is_fetched(raw_dir)
    rf = Roboflow(api_key=key)
    project = rf.workspace(ws).project(proj)
    version = spec.version if spec.version is not None else latest_roboflow_version(project)
    log.info("roboflow: %s version %s -> %s (format yolov8, overwrite=%s)", spec.ref, version, raw_dir, overwrite)
    project.version(int(version)).download("yolov8", location=str(raw_dir), overwrite=overwrite)
    extract_archives(raw_dir, force=opts.force)
    return raw_dir


def fetch_kaggle(spec: DatasetSpec, raw_dir: Path, opts: Options) -> Path:
    ok, note = kaggle_credentials()
    if not ok:
        raise CredentialsError(note)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log.info("kaggle: dataset %s -> %s (credentials: %s)", spec.ref, raw_dir, note)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # lazy: authenticates on import

        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(spec.ref, path=str(raw_dir), unzip=True, quiet=False, force=bool(opts.force))
    except SystemExit as exc:  # kaggle >= 2 prints its auth help and calls exit(1) when no credential works
        raise CredentialsError(f"kaggle rejected the configured credentials ({note}); "
                               f"{kaggle_credentials_hint()}") from exc
    except Exception as exc:  # noqa: BLE001 - any failure -> try the CLI once
        if not shutil.which("kaggle"):
            raise
        log.warning("kaggle python API failed (%s); falling back to the `kaggle` CLI", exc)
        cmd = ["kaggle", "datasets", "download", "-d", spec.ref, "--unzip", "-p", str(raw_dir)]
        if opts.force:
            cmd.append("--force")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise DownloadError(f"kaggle CLI failed ({proc.returncode}): {(proc.stderr or proc.stdout).strip()[:300]}") from exc
    extract_archives(raw_dir, force=opts.force)
    return raw_dir


def fetch_url(spec: DatasetSpec, raw_dir: Path, opts: Options) -> Path:
    """Download ``spec.ref`` into ``raw_dir`` and extract it if it is an archive."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    name = Path(urlparse(spec.ref).path).name or "download.bin"
    dest = raw_dir / name
    if dest.exists() and not opts.force:
        log.info("url: %s already present", dest)
    else:
        log.info("url: %s -> %s", spec.ref, dest)
        download_file(spec.ref, dest, retries=opts.retries, timeout=opts.timeout)
    if is_archive(dest):
        extract_archive(dest, force=opts.force)
    return raw_dir


def card_art_files() -> Dict[str, str]:
    """``{output_stem: upstream_file_name}`` for the 52 cards + jokers + back."""
    out = {name: cards.card_asset_filename(name) for name in cards.CARD_CLASSES}
    out.update(CARD_ART_EXTRAS)
    return out


def fetch_card_art(dest_dir: Path, force: bool = False, retries: int = 3, timeout: float = 60.0) -> List[Path]:
    """Fetch the public-domain deck into ``dest_dir/<CANONICAL>.png`` (+ ``JOKER_RED/JOKER_BLACK/BACK.png``).

    Existing files are kept unless ``force``.  Every file is attempted; if any failed a
    ``DownloadError`` listing them is raised at the end.  Returns the 55 paths on success.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    files = card_art_files()
    todo = {stem: fn for stem, fn in files.items() if force or not (dest_dir / f"{stem}.png").exists()}
    if todo:
        log.info("card art: fetching %d/%d PNG(s) from %s into %s", len(todo), len(files), CARD_ART_BASE_URL, dest_dir)
    failures: List[str] = []
    try:
        from tqdm import tqdm
        items = tqdm(sorted(todo.items()), desc="card art", unit="png", leave=False, disable=len(todo) < 4)
    except ImportError:  # pragma: no cover
        items = sorted(todo.items())
    for stem, fn in items:
        url = CARD_ART_BASE_URL + fn
        dest = dest_dir / f"{stem}.png"
        try:
            download_file(url, dest, retries=retries, timeout=timeout, progress=False)
            with open(dest, "rb") as f:
                magic = f.read(8)
            if magic != PNG_MAGIC:
                dest.unlink(missing_ok=True)
                raise DownloadError(f"{url} did not return a PNG (got {magic!r})")
        except DownloadError as exc:
            log.error("card art: %s", exc)
            failures.append(f"{stem}.png: {exc}")
    if failures:
        raise DownloadError(f"{len(failures)} card art file(s) failed: " + "; ".join(failures[:3])
                            + (" ..." if len(failures) > 3 else ""))
    return [dest_dir / f"{stem}.png" for stem in files]


def fetch_github_raw(spec: DatasetSpec, raw_dir: Path, opts: Options) -> Path:
    """Card art goes straight to ``Paths.assets_cards`` (that *is* the usable form)."""
    dest = opts.paths.assets_cards
    fetch_card_art(dest, force=opts.force, retries=opts.retries, timeout=opts.timeout)
    return dest


#: source -> fetcher; tests replace entries with offline fakes.
FETCHERS: Dict[str, Callable[[DatasetSpec, Path, Options], Path]] = {
    "huggingface": fetch_huggingface,
    "roboflow": fetch_roboflow,
    "kaggle": fetch_kaggle,
    "url": fetch_url,
    "github_raw": fetch_github_raw,
}


def missing_card_art(dest_dir: Path) -> List[str]:
    return [f"{stem}.png" for stem in card_art_files() if not (Path(dest_dir) / f"{stem}.png").exists()]


# --------------------------------------------------------------------------- backgrounds (DTD)
def collect_backgrounds(raw_dir: Path, dest_dir: Path, copy: bool = False, exts: Sequence[str] = (".jpg", ".jpeg", ".png")) -> int:
    """Link every image under an ``images`` directory of ``raw_dir`` into ``dest_dir``.

    Files are named ``<category>_<file>`` (parent directory prefix) to avoid collisions.
    Returns the number of images present in ``dest_dir`` afterwards.
    """
    raw_dir, dest_dir = Path(raw_dir), Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    n_new = 0
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in exts or "images" not in p.parts:
            continue
        rel = p.relative_to(raw_dir).parts
        prefix = rel[-2] if len(rel) >= 2 and rel[-2] != "images" else ""
        dst = dest_dir / (f"{prefix}_{p.name}" if prefix else p.name)
        if not (dst.exists() or dst.is_symlink()):
            labels.link_or_copy(p, dst, copy=copy)
            n_new += 1
    total = len(labels.list_images(dest_dir))
    log.info("backgrounds: %d new, %d total in %s", n_new, total, dest_dir)
    return total


# --------------------------------------------------------------------------- label helpers
def parse_label_line(line: str) -> Optional[labels.Box]:
    """Parse ``cls cx cy w h`` or a polygon ``cls x1 y1 x2 y2 ...`` (converted to its AABB)."""
    parts = line.split()
    if len(parts) < 5:
        return None
    try:
        vals = [float(v) for v in parts[1:]]
        c = int(float(parts[0]))
    except ValueError:
        return None
    if len(vals) == 4:
        return labels.Box(c, *vals)
    if len(vals) >= 6 and len(vals) % 2 == 0:  # polygon / OBB -> axis-aligned box
        pts = [(vals[i], vals[i + 1]) for i in range(0, len(vals), 2)]
        x1, y1, x2, y2 = labels.points_aabb(pts)
        if x2 <= x1 or y2 <= y1:
            return None
        return labels.Box(c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
    return labels.Box(c, *vals[:4])


def read_raw_boxes(path: Path) -> List[labels.Box]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            b = parse_label_line(line)
            if b is not None:
                out.append(b)
    return out


def detect_box_semantics(root: Path, max_images: int = 2000) -> str:
    """``corner`` if median box area / image area < 0.02, ``card`` if > 0.05, else ``unknown``."""
    areas: List[float] = []
    for split in labels.SPLITS:
        _, lbl_dir = labels.split_dirs(root, split)
        if not lbl_dir.is_dir():
            continue
        for i, lp in enumerate(sorted(lbl_dir.glob("*.txt"))):
            if i >= max_images:
                break
            areas.extend(b.area() for b in labels.read_boxes(lp))
    if not areas:
        return "unknown"
    med = statistics.median(areas)
    return "corner" if med < 0.02 else "card" if med > 0.05 else "unknown"


def mapping_rows_to_manifest(rows: Iterable[Tuple[int, str, Optional[str], Optional[int]]]) -> Tuple[Dict[str, Dict], List[str]]:
    mapping: Dict[str, Dict] = {}
    unmapped: List[str] = []
    for idx, raw, canon, cid in rows:
        mapping[str(idx)] = {"raw": raw, "canonical": canon, "id": cid}
        if cid is None:
            unmapped.append(raw)
    return mapping, unmapped


def log_mapping(rows: Iterable[Tuple[int, str, Optional[str], Optional[int]]], key: str) -> None:
    rows = list(rows)
    log.info("%s: class mapping (%d raw classes)", key, len(rows))
    for idx, raw, canon, cid in rows:
        log.debug("  %3d %-22s -> %-16s %s", idx, raw, canon or "-", cid if cid is not None else "DROPPED")
    dropped = [raw for _, raw, _, cid in rows if cid is None]
    if dropped:
        log.warning("%s: %d class(es) not in the target space, their boxes are dropped: %s", key, len(dropped), dropped)


def build_manifest(spec: DatasetSpec, out_dir: Path, mapping_rows: Iterable[Tuple[int, str, Optional[str], Optional[int]]],
                   views: Dict[str, str], config: Dict[str, Any], raw_names: Sequence[str]) -> Dict[str, Any]:
    mapping, unmapped = mapping_rows_to_manifest(mapping_rows)
    return {
        "key": spec.key, "source": spec.source, "ref": spec.ref, "version": spec.version,
        "class_space": spec.class_space, "box_semantics": spec.box_semantics,
        "box_semantics_detected": detect_box_semantics(out_dir),
        "names": cards.class_names_for_space(spec.class_space),
        "raw_names": list(raw_names),
        "class_mapping": mapping, "unmapped": unmapped,
        "stats": labels.dataset_stats(out_dir),
        "views": views, "created_by": "download_datasets", "config": config,
    }


# --------------------------------------------------------------------------- generic YOLO converter
@dataclass
class YoloLayout:
    base: Path
    data_yaml: Optional[Path]
    names: List[str]
    splits: Dict[str, Tuple[Path, Optional[Path]]] = field(default_factory=dict)  # split -> (images, labels|None)


def _depth(p: Path, root: Path) -> int:
    return len(p.relative_to(root).parts)


def _iter_visible(root: Path, pattern: str) -> List[Path]:
    return sorted((p for p in root.rglob(pattern)
                   if not any(part.startswith(".") for part in p.relative_to(root).parts)),
                  key=lambda p: (_depth(p, root), str(p)))


def find_data_yaml(raw_dir: Path) -> Optional[Path]:
    """Shallowest ``*.yaml|*.yml`` with a ``names`` key (roboflow / ultralytics exports)."""
    for p in _iter_visible(raw_dir, "*.y*ml"):
        if p.suffix.lower() not in (".yaml", ".yml") or not p.is_file():
            continue
        try:
            data = labels.read_data_yaml(p)
        except Exception:  # noqa: BLE001 - not a dataset yaml
            continue
        if isinstance(data, dict) and "names" in data:
            return p
    return None


def _parse_names_text(text: str) -> List[str]:
    """One name per line, optionally ``idx name`` / ``idx: name``; index order is respected."""
    indexed: Dict[int, str] = {}
    plain: List[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = re.match(r"^(\d+)\s*[:=.)\-]\s*(\S.*)$", s) or re.match(r"^(\d+)\s+(\S.*)$", s)
        if m:
            indexed[int(m.group(1))] = m.group(2).strip().strip("'\",")
        else:
            plain.append(s.strip("'\","))
    if indexed and len(indexed) >= len(plain):
        return [indexed[i] for i in sorted(indexed)]
    return plain


def infer_names(base: Path) -> List[str]:
    """Names from ``classes.txt`` / ``obj.names`` / ``labels.txt`` / ``names.txt`` or a README."""
    for fn in ("classes.txt", "obj.names", "labels.txt", "names.txt", "classes.names"):
        for p in _iter_visible(base, fn):
            names = _parse_names_text(p.read_text(errors="replace"))
            if names:
                log.info("names inferred from %s (%d classes)", p, len(names))
                return names
    for p in _iter_visible(base, "README*"):
        text = p.read_text(errors="replace")
        m = re.search(r"names\s*:\s*(\[[^\]]*\])", text)
        if m:
            try:
                import yaml
                names = [str(n) for n in yaml.safe_load(m.group(1))]
                if names:
                    log.info("names inferred from %s (%d classes)", p, len(names))
                    return names
            except Exception:  # noqa: BLE001
                pass
    return []


def _resolve_yaml_split(base: Path, value: Any) -> Optional[Path]:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    candidates = [base / v, base.parent / v, base / v.lstrip("./").lstrip("../"), base / Path(v).name]
    stripped = re.sub(r"^(\.\./|\./)+", "", v)
    candidates.append(base / stripped)
    for c in candidates:
        try:
            if c.is_dir():
                return c.resolve()
        except OSError:
            continue
    return None


def labels_dir_for(images_dir: Path) -> Optional[Path]:
    """``.../images/<split>`` -> ``.../labels/<split>`` (ultralytics substitution); ``None`` if absent."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            cand = Path(*parts)
            return cand if cand.is_dir() else None
    sib = images_dir.parent / "labels"
    return sib if sib.is_dir() else None


def find_yolo_layout(raw_dir: Path) -> YoloLayout:
    """Locate names + split image/label directories inside a raw YOLO export."""
    raw_dir = Path(raw_dir)
    data_yaml = find_data_yaml(raw_dir)
    base = data_yaml.parent if data_yaml else raw_dir
    data = labels.read_data_yaml(data_yaml) if data_yaml else {}
    names = labels.yaml_names_list(data) if data else []
    if not names:
        names = infer_names(base) or infer_names(raw_dir)
    layout = YoloLayout(base=base, data_yaml=data_yaml, names=names)

    for split, aliases in SPLIT_ALIASES.items():
        found: Optional[Path] = None
        for key in aliases:
            if key in data:
                found = _resolve_yaml_split(base, data[key])
                if found:
                    break
        if found is None:
            for alias in aliases:
                for cand in (base / alias / "images", base / "images" / alias, base / alias):
                    if cand.is_dir() and labels.list_images(cand):
                        found = cand
                        break
                if found:
                    break
        if found is not None:
            layout.splits[split] = (found, labels_dir_for(found))

    if not layout.splits:  # last resort: any ``images`` dir with pictures inside
        for d in _iter_visible(raw_dir, "images"):
            if not d.is_dir():
                continue
            sub = [c for c in sorted(d.iterdir()) if c.is_dir()]
            for c in sub:
                split = next((s for s, al in SPLIT_ALIASES.items() if c.name.lower() in al), None)
                if split and split not in layout.splits and labels.list_images(c):
                    layout.splits[split] = (c, labels_dir_for(c))
            if labels.list_images(d):
                split = next((s for s, al in SPLIT_ALIASES.items() if d.parent.name.lower() in al), "train")
                layout.splits.setdefault(split, (d, labels_dir_for(d)))
            if layout.splits:
                break
    return layout


def convert_yolo(spec: DatasetSpec, raw_dir: Path, out_dir: Path, copy: bool = False, val_ratio: float = 0.1) -> Dict[str, Any]:
    """Convert a YOLO export in ``raw_dir`` into the canonical layout at ``out_dir`` (recreated).

    Returns the manifest (also written to ``out_dir/manifest.json``).
    """
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    layout = find_yolo_layout(raw_dir)
    if not layout.splits:
        raise ConversionError(f"no images/labels split directories found under {raw_dir} "
                              "(expected <split>/images or images/<split>)")
    if not layout.names:
        raise ConversionError(f"no class names found under {raw_dir} (data.yaml / classes.txt / README)")
    for split, (img_dir, lbl_dir) in layout.splits.items():
        log.info("%s: split %-5s images=%s labels=%s", spec.key, split, img_dir, lbl_dir or "MISSING")
        if lbl_dir is None:
            log.warning("%s: split %s has no labels directory; its images become background-only", spec.key, split)

    rows = cards.describe_mapping(layout.names, spec.name_style, spec.class_space)
    mapping = cards.map_dataset_names(layout.names, spec.name_style, spec.class_space)
    log_mapping(rows, spec.key)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    carve_val = "val" not in layout.splits and val_ratio > 0 and "train" in layout.splits
    every = max(2, int(round(1.0 / val_ratio))) if carve_val else 0
    if carve_val:
        log.warning("%s: no val split in the export; every %d-th train image becomes val", spec.key, every)
    out_splits = sorted(set(layout.splits) | ({"val"} if carve_val else set()), key=labels.SPLITS.index)
    labels.ensure_layout(out_dir, out_splits)

    used: Dict[str, set] = {s: set() for s in out_splits}
    dropped_total = 0
    unknown_ids: Dict[int, int] = {}
    n_images = 0
    for split in sorted(layout.splits, key=labels.SPLITS.index):
        img_dir, lbl_dir = layout.splits[split]
        for i, img in enumerate(labels.list_images(img_dir)):
            target_split = "val" if (carve_val and split == "train" and i % every == 0) else split
            stem = img.stem
            k = 1
            while stem in used[target_split]:
                k += 1
                stem = f"{img.stem}_{k}"
            used[target_split].add(stem)
            o_img_dir, o_lbl_dir = labels.split_dirs(out_dir, target_split)
            labels.link_or_copy(img, o_img_dir / (stem + img.suffix.lower()), copy=copy)
            raw_boxes = read_raw_boxes(lbl_dir / (img.stem + ".txt")) if lbl_dir else []
            for b in raw_boxes:
                if b.cls not in mapping:
                    unknown_ids[b.cls] = unknown_ids.get(b.cls, 0) + 1
            kept, dropped = labels.remap_boxes(raw_boxes, mapping)
            dropped_total += dropped
            labels.write_boxes(o_lbl_dir / (stem + ".txt"), kept)
            n_images += 1
    if unknown_ids:
        log.warning("%s: %d box(es) with class ids missing from names (%s) were dropped", spec.key,
                    sum(unknown_ids.values()), dict(sorted(unknown_ids.items())))
    log.info("%s: %d images converted, %d box(es) dropped", spec.key, n_images, dropped_total)

    names = cards.class_names_for_space(spec.class_space)
    labels.write_data_yaml(out_dir / "data.yaml", names, "images/train", "images/val",
                           test="images/test" if "test" in out_splits else None, root=out_dir)
    config = {"converter": "yolo", "raw_dir": str(raw_dir), "data_yaml": str(layout.data_yaml) if layout.data_yaml else None,
              "copy": copy, "val_carved_every": every, "dropped_boxes": dropped_total,
              "unknown_class_ids": {str(k): v for k, v in sorted(unknown_ids.items())}}
    manifest = build_manifest(spec, out_dir, rows, views={}, config=config, raw_names=layout.names)
    labels.write_manifest(out_dir, manifest)
    _log_detected(spec, manifest)
    return manifest


def _log_detected(spec: DatasetSpec, manifest: Dict[str, Any]) -> None:
    det = manifest.get("box_semantics_detected")
    if det not in ("unknown", spec.box_semantics) and spec.box_semantics != "parts":
        log.warning("%s: declared box_semantics=%s but the area heuristic says %s - check with --preview",
                    spec.key, spec.box_semantics, det)
    else:
        log.info("%s: box_semantics=%s (detected %s)", spec.key, spec.box_semantics, det)


# --------------------------------------------------------------------------- JackFurby converter
def load_jackfurby_classes(path: Optional[Path] = None) -> Dict[int, str]:
    """``{label_index: name}`` from ``vision/configs/jackfurby_card_classes.txt`` (``2C 0`` ... ``AS 51``)."""
    txt = Path(path or JACKFURBY_CLASSES_FILE).read_text().split()
    return {int(i): n for n, i in zip(txt[0::2], txt[1::2])}


def jackfurby_quad_points(points: Sequence[Sequence[float]]) -> List[Tuple[float, float]]:
    """JackFurby stores ``[TL, TR, BL, BR]``; return ``[TL, TR, BR, BL]`` (clockwise) as float tuples."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    if len(pts) != 4:
        raise ConversionError(f"expected 4 card corners, got {len(pts)}")
    tl, tr, bl, br = pts
    return [tl, tr, br, bl]


def jackfurby_points_to_labels(card_points: Iterable[Sequence[Any]], classes: Dict[int, str], img_w: int, img_h: int,
                               space: str = "cards52") -> Tuple[List[labels.Box], List[labels.Quad], int]:
    """``card_points`` entries ``[[TL,TR,BL,BR], label]`` -> (AABB boxes, OBB quads, dropped count)."""
    boxes: List[labels.Box] = []
    quads: List[labels.Quad] = []
    dropped = 0
    for item in card_points:
        pts, label = item[0], int(item[1])
        name = classes.get(label)
        cid = cards.canonical_id(cards.normalize_class_name(name) if name else None, space)
        if cid is None:
            dropped += 1
            continue
        ordered = jackfurby_quad_points(pts)
        box = labels.Box.from_points(cid, ordered, img_w, img_h)
        if box is None:
            dropped += 1
            continue
        boxes.append(box)
        quads.append(labels.Quad.from_pixels(cid, ordered, img_w, img_h))
    return boxes, quads, dropped


def find_jackfurby_annotations(raw_dir: Path) -> List[Tuple[str, str, Path]]:
    """``(subset, split, file)`` for every ``*.pkl`` (or ``*.json`` without a pkl twin) below ``raw_dir``."""
    raw_dir = Path(raw_dir)
    found: Dict[Tuple[str, str], Path] = {}
    for p in _iter_visible(raw_dir, "*.pkl") + _iter_visible(raw_dir, "*.json"):
        if not p.is_file():
            continue
        stem = p.stem.lower()
        split = "train" if "train" in stem else "val" if any(t in stem for t in ("val", "test")) else None
        if split is None:
            continue
        subset = p.parent.name if p.parent != raw_dir else raw_dir.name
        subset = re.sub(r"[^A-Za-z0-9]+", "_", subset).strip("_") or "main"
        found.setdefault((subset, split), p)  # pkl listed first wins over json
    return sorted((s, sp, p) for (s, sp), p in found.items())


def load_jackfurby_samples(path: Path) -> Dict[Any, Dict[str, Any]]:
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
    else:
        with open(path, "rb") as f:
            data = pickle.load(f)
    if isinstance(data, list):
        data = {i: s for i, s in enumerate(data)}
    if not isinstance(data, dict):
        raise ConversionError(f"{path}: expected a dict of samples, got {type(data).__name__}")
    return data


def _sample_sort_key(k: Any) -> Tuple[int, Any]:
    try:
        return (0, int(k))
    except (TypeError, ValueError):
        return (1, str(k))


class _ImageIndex:
    """Lazy ``file name -> paths`` index of the images below a directory (resolution fallback)."""

    def __init__(self, root: Path):
        self.root = root
        self._by_name: Optional[Dict[str, List[Path]]] = None

    def lookup(self, rel: str) -> Optional[Path]:
        if self._by_name is None:
            self._by_name = {}
            for p in sorted(self.root.rglob("*")):
                if p.is_file() and p.suffix.lower() in labels.IMG_EXTS:
                    self._by_name.setdefault(p.name, []).append(p)
        cands = self._by_name.get(Path(rel).name, [])
        if not cands:
            return None
        tail = Path(rel).parts[-2:]
        for c in cands:
            if tuple(c.parts[-len(tail):]) == tuple(tail):
                return c
        return cands[0]


def resolve_jackfurby_image(img_path: str, ann_dir: Path, raw_dir: Path, subset: str, index: _ImageIndex) -> Optional[Path]:
    rel = str(img_path).replace("\\", "/").lstrip("./")
    name = Path(rel).name
    for cand in (ann_dir / rel, ann_dir.parent / rel, raw_dir / rel, ann_dir / "imgs" / subset / name,
                 ann_dir / "imgs" / name, ann_dir / name):
        if cand.is_file():
            return cand
    return index.lookup(rel)


def image_size(path: Path) -> Tuple[int, int]:
    from PIL import Image  # lazy; header-only read

    with Image.open(path) as im:
        return im.size


def convert_jackfurby(spec: DatasetSpec, raw_dir: Path, out_dir: Path, copy: bool = False,
                      classes_file: Optional[Path] = None) -> Dict[str, Any]:
    """Convert JackFurby pickle annotations to whole-card AABB labels + an OBB view."""
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    classes = load_jackfurby_classes(classes_file)
    ann = find_jackfurby_annotations(raw_dir)
    if not ann:
        raise ConversionError(f"no train/val .pkl (or .json) annotation files found under {raw_dir}")
    raw_names = [classes[i] for i in sorted(classes)]
    rows = cards.describe_mapping(raw_names, "english", spec.class_space)
    log_mapping(rows, spec.key)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    labels.ensure_layout(out_dir, ("train", "val"))
    obb_root = out_dir / "views" / "obb"
    for s in ("train", "val"):
        (obb_root / "labels" / s).mkdir(parents=True, exist_ok=True)
        (obb_root / "images" / s).mkdir(parents=True, exist_ok=True)

    n_images = missing = dropped_total = 0
    for subset, split, path in ann:
        samples = load_jackfurby_samples(path)
        index = _ImageIndex(path.parent)
        log.info("%s: %s/%s -> %d samples (%s)", spec.key, subset, split, len(samples), path)
        for sid in sorted(samples, key=_sample_sort_key):
            sample = samples[sid]
            img = resolve_jackfurby_image(sample.get("img_path", ""), path.parent, raw_dir, subset, index)
            if img is None:
                missing += 1
                if missing <= 5:
                    log.warning("%s: image not found for sample %s: %s", spec.key, sid, sample.get("img_path"))
                continue
            w, h = image_size(img)
            boxes, quads, dropped = jackfurby_points_to_labels(sample.get("card_points", []), classes, w, h, spec.class_space)
            dropped_total += dropped
            stem = f"{subset}_{sid}"
            o_img_dir, o_lbl_dir = labels.split_dirs(out_dir, split)
            img_name = stem + img.suffix.lower()
            labels.link_or_copy(img, o_img_dir / img_name, copy=copy)
            labels.write_boxes(o_lbl_dir / (stem + ".txt"), boxes)
            # views/obb/images/<split>/ is a real directory of per-file links (never one directory
            # symlink: ultralytics resolves it and would then read the primary labels, see labels.make_view)
            labels.link_or_copy(o_img_dir / img_name, obb_root / "images" / split / img_name, copy=copy)
            labels.write_quads(obb_root / "labels" / split / (stem + ".txt"), quads)
            n_images += 1
    if missing:
        log.warning("%s: %d sample(s) skipped because their image was not found", spec.key, missing)
    if n_images == 0:
        raise ConversionError(f"{spec.key}: no images could be resolved from {len(ann)} annotation file(s)")
    log.info("%s: %d images converted (%d card(s) dropped)", spec.key, n_images, dropped_total)

    names = cards.class_names_for_space(spec.class_space)
    labels.write_data_yaml(out_dir / "data.yaml", names, "images/train", "images/val", root=out_dir)
    labels.write_data_yaml(obb_root / "data.yaml", names, "images/train", "images/val", root=obb_root)
    config = {"converter": "jackfurby_pkl", "raw_dir": str(raw_dir), "copy": copy, "missing_images": missing,
              "dropped_cards": dropped_total, "annotation_files": [str(p) for _, _, p in ann],
              "classes_file": str(classes_file or JACKFURBY_CLASSES_FILE)}
    manifest = build_manifest(spec, out_dir, rows, views={"obb": "views/obb"}, config=config, raw_names=raw_names)
    labels.write_manifest(out_dir, manifest)
    _log_detected(spec, manifest)
    return manifest


CONVERTERS: Dict[str, Callable[..., Dict[str, Any]]] = {"yolo": convert_yolo, "jackfurby_pkl": convert_jackfurby}


# --------------------------------------------------------------------------- previews
def _class_color(cid: int) -> Tuple[int, int, int]:
    h = (cid * 47) % 180
    import cv2  # lazy
    import numpy as np

    hsv = np.uint8([[[h, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_annotations(img: Any, boxes: Sequence[labels.Box], names: Sequence[str], quads: Sequence[labels.Quad] = ()) -> Any:
    """Draw boxes (+ optional OBB quads) with class names onto a BGR image (returns a copy)."""
    import cv2  # lazy
    import numpy as np

    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(w, h) / 800.0)
    for q in quads:
        pts = np.array(q.to_pixels(w, h), dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (255, 0, 255), max(1, int(2 * scale)))  # magenta = OBB view
    for b in boxes:
        x1, y1, x2, y2 = (int(round(v)) for v in b.to_xyxy(w, h))
        color = _class_color(b.cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(1, int(2 * scale)))
        name = names[b.cls] if 0 <= b.cls < len(names) else str(b.cls)
        cv2.putText(out, name, (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, color, max(1, int(1.5 * scale)))
    return out


def write_previews(root: Path, n: int, names: Optional[Sequence[str]] = None, out_dir: Optional[Path] = None) -> List[Path]:
    """Render the first ``n`` labelled images (train first, then val) to ``<root>/preview/``."""
    import cv2  # lazy

    root = Path(root)
    out_dir = Path(out_dir) if out_dir else root / "preview"
    if names is None:
        names = labels.read_manifest(root).get("names") or labels.yaml_names_list(labels.read_data_yaml(root / "data.yaml"))
    views = labels.read_manifest(root).get("views", {}) or {}
    obb_dir = root / views["obb"] / "labels" if "obb" in views else None
    written: List[Path] = []
    for split in ("train", "val", "test"):
        for img_path, lbl_path in labels.iter_split(root, split):
            if len(written) >= n:
                return written
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img is None:
                log.warning("preview: cannot read %s", img_path)
                continue
            quads = labels.read_quads(obb_dir / split / lbl_path.name) if obb_dir else []
            out = draw_annotations(img, labels.read_boxes(lbl_path), names, quads)
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f"{split}_{img_path.stem}.jpg"
            cv2.imwrite(str(dst), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
            written.append(dst)
    return written


# --------------------------------------------------------------------------- planning
def target_dir(spec: DatasetSpec, paths: Paths) -> Path:
    if spec.kind == "card_art":
        return paths.assets_cards
    if spec.kind == "backgrounds":
        return paths.assets_backgrounds
    return paths.dataset(spec.key)


def build_plan(keys: Sequence[str], opts: Options) -> List[PlanEntry]:
    plan: List[PlanEntry] = []
    for key in keys:
        spec = DATASETS[key]
        raw_dir = opts.paths.raw / key
        target = target_dir(spec, opts.paths)
        ok, note = credentials_status(spec, opts.roboflow_key)
        if spec.kind == "card_art":
            raw_present = converted = not missing_card_art(target)
        elif spec.kind == "backgrounds":
            raw_present = raw_is_fetched(raw_dir)
            converted = bool(labels.list_images(target))
        else:
            raw_present = raw_is_fetched(raw_dir)
            converted = (target / "manifest.json").exists()
        plan.append(PlanEntry(spec, raw_dir, target, source_host(spec), tuple(spec.requires_env), ok, note, raw_present, converted))
    return plan


def format_plan(plan: Sequence[PlanEntry], opts: Options) -> str:
    lines = [f"Download plan ({len(plan)} dataset(s)); root = {opts.paths.root}", ""]
    for e in plan:
        s = e.spec
        lines.append(f"[{s.key}]  source={s.source}  host={e.host}  kind={s.kind}")
        lines.append(f"    ref:        {s.ref}")
        lines.append(f"    version:    {s.version if s.version is not None else 'latest'}    label_style={s.label_style}  "
                     f"name_style={s.name_style}  class_space={s.class_space}  box_semantics={s.box_semantics}")
        lines.append(f"    raw dir:    {e.raw_dir}  ({'present' if e.raw_present else 'not fetched'})")
        lines.append(f"    target:     {e.target}  ({'converted' if e.converted else 'not converted'})")
        env = ", ".join(e.env_vars) if e.env_vars else "none"
        lines.append(f"    env vars:   {env}    credentials: {'yes' if e.creds_ok else 'MISSING'} ({e.creds_note})")
        action = []
        if not opts.convert_only:
            action.append("fetch" if (opts.force or not e.raw_present) else "fetch: cached")
        if not opts.skip_convert and s.kind == "detection":
            action.append("convert" if (opts.force or not e.converted) else "convert: up-to-date")
        elif not opts.skip_convert:
            action.append("collect assets")
        if not e.creds_ok:
            action.append("-> WILL BE SKIPPED (credentials missing)")
        lines.append(f"    action:     {'; '.join(action) or 'nothing'}")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- orchestration
def _fill_counts(res: DatasetResult, manifest: Dict[str, Any], spec: DatasetSpec) -> None:
    stats = manifest.get("stats", {}) or {}
    res.images = sum(int(s.get("images", 0)) for s in stats.values())
    res.boxes = sum(int(s.get("boxes", 0)) for s in stats.values())
    res.semantics = f"{spec.box_semantics}/{manifest.get('box_semantics_detected', '?')}"


def process_dataset(spec: DatasetSpec, opts: Options) -> DatasetResult:
    """Run fetch + convert (+ preview) for one dataset; never raises."""
    raw_dir = opts.paths.raw / spec.key
    out_dir = target_dir(spec, opts.paths)
    res = DatasetResult(spec.key, spec.source, raw_dir=raw_dir, out_dir=out_dir)
    log.info("=== %s (%s: %s) ===", spec.key, spec.source, spec.ref)

    # ---- stage 1: fetch
    if opts.convert_only:
        res.fetch = "skipped"
    else:
        uses_raw = spec.kind != "card_art"
        cached = raw_is_fetched(raw_dir) if uses_raw else not missing_card_art(out_dir)
        if cached and not opts.force:
            res.fetch = "cached"
            log.info("%s: raw data present, not re-downloading (use --force)", spec.key)
        else:
            if uses_raw and not opts.force and raw_has_content(raw_dir):
                log.info("%s: %s holds an incomplete fetch, fetching again", spec.key, raw_dir)
            ok, note = credentials_status(spec, opts.roboflow_key)
            if not ok:
                res.fetch, res.message = "no-credentials", note
                log.warning("%s: skipped, credentials missing: %s", spec.key, note)
            else:
                if uses_raw:
                    _unlink_quiet(raw_dir / FETCHED_MARKER)   # stale if this run gets interrupted
                try:
                    FETCHERS[spec.source](spec, raw_dir, opts)
                    res.fetch = "ok"
                    if uses_raw and raw_has_content(raw_dir):
                        mark_fetched(raw_dir)
                except CredentialsError as exc:
                    res.fetch, res.message = "no-credentials", str(exc)
                    log.warning("%s: skipped, credentials missing: %s", spec.key, exc)
                except (Exception, SystemExit) as exc:  # noqa: BLE001 - keep going with the other datasets
                    res.fetch, res.message = "failed", describe_fetch_error(spec, exc, raw_dir)
                    log.error("%s: %s", spec.key, res.message)
                    log.debug("traceback", exc_info=True)

    # ---- stage 2: convert / collect
    if res.fetch in ("failed", "no-credentials") or opts.skip_convert:
        res.convert = "skipped"
        return res
    try:
        if spec.kind == "card_art":
            missing = missing_card_art(out_dir)
            if missing:
                raise ConversionError(f"{len(missing)} card art file(s) missing in {out_dir}: {missing[:5]}")
            res.convert, res.images = "n/a", len(card_art_files())
            res.message = res.message or f"{res.images} PNGs in {out_dir}"
        elif spec.kind == "backgrounds":
            if not raw_has_content(raw_dir):
                raise ConversionError(f"nothing to collect: {raw_dir} is empty")
            extract_archives(raw_dir)
            res.images = collect_backgrounds(raw_dir, out_dir, copy=opts.copy)
            if res.images == 0:
                raise ConversionError(f"no images found under {raw_dir}")
            res.convert = "ok"
            res.message = res.message or f"{res.images} backgrounds in {out_dir}"
        else:
            manifest = labels.read_manifest(out_dir)
            if manifest and not opts.force:
                res.convert = "up-to-date"
                log.info("%s: already converted at %s (use --force to redo)", spec.key, out_dir)
            else:
                if not raw_dir.is_dir():
                    raise ConversionError(f"raw dir missing, run without --convert-only first: {raw_dir}")
                converter = CONVERTERS.get(spec.label_style)
                if converter is None:
                    raise ConversionError(f"no converter for label_style {spec.label_style!r}")
                kwargs = {"val_ratio": opts.val_ratio} if spec.label_style == "yolo" else {}
                manifest = converter(spec, raw_dir, out_dir, copy=opts.copy, **kwargs)
                res.convert = "ok"
            _fill_counts(res, manifest, spec)
            if opts.preview > 0:
                written = write_previews(out_dir, opts.preview, manifest.get("names"))
                log.info("%s: %d preview image(s) in %s", spec.key, len(written), out_dir / "preview")
    except Exception as exc:  # noqa: BLE001
        res.convert = "failed"
        res.message = f"{type(exc).__name__}: {exc}"[:300]
        log.error("%s: conversion failed: %s", spec.key, res.message)
        log.debug("traceback", exc_info=True)
    return res


def format_summary(results: Sequence[DatasetResult]) -> str:
    header = ("dataset", "source", "fetch", "convert", "images", "boxes", "semantics", "note")
    rows = [header] + [(r.key, r.source, r.fetch, r.convert, str(r.images or ""), str(r.boxes or ""), r.semantics or "-",
                        (r.message or "")[:110]) for r in results]
    widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
    fmt = "  ".join("{:<%d}" % w for w in widths)
    lines = [fmt.format(*header).rstrip(), fmt.format(*("-" * w for w in widths)).rstrip()]
    lines += [fmt.format(*row).rstrip() for row in rows[1:]]
    n_fail = sum(r.failed for r in results)
    n_nocred = sum(r.fetch == "no-credentials" for r in results)
    lines.append("")
    lines.append(f"{len(results)} dataset(s): {sum(r.succeeded for r in results)} ok, {n_nocred} skipped (credentials), {n_fail} failed")
    return "\n".join(lines)


def exit_code(results: Sequence[DatasetResult]) -> int:
    if not results:
        return EXIT_USAGE
    if any(r.failed for r in results):
        return EXIT_FAILED
    if all(r.fetch == "no-credentials" for r in results):
        return EXIT_FAILED
    return EXIT_OK


# --------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m vision.download_datasets", description=__doc__.split("Usage")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default="free",
                   help="group (%s) or comma-separated keys (%s). Default: free" % ("|".join(DATASET_GROUPS), ", ".join(DATASETS)))
    p.add_argument("--root", type=Path, default=None, help="data root (default: data/vision or $RUMMY_VISION_DATA)")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit without any network / disk I/O")
    p.add_argument("--convert-only", action="store_true", help="skip fetching; convert what is already in raw/")
    p.add_argument("--skip-convert", action="store_true", help="fetch only")
    p.add_argument("--copy", action="store_true", help="copy images instead of symlinking them")
    p.add_argument("--preview", type=int, default=0, metavar="N", help="write N annotated images per dataset to <dataset>/preview/")
    p.add_argument("--roboflow-key", default=None, metavar="K", help="Roboflow API key (else $ROBOFLOW_API_KEY)")
    p.add_argument("--force", action="store_true", help="re-download and re-convert even if present")
    p.add_argument("--retries", type=int, default=3, help="HTTP retries for plain downloads (default 3)")
    p.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds (default 60)")
    p.add_argument("--val-ratio", type=float, default=0.1, help="val fraction carved from train when an export has no val split")
    p.add_argument("--verbose", "-v", action="store_true", help="DEBUG logging (incl. the full class mapping table)")
    return p


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    root.setLevel(min(root.level or level, level) if root.level else level)
    log.setLevel(level)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    try:
        keys = resolve_dataset_keys(args.datasets)
    except ValueError as exc:
        log.error("%s", exc)
        return EXIT_USAGE
    if not keys:
        log.error("no datasets selected")
        return EXIT_USAGE
    paths = Paths(root=Path(args.root).expanduser().resolve()) if args.root else Paths()
    opts = Options(paths=paths, force=args.force, copy=args.copy, roboflow_key=args.roboflow_key, retries=args.retries,
                   timeout=args.timeout, preview=args.preview, convert_only=args.convert_only,
                   skip_convert=args.skip_convert, val_ratio=args.val_ratio)
    plan = build_plan(keys, opts)
    if args.dry_run:
        print(format_plan(plan, opts))
        return EXIT_OK
    paths.mkdirs()
    results = [process_dataset(e.spec, opts) for e in plan]
    print(format_summary(results))
    return exit_code(results)


if __name__ == "__main__":
    sys.exit(main())
