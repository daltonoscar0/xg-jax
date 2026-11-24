"""Download the StatsBomb open data reproducibly.

The commit the download came from is resolved first and written to
``data/raw/open_data_commit.txt``, so a run can be reproduced exactly even after
StatsBomb add a competition. Pass ``--commit`` to pin an earlier one.

The data itself is never committed to this repository. The StatsBomb Public Data User
Agreement (clause 1.2.1) forbids distributing or reproducing the data to third parties;
a public repo carrying the JSON would be doing exactly that.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

__all__ = ["CODELOAD", "fetch_open_data", "resolve_commit"]

API = "https://api.github.com/repos/statsbomb/open-data/commits/master"
CODELOAD = "https://codeload.github.com/statsbomb/open-data/tar.gz/{commit}"
USER_AGENT = "xg-jax (+https://github.com/daltonoscar0/xg-jax)"


def resolve_commit() -> str:
    """The current ``master`` commit of statsbomb/open-data.

    Tries the unauthenticated API first and falls back to the ``gh`` CLI, because the
    anonymous API rate limit is low enough to hit on a second run in the same hour.
    """
    request = urllib.request.Request(API, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/vnd.github+json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)["sha"]
    except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError):
        pass
    try:
        out = subprocess.run(["gh", "api", "repos/statsbomb/open-data/commits/master",
                              "--jq", ".sha"], capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "could not resolve the open-data commit; pass --commit explicitly"
        ) from exc


#: Small directories unpacked eagerly: competition and match metadata, about 12 MB.
#: The 12.8 GB of event JSON stays inside the archive and is streamed on demand.
METADATA_DIRS = ("matches",)
METADATA_FILES = ("competitions.json",)


def fetch_open_data(
    root: Path,
    *,
    commit: str | None = None,
    force: bool = False,
    extract_events: bool = False,
) -> str:
    """Download the open data under ``root``'s parent. Returns the commit used.

    By default only the competition and match metadata is unpacked and the event files
    are left compressed, because unpacked they are 12.8 GB. Pass ``extract_events`` if
    you have the disk and would rather have them as files.
    """
    root = Path(root)
    archive = root.parent / "open-data.tar.gz"
    stamp = root.parent / "open_data_commit.txt"
    commit = commit or resolve_commit()

    have = stamp.read_text().strip() if stamp.exists() else ""
    if have == commit and archive.exists() and (root / "data" / "competitions.json").exists() \
            and not force:
        print(f"already at {commit[:10]}; nothing to do")
        return commit

    root.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists() or force:
        url = CODELOAD.format(commit=commit)
        print(f"downloading {url}")
        partial = archive.with_suffix(".part")
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(request, timeout=300) as response, \
                partial.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1 << 20)
        partial.replace(archive)
    print(f"  archive {archive.stat().st_size / 1e9:.2f} GB")

    wanted = _extract(archive, root, extract_events=extract_events)
    print(f"  unpacked {wanted} metadata files")

    stamp.write_text(commit + "\n")
    print(f"open data at {root} (commit {commit})")
    return commit


def _extract(archive: Path, root: Path, *, extract_events: bool) -> int:
    """Unpack the metadata (and optionally the events) from the streamed archive."""
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve()
    count = 0
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            # Strip the "open-data-<sha>/" prefix GitHub puts on tarball members.
            relative = Path(*Path(member.name).parts[1:])
            parts = relative.parts
            if len(parts) < 2 or parts[0] != "data":
                continue
            keep = parts[1] in METADATA_DIRS or parts[-1] in METADATA_FILES
            if not keep and not (extract_events and parts[1] == "events"):
                continue

            target = (root / relative).resolve()
            if not str(target).startswith(str(resolved)):
                raise RuntimeError(f"refusing path traversal in archive: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            with source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=1 << 20)
            count += 1
    return count
