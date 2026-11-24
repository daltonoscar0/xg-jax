"""Turn the StatsBomb open-data JSON dump into one shot-level table.

Event loading is delegated to `sportio <https://github.com/daltonoscar0/sportio>`_, which
already normalises StatsBomb into a canonical ``events`` frame: metres on a 105 x 68 pitch,
the attacking team always going +x, and every shot sub-dict (freeze frame, technique,
key-pass id, StatsBomb's own xG) preserved in the JSON ``extra`` column. This module reads
the local clone match by match, keeps the shots, and flattens ``extra`` into columns.

Nothing here looks past the shot event. ``shot_statsbomb_xg`` is carried along as a
comparison column and is never offered to the feature builder.
"""

from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import sportio

__all__ = [
    "OPEN_DATA_REPO",
    "build_shot_table",
    "competitions",
    "data_summary",
    "default_archive",
    "default_root",
    "match_index",
    "open_data_commit",
    "shots_for_match",
]

OPEN_DATA_REPO = "https://github.com/statsbomb/open-data.git"

#: StatsBomb shot outcome whose name means the ball went in.
GOAL_OUTCOME = "goal"

#: Shot sub-dict flags StatsBomb writes only when true.
SHOT_FLAGS = ("first_time", "one_on_one", "open_goal", "aerial_won", "deflected",
              "follows_dribble", "redirect", "saved_to_post", "saved_off_target")

#: Key-pass attributes that describe the assist, all known before the shot is taken.
PASS_FLAGS = ("cross", "cut_back", "through_ball", "switch", "shot_assist", "goal_assist")


def default_root() -> Path:
    """Where ``xg fetch`` puts the open-data clone."""
    env = os.environ.get("XG_DATA_ROOT")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2] / "data" / "raw" / "open-data"


def default_archive(root: Path | None = None) -> Path:
    """The downloaded tarball.

    The open data is 16 GB once unpacked, 12.8 GB of it event JSON, for about 150,000
    shots. Keeping it compressed and streaming the event files out in chunks turns a
    16 GB working set into 1.7 GB, which is the difference between this running on a
    laptop and not. Pass ``--extract`` to ``xg fetch`` if you would rather have the
    files on disk.
    """
    return Path(root or default_root()).parent / "open-data.tar.gz"


def open_data_commit(root: Path | None = None) -> str:
    """The open-data commit the local clone is pinned to, as recorded by ``xg fetch``."""
    root = Path(root or default_root())
    stamp = root.parent / "open_data_commit.txt"
    if stamp.exists():
        return stamp.read_text().strip()
    return "unknown"


def competitions(root: Path | None = None) -> pd.DataFrame:
    """Every competition/season pair in the local clone."""
    root = Path(root or default_root())
    payload = json.loads((root / "data" / "competitions.json").read_text())
    return pd.DataFrame(payload)


def match_index(root: Path | None = None) -> pd.DataFrame:
    """One row per match across every competition and season, with file paths attached."""
    root = Path(root or default_root())
    rows = []
    for comp in competitions(root).to_dict(orient="records"):
        cid, sid = int(comp["competition_id"]), int(comp["season_id"])
        path = root / "data" / "matches" / str(cid) / f"{sid}.json"
        if not path.exists():
            continue
        for match in json.loads(path.read_text()):
            mid = int(match["match_id"])
            events = root / "data" / "events" / f"{mid}.json"
            rows.append({
                "match_id": mid,
                "competition_id": cid,
                "season_id": sid,
                "competition": comp["competition_name"],
                "country": comp["country_name"],
                "gender": comp.get("competition_gender"),
                "season": comp["season_name"],
                "comp_season": f"{comp['competition_name']} {comp['season_name']}",
                "match_date": match.get("match_date"),
                "home_team": (match.get("home_team") or {}).get("home_team_name"),
                "away_team": (match.get("away_team") or {}).get("away_team_name"),
                "events_file": str(events) if events.exists() else None,
                "three_sixty_file": None,
            })
    index = pd.DataFrame(rows).drop_duplicates(subset="match_id")
    return index.sort_values("match_id").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# per-match shot extraction
# --------------------------------------------------------------------------------------


def shots_for_match(match: dict) -> pd.DataFrame:
    """Every non-shootout shot in one match, flattened.

    ``match`` is one record from :func:`match_index`. Returns an empty frame rather than
    raising when a file is malformed, so one bad match cannot sink a 3,000-match build.
    """
    try:
        events = sportio.load_events(
            "statsbomb",
            competition=match["comp_season"],
            season=str(match["season"]),
            match=match["match_id"],
            events_file=match["events_file"],
            three_sixty_file=match.get("three_sixty_file"),
            # Landmark rather than linear: a linear 120x80 -> 105x68 rescale shrinks the
            # goal to 6.80 m and the penalty spot to 10.50 m, and distance to goal is the
            # feature the whole model turns on.
            pitch="landmark",
            frame="possession",
        )
    except Exception:  # noqa: BLE001 - a single unreadable match must not stop the build
        return pd.DataFrame()

    extras = events["extra"].map(_loads)
    by_id = {blob.get("id"): (row, blob)
             for row, blob in zip(events.to_dict("records"), extras, strict=True)
             if blob.get("id")}

    rows = [
        _shot_row(row, blob, match, by_id)
        for row, blob in zip(events.to_dict("records"), extras, strict=True)
        # Period 5 is the penalty shootout; those are not shots from open play at all.
        if row["type"] == "shot" and row["period"] != 5
    ]
    return pd.DataFrame(rows)


def _loads(blob: object) -> dict:
    if not isinstance(blob, str) or not blob:
        return {}
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return {}


def _shot_row(row: dict, blob: dict, match: dict, by_id: dict) -> dict:
    out = {
        "shot_id": blob.get("id"),
        "match_id": match["match_id"],
        "competition_id": match["competition_id"],
        "season_id": match["season_id"],
        "comp_season": match["comp_season"],
        "competition": match["competition"],
        "season": match["season"],
        "gender": match["gender"],
        "team_id": row["team_id"],
        "team": row["team"],
        "player_id": row["player_id"],
        "player": row["player"],
        "period": row["period"],
        "minute": blob.get("minute"),
        "second": blob.get("second"),
        "position": blob.get("position"),
        "x": row["x"],
        "y": row["y"],
        "shot_type": row["subtype"] or "open_play",
        "technique": _snake(blob.get("technique")),
        "play_pattern": blob.get("play_pattern"),
        "body_part": row["body_part"],
        "outcome": row["outcome"],
        "goal": int(row["outcome"] == GOAL_OUTCOME),
        "under_pressure": bool(row["pressure"]),
        # Comparison only. Never a feature -- see features.LEAKY_COLUMNS.
        "shot_statsbomb_xg": blob.get("statsbomb_xg"),
    }
    for flag in SHOT_FLAGS:
        out[f"shot_{flag}"] = bool(blob.get(flag, False))

    out.update(_key_pass(blob.get("key_pass_id"), by_id))
    out["freeze_frame"] = json.dumps(blob.get("shot_freeze_frame") or [],
                                     separators=(",", ":"))
    out["has_freeze_frame"] = bool(blob.get("shot_freeze_frame"))
    return out


def _key_pass(key_pass_id: object, by_id: dict) -> dict:
    """Assist attributes from the pass that set the shot up, if there was one."""
    out = {"has_key_pass": False, "pass_height": None, "pass_type": None,
           "pass_technique": None, "pass_length": None}
    out.update({f"pass_{flag}": False for flag in PASS_FLAGS})

    found = by_id.get(key_pass_id)
    if found is None:
        return out
    prow, pblob = found
    out["has_key_pass"] = True
    out["pass_height"] = _snake(pblob.get("height"))
    out["pass_type"] = prow["subtype"]
    out["pass_technique"] = _snake(pblob.get("technique"))
    out["pass_length"] = pblob.get("length")
    for flag in PASS_FLAGS:
        out[f"pass_{flag}"] = bool(pblob.get(flag, False))
    return out


def _snake(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip().lower().replace(" ", "_").replace("-", "_")


# --------------------------------------------------------------------------------------
# the whole dump
# --------------------------------------------------------------------------------------


#: Matches unpacked from the archive at a time. 200 event files is roughly 600 MB on
#: disk, which keeps the transient working set small while still giving the worker pool
#: enough to chew on.
CHUNK_MATCHES = 200


def build_shot_table(
    root: Path | None = None,
    *,
    cache: Path | None = None,
    archive: Path | None = None,
    workers: int | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Every shot in the open data, cached to parquet.

    Reads extracted event files when they are on disk, and otherwise streams them out of
    the downloaded tarball a chunk at a time. The cache is gitignored: the StatsBomb user
    agreement does not permit redistributing the data, and a shot table is still the data.
    """
    root = Path(root or default_root())
    cache = Path(cache) if cache else root.parent.parent / "cache" / "shots.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    index = match_index(root)
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    extracted = index["events_file"].notna()

    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        if extracted.any():
            records = index.loc[extracted].to_dict(orient="records")
            print(f"reading {len(records)} extracted matches")
            frames += _run(pool, records, len(records), 0)

        missing = index.loc[~extracted]
        if not missing.empty:
            archive = Path(archive) if archive else default_archive(root)
            if not archive.exists():
                raise FileNotFoundError(
                    f"{len(missing)} matches are neither extracted nor in an archive; "
                    f"run `xg fetch` (expected {archive})"
                )
            print(f"streaming {len(missing)} matches from {archive.name}")
            lookup = {int(r["match_id"]): r for r in missing.to_dict(orient="records")}
            done = 0
            for batch, tmp in _archive_chunks(archive, lookup):
                frames += _run(pool, batch, len(lookup), done)
                done += len(batch)
                shutil.rmtree(tmp, ignore_errors=True)

    if not frames:
        raise RuntimeError("no shots were extracted; is the open data present?")
    shots = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    shots.to_parquet(cache, index=False)
    print(f"{len(shots):,} shots -> {cache}")
    return shots


def _run(pool, records: list[dict], total: int, done: int) -> list[pd.DataFrame]:
    out = []
    for i, frame in enumerate(pool.map(shots_for_match, records, chunksize=4), start=1):
        if not frame.empty:
            out.append(frame)
        if (done + i) % 500 == 0:
            print(f"  {done + i}/{total} matches", flush=True)
    return out


def _archive_chunks(archive: Path, lookup: dict[int, dict]):
    """Yield ``(records, tmpdir)`` batches of event files unpacked from the tarball.

    ``r|gz`` is the streaming mode: the archive is read once, front to back, and never
    seeks. Each batch is written to its own temp directory so the caller can delete it
    as soon as the workers are done with it.
    """
    staging = archive.parent / "_stream"
    staging.mkdir(parents=True, exist_ok=True)
    batch: list[dict] = []
    tmp = Path(tempfile.mkdtemp(dir=staging))
    try:
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                parts = member.name.split("/")
                if len(parts) < 4 or parts[-2] != "events":
                    continue
                try:
                    match_id = int(Path(parts[-1]).stem)
                except ValueError:
                    continue
                record = lookup.get(match_id)
                if record is None:
                    continue

                dest = tmp / f"{match_id}.json"
                source = tar.extractfile(member)
                if source is None:
                    continue
                with source, dest.open("wb") as handle:
                    shutil.copyfileobj(source, handle, length=1 << 20)
                batch.append({**record, "events_file": str(dest)})

                if len(batch) >= CHUNK_MATCHES:
                    yield batch, tmp
                    batch, tmp = [], Path(tempfile.mkdtemp(dir=staging))
        if batch:
            yield batch, tmp
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def data_summary(shots: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Per competition-season counts, for ``results/data_summary.csv``."""
    matches = (index.groupby(["competition", "season"], as_index=False)
               .agg(matches=("match_id", "nunique")))
    rows = (shots.assign(is_penalty=shots["shot_type"].eq("penalty"))
            .groupby(["competition", "season"], as_index=False)
            .agg(shots=("shot_id", "size"),
                 goals=("goal", "sum"),
                 penalties=("is_penalty", "sum"),
                 with_freeze_frame=("has_freeze_frame", "sum")))
    out = matches.merge(rows, on=["competition", "season"], how="left").fillna(0)
    for col in ("shots", "goals", "penalties", "with_freeze_frame"):
        out[col] = out[col].astype("int64")
    out["non_penalty_shots"] = out["shots"] - out["penalties"]
    out["freeze_frame_pct"] = (100 * out["with_freeze_frame"]
                               / out["shots"].clip(lower=1)).round(1)
    return out.sort_values(["competition", "season"]).reset_index(drop=True)


def iter_match_files(root: Path | None = None) -> Iterator[Path]:
    root = Path(root or default_root())
    yield from sorted((root / "data" / "events").glob("*.json"))
