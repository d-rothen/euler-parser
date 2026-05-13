"""Helpers for euler-loading modality path selectors in config files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl

from ds_crawler import validate_metadata_scope
from ds_crawler.zip_utils import validate_split_name
from euler_loading import Modality


@dataclass(frozen=True)
class ModalityPath:
    path: str
    split: str | None = None
    metadata_scope: str | None = None


def _parse_inline_scope(path: str) -> tuple[str, str | None]:
    candidate_path, sep, fragment = path.rpartition("#")
    if not sep:
        return path, None

    params = parse_qsl(fragment, keep_blank_values=True)
    scope_values = [value for key, value in params if key == "scope"]
    if not scope_values:
        return path, None

    unsupported = [key for key, _ in params if key != "scope"]
    if unsupported:
        raise ValueError(
            "Unsupported modality path fragment parameter(s): "
            f"{', '.join(sorted(set(unsupported)))}. "
            "Only '#scope=<metadata_scope>' is supported."
        )
    if len(scope_values) > 1:
        raise ValueError("Modality path may contain at most one '#scope=' selector.")

    return candidate_path, validate_metadata_scope(scope_values[0])


def _parse_inline_split(path: str) -> tuple[str, str | None]:
    colon_pos = path.rfind(":")
    if colon_pos <= 1:
        return path, None

    candidate_split = path[colon_pos + 1 :]
    candidate_path = path[:colon_pos]

    try:
        split = validate_split_name(candidate_split)
    except ValueError:
        return path, None

    return candidate_path, split


def parse_modality_path(
    path: str | Path,
    *,
    split: str | None = None,
    metadata_scope: str | None = None,
) -> ModalityPath:
    """Parse euler-loading inline split/scope selectors from a config path."""
    raw_path = str(path)
    parsed_path, inline_scope = _parse_inline_scope(raw_path)
    parsed_path, inline_split = _parse_inline_split(parsed_path)

    if inline_split is not None and split is not None:
        raise ValueError(
            f"Modality path {raw_path!r} contains an inline split "
            f"({inline_split!r}) but an explicit split={split!r} was also "
            "provided. Use one or the other, not both."
        )

    normalized_scope = (
        validate_metadata_scope(metadata_scope)
        if metadata_scope is not None
        else None
    )
    if inline_scope is not None:
        if normalized_scope is not None and normalized_scope != inline_scope:
            raise ValueError(
                f"Modality path {raw_path!r} contains an inline metadata scope "
                f"({inline_scope!r}) but an explicit "
                f"metadata_scope={metadata_scope!r} was also provided. Use one "
                "metadata scope selector."
            )
        normalized_scope = inline_scope

    return ModalityPath(
        path=parsed_path,
        split=split if split is not None else inline_split,
        metadata_scope=normalized_scope,
    )


def normalize_modality_path(path: str | Path, *, split: str | None = None) -> Path:
    """Return the filesystem/archive root for a config modality path."""
    return Path(parse_modality_path(path, split=split).path)


def build_modality(
    *,
    path: str | Path,
    modality_key: str,
    split: str | None = None,
    loader=None,
    used_as: str | None = None,
) -> Modality:
    """Create a Modality after parsing inline config selectors locally."""
    parsed = parse_modality_path(
        path,
        split=split,
    )
    return Modality(
        path=parsed.path,
        loader=loader,
        used_as=used_as,
        modality_type=modality_key,
        metadata_scope=parsed.metadata_scope or modality_key,
        split=parsed.split,
    )
