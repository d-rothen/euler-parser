"""
Hierarchy Parser Module

Provides utilities for placing and retrieving values in hierarchically
structured data with nested children dictionaries and file entries.
"""

from typing import Any, Optional


def set_value(
    target: dict,
    hierarchy: list[str],
    file_id: str,
    value: Any
) -> dict:
    """
    Place a value object into the target hierarchy at the specified location.

    Creates intermediate hierarchy levels if they don't exist.
    Adds or updates the file entry with the given id.

    Args:
        target: The target object to mutate
        hierarchy: List of keys forming the path (e.g., ["level_1", "level_2"])
        file_id: The id of the file entry to set
        value: The value object to place (should contain at least 'path')

    Returns:
        The mutated target object
    """
    # Navigate/create the hierarchy path
    current = target

    for level in hierarchy:
        if "children" not in current:
            current["children"] = {}
        if level not in current["children"]:
            current["children"][level] = {}
        current = current["children"][level]

    # Ensure files array exists
    if "files" not in current:
        current["files"] = []

    # Find existing entry with this id or add new one
    for i, file_entry in enumerate(current["files"]):
        if isinstance(file_entry, dict) and file_entry.get("id") == file_id:
            current["files"][i] = value
            return target

    # No existing entry found, append new one
    current["files"].append(value)
    return target


def retrieve_value(
    source: dict,
    hierarchy: list[str],
    file_id: str
) -> Optional[Any]:
    """
    Retrieve the value at a specific hierarchy path and file id.

    Args:
        source: The source object to search
        hierarchy: List of keys forming the path
        file_id: The id of the file entry to retrieve

    Returns:
        The file entry object if found, None otherwise
    """
    current = source

    # Navigate the hierarchy
    for level in hierarchy:
        children = current.get("children", {})
        if not isinstance(children, dict) or level not in children:
            return None
        current = children[level]
        if not isinstance(current, dict):
            return None

    # Search for file with matching id
    files = current.get("files", [])
    for file_entry in files:
        if isinstance(file_entry, dict) and file_entry.get("id") == file_id:
            return file_entry

    return None


def retrieve_values(
    source: dict,
    hierarchy: list[str]
) -> Optional[dict]:
    """
    Retrieve the mapping (node) at a specific hierarchy level.

    Args:
        source: The source object to search
        hierarchy: List of keys forming the path

    Returns:
        The node at that hierarchy level, or None if path doesn't exist
    """
    current = source

    # Navigate the hierarchy
    for level in hierarchy:
        children = current.get("children", {})
        if not isinstance(children, dict) or level not in children:
            return None
        current = children[level]
        if not isinstance(current, dict):
            return None

    return current
