"""Tests for save_results with zip and directory multiplexing."""

import json
import zipfile

import pytest

from euler_eval.cli import _find_zip_ancestor, _save_json_to_zip, save_results
from pathlib import Path


# ---------------------------------------------------------------------------
# _find_zip_ancestor
# ---------------------------------------------------------------------------


class TestFindZipAncestor:
    def test_no_zip_in_path(self, tmp_path):
        p = tmp_path / "some" / "dir" / "eval.json"
        assert _find_zip_ancestor(p) == (None, None)

    def test_zip_file_detected(self, tmp_path):
        zp = tmp_path / "dataset.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "hello")

        p = zp / "eval.json"
        zip_path, internal = _find_zip_ancestor(p)
        assert zip_path == zp
        assert internal == "eval.json"

    def test_zip_with_nested_internal_path(self, tmp_path):
        zp = tmp_path / "data.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "hello")

        p = zp / "sub" / "dir" / "eval.json"
        zip_path, internal = _find_zip_ancestor(p)
        assert zip_path == zp
        assert internal == "sub/dir/eval.json"

    def test_zip_suffix_but_not_a_file(self, tmp_path):
        """A directory named .zip should not be detected."""
        fake_zip = tmp_path / "not_a.zip"
        fake_zip.mkdir()
        p = fake_zip / "eval.json"
        assert _find_zip_ancestor(p) == (None, None)

    def test_zip_is_the_leaf(self, tmp_path):
        """If the path itself ends in .zip with no child, internal is empty."""
        zp = tmp_path / "archive.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "hello")
        zip_path, internal = _find_zip_ancestor(zp)
        assert zip_path == zp
        assert internal == ""


# ---------------------------------------------------------------------------
# _save_json_to_zip
# ---------------------------------------------------------------------------


class TestSaveJsonToZip:
    def test_append_to_zip(self, tmp_path):
        zp = tmp_path / "archive.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("existing.txt", "keep me")

        data = {"score": 0.95}
        _save_json_to_zip(zp, "eval.json", data)

        with zipfile.ZipFile(zp, "r") as zf:
            assert "existing.txt" in zf.namelist()
            assert "eval.json" in zf.namelist()
            result = json.loads(zf.read("eval.json"))
            assert result == data

    def test_replace_existing_entry(self, tmp_path):
        zp = tmp_path / "archive.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("existing.txt", "keep me")
            zf.writestr("eval.json", '{"old": true}')

        new_data = {"new": True, "score": 1.0}
        _save_json_to_zip(zp, "eval.json", new_data)

        with zipfile.ZipFile(zp, "r") as zf:
            names = zf.namelist()
            # No duplicates
            assert names.count("eval.json") == 1
            assert "existing.txt" in names
            result = json.loads(zf.read("eval.json"))
            assert result == new_data

    def test_preserves_other_entries(self, tmp_path):
        zp = tmp_path / "archive.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("a.txt", "aaa")
            zf.writestr("b.txt", "bbb")
            zf.writestr("eval.json", '{"old": true}')

        _save_json_to_zip(zp, "eval.json", {"replaced": True})

        with zipfile.ZipFile(zp, "r") as zf:
            assert zf.read("a.txt") == b"aaa"
            assert zf.read("b.txt") == b"bbb"
            assert json.loads(zf.read("eval.json")) == {"replaced": True}


# ---------------------------------------------------------------------------
# save_results  (directory path)
# ---------------------------------------------------------------------------


class TestSaveResultsDirectory:
    def test_saves_to_depth_path(self, tmp_path):
        depth_dir = tmp_path / "pred_depth"
        depth_dir.mkdir()

        results = {"depth": {"rmse": 0.5}}
        config = {"name": "model", "depth": {"path": str(depth_dir)}}
        out = save_results(results, config)

        assert out == depth_dir / "eval.json"
        assert out.exists()
        assert json.loads(out.read_text()) == results

    def test_saves_to_rgb_path_when_no_depth(self, tmp_path):
        rgb_dir = tmp_path / "pred_rgb"
        rgb_dir.mkdir()

        results = {"rgb": {"psnr": 30.0}}
        config = {"name": "model", "rgb": {"path": str(rgb_dir)}}
        out = save_results(results, config)

        assert out == rgb_dir / "eval.json"

    def test_explicit_output_file(self, tmp_path):
        out_file = tmp_path / "custom" / "results.json"
        results = {"depth": {"rmse": 0.5}}
        config = {"name": "model", "output_file": str(out_file)}
        out = save_results(results, config)

        assert out == out_file
        assert out.exists()

    def test_fallback_to_cwd(self):
        results = {"depth": {"rmse": 0.5}}
        config = {"name": "model"}
        out = save_results(results, config)

        assert out == Path("eval.json")
        assert out.exists()
        out.unlink()  # cleanup

    def test_modality_selects_specific_path(self, tmp_path):
        """When modality is set, save to that modality's path."""
        depth_dir = tmp_path / "pred_depth"
        rgb_dir = tmp_path / "pred_rgb"
        depth_dir.mkdir()
        rgb_dir.mkdir()

        config = {
            "name": "model",
            "depth": {"path": str(depth_dir)},
            "rgb": {"path": str(rgb_dir)},
        }

        depth_results = {"depth": {"rmse": 0.5}}
        rgb_results = {"rgb": {"psnr": 30.0}}

        depth_out = save_results(depth_results, config, modality="depth")
        rgb_out = save_results(rgb_results, config, modality="rgb")

        assert depth_out == depth_dir / "eval.json"
        assert rgb_out == rgb_dir / "eval.json"
        assert json.loads(depth_out.read_text()) == depth_results
        assert json.loads(rgb_out.read_text()) == rgb_results

    def test_modality_ignored_when_output_file_set(self, tmp_path):
        """Explicit output_file takes precedence over modality."""
        out_file = tmp_path / "custom" / "results.json"
        config = {
            "name": "model",
            "depth": {"path": str(tmp_path / "depth")},
            "output_file": str(out_file),
        }
        out = save_results({"depth": {"rmse": 0.5}}, config, modality="depth")
        assert out == out_file


# ---------------------------------------------------------------------------
# save_results  (zip path)
# ---------------------------------------------------------------------------


class TestSaveResultsZip:
    def test_writes_eval_json_into_zip(self, tmp_path):
        zp = tmp_path / "dataset.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("image_001.png", b"\x89PNG")

        results = {"depth": {"rmse": 0.42}}
        config = {"name": "model", "depth": {"path": str(zp)}}
        out = save_results(results, config)

        # Return value is the logical path
        assert str(zp) in str(out)

        # eval.json is inside the zip
        with zipfile.ZipFile(zp, "r") as zf:
            assert "eval.json" in zf.namelist()
            stored = json.loads(zf.read("eval.json"))
            assert stored == results

    def test_preserves_existing_zip_contents(self, tmp_path):
        zp = tmp_path / "dataset.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("image_001.png", b"\x89PNG")
            zf.writestr("image_002.png", b"\x89PNG")

        results = {"rgb": {"psnr": 28.5}}
        config = {"name": "model", "depth": {"path": str(zp)}}
        save_results(results, config)

        with zipfile.ZipFile(zp, "r") as zf:
            names = zf.namelist()
            assert "image_001.png" in names
            assert "image_002.png" in names
            assert "eval.json" in names

    def test_replaces_existing_eval_json_in_zip(self, tmp_path):
        zp = tmp_path / "dataset.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("image_001.png", b"\x89PNG")
            zf.writestr("eval.json", '{"old": true}')

        new_results = {"depth": {"rmse": 0.1}}
        config = {"name": "model", "depth": {"path": str(zp)}}
        save_results(new_results, config)

        with zipfile.ZipFile(zp, "r") as zf:
            assert zf.namelist().count("eval.json") == 1
            stored = json.loads(zf.read("eval.json"))
            assert stored == new_results

    def test_explicit_output_through_zip(self, tmp_path):
        """When output_file explicitly passes through a zip."""
        zp = tmp_path / "output.zip"
        with zipfile.ZipFile(zp, "w") as zf:
            zf.writestr("dummy.txt", "x")

        out_path = str(zp / "results" / "eval.json")
        results = {"depth": {"rmse": 0.3}}
        config = {"name": "model", "output_file": out_path}
        save_results(results, config)

        with zipfile.ZipFile(zp, "r") as zf:
            assert "results/eval.json" in zf.namelist()
            stored = json.loads(zf.read("results/eval.json"))
            assert stored == results
