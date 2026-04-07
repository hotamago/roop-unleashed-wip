import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import roop.globals
import ui.globals
from roop.FaceSet import FaceSet
from roop.ProcessEntry import ProcessEntry
import ui.tabs.faceswap_tab as faceswap_tab


def make_face(mask_offsets=None):
    return SimpleNamespace(mask_offsets=list(mask_offsets or faceswap_tab.default_mask_offsets()))


def test_build_resume_payload_captures_sources_targets_and_settings(monkeypatch):
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    roop.globals.INPUT_FACESETS.append(source_face_set)
    roop.globals.TARGET_FACES.append(SimpleNamespace(embedding=[0.1, 0.2, 0.3]))
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": "C:/source.png", "face_index": 0})
    ui.globals.ui_target_face_refs.append({"path": "C:/target.mp4", "frame_number": 12, "face_index": 1})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 5, 25, 30.0)]
    faceswap_tab.selected_preview_index = 0
    faceswap_tab.SELECTED_INPUT_FACE_INDEX = 0
    faceswap_tab.SELECTED_TARGET_FACE_INDEX = 0

    payload = faceswap_tab.build_resume_payload({"output_method": "File", "detection": "Selected face"})

    assert payload["sources"][0]["type"] == "image_face"
    assert payload["sources"][0]["mask_offsets"] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert payload["targets"]["files"][0]["startframe"] == 5
    assert payload["targets"]["selected_faces"][0]["frame_number"] == 12
    assert np.allclose(payload["targets"]["selected_faces"][0]["face_embedding"], [0.1, 0.2, 0.3])
    assert payload["settings"]["output_method"] == "File"


def test_write_and_read_resume_payload_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": "C:/source.png", "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    payload = faceswap_tab.build_resume_payload({"output_method": "File"})
    resume_path = faceswap_tab.write_resume_payload(payload)
    reloaded = faceswap_tab.read_resume_payload(resume_path)

    assert resume_path.endswith(".json")
    assert reloaded["version"] == faceswap_tab.RESUME_CACHE_VERSION
    assert reloaded["targets"]["files"][0]["filename"].endswith("target.mp4")


def test_resume_payload_signature_ignores_performance_only_settings():
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    roop.globals.TARGET_FACES.append(SimpleNamespace())
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": "C:/source.png", "face_index": 0})
    ui.globals.ui_target_face_refs.append({"path": "C:/target.mp4", "frame_number": 12, "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    payload_a = faceswap_tab.build_resume_payload({"output_method": "File"})
    payload_b = faceswap_tab.build_resume_payload({"output_method": "File"})
    payload_a["settings"]["mask_batch_size"] = 32
    payload_b["settings"]["mask_batch_size"] = 84
    payload_b["settings"]["swap_batch_size"] = 16

    assert faceswap_tab.get_resume_payload_signature(payload_a) == faceswap_tab.get_resume_payload_signature(payload_b)


def test_resume_job_signature_ignores_settings_changes():
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    roop.globals.TARGET_FACES.append(SimpleNamespace())
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": "C:/source.png", "face_index": 0})
    ui.globals.ui_target_face_refs.append({"path": "C:/target.mp4", "frame_number": 12, "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    payload_a = faceswap_tab.build_resume_payload({"output_method": "File", "enhancer": "GFPGAN"})
    payload_b = faceswap_tab.build_resume_payload({"output_method": "Virtual Camera", "enhancer": "CodeFormer"})

    assert faceswap_tab.get_resume_payload_signature(payload_a) != faceswap_tab.get_resume_payload_signature(payload_b)
    assert faceswap_tab.get_resume_job_signature(payload_a) == faceswap_tab.get_resume_job_signature(payload_b)


def test_resume_signatures_change_when_selected_target_face_identity_changes():
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": "C:/source.png", "face_index": 0})
    ui.globals.ui_target_face_refs.append({"path": "C:/target.mp4", "frame_number": 12, "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    roop.globals.TARGET_FACES[:] = [SimpleNamespace(embedding=np.array([1.0, 0.0], dtype=np.float32))]
    payload_a = faceswap_tab.build_resume_payload({"output_method": "File"})

    roop.globals.TARGET_FACES[:] = [SimpleNamespace(embedding=np.array([0.0, 1.0], dtype=np.float32))]
    payload_b = faceswap_tab.build_resume_payload({"output_method": "File"})

    assert faceswap_tab.get_resume_payload_signature(payload_a) != faceswap_tab.get_resume_payload_signature(payload_b)
    assert faceswap_tab.get_resume_job_signature(payload_a) != faceswap_tab.get_resume_job_signature(payload_b)


def test_write_resume_payload_snapshots_source_files_into_resume_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_path = tmp_path / "gradio" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source-image")
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": str(source_path), "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    payload = faceswap_tab.build_resume_payload({"output_method": "File"})
    resume_path = faceswap_tab.write_resume_payload(payload)
    reloaded = faceswap_tab.read_resume_payload(resume_path)

    cached_path = reloaded["sources"][0]["resume_cached_path"]
    assert cached_path.startswith(str(tmp_path))
    assert Path(cached_path).read_bytes() == b"source-image"


def test_restore_input_faces_from_resume_uses_cached_snapshot_when_original_path_is_missing(tmp_path, monkeypatch):
    cached_source = tmp_path / "resume_cache" / "cached-source.png"
    cached_source.parent.mkdir(parents=True, exist_ok=True)
    cached_source.write_bytes(b"cached-image")
    restored_paths = []

    def fake_append_image_source(source_path, source_ref):
        restored_paths.append(source_path)
        face_set = FaceSet()
        face_set.faces.append(make_face(source_ref.get("mask_offsets")))
        roop.globals.INPUT_FACESETS.append(face_set)
        ui.globals.ui_input_face_refs.append(dict(source_ref))

    monkeypatch.setattr(faceswap_tab, "append_image_source", fake_append_image_source)

    faceswap_tab.restore_input_faces_from_resume([
        {
            "type": "image_face",
            "path": str(tmp_path / "missing" / "source.png"),
            "resume_cached_path": str(cached_source),
            "face_index": 0,
            "mask_offsets": list(faceswap_tab.default_mask_offsets()),
        }
    ])

    assert restored_paths == [str(cached_source)]


def test_write_resume_payload_prefers_ui_resume_bound_path_over_last_path(tmp_path, monkeypatch):
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_path = tmp_path / "gradio" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source-image")
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": str(source_path), "face_index": 0})
    roop.globals.TARGET_FACES.append(SimpleNamespace(embedding=np.array([0.1, 0.2], dtype=np.float32)))
    ui.globals.ui_target_face_refs.append({"path": "C:/target.mp4", "frame_number": 0, "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    bound_file = tmp_path / "20260406_8d8d244889be.json"
    bound_file.write_text("{}", encoding="utf-8")
    wrong_last = tmp_path / "wrong_other.json"
    wrong_last.write_text("{}", encoding="utf-8")
    ui.globals.ui_resume_bound_path = str(bound_file)
    ui.globals.ui_resume_last_path = str(wrong_last)

    faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))

    assert bound_file.is_file()
    assert faceswap_tab.read_resume_payload(str(bound_file)).get("version") == faceswap_tab.RESUME_CACHE_VERSION
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_write_resume_payload_sticks_to_ui_resume_last_path_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_path = tmp_path / "gradio" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source-image")
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": str(source_path), "face_index": 0})
    roop.globals.TARGET_FACES.append(SimpleNamespace(embedding=np.array([0.5, 0.25], dtype=np.float32)))
    ui.globals.ui_target_face_refs.append({"path": "C:/target.mp4", "frame_number": 0, "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    loaded = tmp_path / "20260406_8d8d244889be.json"
    loaded.write_text("{}", encoding="utf-8")
    ui.globals.ui_resume_bound_path = None
    ui.globals.ui_resume_last_path = str(loaded)

    faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))

    assert loaded.is_file()
    assert len(list(tmp_path.glob("*.json"))) == 1
    reloaded = faceswap_tab.read_resume_payload(str(loaded))
    assert reloaded.get("version") == faceswap_tab.RESUME_CACHE_VERSION


def test_resolve_equivalent_resume_path_reuses_loaded_file_after_stripping_extra_json_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_path = tmp_path / "gradio" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source-image")
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": str(source_path), "face_index": 0})
    roop.globals.TARGET_FACES.append(SimpleNamespace(embedding=np.array([0.1, 0.2], dtype=np.float32)))
    ui.globals.ui_target_face_refs.append({"path": "C:/target.mp4", "frame_number": 12, "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    first_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))
    data = json.loads(Path(first_path).read_text(encoding="utf-8"))
    data["client_debug_meta"] = {"x": 1}
    Path(first_path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    ui.globals.ui_resume_last_path = first_path
    fresh = faceswap_tab.build_resume_payload({"output_method": "File"})
    desired_path, _ = faceswap_tab.get_resume_payload_path(fresh)
    resolved = faceswap_tab.resolve_equivalent_resume_path(fresh, desired_path)

    assert os.path.normcase(os.path.normpath(resolved)) == os.path.normcase(os.path.normpath(first_path))


def test_write_resume_payload_reuses_existing_file_for_equivalent_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(tmp_path))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_path = tmp_path / "gradio" / "source.png"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_bytes(b"source-image")
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": str(source_path), "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    now_values = iter([1111111111, 2222222222])
    stamp_values = iter(["20260101_010101", "20270101_020202"])
    monkeypatch.setattr(faceswap_tab.time, "time", lambda: next(now_values))
    monkeypatch.setattr(faceswap_tab.time, "strftime", lambda fmt: next(stamp_values))

    first_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))
    second_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))

    assert first_resume_path == second_resume_path
    assert len(list(tmp_path.glob("*.json"))) == 1
    assert len(list(tmp_path.glob("*_assets"))) == 1


def test_write_resume_payload_snapshots_transient_target_files_into_resume_cache(tmp_path, monkeypatch):
    resume_root = tmp_path / "resume_cache"
    gradio_root = tmp_path / "temp" / "gradio"
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(resume_root))
    monkeypatch.setattr(faceswap_tab, "get_gradio_temp_root", lambda: gradio_root)
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    target_path = gradio_root / "upload" / "clip.mp4"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"target-video")
    roop.globals.TARGET_FACES.append(SimpleNamespace())
    ui.globals.ui_target_face_refs.append({"path": str(target_path), "frame_number": 12, "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry(str(target_path), 0, 100, 24.0)]

    payload = faceswap_tab.build_resume_payload({"output_method": "File"})
    resume_path = faceswap_tab.write_resume_payload(payload)
    reloaded = faceswap_tab.read_resume_payload(resume_path)

    cached_file_path = reloaded["targets"]["files"][0]["resume_cached_path"]
    cached_face_path = reloaded["targets"]["selected_faces"][0]["resume_cached_path"]
    assert cached_file_path.startswith(str(resume_root))
    assert cached_face_path == cached_file_path
    assert Path(cached_file_path).read_bytes() == b"target-video"


def test_restore_process_entries_uses_cached_snapshot_when_original_path_is_missing(tmp_path):
    cached_target = tmp_path / "resume_cache" / "cached-target.mp4"
    cached_target.parent.mkdir(parents=True, exist_ok=True)
    cached_target.write_bytes(b"cached-target")

    target_paths = faceswap_tab.restore_process_entries([
        {
            "filename": str(tmp_path / "missing" / "clip.mp4"),
            "resume_cached_path": str(cached_target),
            "startframe": 5,
            "endframe": 25,
            "fps": 30.0,
        }
    ])

    assert target_paths == [str(cached_target)]
    assert faceswap_tab.list_files_process[0].filename == str(cached_target)


def test_append_target_face_from_resume_uses_cached_snapshot_when_original_path_is_missing(tmp_path, monkeypatch):
    cached_target = tmp_path / "resume_cache" / "cached-target.png"
    cached_target.parent.mkdir(parents=True, exist_ok=True)
    cached_target.write_bytes(b"cached-target")
    monkeypatch.setattr(
        faceswap_tab,
        "extract_face_images",
        lambda target_path, video_info: [(SimpleNamespace(id="face"), "thumb-image")],
    )
    monkeypatch.setattr(faceswap_tab.util, "convert_to_gradio", lambda image: image)

    faceswap_tab.append_target_face_from_resume({
        "path": str(tmp_path / "missing" / "target.png"),
        "resume_cached_path": str(cached_target),
        "frame_number": 0,
        "face_index": 0,
    })

    assert len(roop.globals.TARGET_FACES) == 1
    assert ui.globals.ui_target_face_refs[0]["path"] == str(cached_target)


def test_append_target_face_from_resume_prefers_embedding_match_over_saved_index(tmp_path, monkeypatch):
    cached_target = tmp_path / "resume_cache" / "cached-target.png"
    cached_target.parent.mkdir(parents=True, exist_ok=True)
    cached_target.write_bytes(b"cached-target")
    wrong_face = SimpleNamespace(id="wrong", embedding=np.array([1.0, 0.0], dtype=np.float32))
    right_face = SimpleNamespace(id="right", embedding=np.array([0.0, 1.0], dtype=np.float32))
    monkeypatch.setattr(
        faceswap_tab,
        "extract_face_images",
        lambda target_path, video_info: [
            (wrong_face, "wrong-thumb"),
            (right_face, "right-thumb"),
        ],
    )
    monkeypatch.setattr(faceswap_tab.util, "convert_to_gradio", lambda image: image)

    faceswap_tab.append_target_face_from_resume({
        "path": str(tmp_path / "missing" / "target.png"),
        "resume_cached_path": str(cached_target),
        "frame_number": 0,
        "face_index": 0,
        "face_embedding": [0.0, 1.0],
    })

    assert roop.globals.TARGET_FACES[0] is right_face
    assert ui.globals.ui_target_face_refs[0]["face_index"] == 1


def test_write_resume_payload_refreshes_existing_resume_file_with_missing_target_snapshot(tmp_path, monkeypatch):
    resume_root = tmp_path / "resume_cache"
    gradio_root = tmp_path / "temp" / "gradio"
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(resume_root))
    monkeypatch.setattr(faceswap_tab, "get_gradio_temp_root", lambda: gradio_root)
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    target_path = gradio_root / "upload" / "clip.mp4"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"target-video")
    faceswap_tab.list_files_process[:] = [ProcessEntry(str(target_path), 0, 100, 24.0)]

    payload = faceswap_tab.build_resume_payload({"output_method": "File"})
    resume_path, resume_key = faceswap_tab.get_resume_payload_path(payload)
    stale_payload = dict(payload)
    stale_payload["resume_key"] = resume_key
    Path(resume_path).parent.mkdir(parents=True, exist_ok=True)
    Path(resume_path).write_text(json.dumps(stale_payload, indent=2), encoding="utf-8")

    rewritten_path = faceswap_tab.write_resume_payload(payload)
    reloaded = faceswap_tab.read_resume_payload(rewritten_path)

    assert rewritten_path == resume_path
    assert reloaded["targets"]["files"][0]["resume_cached_path"].startswith(str(resume_root))


def test_write_resume_payload_keeps_same_signature_path_between_temp_and_cached_target_paths(tmp_path, monkeypatch):
    resume_root = tmp_path / "resume_cache"
    gradio_root = tmp_path / "temp" / "gradio"
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(resume_root))
    monkeypatch.setattr(faceswap_tab, "get_gradio_temp_root", lambda: gradio_root)
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    stable_source = tmp_path / "source.png"
    stable_source.write_bytes(b"source-image")
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": str(stable_source), "face_index": 0})
    target_temp = gradio_root / "upload" / "clip.mp4"
    target_temp.parent.mkdir(parents=True, exist_ok=True)
    target_temp.write_bytes(b"target-video")
    roop.globals.TARGET_FACES.append(SimpleNamespace())
    ui.globals.ui_target_face_refs.append({"path": str(target_temp), "frame_number": 12, "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry(str(target_temp), 0, 100, 24.0)]

    first_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))
    reloaded = faceswap_tab.read_resume_payload(first_resume_path)
    cached_target_path = reloaded["targets"]["files"][0]["resume_cached_path"]

    faceswap_tab.list_files_process[:] = [ProcessEntry(str(cached_target_path), 0, 100, 24.0)]
    ui.globals.ui_target_face_refs[:] = [{
        "path": str(cached_target_path),
        "frame_number": 12,
        "face_index": 0,
    }]

    second_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))

    assert first_resume_path == second_resume_path


def test_write_resume_payload_does_not_duplicate_source_assets_on_resume(tmp_path, monkeypatch):
    resume_root = tmp_path / "resume_cache"
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(resume_root))
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"source-image")
    source_face_set = FaceSet()
    source_face_set.faces.append(make_face())
    roop.globals.INPUT_FACESETS.append(source_face_set)
    ui.globals.ui_input_face_refs.append({"type": "image_face", "path": str(source_path), "face_index": 0})
    faceswap_tab.list_files_process[:] = [ProcessEntry("C:/target.mp4", 0, 100, 24.0)]

    first_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))
    reloaded = faceswap_tab.read_resume_payload(first_resume_path)
    cached_source_path = reloaded["sources"][0]["resume_cached_path"]
    ui.globals.ui_input_face_refs[:] = [{
        "type": "image_face",
        "path": str(cached_source_path),
        "resume_cached_path": str(cached_source_path),
        "face_index": 0,
        "mask_offsets": list(faceswap_tab.default_mask_offsets()),
    }]

    second_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))
    source_assets = list((Path(str(first_resume_path).replace(".json", "_assets")) / "sources").glob("*"))

    assert first_resume_path == second_resume_path
    assert len(source_assets) == 1


def test_write_resume_payload_does_not_duplicate_target_assets_on_resume(tmp_path, monkeypatch):
    resume_root = tmp_path / "resume_cache"
    gradio_root = tmp_path / "temp" / "gradio"
    monkeypatch.setattr(faceswap_tab, "get_resume_cache_root", lambda: str(resume_root))
    monkeypatch.setattr(faceswap_tab, "get_gradio_temp_root", lambda: gradio_root)
    monkeypatch.setattr(faceswap_tab, "list_resume_cache_files", lambda: [], raising=False)
    target_path = gradio_root / "upload" / "clip.mp4"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(b"target-video")
    faceswap_tab.list_files_process[:] = [ProcessEntry(str(target_path), 0, 100, 24.0)]
    roop.globals.TARGET_FACES.append(SimpleNamespace())
    ui.globals.ui_target_face_refs.append({"path": str(target_path), "frame_number": 12, "face_index": 0})

    first_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))
    reloaded = faceswap_tab.read_resume_payload(first_resume_path)
    cached_target_path = reloaded["targets"]["files"][0]["resume_cached_path"]
    faceswap_tab.list_files_process[:] = [ProcessEntry(str(cached_target_path), 0, 100, 24.0)]
    ui.globals.ui_target_face_refs[:] = [{
        "path": str(cached_target_path),
        "resume_cached_path": str(cached_target_path),
        "frame_number": 12,
        "face_index": 0,
    }]

    second_resume_path = faceswap_tab.write_resume_payload(faceswap_tab.build_resume_payload({"output_method": "File"}))
    target_assets = list((Path(str(first_resume_path).replace(".json", "_assets")) / "targets").glob("*"))

    assert first_resume_path == second_resume_path
    assert len(target_assets) == 1
