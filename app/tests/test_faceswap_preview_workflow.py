from types import SimpleNamespace

import numpy as np

import roop.config.globals
import ui.globals
import ui.tabs.faceswap_tab as faceswap_tab
from roop.pipeline.faceset import FaceSet


def test_on_srcfile_changed_extracts_source_faces_with_recognition(monkeypatch, tmp_path):
    recorded_modules = []
    source_path = tmp_path / "source.png"
    source_path.write_bytes(b"fake")

    def fake_extract(_path, _video_info):
        recorded_modules.append(tuple(getattr(roop.config.globals, "g_desired_face_analysis", []) or []))
        return [(SimpleNamespace(embedding=np.array([1.0], dtype=np.float32)), "thumb")]

    monkeypatch.setattr(faceswap_tab, "extract_face_images", fake_extract)
    monkeypatch.setattr(faceswap_tab, "normalize_source_path", lambda path, warn=True: str(source_path))
    monkeypatch.setattr(faceswap_tab.util, "has_image_extension", lambda path: True)
    monkeypatch.setattr(faceswap_tab.util, "convert_to_gradio", lambda image: image)

    faceswap_tab.on_srcfile_changed([SimpleNamespace(name=str(source_path))])

    assert recorded_modules == [("landmark_3d_68", "landmark_2d_106", "detection", "recognition")]
    assert len(roop.config.globals.INPUT_FACESETS) == 1


def test_on_use_face_from_selected_extracts_target_faces_with_recognition(monkeypatch):
    recorded_modules = []
    monkeypatch.setattr(faceswap_tab, "list_target_paths", lambda _files: ["target.png"])
    monkeypatch.setattr(faceswap_tab.util, "is_image", lambda _path: True)
    monkeypatch.setattr(faceswap_tab.util, "convert_to_gradio", lambda image: image)
    monkeypatch.setattr(
        faceswap_tab,
        "extract_face_images",
        lambda _path, _video_info: (
            recorded_modules.append(tuple(getattr(roop.config.globals, "g_desired_face_analysis", []) or []))
            or [(SimpleNamespace(embedding=np.array([1.0], dtype=np.float32)), "thumb")]
        ),
    )

    faceswap_tab.on_use_face_from_selected(["target.png"], 1)

    assert recorded_modules == [("landmark_3d_68", "landmark_2d_106", "detection", "recognition")]
    assert len(roop.config.globals.TARGET_FACES) == 1


def test_ensure_loaded_face_embeddings_repairs_faces_from_refs(monkeypatch):
    source_face = SimpleNamespace(embedding=None, normed_embedding=None)
    source_set = FaceSet()
    source_set.faces.append(source_face)
    roop.config.globals.INPUT_FACESETS[:] = [source_set]
    ui.globals.ui_input_face_refs[:] = [{"type": "image_face", "path": "source.png", "face_index": 0}]

    roop.config.globals.TARGET_FACES[:] = [SimpleNamespace(embedding=None, normed_embedding=None)]
    ui.globals.ui_target_face_refs[:] = [{"path": "target.png", "frame_number": 0, "face_index": 0}]

    def fake_restore_input_faces(source_refs):
        restored_set = FaceSet()
        restored_set.faces.append(
            SimpleNamespace(
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                normed_embedding=np.array([1.0, 0.0], dtype=np.float32),
            )
        )
        roop.config.globals.INPUT_FACESETS[:] = [restored_set]
        ui.globals.ui_input_face_refs[:] = list(source_refs)

    def fake_append_target_face(face_ref):
        roop.config.globals.TARGET_FACES.append(
            SimpleNamespace(
                embedding=np.array([0.0, 1.0], dtype=np.float32),
                normed_embedding=np.array([0.0, 1.0], dtype=np.float32),
            )
        )
        ui.globals.ui_target_face_refs.append(dict(face_ref))

    monkeypatch.setattr(faceswap_tab, "restore_input_faces_from_resume", fake_restore_input_faces)
    monkeypatch.setattr(faceswap_tab, "append_target_face_from_resume", fake_append_target_face)

    repaired = faceswap_tab.ensure_loaded_face_embeddings()

    assert repaired is True
    assert faceswap_tab.get_face_embedding_vector(roop.config.globals.INPUT_FACESETS[0].faces[0]) is not None
    assert faceswap_tab.get_face_embedding_vector(roop.config.globals.TARGET_FACES[0]) is not None
