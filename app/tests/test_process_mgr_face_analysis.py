from types import SimpleNamespace

from roop.pipeline.batch_executor import ProcessMgr


def test_process_mgr_limits_face_analysis_modules_for_non_matching_modes():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="all")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_3d_68", "landmark_2d_106", "detection"]


def test_process_mgr_adds_recognition_only_for_selected_matching():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="selected")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_3d_68", "landmark_2d_106", "detection", "recognition"]


def test_process_mgr_adds_genderage_for_gender_filtering():
    mgr = ProcessMgr(None)
    mgr.options = SimpleNamespace(swap_mode="all_female")

    modules = mgr.resolve_face_analysis_modules()

    assert modules == ["landmark_3d_68", "landmark_2d_106", "detection", "genderage"]
