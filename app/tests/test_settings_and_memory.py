from types import SimpleNamespace

import roop.config.globals
from roop.config.settings import Settings
from roop.face_analytics_models import (
    get_face_detector_model_choices,
    get_face_landmarker_model_choices,
    get_face_masker_model_choices,
)
from roop.face_swap_models import (
    get_face_swap_model_choices,
    get_face_swap_upscale_choices,
    parse_face_swap_upscale_size,
)
from roop.memory.planner import (
    describe_memory_plan,
    resolve_detect_single_batch_workers,
    resolve_gpu_single_batch_worker_cap,
    resolve_memory_plan,
    resolve_single_batch_workers,
)
from roop.pipeline.options import ProcessOptions
from roop.pipeline.staged_executor.cache import get_staged_cache_options_snapshot
from ui.tabs.settings_tab import on_face_swap_model_changed


def test_settings_loads_and_persists_manual_stage_tuning(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("provider: cuda\n", encoding="utf-8")

    cfg = Settings(str(config_path))

    assert cfg.detect_pack_frame_count == 256
    assert cfg.staged_chunk_size == 96
    assert cfg.prefetch_frames == 24
    assert cfg.detect_batch_size == 8
    assert cfg.detect_single_batch_workers == 1
    assert cfg.swap_batch_size == 32
    assert cfg.mask_batch_size == 64
    assert cfg.enhance_batch_size == 8
    assert cfg.single_batch_workers == 1
    assert cfg.face_detector_model == "insightface"
    assert cfg.face_landmarker_model == "insightface_2d106"
    assert cfg.face_masker_model == "legacy_xseg"
    assert cfg.face_swap_model == "inswapper_128"
    assert cfg.subsample_upscale == "256px"

    cfg.detect_pack_frame_count = 512
    cfg.staged_chunk_size = 128
    cfg.prefetch_frames = 48
    cfg.detect_batch_size = 12
    cfg.detect_single_batch_workers = 2
    cfg.swap_batch_size = 40
    cfg.mask_batch_size = 96
    cfg.enhance_batch_size = 12
    cfg.single_batch_workers = 3
    cfg.face_detector_model = "retinaface"
    cfg.face_landmarker_model = "2dfan4"
    cfg.face_masker_model = "bisenet_resnet_18"
    cfg.face_swap_model = "hyperswap_1a_256"
    cfg.subsample_upscale = "512px"
    cfg.save()

    reloaded = Settings(str(config_path))
    assert reloaded.detect_pack_frame_count == 512
    assert reloaded.staged_chunk_size == 128
    assert reloaded.prefetch_frames == 48
    assert reloaded.detect_batch_size == 12
    assert reloaded.detect_single_batch_workers == 2
    assert reloaded.swap_batch_size == 40
    assert reloaded.mask_batch_size == 96
    assert reloaded.enhance_batch_size == 12
    assert reloaded.single_batch_workers == 3
    assert reloaded.face_detector_model == "retinaface"
    assert reloaded.face_landmarker_model == "2dfan4"
    assert reloaded.face_masker_model == "bisenet_resnet_18"
    assert reloaded.face_swap_model == "hyperswap_1a_256"
    assert reloaded.subsample_upscale == "512px"


def test_settings_normalizes_subsample_for_256_face_swap_models(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "face_swap_model: hyperswap_1b_256\nsubsample_upscale: 128px\n",
        encoding="utf-8",
    )

    cfg = Settings(str(config_path))

    assert cfg.face_swap_model == "hyperswap_1b_256"
    assert cfg.subsample_upscale == "256px"


def test_settings_rounds_hyperswap_upscale_to_supported_choice(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "face_swap_model: hyperswap_1c_256\nsubsample_upscale: 384px\n",
        encoding="utf-8",
    )

    cfg = Settings(str(config_path))

    assert cfg.face_swap_model == "hyperswap_1c_256"
    assert cfg.subsample_upscale == "512px"
    assert get_face_swap_upscale_choices(cfg.face_swap_model) == ["256px", "512px", "768px", "1024px"]


def test_face_swap_model_choices_include_inswapper_fp16():
    assert "inswapper_128_fp16" in get_face_swap_model_choices()
    assert get_face_swap_upscale_choices("inswapper_128_fp16") == ["128px", "256px", "384px", "512px", "768px", "1024px"]


def test_on_face_swap_model_changed_downloads_and_releases_runtime(monkeypatch):
    calls = []
    monkeypatch.setattr("ui.tabs.settings_tab.ensure_face_swap_model_downloaded", lambda model: calls.append(("download", model)) or f"{model}.onnx")
    monkeypatch.setattr("ui.tabs.settings_tab.update_memory_status", lambda: "memory-status")
    monkeypatch.setattr("roop.core.app.release_resources", lambda: calls.append(("release", None)))
    monkeypatch.setattr(
        roop.config.globals,
        "CFG",
        SimpleNamespace(face_swap_model="hyperswap_1a_256", subsample_upscale="256px"),
        raising=False,
    )
    monkeypatch.setattr(roop.config.globals, "subsample_size", 256, raising=False)

    memory_status, upscale_dropdown, hint = on_face_swap_model_changed("hyperswap_1c_256", "256px")

    assert calls == [("download", "hyperswap_1c_256"), ("release", None)]
    assert memory_status == "memory-status"
    assert roop.config.globals.CFG.face_swap_model == "hyperswap_1c_256"
    assert roop.config.globals.CFG.subsample_upscale == "256px"
    assert roop.config.globals.subsample_size == 256
    assert getattr(upscale_dropdown, "value", None) == "256px"
    assert "hyperswap_1c_256" in getattr(hint, "value", "")


def test_face_analytics_model_choices_include_facefusion_variants():
    assert {"insightface", "retinaface", "scrfd", "yolo_face", "yunet"} <= set(get_face_detector_model_choices())
    assert {"insightface_2d106", "2dfan4", "peppa_wutz", "fan_68_5"} <= set(get_face_landmarker_model_choices())
    assert {"legacy_xseg", "xseg_1", "xseg_2", "xseg_3", "bisenet_resnet_18", "bisenet_resnet_34"} <= set(get_face_masker_model_choices())


def test_resolve_memory_plan_uses_manual_stage_tuning(monkeypatch):
    monkeypatch.setattr("roop.memory.planner.provider_uses_gpu", lambda: False)
    monkeypatch.setattr(
        roop.config.globals,
        "CFG",
        SimpleNamespace(
            detect_pack_frame_count=320,
            staged_chunk_size=96,
            prefetch_frames=120,
            detect_batch_size=10,
            detect_single_batch_workers=2,
            swap_batch_size=48,
            mask_batch_size=96,
            enhance_batch_size=12,
            single_batch_workers=3,
        ),
        raising=False,
    )
    monkeypatch.setattr("roop.memory.planner.get_available_ram_gb", lambda: 24.0)
    monkeypatch.setattr("roop.memory.planner.get_available_vram_gb", lambda: 10.0)

    plan = resolve_memory_plan(1920, 1080)

    assert plan["chunk_size"] == 96
    assert plan["prefetch_frames"] == 96
    assert plan["detect_batch_size"] == 10
    assert plan["detect_single_batch_workers"] == 2
    assert plan["swap_batch_size"] == 48
    assert plan["mask_batch_size"] == 96
    assert plan["enhance_batch_size"] == 12
    assert plan["single_batch_workers"] == 3
    assert plan["detect_pack_frame_count"] == 320
    assert "detect batch=10" in describe_memory_plan(plan)
    assert "detect workers=2" in describe_memory_plan(plan)
    assert "single-batch workers=3" in describe_memory_plan(plan)


def test_resolve_memory_plan_allows_gpu_single_batch_workers_when_vram_is_available(monkeypatch):
    monkeypatch.setattr("roop.memory.planner.provider_uses_gpu", lambda: True)
    monkeypatch.setattr(
        roop.config.globals,
        "CFG",
        SimpleNamespace(
            detect_pack_frame_count=256,
            staged_chunk_size=128,
            prefetch_frames=48,
            detect_batch_size=8,
            detect_single_batch_workers=2,
            swap_batch_size=8,
            mask_batch_size=8,
            enhance_batch_size=8,
            single_batch_workers=2,
        ),
        raising=False,
    )
    monkeypatch.setattr("roop.memory.planner.get_available_ram_gb", lambda: 24.0)
    monkeypatch.setattr("roop.memory.planner.get_available_vram_gb", lambda: 10.0)

    plan = resolve_memory_plan(1920, 1080)

    assert plan["single_batch_workers"] == 2
    assert plan["detect_single_batch_workers"] == 2
    assert plan["requested_single_batch_workers"] == 2
    assert plan["single_batch_workers_reason"] is None
    assert "single-batch workers=2" in describe_memory_plan(plan)


def test_resolve_memory_plan_caps_gpu_single_batch_workers_on_low_vram(monkeypatch):
    monkeypatch.setattr("roop.memory.planner.provider_uses_gpu", lambda: True)
    monkeypatch.setattr(
        roop.config.globals,
        "CFG",
        SimpleNamespace(
            detect_pack_frame_count=256,
            staged_chunk_size=128,
            prefetch_frames=48,
            detect_batch_size=8,
            detect_single_batch_workers=3,
            swap_batch_size=8,
            mask_batch_size=8,
            enhance_batch_size=8,
            single_batch_workers=3,
        ),
        raising=False,
    )
    monkeypatch.setattr("roop.memory.planner.get_available_ram_gb", lambda: 24.0)
    monkeypatch.setattr("roop.memory.planner.get_available_vram_gb", lambda: 7.5)

    plan = resolve_memory_plan(1920, 1080)

    assert plan["single_batch_workers"] == 1
    assert plan["requested_single_batch_workers"] == 3
    assert plan["single_batch_workers_reason"] == "GPU VRAM cap 1"
    assert plan["detect_single_batch_workers"] == 1
    assert plan["requested_detect_single_batch_workers"] == 3
    assert plan["detect_single_batch_workers_reason"] == "GPU VRAM cap 1"
    assert "detect workers=1 (requested 3, GPU VRAM cap 1)" in describe_memory_plan(plan)
    assert "single-batch workers=1 (requested 3, GPU VRAM cap 1)" in describe_memory_plan(plan)


def test_resolve_single_batch_workers_keeps_cpu_parallelism(monkeypatch):
    monkeypatch.setattr("roop.memory.planner.provider_uses_gpu", lambda: False)

    effective_workers, requested_workers, reason = resolve_single_batch_workers(4)

    assert effective_workers == 4
    assert requested_workers == 4
    assert reason is None


def test_resolve_detect_single_batch_workers_keeps_cpu_parallelism(monkeypatch):
    monkeypatch.setattr("roop.memory.planner.provider_uses_gpu", lambda: False)

    effective_workers, requested_workers, reason = resolve_detect_single_batch_workers(4)

    assert effective_workers == 4
    assert requested_workers == 4
    assert reason is None


def test_resolve_gpu_single_batch_worker_cap_scales_with_vram():
    assert resolve_gpu_single_batch_worker_cap(None) == 2
    assert resolve_gpu_single_batch_worker_cap(9.0) == 1
    assert resolve_gpu_single_batch_worker_cap(10.0) == 2
    assert resolve_gpu_single_batch_worker_cap(14.0) == 3
    assert resolve_gpu_single_batch_worker_cap(20.0) == 4


def test_process_options_coerce_subsample_size_for_256_face_swap_models():
    options = ProcessOptions(
        {"faceswap": {}},
        0.6,
        0.5,
        "all",
        0,
        "",
        None,
        1,
        128,
        False,
        False,
        face_swap_model="hyperswap_1a_256",
    )

    assert options.face_swap_model == "hyperswap_1a_256"
    assert options.face_swap_tile_size == 256
    assert options.subsample_size == 256


def test_process_options_rounds_to_supported_hyperswap_subsample_size():
    options = ProcessOptions(
        {"faceswap": {}},
        0.6,
        0.5,
        "all",
        0,
        "",
        None,
        1,
        384,
        False,
        False,
        face_swap_model="hyperswap_1a_256",
    )

    assert options.subsample_size == 512


def test_parse_face_swap_upscale_size_keeps_four_digit_values():
    assert parse_face_swap_upscale_size("1024px", "hyperswap_1a_256") == 1024


def test_staged_cache_snapshot_changes_when_face_swap_model_changes():
    base = {
        "processors": {"faceswap": {}},
        "face_distance_threshold": 0.6,
        "blend_ratio": 0.5,
        "swap_mode": "all",
        "selected_index": 0,
        "masking_text": "",
        "num_swap_steps": 1,
        "subsample_size": 256,
        "restore_original_mouth": False,
        "show_face_masking": False,
    }
    options_a = SimpleNamespace(face_swap_model="inswapper_128", **base)
    options_b = SimpleNamespace(face_swap_model="hyperswap_1b_256", **base)

    assert get_staged_cache_options_snapshot(options_a) != get_staged_cache_options_snapshot(options_b)


def test_staged_cache_snapshot_changes_when_face_analytics_models_change(monkeypatch):
    base = {
        "processors": {"faceswap": {}},
        "face_swap_model": "inswapper_128",
        "face_distance_threshold": 0.6,
        "blend_ratio": 0.5,
        "swap_mode": "all",
        "selected_index": 0,
        "masking_text": "",
        "num_swap_steps": 1,
        "subsample_size": 256,
        "restore_original_mouth": False,
        "show_face_masking": False,
    }
    options = SimpleNamespace(**base)

    monkeypatch.setattr(
        roop.config.globals,
        "CFG",
        SimpleNamespace(
            face_detector_model="insightface",
            face_landmarker_model="insightface_2d106",
            face_masker_model="legacy_xseg",
        ),
        raising=False,
    )
    snapshot_a = get_staged_cache_options_snapshot(options)

    monkeypatch.setattr(
        roop.config.globals,
        "CFG",
        SimpleNamespace(
            face_detector_model="scrfd",
            face_landmarker_model="fan_68_5",
            face_masker_model="bisenet_resnet_34",
        ),
        raising=False,
    )
    snapshot_b = get_staged_cache_options_snapshot(options)

    assert snapshot_a != snapshot_b
