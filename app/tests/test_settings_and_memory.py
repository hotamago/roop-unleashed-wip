from types import SimpleNamespace

import roop.globals
from roop.memory import describe_memory_plan, resolve_memory_plan
from settings import Settings


def test_settings_loads_legacy_memory_and_persists_new_cache_knobs(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("memory_limit: 12\nprovider: cuda\n", encoding="utf-8")

    cfg = Settings(str(config_path))

    assert cfg.max_ram_gb == 12
    assert cfg.memory_limit == 12
    assert cfg.detect_pack_frame_count == 256
    assert cfg.staged_chunk_size == 0

    cfg.detect_pack_frame_count = 512
    cfg.staged_chunk_size = 96
    cfg.save()

    reloaded = Settings(str(config_path))
    assert reloaded.detect_pack_frame_count == 512
    assert reloaded.staged_chunk_size == 96


def test_resolve_memory_plan_respects_chunk_override_and_detect_pack(monkeypatch):
    monkeypatch.setattr(
        roop.globals,
        "CFG",
        SimpleNamespace(
            memory_mode="smart",
            max_ram_gb=0,
            max_vram_gb=0,
            detect_pack_frame_count=320,
            staged_chunk_size=96,
        ),
        raising=False,
    )
    monkeypatch.setattr("roop.memory.get_available_ram_gb", lambda: 24.0)
    monkeypatch.setattr("roop.memory.get_available_vram_gb", lambda: 10.0)

    plan = resolve_memory_plan(1920, 1080)

    assert plan["chunk_size"] == 96
    assert plan["detect_pack_frame_count"] == 320
    assert plan["vram_budget_gb"] == 9.25
    assert "detect pack=320" in describe_memory_plan(plan)
