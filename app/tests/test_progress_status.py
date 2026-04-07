import roop.globals
from roop.progress_status import (
    get_processing_status_line,
    get_processing_status_markdown,
    publish_processing_progress,
    reset_processing_status,
    set_processing_message,
)


def test_publish_processing_progress_exposes_pipeline_steps():
    reset_processing_status()

    publish_processing_progress(
        stage="swap",
        completed=5,
        total=10,
        unit="frames",
        current_step=3,
        total_steps=7,
        step_completed=2,
        step_total=4,
        step_unit="faces",
        rate=1.5,
        eta=6.0,
        detail="Running batched face swap",
    )

    line = get_processing_status_line()
    markdown = get_processing_status_markdown()

    assert "Pipeline: 3/7" in line
    assert "Step: 2/4 faces" in line
    assert "- Pipeline step: 3/7" in markdown
    assert roop.globals.runtime_processing_state["current_step"] == 3
    assert roop.globals.runtime_processing_state["total_steps"] == 7


def test_set_processing_message_preserves_pipeline_steps_when_stage_changes():
    reset_processing_status()

    set_processing_message("Preparing", stage="prepare", current_step=1, total_steps=6)
    set_processing_message("Detecting", stage="detect", detail="Packed detect cache")

    state = roop.globals.runtime_processing_state
    assert state["current_step"] == 1
    assert state["total_steps"] == 6
    assert state["stage"] == "detect"
    assert "Pipeline: 1/6" in get_processing_status_line()
