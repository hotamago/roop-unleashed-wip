import time

import roop.config.globals
import roop.utils as util
from roop.progress.status import get_processing_status_line, publish_processing_progress


def get_pipeline_steps(executor):
    steps = ["prepare", "detect"]
    if executor.swap_enabled:
        steps.append("swap")
    if executor.mask_name is not None:
        steps.append("mask")
    if executor.enhancer_name is not None:
        steps.append("enhance")
    steps.append("composite")
    is_video = executor.current_entry is not None and not util.has_image_extension(executor.current_entry.filename)
    if is_video and executor.output_method != "Virtual Camera":
        steps.append("mux")
    return steps


def get_stage_step_info(executor, stage):
    steps = executor.get_pipeline_steps()
    stage_alias = {
        "resume": "prepare",
        "image": "composite",
        "legacy": "prepare",
    }
    stage_key = stage_alias.get(stage, stage)
    if stage_key not in steps:
        if stage == "prepare":
            return 1, len(steps)
        return None, len(steps)
    return steps.index(stage_key) + 1, len(steps)


def update_progress(executor, stage, detail=None, step_completed=None, step_total=None, step_unit="items", force_log=False):
    if stage != executor.current_stage or step_completed in (None, 0):
        executor.current_stage = stage
        executor.stage_started_at = time.time()
    stage_elapsed = None
    stage_rate = None
    stage_eta = None
    if executor.stage_started_at is not None:
        stage_elapsed = max(time.time() - executor.stage_started_at, 0.0)
    if step_completed is not None and step_completed > 0 and stage_elapsed and stage_elapsed > 0:
        stage_rate = step_completed / stage_elapsed
    if stage_rate is not None and step_total is not None and step_total > 0:
        stage_eta = max(step_total - step_completed, 0) / stage_rate
    target_name = None
    if executor.current_entry is not None:
        target_name = executor.current_entry.filename
    current_step, total_steps = executor.get_stage_step_info(stage)
    publish_processing_progress(
        stage=stage,
        completed=executor.completed_units,
        total=executor.total_units,
        unit="units",
        target_name=target_name,
        file_index=executor.current_file_index,
        total_files=executor.total_files,
        chunk_index=executor.current_chunk_index,
        total_chunks=executor.current_total_chunks,
        current_step=current_step,
        total_steps=total_steps,
        step_completed=step_completed,
        step_total=step_total,
        step_unit=step_unit,
        rate=stage_rate,
        rate_unit=step_unit,
        elapsed=stage_elapsed,
        eta=stage_eta,
        detail=detail,
        memory_status=roop.config.globals.runtime_memory_status,
        force_log=force_log,
    )
    if executor.progress is not None:
        executor.progress(
            (executor.completed_units, executor.total_units),
            desc=get_processing_status_line(),
            total=executor.total_units,
            unit="units",
        )


__all__ = ["get_pipeline_steps", "get_stage_step_info", "update_progress"]
