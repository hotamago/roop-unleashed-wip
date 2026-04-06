import math
import os
import time

import roop.globals


def _default_state():
    return {
        "status": "idle",
        "message": "Idle",
        "stage": None,
        "target_name": None,
        "file_index": None,
        "total_files": None,
        "chunk_index": None,
        "total_chunks": None,
        "completed": 0,
        "total": 0,
        "unit": "units",
        "step_completed": None,
        "step_total": None,
        "step_unit": None,
        "rate": None,
        "rate_unit": None,
        "eta": None,
        "elapsed": 0.0,
        "detail": None,
        "started_at": None,
        "memory_status": roop.globals.runtime_memory_status,
    }


def _ensure_state():
    state = roop.globals.runtime_processing_state
    if not isinstance(state, dict) or not state:
        return _default_state()
    merged = _default_state()
    merged.update(state)
    return merged


def _is_number(value):
    return isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value)


def format_duration(seconds):
    if not _is_number(seconds) or seconds < 0:
        return "--:--"
    total_seconds = int(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_stage(stage):
    if not stage:
        return None
    return str(stage).replace("_", " ").strip().title()


def _format_target_name(target_name):
    if not target_name:
        return None
    return os.path.basename(str(target_name))


def _format_progress_value(completed, total, unit):
    if _is_number(total) and total > 0:
        percent = max(0.0, min(100.0, (float(completed) / float(total)) * 100.0))
        return f"{int(completed)}/{int(total)} {unit} ({percent:.1f}%)"
    if _is_number(completed) and completed > 0:
        return f"{int(completed)} {unit}"
    return None


def render_status_line(state):
    message = state.get("message") or state.get("status", "idle").title()
    parts = [message]

    stage = _format_stage(state.get("stage"))
    if stage:
        parts.append(f"Stage: {stage}")

    target_name = _format_target_name(state.get("target_name"))
    file_index = state.get("file_index")
    total_files = state.get("total_files")
    if _is_number(file_index) and _is_number(total_files) and total_files > 0:
        if target_name:
            parts.append(f"File: {int(file_index)}/{int(total_files)} {target_name}")
        else:
            parts.append(f"Files: {int(file_index)}/{int(total_files)}")
    elif target_name:
        parts.append(f"Target: {target_name}")

    chunk_index = state.get("chunk_index")
    total_chunks = state.get("total_chunks")
    if _is_number(chunk_index) and _is_number(total_chunks) and total_chunks > 0:
        parts.append(f"Chunk: {int(chunk_index)}/{int(total_chunks)}")

    progress_text = _format_progress_value(state.get("completed", 0), state.get("total", 0), state.get("unit", "units"))
    if progress_text:
        parts.append(f"Progress: {progress_text}")

    step_progress = _format_progress_value(
        state.get("step_completed", 0) if state.get("step_completed") is not None else 0,
        state.get("step_total", 0) if state.get("step_total") is not None else 0,
        state.get("step_unit", "items"),
    )
    if step_progress:
        parts.append(f"Step: {step_progress}")

    rate = state.get("rate")
    if _is_number(rate) and rate > 0:
        parts.append(f"Speed: {rate:.2f} {state.get('rate_unit') or state.get('unit', 'units')}/s")

    eta = state.get("eta")
    if _is_number(eta) and eta >= 0:
        parts.append(f"ETA: {format_duration(eta)}")

    elapsed = state.get("elapsed")
    if _is_number(elapsed) and elapsed >= 0:
        parts.append(f"Elapsed: {format_duration(elapsed)}")

    detail = state.get("detail")
    if detail:
        parts.append(str(detail))

    return " | ".join(parts)


def render_status_markdown(state):
    lines = ["**Process Info**"]
    lines.append(f"- Status: {state.get('message') or state.get('status', 'idle').title()}")

    stage = _format_stage(state.get("stage"))
    if stage:
        lines.append(f"- Stage: {stage}")

    target_name = _format_target_name(state.get("target_name"))
    file_index = state.get("file_index")
    total_files = state.get("total_files")
    if _is_number(file_index) and _is_number(total_files) and total_files > 0:
        if target_name:
            lines.append(f"- File: {int(file_index)}/{int(total_files)} `{target_name}`")
        else:
            lines.append(f"- Files: {int(file_index)}/{int(total_files)}")
    elif target_name:
        lines.append(f"- Target: `{target_name}`")

    chunk_index = state.get("chunk_index")
    total_chunks = state.get("total_chunks")
    if _is_number(chunk_index) and _is_number(total_chunks) and total_chunks > 0:
        lines.append(f"- Chunk: {int(chunk_index)}/{int(total_chunks)}")

    progress_text = _format_progress_value(state.get("completed", 0), state.get("total", 0), state.get("unit", "units"))
    if progress_text:
        lines.append(f"- Progress: {progress_text}")

    step_progress = _format_progress_value(
        state.get("step_completed", 0) if state.get("step_completed") is not None else 0,
        state.get("step_total", 0) if state.get("step_total") is not None else 0,
        state.get("step_unit", "items"),
    )
    if step_progress:
        lines.append(f"- Step progress: {step_progress}")

    rate = state.get("rate")
    if _is_number(rate) and rate > 0:
        lines.append(f"- Speed: {rate:.2f} {state.get('rate_unit') or state.get('unit', 'units')}/s")

    eta = state.get("eta")
    if _is_number(eta) and eta >= 0:
        lines.append(f"- ETA: {format_duration(eta)}")

    elapsed = state.get("elapsed")
    if _is_number(elapsed) and elapsed >= 0:
        lines.append(f"- Elapsed: {format_duration(elapsed)}")

    detail = state.get("detail")
    if detail:
        lines.append(f"- Detail: {detail}")

    memory_status = state.get("memory_status") or roop.globals.runtime_memory_status
    if memory_status:
        lines.append(f"- {memory_status}")

    return "\n".join(lines)


def _apply_state(state, force_log=False):
    merged = _default_state()
    merged.update(state)
    roop.globals.runtime_processing_state = merged
    roop.globals.runtime_processing_status = render_status_line(merged)
    roop.globals.runtime_processing_markdown = render_status_markdown(merged)

    now = time.time()
    should_log = force_log
    if merged.get("status") in ("running", "stopping"):
        if now - roop.globals.runtime_processing_last_log_at >= 1.0:
            should_log = True
        total = merged.get("total", 0)
        completed = merged.get("completed", 0)
        if _is_number(total) and total > 0 and _is_number(completed) and completed >= total:
            should_log = True
    if should_log:
        print(f"[processing] {roop.globals.runtime_processing_status}", flush=True)
        roop.globals.runtime_processing_last_log_at = now
    return roop.globals.runtime_processing_markdown


def reset_processing_status(message="Idle"):
    state = _default_state()
    state["message"] = message
    return _apply_state(state, force_log=False)


def start_processing_status(message="Preparing faceswap job", total=None, unit="units", total_files=None, memory_status=None):
    now = time.time()
    state = _default_state()
    state.update({
        "status": "running",
        "message": message,
        "total": total or 0,
        "unit": unit,
        "total_files": total_files,
        "started_at": now,
        "elapsed": 0.0,
        "memory_status": memory_status or roop.globals.runtime_memory_status,
    })
    return _apply_state(state, force_log=True)


def set_processing_message(message, status=None, stage=None, target_name=None, file_index=None, total_files=None, chunk_index=None, total_chunks=None, detail=None, memory_status=None, force_log=False):
    state = _ensure_state()
    if status is not None:
        state["status"] = status
    if message is not None:
        state["message"] = message
    if stage is not None:
        state["stage"] = stage
    if target_name is not None:
        state["target_name"] = target_name
    if file_index is not None:
        state["file_index"] = file_index
    if total_files is not None:
        state["total_files"] = total_files
    if chunk_index is not None:
        state["chunk_index"] = chunk_index
    if total_chunks is not None:
        state["total_chunks"] = total_chunks
    if detail is not None:
        state["detail"] = detail
    if memory_status is not None:
        state["memory_status"] = memory_status
    if state.get("started_at") is not None:
        state["elapsed"] = max(time.time() - state["started_at"], 0.0)
    return _apply_state(state, force_log=force_log)


def set_memory_status(memory_status):
    roop.globals.runtime_memory_status = memory_status
    state = _ensure_state()
    state["memory_status"] = memory_status
    return _apply_state(state, force_log=False)


def publish_processing_progress(stage=None, completed=None, total=None, unit=None, target_name=None, file_index=None, total_files=None, chunk_index=None, total_chunks=None, step_completed=None, step_total=None, step_unit=None, rate=None, rate_unit=None, elapsed=None, eta=None, detail=None, memory_status=None, force_log=False):
    state = _ensure_state()
    now = time.time()

    if state.get("started_at") is None:
        state["started_at"] = now

    state["status"] = "running"
    if stage is not None:
        state["stage"] = stage
    if completed is not None:
        state["completed"] = completed
    if total is not None:
        state["total"] = total
    if unit is not None:
        state["unit"] = unit
    if target_name is not None:
        state["target_name"] = target_name
    if file_index is not None:
        state["file_index"] = file_index
    if total_files is not None:
        state["total_files"] = total_files
    if chunk_index is not None:
        state["chunk_index"] = chunk_index
    if total_chunks is not None:
        state["total_chunks"] = total_chunks
    if step_completed is not None:
        state["step_completed"] = step_completed
    if step_total is not None:
        state["step_total"] = step_total
    if step_unit is not None:
        state["step_unit"] = step_unit
    if rate_unit is not None:
        state["rate_unit"] = rate_unit
    if detail is not None:
        state["detail"] = detail
    if memory_status is not None:
        state["memory_status"] = memory_status

    if elapsed is None:
        elapsed = max(now - state["started_at"], 0.0)
    state["elapsed"] = elapsed

    if rate is None:
        total_completed = state.get("completed", 0)
        if _is_number(total_completed) and total_completed > 0 and _is_number(elapsed) and elapsed > 0:
            rate = total_completed / elapsed
    state["rate"] = rate

    if eta is None:
        total_value = state.get("total", 0)
        completed_value = state.get("completed", 0)
        if _is_number(rate) and rate > 0 and _is_number(total_value) and total_value > 0:
            remaining = max(float(total_value) - float(completed_value), 0.0)
            eta = remaining / rate
    state["eta"] = eta

    if not state.get("message") or state.get("message") == "Idle":
        stage_name = _format_stage(stage) or "Processing"
        state["message"] = f"{stage_name} in progress"

    return _apply_state(state, force_log=force_log)


def finish_processing_status(message, status="completed"):
    state = _ensure_state()
    state["status"] = status
    state["message"] = message
    if status == "completed" and _is_number(state.get("total")) and state.get("total", 0) > 0:
        state["completed"] = state["total"]
        state["eta"] = 0.0
    elif status in ("stopped", "error"):
        state["eta"] = None
    if state.get("started_at") is not None:
        state["elapsed"] = max(time.time() - state["started_at"], 0.0)
    return _apply_state(state, force_log=True)


def get_processing_status_markdown():
    if not roop.globals.runtime_processing_markdown:
        return render_status_markdown(_ensure_state())
    return roop.globals.runtime_processing_markdown


def get_processing_status_line():
    if not roop.globals.runtime_processing_status:
        return render_status_line(_ensure_state())
    return roop.globals.runtime_processing_status
