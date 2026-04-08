from .video_iter import iter_video_chunk


def flatten_tasks(_executor, chunk_meta):
    tasks = []
    for frame_meta in chunk_meta["frames"]:
        for task_meta in frame_meta["tasks"]:
            tasks.append((frame_meta, task_meta))
    return tasks


def iter_chunk_source_frames_with_meta(_executor, chunk_meta, memory_plan, video_path=None, source_image=None):
    if source_image is not None:
        frame_meta = chunk_meta["frames"][0] if chunk_meta["frames"] else {"frame_number": 0, "fallback": True, "tasks": []}
        yield frame_meta.get("frame_number", 0), source_image, frame_meta
        return
    frame_lookup = {frame_meta["frame_number"]: frame_meta for frame_meta in chunk_meta["frames"]}
    for frame_number, frame in iter_video_chunk(video_path, chunk_meta["start"], chunk_meta["end"], memory_plan["prefetch_frames"]):
        frame_meta = frame_lookup.get(frame_number, {"frame_number": frame_number, "fallback": True, "tasks": []})
        yield frame_number, frame, frame_meta


__all__ = ["flatten_tasks", "iter_chunk_source_frames_with_meta"]
