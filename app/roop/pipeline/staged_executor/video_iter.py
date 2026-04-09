import queue
from threading import Thread

import cv2

import roop.config.globals
from roop.media.video_io import open_video_capture


def iter_video_chunk_cv2(video_path, frame_start, frame_end, prefetch_frames):
    q = queue.Queue(maxsize=max(2, prefetch_frames))
    cap = open_video_capture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    def producer():
        current = frame_start
        while current < frame_end and roop.config.globals.processing:
            ok, frame = cap.read()
            if not ok:
                break
            q.put((current, frame), block=True)
            current += 1
        q.put(None)

    thread = Thread(target=producer, daemon=True)
    thread.start()
    while True:
        item = q.get()
        if item is None:
            break
        yield item
    thread.join()
    cap.release()


def iter_video_chunk(video_path, frame_start, frame_end, prefetch_frames):
    yield from iter_video_chunk_cv2(video_path, frame_start, frame_end, prefetch_frames)


__all__ = ["iter_video_chunk"]
