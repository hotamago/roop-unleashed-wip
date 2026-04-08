# Big Refactor: Professional Project Structure & Pipeline Optimization

## User Requirements

Big refactor, Phân chia, cấu trúc hóa lại các code trong app, hiện tại có nhiều file chứa quá nhiều thứ, tách ra phân chia thư mục, cấu trúc project professional. Thẳng tay xóa những file không dùng hoặc đã bị phân chia. Đảm bảo thư mục gọn gàng
Đảm bảo rằng project, code dễ scale và phát triển sau này, simplify but effect.
- optimize từng step process kỹ càng (detect face, step Streaming source decode + mask stage,  step face swap, Optimize step enhancement stage)
- Tận dụng tối đa tính toán nặng trên GPU, ngoài ra tận dụng tối đa tính toán song song nếu phải dùng CPU

## Problem Summary

The codebase has several critical structural issues:

1. **Monster files**: `ProcessMgr.py` (1449 lines), `staged_executor.py` (1617 lines), `core.py` (556 lines) each contain too many responsibilities
2. **Flat directory structure**: Everything dumped in `roop/` — no logical grouping
3. **Empty `pipeline/` and `image_processing/` dirs**: Planned but never populated
4. **Duplicate logic**: Face detection, swap, rotation, compositing code duplicated between `ProcessMgr` and `StagedBatchExecutor`
5. **No clear GPU optimization strategy**: GPU batch processing buried inside massive classes
6. **Unused global state**: Several globals (`IMAGE_CHAIN_PROCESSOR`, `VIDEO_CHAIN_PROCESSOR`, etc.) never used

## Proposed New Directory Structure

```
app/roop/
├── __init__.py
├── config/                      # Configuration & settings
│   ├── __init__.py
│   ├── globals.py               # Runtime global state (cleaned)
│   ├── settings.py              # YAML settings loader (moved from app/)
│   └── types.py                 # Type definitions (Face, Frame, FaceSet)
│
├── core/                        # Application entry & orchestration  
│   ├── __init__.py
│   ├── app.py                   # run(), parse_args(), pre_check() — from core.py
│   ├── providers.py             # Execution provider logic — from core.py
│   └── resources.py             # Resource limits & cleanup — from core.py
│
├── face/                        # Face detection, analysis, alignment
│   ├── __init__.py
│   ├── analyser.py              # Face analyser singleton — from face_util.py
│   ├── alignment.py             # align_crop, estimate_norm — from face_util.py
│   ├── detection.py             # get_first_face, get_all_faces, extract_face_images
│   ├── rotation.py              # Rotation helpers — from face_util.py
│   └── geometry.py              # clamp_cut_values, cutout, paste — from face_util.py + ProcessMgr
│
├── pipeline/                    # Processing pipeline stages
│   ├── __init__.py
│   ├── options.py               # ProcessOptions — from ProcessOptions.py
│   ├── entry.py                 # ProcessEntry — from ProcessEntry.py  
│   ├── faceset.py               # FaceSet — from FaceSet.py
│   ├── face_targeting.py        # Face matching/selection logic — from ProcessMgr
│   ├── face_serializer.py       # serialize/deserialize face — from ProcessMgr
│   ├── swap_stage.py            # Swap logic: run_swap_task, pixel boost — from ProcessMgr
│   ├── mask_stage.py            # Mask application — from ProcessMgr
│   ├── enhance_stage.py         # Enhancement stage — from ProcessMgr
│   ├── compose_stage.py         # paste_upscale, compose_task, mouth — from ProcessMgr
│   ├── batch_executor.py        # Legacy in-memory batch — from ProcessMgr (run_batch_inmem)
│   └── staged_executor/         # Modern staged pipeline — from staged_executor.py
│       ├── __init__.py
│       ├── executor.py          # StagedBatchExecutor main class (slimmed)
│       ├── cache.py             # Cache read/write, manifest, hashing
│       ├── video_cache.py       # [NEW] Video-encoded stage cache (replaces raw pickle)
│       ├── detect_stage.py      # Detect stage (packed detect)
│       ├── swap_stage.py        # Full swap stage
│       ├── mask_stage.py        # Full mask stage
│       ├── enhance_stage.py     # Full enhance stage  
│       ├── compose_stage.py     # Full compose + encode stage
│       ├── chunk_processor.py   # Streaming chunk logic
│       ├── video_iter.py        # iter_video_chunk helper
│       └── progress.py          # Progress reporting for staged pipeline
│
├── processors/                  # ONNX model processors (unchanged structure)
│   ├── __init__.py
│   ├── base.py                  # [NEW] Abstract base processor class
│   ├── FaceSwapInsightFace.py
│   ├── Enhance_CodeFormer.py
│   ├── Enhance_DMDNet.py
│   ├── Enhance_GFPGAN.py
│   ├── Enhance_GPEN.py
│   ├── Enhance_RestoreFormerPPlus.py
│   ├── Frame_Colorizer.py
│   ├── Frame_Filter.py
│   ├── Frame_Masking.py
│   ├── Frame_Upscale.py
│   ├── Mask_Clip2Seg.py
│   └── Mask_XSeg.py
│
├── media/                       # Video/Image I/O
│   ├── __init__.py
│   ├── capturer.py              # Frame capture — from capturer.py
│   ├── ffmpeg_ops.py            # FFmpeg operations — from util_ffmpeg.py
│   ├── ffmpeg_writer.py         # FFMPEG_VideoWriter — from ffmpeg_writer.py
│   ├── video_io.py              # video reader + GPU encoder config — from video_io.py
│   └── stream_writer.py         # Virtual cam output — from StreamWriter.py
│
├── onnx/                        # ONNX runtime helpers
│   ├── __init__.py
│   ├── batch.py                 # Native batch model patching — from onnx_batch.py
│   └── runtime.py               # Provider/model resolution — from onnx_runtime.py
│
├── memory/                      # Memory & resource management
│   ├── __init__.py
│   └── planner.py               # Memory plan + VRAM — from memory.py
│
├── progress/                    # Progress & status reporting
│   ├── __init__.py
│   └── status.py                # All progress formatting — from progress_status.py
│
└── utils/                       # Generic utilities
    ├── __init__.py
    ├── io.py                    # File I/O, path helpers — from utilities.py
    ├── download.py              # conditional_download — from utilities.py  
    ├── platform.py              # Platform detection — from utilities.py
    ├── cache_paths.py           # Cache directory paths — from cache_paths.py
    ├── template_parser.py       # Template parsing — from template_parser.py
    └── vr.py                    # VR lens distortion — from vr_util.py
```

## Files to DELETE

| File | Reason |
|------|--------|
| `roop/ProcessMgr.py` | Decomposed into `pipeline/` modules |
| `roop/staged_executor.py` | Decomposed into `pipeline/staged_executor/` |
| `roop/core.py` | Decomposed into `core/` modules |
| `roop/face_util.py` | Decomposed into `face/` modules |
| `roop/utilities.py` | Decomposed into `utils/` modules |
| `roop/globals.py` | Moved to `config/globals.py` |
| `roop/typing.py` | Moved to `config/types.py` |
| `roop/ProcessOptions.py` | Moved to `pipeline/options.py` |
| `roop/ProcessEntry.py` | Moved to `pipeline/entry.py` |
| `roop/FaceSet.py` | Moved to `pipeline/faceset.py` |
| `roop/memory.py` | Moved to `memory/planner.py` |
| `roop/progress_status.py` | Moved to `progress/status.py` |
| `roop/capturer.py` | Moved to `media/capturer.py` |
| `roop/ffmpeg_writer.py` | Moved to `media/ffmpeg_writer.py` |
| `roop/video_io.py` | Moved to `media/video_io.py` |
| `roop/StreamWriter.py` | Moved to `media/stream_writer.py` |
| `roop/util_ffmpeg.py` | Moved to `media/ffmpeg_ops.py` |
| `roop/onnx_batch.py` | Moved to `onnx/batch.py` |
| `roop/onnx_runtime.py` | Moved to `onnx/runtime.py` |
| `roop/cache_paths.py` | Moved to `utils/cache_paths.py` |
| `roop/template_parser.py` | Moved to `utils/template_parser.py` |
| `roop/vr_util.py` | Moved to `utils/vr.py` |
| `roop/metadata.py` | Content absorbed into `config/` |
| `roop/image_processing/` | Empty directory — delete |
| `roop/pipeline/__pycache__` | Build artifact — delete |
| `app/settings.py` | Moved to `roop/config/settings.py` |

---

## Proposed Changes (by Component)

### 1. Config Layer (`roop/config/`)

#### [NEW] [globals.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/config/globals.py)
- Clean up unused globals: remove `IMAGE_CHAIN_PROCESSOR`, `VIDEO_CHAIN_PROCESSOR`, `BATCH_IMAGE_CHAIN_PROCESSOR`
- Keep essential runtime state

#### [NEW] [types.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/config/types.py)
- Merge `typing.py` + `FaceSet.py` type definitions

#### [NEW] [settings.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/config/settings.py)
- Move `Settings` class from `app/settings.py`

---

### 2. Core Application (`roop/core/`)

#### [NEW] [app.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/core/app.py)
- `run()`, `parse_args()`, `pre_check()`, `start()`, `destroy()`
- `batch_process_regular()`, `batch_process_legacy()` 
- Main orchestration logic

#### [NEW] [providers.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/core/providers.py)
- All execution provider logic: `encode_execution_providers()`, `decode_execution_providers()`, `_build_cuda_execution_provider()`, `_build_tensorrt_execution_provider()`

#### [NEW] [resources.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/core/resources.py)
- `release_resources()`, `limit_resources()`, `suggest_*()` functions

---

### 3. Face Module (`roop/face/`)

Split `face_util.py` (318 lines) into focused modules:

#### [NEW] [analyser.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/face/analyser.py)
- `get_face_analyser()`, `release_face_analyser()` — singleton with thread-safe init

#### [NEW] [detection.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/face/detection.py)
- `get_first_face()`, `get_all_faces()`, `extract_face_images()`

#### [NEW] [alignment.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/face/alignment.py)
- `align_crop()`, `estimate_norm()`, `square_crop()`, `transform()`, arcface constants

#### [NEW] [rotation.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/face/rotation.py)
- `rotate_anticlockwise()`, `rotate_clockwise()`, `rotate_image_90()`, `rotate_image_180()`

#### [NEW] [geometry.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/face/geometry.py)
- `clamp_cut_values()`, `resize_image_keep_content()`, `create_blank_image()`
- `cutout()`, `paste_simple()` — extracted from ProcessMgr

---

### 4. Pipeline Module (`roop/pipeline/`)

Decompose `ProcessMgr.py` (1449 lines) into focused stage modules:

#### [NEW] [face_targeting.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/face_targeting.py)
- `get_frame_face_targets()` — face matching/selection per swap mode
- `eNoFaceAction` enum

#### [NEW] [face_serializer.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/face_serializer.py)
- `serialize_face()`, `deserialize_face()`, `FaceProxy`

#### [NEW] [swap_stage.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/swap_stage.py)
- `run_swap_task()`, `run_swap_tasks_batch()`, `run_swap_tasks_parallel_single_batch()`
- `prepare_crop_frame()`, `normalize_swap_frame()`
- `implode_pixel_boost()`, `explode_pixel_boost()`
- **GPU optimization**: Pre-allocate numpy arrays, use contiguous memory, batch GPU transfers

#### [NEW] [mask_stage.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/mask_stage.py)
- `run_mask_task()`, `process_mask()`

#### [NEW] [enhance_stage.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/enhance_stage.py)
- `run_enhance_task()`, `run_enhance_tasks_batch()`

#### [NEW] [compose_stage.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/compose_stage.py)
- `compose_task()`, `paste_upscale()`, `create_landmark_mask()`
- `create_mouth_mask()`, `apply_mouth_area()`, `apply_color_transfer()`
- `blur_area()`, `simple_blend_with_mask()`

---

### 5. Staged Executor Decomposition (`roop/pipeline/staged_executor/`)

Split `staged_executor.py` (1617 lines) into:

#### [NEW] [executor.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/staged_executor/executor.py)
- Slim `StagedBatchExecutor` class — ~200 lines max
- `run()`, `process_image_entry()`, `process_video_entry()`

#### [NEW] [cache.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/staged_executor/cache.py)
- All JSON/pickle I/O helpers
- Manifest management, hashing, signatures
- `prepare_job()`, `cleanup_job_dir()`

#### [NEW] [detect_stage.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/staged_executor/detect_stage.py)
- `ensure_full_detect_stage()`, `ensure_detect_cache()`, `ensure_chunk_detect()`
- Packed detect iteration

#### [NEW] [swap_stage.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/staged_executor/swap_stage.py)
- `ensure_full_swap_stage()`, `ensure_swap_stage()`

#### [NEW] [mask_stage.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/staged_executor/mask_stage.py)
- `ensure_full_mask_stage()`, `ensure_mask_stage()`

#### [NEW] [enhance_stage.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/staged_executor/enhance_stage.py)
- `ensure_full_enhance_stage()`, `ensure_enhance_stage()`

#### [NEW] [compose_stage.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/staged_executor/compose_stage.py)
- `ensure_full_compose_stage()`, `compose_chunk()`, `compose_image_from_cache()`

#### [NEW] [video_iter.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/staged_executor/video_iter.py)
- `iter_video_chunk()` — threaded video frame producer

#### [NEW] [progress.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/pipeline/staged_executor/progress.py)
- `update_progress()`, `get_pipeline_steps()`, `get_stage_step_info()`

---

### 6. Processor Base Class (`roop/processors/base.py`)

#### [NEW] [base.py](file:///d:/backup/AI/roop-unleashed-wip/app/roop/processors/base.py)
- Abstract base class with clear interface: `Initialize()`, `Run()`, `RunBatch()`, `Release()`, `CreateWorkerProcessor()`
- Common `_resolve_batch_size_limit()`, `_effective_batch_size()` methods
- Each existing processor inherits from this — removes code duplication across ~12 processor files

---

## GPU & Parallel Processing Optimizations

> [!IMPORTANT]
> These optimizations are integrated into the new module structure, not separate steps.

### 1. Face Detection Stage
- **Current**: Sequential frame decode → sequential face detection
- **Optimized**: Use `ThreadPoolExecutor` for parallel CPU decode while GPU runs detection
- **In**: `pipeline/staged_executor/detect_stage.py`

### 2. Streaming Source Decode + Mask Stage
- **Current**: Read frame → rebuild alignment → process mask → next frame (serial)
- **Optimized**: Pre-fetch frames in background thread, batch mask operations on GPU
- **In**: `pipeline/staged_executor/mask_stage.py`

### 3. Face Swap Stage
- **Current**: Supports batching but with per-task array allocation overhead
- **Optimized**: Pre-allocate batch arrays, minimize CPU↔GPU copies, use `np.ascontiguousarray` on all inputs
- **In**: `pipeline/swap_stage.py`

### 4. Enhancement Stage
- **Current**: Good batching but no pre-allocation
- **Optimized**: Batch GPU transfers, pre-allocate output arrays
- **In**: `pipeline/enhance_stage.py`

### 5. Compose Stage (CPU-heavy)
- **Current**: Multi-threaded compositing with `ThreadPoolExecutor`
- **Optimized**: Use `concurrent.futures.ProcessPoolExecutor` for true parallel CPU compositing when worker count > 1, with shared memory for frame data
- **In**: `pipeline/staged_executor/compose_stage.py`

### 6. Video I/O
- **Current**: Already uses GPU-accelerated NVENC when available
- **Optimized**: Ensure hardware decoding via `cv2.CAP_PROP_HW_ACCELERATION` is always attempted
- **In**: `media/video_io.py`

---

## Cache Storage Optimization (Video-Encoded Cache)

> [!IMPORTANT]
> **Problem**: 2GB video → 50GB+ cache. Currently each swap/mask/enhance stage stores face crops as raw numpy arrays via `pickle.dumps()`. A 512×512×3 uint8 crop = 768KB raw. For a 10-min 30fps video with 1 face per frame = 18,000 crops × 768KB = **~14GB per stage**. With swap + mask + enhance = **~42GB** for a single video.

### Current Storage (per stage pack)
```
# write_stage_cache_map() in staged_executor.py
cache_map = {"f000001_t000": numpy_array_512x512x3, ...}  # raw uint8
pickle.dumps({"images": cache_map})  # no compression!
```

### New Storage: Video-Encoded Cache

#### [NEW] `pipeline/staged_executor/video_cache.py`

Replace raw pickle blobs with video-encoded cache files. Each stage pack stores its face crops as a **lossless/near-lossless video** file:

```python
class VideoStageCache:
    """
    Stores face crop images as a video file (H.264 CRF=0 for lossless,
    or CRF=4 for near-lossless ~20x compression).
    
    Layout per pack:
        swap/packs/000001_000256.mp4     # video of all crops grid-packed
        swap/packs/000001_000256.idx.bin  # index: {cache_key -> frame_number}
    
    Grid packing: Multiple 512×512 crops are tiled into larger frames
    (e.g., 4×4 grid = 2048×2048 frame) to maximize video codec efficiency.
    """
    
    def write(self, cache_path, cache_map: dict[str, np.ndarray]):
        # 1. Sort crops by cache_key for deterministic order
        # 2. Grid-pack crops into larger frames (efficient for H.264)
        # 3. Encode as video with FFMPEG_VideoWriter (CRF=4, libx264)
        # 4. Write index mapping cache_key -> (frame_idx, grid_x, grid_y)
    
    def read(self, cache_path) -> dict[str, np.ndarray]:
        # 1. Read index file
        # 2. Decode video frames
        # 3. Unpack grid into individual crops
        # 4. Return cache_map
    
    def read_keys(self, cache_path, keys: list[str]) -> dict[str, np.ndarray]:
        # Random access: seek to specific frames, extract specific grid cells
        # Enables partial cache reads without loading everything
```

### Compression Estimates

| Storage Method | Per Crop (512×512) | 18K Crops | Ratio |
|---|---|---|---|
| Raw pickle (current) | 768 KB | 13.5 GB | 1x |
| PNG compressed | ~200 KB | 3.5 GB | 3.8x |
| H.264 CRF=0 (lossless video) | ~40 KB | 700 MB | 19x |
| H.264 CRF=4 (visually lossless) | ~15 KB | 263 MB | 51x |

Using **CRF=4** (visually lossless, indistinguishable from original):
- 3 stages × 263 MB = **~800 MB total** vs 42 GB current = **52x reduction**
- Video seek for resume is faster than deserializing giant pickle blobs

### Grid Packing Strategy

Face crops are small (128×128 or 512×512). Video codecs work poorly on tiny frames. Solution: tile crops into larger composite frames:

```
┌──────┬──────┬──────┬──────┐
│crop_0│crop_1│crop_2│crop_3│  
├──────┼──────┼──────┼──────┤  4×4 grid of 512×512 crops
│crop_4│crop_5│crop_6│crop_7│  = 2048×2048 video frame
├──────┼──────┼──────┼──────┤  
│crop_8│crop_9│cr_10 │cr_11 │  16 crops per video frame
├──────┼──────┼──────┼──────┤  → 18000/16 = 1125 video frames
│cr_12 │cr_13 │cr_14 │cr_15 │  
└──────┴──────┴──────┴──────┘
```

### Backward Compatibility

- New cache reads check for `.mp4` + `.idx.bin` first, falls back to `.bin` (pickle)
- Old caches are auto-migrated on first access (read pickle → write video)
- `PIPELINE_VERSION` bump invalidates stale caches automatically

---

## Backward Compatibility

> [!WARNING]
> All import paths change. These files need import updates:

| Consumer | Nature of updates |
|----------|------------------|
| `app/run.py` | Update `from roop import core` → `from roop.core import app` |
| `app/ui/main.py` | Update all roop imports |
| `app/ui/tabs/*.py` | Update all roop imports |
| `app/tests/*.py` | Update all test imports |

All existing public API remains the same (same function names, same signatures). Only the import paths change.

---

## Resolved Decisions

1. **Legacy `batch_process_legacy()`**: Keep as fallback for "Legacy extract frames" mode in UI. Isolate in `pipeline/batch_executor.py`.
2. **`settings.py`**: Move to `roop/config/settings.py`. No external consumers identified.
3. **VR mode**: Keep but isolate in `utils/vr.py`. Minimal code footprint.

---

## Verification Plan

### Automated Tests
```bash
# Run existing test suite (all 17 test files must pass)
cd d:\backup\AI\roop-unleashed-wip\app
python -m pytest tests/ -v

# Import validation — ensure all new modules import cleanly
python -c "from roop.core.app import run; from roop.pipeline.staged_executor.executor import StagedBatchExecutor; from roop.face.detection import get_first_face; print('All imports OK')"
```

### Manual Verification
- Start the Gradio UI and verify all tabs load
- Process one image faceswap to verify pipeline works end-to-end
- Process one short video to verify staged executor works
- Check GPU utilization during processing matches or exceeds pre-refactor levels
- **Verify cache size**: Process a 2-min video, check `processing_cache/jobs/` size is <500MB (was ~5GB+)
