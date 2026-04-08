# Refactor Task Checklist

## Phase 1: Create Directory Structure
- [ ] Create all new directories with `__init__.py` files
- [ ] Create `roop/config/`, `roop/core/`, `roop/face/`, `roop/pipeline/staged_executor/`, `roop/media/`, `roop/onnx/`, `roop/memory/`, `roop/progress/`, `roop/utils/`

## Phase 2: Config & Types Layer
- [ ] Create `roop/config/globals.py` (cleaned globals)
- [ ] Create `roop/config/types.py` (merged typing + FaceSet)
- [ ] Create `roop/config/settings.py` (moved from app/settings.py)

## Phase 3: Utils, Media, ONNX, Memory, Progress
- [ ] Create `roop/utils/io.py` (file helpers from utilities.py)
- [ ] Create `roop/utils/download.py` (conditional_download)
- [ ] Create `roop/utils/platform.py` (platform detection)
- [ ] Create `roop/utils/cache_paths.py` (from cache_paths.py)
- [ ] Create `roop/utils/template_parser.py` (from template_parser.py)
- [ ] Create `roop/utils/vr.py` (from vr_util.py)
- [ ] Create `roop/media/capturer.py` (from capturer.py)
- [ ] Create `roop/media/ffmpeg_ops.py` (from util_ffmpeg.py)
- [ ] Create `roop/media/ffmpeg_writer.py` (from ffmpeg_writer.py)
- [ ] Create `roop/media/video_io.py` (from video_io.py)
- [ ] Create `roop/media/stream_writer.py` (from StreamWriter.py)
- [ ] Create `roop/onnx/batch.py` (from onnx_batch.py)
- [ ] Create `roop/onnx/runtime.py` (from onnx_runtime.py)
- [ ] Create `roop/memory/planner.py` (from memory.py)
- [ ] Create `roop/progress/status.py` (from progress_status.py)

## Phase 4: Face Module
- [ ] Create `roop/face/analyser.py`
- [ ] Create `roop/face/detection.py`
- [ ] Create `roop/face/alignment.py`
- [ ] Create `roop/face/rotation.py`
- [ ] Create `roop/face/geometry.py`

## Phase 5: Pipeline Module (ProcessMgr decomposition)
- [ ] Create `roop/pipeline/options.py`
- [ ] Create `roop/pipeline/entry.py`
- [ ] Create `roop/pipeline/faceset.py`
- [ ] Create `roop/pipeline/face_targeting.py`
- [ ] Create `roop/pipeline/face_serializer.py`
- [ ] Create `roop/pipeline/swap_stage.py`
- [ ] Create `roop/pipeline/mask_stage.py`
- [ ] Create `roop/pipeline/enhance_stage.py`
- [ ] Create `roop/pipeline/compose_stage.py`
- [ ] Create `roop/pipeline/batch_executor.py`
- [ ] Create `roop/processors/base.py`

## Phase 6: Staged Executor Decomposition
- [ ] Create `roop/pipeline/staged_executor/cache.py`
- [ ] Create `roop/pipeline/staged_executor/video_cache.py`
- [ ] Create `roop/pipeline/staged_executor/video_iter.py`
- [ ] Create `roop/pipeline/staged_executor/progress.py`
- [ ] Create `roop/pipeline/staged_executor/detect_stage.py`
- [ ] Create `roop/pipeline/staged_executor/swap_stage.py`
- [ ] Create `roop/pipeline/staged_executor/mask_stage.py`
- [ ] Create `roop/pipeline/staged_executor/enhance_stage.py`
- [ ] Create `roop/pipeline/staged_executor/compose_stage.py`
- [ ] Create `roop/pipeline/staged_executor/chunk_processor.py`
- [ ] Create `roop/pipeline/staged_executor/executor.py`

## Phase 7: Core Application
- [ ] Create `roop/core/providers.py`
- [ ] Create `roop/core/resources.py`
- [ ] Create `roop/core/app.py`

## Phase 8: Update External Consumers & Cleanup
- [ ] Update `app/run.py` imports
- [ ] Update `app/ui/main.py` imports
- [ ] Update `app/ui/tabs/faceswap_tab.py` imports
- [ ] Update `app/ui/tabs/facemgr_tab.py` imports
- [ ] Update `app/ui/tabs/settings_tab.py` imports
- [ ] Update `app/ui/tabs/extras_tab.py` imports
- [ ] Update `app/ui/globals.py` imports
- [ ] Update processor imports
- [ ] Update test imports
- [ ] Delete old files
- [ ] Clean up `__pycache__` directories

## Phase 9: Verification
- [ ] Run import validation
- [ ] Run test suite
- [ ] Verify UI launch
