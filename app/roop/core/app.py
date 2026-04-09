#!/usr/bin/env python3

import os
import sys
import shutil
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

import warnings
from typing import List
import signal
import torch

try:
    import tensorrt  # registers TensorRT DLL paths on Windows so onnxruntime can find them
except ImportError:
    pass

import onnxruntime as ort
available_providers = ort.get_available_providers()
print("Available ONNX providers at startup:", available_providers)  # Debug

import pathlib
import argparse

from time import time
from roop.utils import print_cuda_info
import roop.config.globals
import roop.utils as util
import roop.media.ffmpeg_ops as ffmpeg
import ui.main as main
from roop.config.settings import Settings
from roop.face import extract_face_images
from roop.face_analytics_models import (
    ensure_face_detector_model_downloaded,
    ensure_face_landmarker_model_downloaded,
    ensure_face_masker_model_downloaded,
)
from roop.pipeline.entry import ProcessEntry
from roop.pipeline.batch_executor import ProcessMgr
from roop.pipeline.options import ProcessOptions
from roop.media.capturer import get_video_frame_total, release_video
from roop.memory import get_available_vram_gb
from roop.progress.status import finish_processing_status, set_processing_message
from roop.pipeline.staged_executor.executor import StagedBatchExecutor
from roop.pipeline.one_chain_executor import OneChainAllExecutor
from roop.core import providers as core_providers
from roop.core import resources as core_resources
from roop.face_swap_models import ensure_face_swap_model_downloaded, parse_face_swap_upscale_size


clip_text = None

call_display_ui = None

process_mgr = None


if 'ROCMExecutionProvider' in roop.config.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    roop.config.globals.headless = False

    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('--server_share', help='Public server', dest='server_share', action='store_true', default=False)
    program.add_argument('--cuda_device_id', help='Index of the cuda gpu to use', dest='cuda_device_id', type=int, default=0)
    roop.config.globals.startup_args = program.parse_args()
    # Always enable all processors when using GUI
    roop.config.globals.frame_processors = ['face_swapper', 'face_enhancer']


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return core_providers.encode_execution_providers(execution_providers)


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return core_providers.decode_execution_providers(execution_providers)
    
# Force GPU if available
# roop.config.globals.execution_providers = decode_execution_providers(['cuda'])
# print("Forced execution providers:", roop.config.globals.execution_providers)  # Debug

def suggest_max_memory() -> int:
    return core_resources.suggest_max_memory()


def suggest_execution_providers() -> List[str]:
    return core_providers.suggest_execution_providers()


def suggest_execution_threads() -> int:
    return core_resources.suggest_execution_threads()


def limit_resources() -> None:
    return core_resources.limit_resources()

def release_resources() -> None:
    global process_mgr
    core_resources.release_resources(process_mgr)
    process_mgr = None


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    
    download_directory_path = util.resolve_relative_path('../models')
    ensure_face_swap_model_downloaded(getattr(roop.config.globals.CFG, "face_swap_model", None))
    ensure_face_detector_model_downloaded(getattr(roop.config.globals.CFG, "face_detector_model", None))
    ensure_face_landmarker_model_downloaded(getattr(roop.config.globals.CFG, "face_landmarker_model", None))
    ensure_face_masker_model_downloaded(getattr(roop.config.globals.CFG, "face_masker_model", None))
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GFPGANv1.4.onnx'])
    util.conditional_download(download_directory_path, ['https://github.com/csxmli2016/DMDNet/releases/download/v1/DMDNet.pth'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/GPEN-BFR-512.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/restoreformer_plus_plus.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/xseg.onnx'])
    download_directory_path = util.resolve_relative_path('../models/CLIP')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/rd64-uni-refined.pth'])
    download_directory_path = util.resolve_relative_path('../models/CodeFormer')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/CodeFormerv0.1.onnx'])
    download_directory_path = util.resolve_relative_path('../models/Frame')
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_artistic.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/deoldify_stable.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/isnet-general-use.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x4.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/real_esrgan_x2.onnx'])
    util.conditional_download(download_directory_path, ['https://huggingface.co/countfloyd/deepfake/resolve/main/lsdir_x4.onnx'])

    print_cuda_info()  # Debug CUDA during pre-check


    if not shutil.which('ffmpeg'):
       update_status('ffmpeg is not installed.')
    return True

def set_display_ui(function):
    global call_display_ui

    call_display_ui = function


def update_status(message: str) -> None:
    global call_display_ui

    print(message)
    if call_display_ui is not None:
        call_display_ui(message)




def start() -> None:
    if roop.config.globals.headless:
        print('Headless mode currently unsupported - starting UI!')
        # faces = extract_face_images(roop.config.globals.source_path,  (False, 0))
        # roop.config.globals.INPUT_FACES.append(faces[roop.config.globals.source_face_index])
        # faces = extract_face_images(roop.config.globals.target_path,  (False, util.has_image_extension(roop.config.globals.target_path)))
        # roop.config.globals.TARGET_FACES.append(faces[roop.config.globals.target_face_index])
        # if 'face_enhancer' in roop.config.globals.frame_processors:
        #     roop.config.globals.selected_enhancer = 'GFPGAN'
       
    # FIX: was batch_process_regular(None, False, None) - only 3 args for a 10-param function.
    # Headless mode is unsupported in this fork; log and fall through to UI launch.
    print('Headless batch processing is not implemented - falling through to UI.')


def get_processing_plugins(masking_engine):
    processors = {  "faceswap": {}}
    if masking_engine is not None:
        processors.update({masking_engine: {}})
    
    if roop.config.globals.selected_enhancer == 'GFPGAN':
        processors.update({"gfpgan": {}})
    elif roop.config.globals.selected_enhancer == 'Codeformer':
        processors.update({"codeformer": {}})
    elif roop.config.globals.selected_enhancer == 'DMDNet':
        processors.update({"dmdnet": {}})
    elif roop.config.globals.selected_enhancer == 'GPEN':
        processors.update({"gpen": {}})
    elif roop.config.globals.selected_enhancer == 'Restoreformer++':
        processors.update({"restoreformer++": {}})
    return processors


def live_swap(frame, options):
    global process_mgr

    if frame is None:
        return frame

    if process_mgr is None:
        process_mgr = ProcessMgr(None)
    
#    if len(roop.config.globals.INPUT_FACESETS) <= selected_index:
#        selected_index = 0
    process_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, options)
    newframe = process_mgr.process_frame(frame)
    if newframe is None:
        return frame
    return newframe


def prepare_output_targets(files: list[ProcessEntry]) -> None:
    for index, entry in enumerate(files):
        fullname = entry.filename
        if util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(fullname, roop.config.globals.output_path, f'.{roop.config.globals.CFG.output_image_format}')
            destination = util.replace_template(destination, index=index)
            pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            entry.finalname = destination
        elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
            entry.finalname = util.get_destfilename_from_path(fullname, roop.config.globals.output_path, f'__temp.{roop.config.globals.CFG.output_video_format}')


def batch_process_regular(output_method, files:list[ProcessEntry], masking_engine:str, new_clip_text:str, processing_mode, imagemask, restore_original_mouth, num_swap_steps, progress, selected_index = 0) -> None:
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    mask = imagemask["layers"][0] if imagemask is not None else None
    if len(roop.config.globals.INPUT_FACESETS) <= selected_index:
        selected_index = 0
    options = ProcessOptions(get_processing_plugins(masking_engine), roop.config.globals.distance_threshold, roop.config.globals.blend_ratio,
                              roop.config.globals.face_swap_mode, selected_index, new_clip_text, mask, num_swap_steps,
                              roop.config.globals.subsample_size, False, restore_original_mouth)
    roop.config.globals.processing = True
    smart_total_steps = 4
    if "faceswap" in options.processors:
        smart_total_steps += 1
    if any(name.startswith("mask_") for name in options.processors):
        smart_total_steps += 1
    if any(name not in ("faceswap",) and not name.startswith("mask_") for name in options.processors):
        smart_total_steps += 1
    if output_method != "Virtual Camera":
        smart_total_steps += 1
    set_processing_message("Preparing faceswap job", stage="prepare", total_files=len(files), current_step=1, total_steps=smart_total_steps, detail="Initializing processor pipeline", memory_status=roop.config.globals.runtime_memory_status, force_log=True)
    if processing_mode == "Legacy extract frames":
        if process_mgr is None:
            process_mgr = ProcessMgr(progress)
        process_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, options)
        set_processing_message("Running legacy extract-frames flow", stage="legacy", current_step=1, total_steps=4, detail="Using compatibility pipeline", force_log=True)
        batch_process_legacy(output_method, files, False)
        return
    if processing_mode == "One-chain-all":
        prepare_output_targets(files)
        if output_method != "File":
            set_processing_message(
                "One-chain-all only supports file outputs; falling back to legacy flow",
                stage="legacy",
                current_step=1,
                total_steps=4,
                detail="Virtual camera streaming still uses the legacy compatibility pipeline",
                force_log=True,
            )
            if process_mgr is None:
                process_mgr = ProcessMgr(progress)
            process_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, options)
            batch_process_legacy(output_method, files, False)
            return
        set_processing_message(
            "Running one-chain-all flow",
            stage="prepare",
            current_step=1,
            total_steps=4,
            detail="Extract frames, run the full chain per frame, then merge with ffmpeg",
            force_log=True,
        )
        executor = OneChainAllExecutor(output_method, progress, options)
        executor.run(files)
        if roop.config.globals.processing:
            end_processing('Finished')
        else:
            end_processing('Processing stopped!')
        return

    prepare_output_targets(files)
    executor = StagedBatchExecutor(output_method, progress, options)
    detail = "Packed detect cache + streaming aligned rebuild" if output_method == "File" else "Chunked packed cache + streaming aligned rebuild"
    set_processing_message("Running smart staged flow", stage="prepare", current_step=1, total_steps=smart_total_steps, detail=detail, force_log=True)
    executor.run(files)
    if roop.config.globals.processing:
        end_processing('Finished')
    else:
        end_processing('Processing stopped!')

def batch_process_with_options(files:list[ProcessEntry], options, progress):
    global clip_text, process_mgr

    release_resources()
    limit_resources()
    if process_mgr is None:
        process_mgr = ProcessMgr(progress)
    process_mgr.initialize(roop.config.globals.INPUT_FACESETS, roop.config.globals.TARGET_FACES, options)
    roop.config.globals.keep_frames = False
    roop.config.globals.wait_after_extraction = False
    roop.config.globals.skip_audio = False
    batch_process_legacy("Files", files, True)



def batch_process_legacy(output_method, files:list[ProcessEntry], use_new_method) -> None:
    global clip_text, process_mgr

    roop.config.globals.processing = True

    # limit threads for some providers
    max_threads = suggest_execution_threads()
    if max_threads == 1:
        roop.config.globals.execution_threads = 1

    imagefiles:list[ProcessEntry] = []
    videofiles:list[ProcessEntry] = []
           
    update_status('Sorting videos/images')
    set_processing_message('Sorting inputs', stage='prepare', current_step=1, total_steps=4, detail='Building image/video queue', force_log=True)


    for index, f in enumerate(files):
        fullname = f.filename
        if util.has_image_extension(fullname):
            destination = util.get_destfilename_from_path(fullname, roop.config.globals.output_path, f'.{roop.config.globals.CFG.output_image_format}')
            destination = util.replace_template(destination, index=index)
            pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)
            f.finalname = destination
            imagefiles.append(f)

        elif util.is_video(fullname) or util.has_extension(fullname, ['gif']):
            destination = util.get_destfilename_from_path(fullname, roop.config.globals.output_path, f'__temp.{roop.config.globals.CFG.output_video_format}')
            f.finalname = destination
            videofiles.append(f)



    if(len(imagefiles) > 0):
        update_status('Processing image(s)')
        set_processing_message('Processing image batch', stage='images', target_name=f'{len(imagefiles)} image(s)', current_step=2, total_steps=2, detail='Legacy image batch pipeline', force_log=True)
        origimages = []
        fakeimages = []
        for f in imagefiles:
            origimages.append(f.filename)
            fakeimages.append(f.finalname)

        process_mgr.set_progress_context('images', f'{len(imagefiles)} image(s)', unit='images')
        process_mgr.run_batch(origimages, fakeimages, roop.config.globals.execution_threads)
        origimages.clear()
        fakeimages.clear()

    if(len(videofiles) > 0):
        for index,v in enumerate(videofiles):
            if not roop.config.globals.processing:
                end_processing('Processing stopped!')
                return
            fps = v.fps if v.fps > 0 else util.detect_fps(v.filename)
            if v.endframe == 0:
                v.endframe = get_video_frame_total(v.filename)

            is_streaming_only = output_method == "Virtual Camera"
            if is_streaming_only == False:
                update_status(f'Creating {os.path.basename(v.finalname)} with {fps} FPS...')

            start_processing = time()
            if is_streaming_only == False and roop.config.globals.keep_frames or not use_new_method:
                util.create_temp(v.filename)
                update_status('Extracting frames...')
                set_processing_message('Extracting frames', stage='extract', target_name=v.filename, file_index=index + 1, total_files=len(videofiles), current_step=2, total_steps=4, detail='Writing source frames to temp storage', force_log=True)
                ffmpeg.extract_frames(v.filename,v.startframe,v.endframe, fps)
                if not roop.config.globals.processing:
                    end_processing('Processing stopped!')
                    return

                temp_frame_paths = util.get_temp_frame_paths(v.filename)
                process_mgr.set_progress_context('legacy_swap', v.filename, index + 1, len(videofiles), unit='frames')
                process_mgr.run_batch(temp_frame_paths, temp_frame_paths, roop.config.globals.execution_threads)
                if not roop.config.globals.processing:
                    end_processing('Processing stopped!')
                    return
                if roop.config.globals.wait_after_extraction:
                    extract_path = os.path.dirname(temp_frame_paths[0])
                    util.open_folder(extract_path)
                    input("Press any key to continue...")
                    print("Resorting frames to create video")
                    util.sort_rename_frames(extract_path)                                    
                
                set_processing_message('Encoding video from processed frames', stage='encode', target_name=v.filename, file_index=index + 1, total_files=len(videofiles), current_step=3, total_steps=4, detail='Creating video from processed frame cache', force_log=True)
                ffmpeg.create_video(v.filename, v.finalname, fps)
                if not roop.config.globals.keep_frames:
                    util.delete_temp_frames(temp_frame_paths[0])
            else:
                if util.has_extension(v.filename, ['gif']):
                    skip_audio = True
                else:
                    skip_audio = roop.config.globals.skip_audio
                process_mgr.set_progress_context('legacy_in_memory', v.filename, index + 1, len(videofiles), unit='frames')
                process_mgr.run_batch_inmem(output_method, v.filename, v.finalname, v.startframe, v.endframe, fps,roop.config.globals.execution_threads, skip_audio)
                
            if not roop.config.globals.processing:
                end_processing('Processing stopped!')
                return
            
            video_file_name = v.finalname
            if os.path.isfile(video_file_name):
                destination = ''
                if util.has_extension(v.filename, ['gif']):
                    gifname = util.get_destfilename_from_path(v.filename, roop.config.globals.output_path, '.gif')
                    destination = util.replace_template(gifname, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    update_status('Creating final GIF')
                    set_processing_message('Creating final GIF', stage='mux', target_name=v.filename, file_index=index + 1, total_files=len(videofiles), current_step=4, total_steps=4, detail='Encoding GIF output', force_log=True)
                    ffmpeg.create_gif_from_video(video_file_name, destination)
                    if os.path.isfile(destination):
                        os.remove(video_file_name)
                else:
                    skip_audio = roop.config.globals.skip_audio
                    destination = util.replace_template(video_file_name, index=index)
                    pathlib.Path(os.path.dirname(destination)).mkdir(parents=True, exist_ok=True)

                    if not skip_audio:
                        set_processing_message('Restoring audio track', stage='mux', target_name=v.filename, file_index=index + 1, total_files=len(videofiles), current_step=4, total_steps=4, detail='Muxing processed video with source audio', force_log=True)
                        ffmpeg.restore_audio(video_file_name, v.filename, v.startframe, v.endframe, destination)
                        if os.path.isfile(destination):
                            os.remove(video_file_name)
                    else:
                        shutil.move(video_file_name, destination)

            elif is_streaming_only == False:
                update_status(f'Failed processing {os.path.basename(v.finalname)}!')
            elapsed_time = time() - start_processing
            average_fps = (v.endframe - v.startframe) / elapsed_time
            update_status(f'\nProcessing {os.path.basename(destination)} took {elapsed_time:.2f} secs, {average_fps:.2f} frames/s')
            import gc
            gc.collect()
            try:
                if torch.cuda.is_available():
                    with torch.cuda.device(roop.config.globals.cuda_device_id):
                        torch.cuda.empty_cache()
            except Exception:
                pass
    end_processing('Finished')


def end_processing(msg:str):
    lowered = msg.lower()
    status = 'completed'
    if 'stopped' in lowered or 'abort' in lowered:
        status = 'stopped'
    elif 'fail' in lowered or 'error' in lowered:
        status = 'error'
    finish_processing_status(msg, status=status)
    update_status(msg)
    roop.config.globals.target_folder_path = None
    roop.config.globals.processing = False
    release_resources()


def destroy() -> None:
    if roop.config.globals.target_path:
        util.clean_temp(roop.config.globals.target_path)
    release_resources()        
    sys.exit()


def run() -> None:
    parse_args()
    roop.config.globals.CFG = Settings('config.yaml')
    roop.config.globals.subsample_size = parse_face_swap_upscale_size(
        getattr(roop.config.globals.CFG, "subsample_upscale", "256px"),
        getattr(roop.config.globals.CFG, "face_swap_model", None),
    )
    if not pre_check():
        return
    roop.config.globals.cuda_device_id = roop.config.globals.startup_args.cuda_device_id
    roop.config.globals.execution_threads = roop.config.globals.CFG.max_threads
    roop.config.globals.video_encoder = roop.config.globals.CFG.output_video_codec
    roop.config.globals.video_quality = roop.config.globals.CFG.video_quality
    roop.config.globals.max_memory = None
    roop.config.globals.max_vram = None
    if roop.config.globals.startup_args.server_share:
        roop.config.globals.CFG.server_share = True
    main.run()

