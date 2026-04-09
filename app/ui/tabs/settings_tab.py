import shutil
import os
import gradio as gr
import roop.config.globals
import ui.globals
from roop.face_analytics_models import (
    get_face_detector_model_choices,
    get_face_detector_model_hint,
    get_face_detector_model_key,
    get_face_landmarker_model_choices,
    get_face_landmarker_model_hint,
    get_face_landmarker_model_key,
    get_face_masker_model_choices,
    get_face_masker_model_hint,
    get_face_masker_model_key,
)
from roop.face_swap_models import (
    ensure_face_swap_model_downloaded,
    get_face_swap_model_choices,
    get_face_swap_model_hint,
    get_face_swap_model_key,
    get_face_swap_upscale_choices,
    get_face_swap_upscale_hint,
    normalize_face_swap_upscale,
    parse_face_swap_upscale_size,
)
from roop.utils.cache_paths import get_jobs_root
from roop.memory.planner import describe_memory_plan, resolve_memory_plan

image_formats = ['jpg','png', 'webp']
video_formats = ['avi','mkv', 'mp4', 'webm']
video_codecs = ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']
providerlist = None
CONFIG_SAVE_DIR = "saved_configs"

settings_controls = []


def build_face_swap_upscale_update(model_name=None, selected_upscale=None):
    current_model = get_face_swap_model_key(
        model_name or getattr(roop.config.globals.CFG, "face_swap_model", None)
    )
    normalized_upscale = normalize_face_swap_upscale(
        selected_upscale if selected_upscale is not None else getattr(roop.config.globals.CFG, "subsample_upscale", "256px"),
        current_model,
    )
    return gr.Dropdown(
        choices=get_face_swap_upscale_choices(current_model),
        value=normalized_upscale,
        info=get_face_swap_upscale_hint(current_model),
        interactive=True,
    )


def on_face_swap_model_changed(selected_model, current_upscale):
    model_key = get_face_swap_model_key(selected_model)
    normalized_upscale = normalize_face_swap_upscale(current_upscale, model_key)
    ensure_face_swap_model_downloaded(model_key)
    roop.config.globals.CFG.face_swap_model = model_key
    roop.config.globals.CFG.subsample_upscale = normalized_upscale
    roop.config.globals.subsample_size = parse_face_swap_upscale_size(normalized_upscale, model_key)
    from roop.core.app import release_resources

    release_resources()
    return (
        update_memory_status(),
        build_face_swap_upscale_update(model_key, normalized_upscale),
        gr.Markdown(value=get_face_swap_model_hint(model_key)),
    )


def on_face_analytics_model_changed(new_value, attribname):
    if attribname == "face_detector_model":
        normalized_value = get_face_detector_model_key(new_value)
        hint = get_face_detector_model_hint(normalized_value)
    elif attribname == "face_landmarker_model":
        normalized_value = get_face_landmarker_model_key(new_value)
        hint = get_face_landmarker_model_hint(normalized_value)
    else:
        normalized_value = get_face_masker_model_key(new_value)
        hint = get_face_masker_model_hint(normalized_value)

    setattr(roop.config.globals.CFG, attribname, normalized_value)

    if attribname in ("face_detector_model", "face_landmarker_model"):
        from roop.face import release_face_analyser

        release_face_analyser()

    return update_memory_status(), gr.Markdown(value=hint)

def settings_tab():
    from roop.core.providers import suggest_execution_providers
    global providerlist

    settings_controls.clear()
    providerlist = suggest_execution_providers()
    initial_memory_status = describe_memory_plan(resolve_memory_plan())
    with gr.Tab("Settings"):
        with gr.Row():
            with gr.Column():
                settings_controls.append(gr.Checkbox(label="Public Server", value=roop.config.globals.CFG.server_share, elem_id='server_share', interactive=True))
                settings_controls.append(gr.Checkbox(label='Clear output folder before each run', value=roop.config.globals.CFG.clear_output, elem_id='clear_output', interactive=True))
                output_template = gr.Textbox(
                    label="Filename Output Template", 
                    info="(file extension is added automatically) | Tokens: {file}, {time}, {index}, {timestamp}", 
                    lines=1, 
                    placeholder='{file}_{timestamp}', 
                    value=roop.config.globals.CFG.output_template
                    )
            with gr.Column():
                input_server_name = gr.Textbox(label="Server Name", lines=1, info="Leave blank to run locally", value=roop.config.globals.CFG.server_name)
            with gr.Column():
                input_server_port = gr.Number(label="Server Port", precision=0, info="Leave at 0 to use default", value=roop.config.globals.CFG.server_port)
        with gr.Row():
            with gr.Column():
                settings_controls.append(gr.Dropdown(providerlist, label="Provider", value=roop.config.globals.CFG.provider, elem_id='provider', interactive=True))
                chk_det_size = gr.Checkbox(label="Use default Det-Size", value=True, elem_id='default_det_size', interactive=True)
                settings_controls.append(gr.Checkbox(label="Force CPU for Face Analyser", value=roop.config.globals.CFG.force_cpu, elem_id='force_cpu', interactive=True))
                ui.globals.ui_face_detector_model = gr.Dropdown(
                    get_face_detector_model_choices(),
                    label="Face Detector Model",
                    info="Pick the detector used before analytics enrichment.",
                    value=get_face_detector_model_key(getattr(roop.config.globals.CFG, "face_detector_model", None)),
                    elem_id='face_detector_model',
                    interactive=True,
                )
                face_detector_model_hint = gr.Markdown(
                    get_face_detector_model_hint(getattr(roop.config.globals.CFG, "face_detector_model", None))
                )
                ui.globals.ui_face_landmarker_model = gr.Dropdown(
                    get_face_landmarker_model_choices(),
                    label="Face Landmarker Model",
                    info="Pick the optional 68-point landmarker added on top of the current compatibility path.",
                    value=get_face_landmarker_model_key(getattr(roop.config.globals.CFG, "face_landmarker_model", None)),
                    elem_id='face_landmarker_model',
                    interactive=True,
                )
                face_landmarker_model_hint = gr.Markdown(
                    get_face_landmarker_model_hint(getattr(roop.config.globals.CFG, "face_landmarker_model", None))
                )
                ui.globals.ui_face_masker_model = gr.Dropdown(
                    get_face_masker_model_choices(),
                    label="Face Masker Model",
                    info="Used when the DFL XSeg mask engine is active.",
                    value=get_face_masker_model_key(getattr(roop.config.globals.CFG, "face_masker_model", None)),
                    elem_id='face_masker_model',
                    interactive=True,
                )
                face_masker_model_hint = gr.Markdown(
                    get_face_masker_model_hint(getattr(roop.config.globals.CFG, "face_masker_model", None))
                )
                ui.globals.ui_face_swap_model = gr.Dropdown(
                    get_face_swap_model_choices(),
                    label="Face Swap Model",
                    info="Pick the ONNX model used for swapping. Models download on first use.",
                    value=get_face_swap_model_key(getattr(roop.config.globals.CFG, "face_swap_model", None)),
                    elem_id='face_swap_model',
                    interactive=True,
                )
                face_swap_model_hint = gr.Markdown(
                    get_face_swap_model_hint(getattr(roop.config.globals.CFG, "face_swap_model", None))
                )
                max_threads = gr.Slider(1, 32, value=roop.config.globals.CFG.max_threads, label="Max. Number of Threads", info='default: 3', step=1.0, interactive=True)
            with gr.Column():
                staged_chunk_size = gr.Slider(8, 480, value=roop.config.globals.CFG.staged_chunk_size, label="Staged Chunk Size", info='Frames scheduled per staged chunk', step=1.0, interactive=True)
                prefetch_frames = gr.Slider(1, 256, value=roop.config.globals.CFG.prefetch_frames, label="Prefetch Frames", info='Bounded decode queue size for staged reads', step=1.0, interactive=True)
                detect_pack_frame_count = gr.Slider(8, 1024, value=roop.config.globals.CFG.detect_pack_frame_count, label="Detect Pack Frame Count", info='Packed detect metadata frames per cache blob', step=8.0, interactive=True)
                detect_batch_size = gr.Slider(1, 128, value=getattr(roop.config.globals.CFG, "detect_batch_size", 8), label="Detect Batch Size", info='Requested detector batch size when the selected detector model supports batched inference.', step=1.0, interactive=True)
                detect_single_batch_workers = gr.Slider(1, 8, value=getattr(roop.config.globals.CFG, "detect_single_batch_workers", 1), label="Detect Single-Batch Workers", info='Parallel detector worker sessions used when the selected detector model only supports batch=1.', step=1.0, interactive=True)
                single_batch_workers = gr.Slider(1, 8, value=roop.config.globals.CFG.single_batch_workers, label="Single-Batch Workers", info='Parallel worker sessions for models limited to batch=1. GPU runtime is capped to 1 worker to avoid session/VRAM thrash.', step=1.0, interactive=True)
                settings_controls.append(gr.Dropdown(image_formats, label="Image Output Format", info='default: png', value=roop.config.globals.CFG.output_image_format, elem_id='output_image_format', interactive=True))
            with gr.Column():
                swap_batch_size = gr.Slider(1, 256, value=roop.config.globals.CFG.swap_batch_size, label="Swap Batch Size", info='Requested batch size for swap-capable models', step=1.0, interactive=True)
                mask_batch_size = gr.Slider(1, 512, value=roop.config.globals.CFG.mask_batch_size, label="Mask Batch Size", info='Requested batch size for masking models', step=1.0, interactive=True)
                enhance_batch_size = gr.Slider(1, 128, value=roop.config.globals.CFG.enhance_batch_size, label="Enhance Batch Size", info='Requested batch size for enhancer models', step=1.0, interactive=True)
                settings_controls.append(gr.Dropdown(video_codecs, label="Video Codec", info='default: libx264', value=roop.config.globals.CFG.output_video_codec, elem_id='output_video_codec', interactive=True))
                settings_controls.append(gr.Dropdown(video_formats, label="Video Output Format", info='default: mp4', value=roop.config.globals.CFG.output_video_format, elem_id='output_video_format', interactive=True))
                video_quality = gr.Slider(0, 100, value=roop.config.globals.CFG.video_quality, label="Video Quality (crf)", info='default: 14', step=1.0, interactive=True)
            with gr.Column():
                with gr.Group():
                    settings_controls.append(gr.Checkbox(label='Use OS temp folder', value=roop.config.globals.CFG.use_os_temp_folder, elem_id='use_os_temp_folder', interactive=True))
                    settings_controls.append(gr.Checkbox(label='Show video in browser (re-encodes output)', value=roop.config.globals.CFG.output_show_video, elem_id='output_show_video', interactive=True))
                    memory_status = gr.Markdown(initial_memory_status)
                button_apply_restart = gr.Button("Restart Server", variant='primary')
                button_clean_temp = gr.Button("Clean temp folder")
                button_clean_cache = gr.Button("Clean Processing Cache")
                button_apply_settings = gr.Button("Apply Settings")
                ui.globals.ui_memory_status = memory_status

    chk_det_size.select(fn=on_option_changed)

    # Settings
    for s in settings_controls:
        s.select(fn=on_settings_changed, outputs=[memory_status])
    ui.globals.ui_face_detector_model.change(
        fn=lambda value: on_face_analytics_model_changed(value, "face_detector_model"),
        inputs=[ui.globals.ui_face_detector_model],
        outputs=[memory_status, face_detector_model_hint],
        show_progress='hidden',
    )
    ui.globals.ui_face_landmarker_model.change(
        fn=lambda value: on_face_analytics_model_changed(value, "face_landmarker_model"),
        inputs=[ui.globals.ui_face_landmarker_model],
        outputs=[memory_status, face_landmarker_model_hint],
        show_progress='hidden',
    )
    ui.globals.ui_face_masker_model.change(
        fn=lambda value: on_face_analytics_model_changed(value, "face_masker_model"),
        inputs=[ui.globals.ui_face_masker_model],
        outputs=[memory_status, face_masker_model_hint],
        show_progress='hidden',
    )
    ui.globals.ui_face_swap_model.change(
        fn=on_face_swap_model_changed,
        inputs=[ui.globals.ui_face_swap_model, ui.globals.ui_upscale],
        outputs=[memory_status, ui.globals.ui_upscale, face_swap_model_hint],
        show_progress='hidden',
    )
    max_threads.input(fn=lambda a,b='max_threads':on_settings_changed_misc(a,b), inputs=[max_threads], outputs=[memory_status])
    staged_chunk_size.input(fn=lambda a,b='staged_chunk_size':on_settings_changed_misc(a,b), inputs=[staged_chunk_size], outputs=[memory_status])
    prefetch_frames.input(fn=lambda a,b='prefetch_frames':on_settings_changed_misc(a,b), inputs=[prefetch_frames], outputs=[memory_status])
    detect_pack_frame_count.input(fn=lambda a,b='detect_pack_frame_count':on_settings_changed_misc(a,b), inputs=[detect_pack_frame_count], outputs=[memory_status])
    detect_batch_size.input(fn=lambda a,b='detect_batch_size':on_settings_changed_misc(a,b), inputs=[detect_batch_size], outputs=[memory_status])
    detect_single_batch_workers.input(fn=lambda a,b='detect_single_batch_workers':on_settings_changed_misc(a,b), inputs=[detect_single_batch_workers], outputs=[memory_status])
    single_batch_workers.input(fn=lambda a,b='single_batch_workers':on_settings_changed_misc(a,b), inputs=[single_batch_workers], outputs=[memory_status])
    swap_batch_size.input(fn=lambda a,b='swap_batch_size':on_settings_changed_misc(a,b), inputs=[swap_batch_size], outputs=[memory_status])
    mask_batch_size.input(fn=lambda a,b='mask_batch_size':on_settings_changed_misc(a,b), inputs=[mask_batch_size], outputs=[memory_status])
    enhance_batch_size.input(fn=lambda a,b='enhance_batch_size':on_settings_changed_misc(a,b), inputs=[enhance_batch_size], outputs=[memory_status])
    video_quality.input(fn=lambda a,b='video_quality':on_settings_changed_misc(a,b), inputs=[video_quality], outputs=[memory_status])

    button_clean_temp.click(fn=clean_temp)
    button_clean_cache.click(fn=clean_processing_cache)
    button_apply_settings.click(apply_settings, inputs=[input_server_name, input_server_port, output_template], outputs=[memory_status])
    button_apply_restart.click(restart)



#SAVE/LOAD CONFIGS
def get_saved_configs():
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)
    return [f.replace(".yaml", "") for f in os.listdir(CONFIG_SAVE_DIR) if f.endswith(".yaml")]

def save_config(name):
    if not name or name.strip() == "":
        gr.Warning("Please enter a config name!")
        return gr.Dropdown(choices=get_saved_configs())
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)
    import shutil
    shutil.copy("config.yaml", os.path.join(CONFIG_SAVE_DIR, f"{name.strip()}.yaml"))
    gr.Info(f"Config '{name}' saved!")
    return gr.Dropdown(choices=get_saved_configs(), value=name.strip())

def load_config(name):
    if not name:
        gr.Warning("No config selected!")
        return
    path = os.path.join(CONFIG_SAVE_DIR, f"{name}.yaml")
    if not os.path.exists(path):
        gr.Warning(f"Config '{name}' not found!")
        return
    from roop.config.settings import Settings
    from roop.face import release_face_analyser
    roop.config.globals.CFG = Settings(path)
    release_face_analyser()
    gr.Info(f"Config '{name}' loaded! Click 'Restart Server' to fully apply.")

def on_option_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.config.globals, attribname):
            setattr(roop.config.globals, attribname, evt.selected)
            if attribname == "default_det_size":
                from roop.face import release_face_analyser

                release_face_analyser()
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.config.globals, attribname):
            setattr(roop.config.globals, attribname, evt.value)
            return
    raise gr.Error(f'Unhandled Setting for {evt.target}')


def on_settings_changed_misc(new_val, attribname):
    if hasattr(roop.config.globals.CFG, attribname):
        if attribname in (
            'max_threads',
            'video_quality',
            'staged_chunk_size',
            'prefetch_frames',
            'detect_pack_frame_count',
            'detect_batch_size',
            'detect_single_batch_workers',
            'single_batch_workers',
            'swap_batch_size',
            'mask_batch_size',
            'enhance_batch_size',
        ):
            new_val = int(new_val)
        setattr(roop.config.globals.CFG, attribname, new_val)
    else:
        print("Didn't find attrib!")
    return update_memory_status()


def on_settings_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.config.globals.CFG, attribname):
            setattr(roop.config.globals.CFG, attribname, evt.selected)
            if attribname == "force_cpu":
                from roop.face import release_face_analyser

                release_face_analyser()
            return update_memory_status()
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.config.globals.CFG, attribname):
            setattr(roop.config.globals.CFG, attribname, evt.value)
            if attribname == "provider":
                from roop.face import release_face_analyser

                release_face_analyser()
            return update_memory_status()
            
    raise gr.Error(f'Unhandled Setting for {evt.target}')

def clean_temp():
    from ui.main import prepare_environment
    
    ui.globals.ui_input_thumbs.clear()
    roop.config.globals.INPUT_FACESETS.clear()
    roop.config.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs = []
    if not roop.config.globals.CFG.use_os_temp_folder:
        shutil.rmtree(os.environ["TEMP"])
    prepare_environment()
    gr.Info('Temp Files removed')
    return None,None,None,None


def apply_settings(input_server_name, input_server_port, output_template):
    from ui.main import show_msg

    roop.config.globals.CFG.server_name = input_server_name
    roop.config.globals.CFG.server_port = input_server_port
    roop.config.globals.CFG.output_template = output_template
    roop.config.globals.CFG.save()
    show_msg('Settings saved')
    return update_memory_status()


def restart():
    ui.globals.ui_restart_server = True


def update_memory_status():
    plan = resolve_memory_plan()
    return gr.Markdown(value=describe_memory_plan(plan))


def clean_processing_cache():
    jobs_dir = str(get_jobs_root())
    if os.path.isdir(jobs_dir):
        shutil.rmtree(jobs_dir)
    os.makedirs(jobs_dir, exist_ok=True)
    gr.Info('Processing cache removed')

