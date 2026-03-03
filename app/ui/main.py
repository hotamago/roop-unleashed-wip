import os
import time
import warnings
import gradio as gr
import roop.globals
import roop.metadata
import roop.utilities as util
import ui.globals as uii
import ui.globals

from ui.tabs.faceswap_tab import faceswap_tab
from ui.tabs.livecam_tab import livecam_tab
from ui.tabs.facemgr_tab import facemgr_tab
from ui.tabs.extras_tab import extras_tab
from ui.tabs.settings_tab import settings_tab

roop.globals.keep_fps = None
roop.globals.keep_frames = None
roop.globals.skip_audio = None
roop.globals.use_batch = None

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

def prepare_environment():
    roop.globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(roop.globals.output_path, exist_ok=True)
    if not roop.globals.CFG.use_os_temp_folder:
        os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(os.environ["TEMP"], exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]
    os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'

def run():
    from roop.core import decode_execution_providers, set_display_ui

    prepare_environment()

    set_display_ui(show_msg)
    if roop.globals.CFG.provider == "cuda" and util.has_cuda_device() == False:
       roop.globals.CFG.provider = "cpu"

    roop.globals.execution_providers = decode_execution_providers([roop.globals.CFG.provider])
    gputype = util.get_device()
    if gputype == 'cuda':
        util.print_cuda_info()
        
    print(f'Using provider {roop.globals.execution_providers} - Device:{gputype}')
    
    run_server = True
    uii.ui_restart_server = False
    mycss = """
        span {color: var(--block-info-text-color)}
        #fixedheight {
            max-height: 238.4px;
            overflow-y: auto !important;
        }
        .image-container.svelte-1l6wqyv {height: 100%}

    """

    while run_server:
        server_name = roop.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        server_port = roop.globals.CFG.server_port
        if server_port <= 0:
            server_port = None
        ssl_verify = True
        with gr.Blocks(title=f'{roop.metadata.name} {roop.metadata.version}', theme=roop.globals.CFG.selected_theme, css=mycss, delete_cache=(60, 86400)) as ui:
            with gr.Row(variant='compact'):
                    gr.Markdown(f"### [{roop.metadata.name} {roop.metadata.version}](https://github.com/C0untFloyd/roop-unleashed)")
                    gr.HTML(util.create_version_html(), elem_id="versions")
                    bt_save_session = gr.Button("💾 Save Settings", size='sm', variant='primary', scale=0)
                    bt_load_session = gr.Button("📂 Load Settings", size='sm', scale=0)
            bt_destfiles = faceswap_tab()
            livecam_tab()
            facemgr_tab()
            extras_tab(bt_destfiles)
            settings_tab()
            # Wire Save/Load after all tabs so ui.globals component refs are populated
            _comps = _session_components()
            bt_save_session.click(fn=save_session, inputs=_comps, outputs=[])
            bt_load_session.click(fn=load_session, inputs=[], outputs=_comps)
        launch_browser = roop.globals.CFG.launch_browser

        uii.ui_restart_server = False
        try:
            ui.queue().launch(inbrowser=launch_browser, server_name=server_name, server_port=server_port, share=roop.globals.CFG.server_share, ssl_verify=ssl_verify, prevent_thread_lock=True, show_error=True)
        except Exception as e:
            print(f'Exception {e} when launching Gradio Server!')
            uii.ui_restart_server = True
            run_server = False
        try:
            while uii.ui_restart_server == False:
                time.sleep(1.0)

        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()


def show_msg(msg: str):
    gr.Info(msg)


_SESSION_CFG_KEYS = [
    'face_detection_mode', 'num_swap_steps', 'selected_enhancer', 'max_face_distance',
    'subsample_upscale', 'blend_ratio', 'video_swapping_method', 'no_face_action',
    'vr_mode', 'autorotate_faces', 'skip_audio', 'keep_frames', 'wait_after_extraction',
    'output_method', 'mask_engine', 'mask_clip_text', 'show_mask_offsets',
    'restore_original_mouth', 'mask_top', 'mask_bottom', 'mask_left', 'mask_right',
    'mask_erosion', 'mask_blur',
]


def _session_components():
    return [
        ui.globals.ui_selected_face_detection,
        ui.globals.ui_num_swap_steps,
        ui.globals.ui_selected_enhancer,
        ui.globals.ui_max_face_distance,
        ui.globals.ui_upscale,
        ui.globals.ui_blend_ratio,
        ui.globals.ui_video_swapping_method,
        ui.globals.ui_no_face_action,
        ui.globals.ui_vr_mode,
        ui.globals.ui_autorotate,
        ui.globals.ui_skip_audio,
        ui.globals.ui_keep_frames,
        ui.globals.ui_wait_after_extraction,
        ui.globals.ui_output_method,
        ui.globals.ui_selected_mask_engine,
        ui.globals.ui_clip_text,
        ui.globals.ui_chk_showmaskoffsets,
        ui.globals.ui_chk_restoreoriginalmouth,
        ui.globals.ui_mask_top,
        ui.globals.ui_mask_bottom,
        ui.globals.ui_mask_left,
        ui.globals.ui_mask_right,
        ui.globals.ui_mask_erosion,
        ui.globals.ui_mask_blur,
    ]


def save_session(*values):
    cfg = roop.globals.CFG
    for key, val in zip(_SESSION_CFG_KEYS, values):
        setattr(cfg, key, val)
    cfg.save()
    gr.Info('Settings saved!')


def load_session():
    roop.globals.CFG.load()
    cfg = roop.globals.CFG
    return tuple(getattr(cfg, key) for key in _SESSION_CFG_KEYS)

