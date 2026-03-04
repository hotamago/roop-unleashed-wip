import os
import gradio as gr
import roop.utilities as util
import roop.util_ffmpeg as ffmpeg
import roop.globals

RESOLUTION_CHOICES = ["1280x720", "1920x1080", "854x480", "3840x2160"]
ROTATION_CHOICES   = ["90° Clockwise", "90° Counter-clockwise", "180°", "Flip Horizontal", "Flip Vertical"]


def extras_tab(bt_destfiles=None):
    with gr.Tab("🎉 Extras"):

        # ── Upload + Preview ──────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=1):
                files_to_process = gr.Files(
                    label="Upload file",
                    file_count="multiple",
                    file_types=["image", "video"],
                )
            with gr.Column(scale=2):
                preview_image = gr.Image(
                    label="Preview", visible=False, interactive=False,
                    show_download_button=False,
                )
                preview_video = gr.Video(
                    label="Preview", visible=False, interactive=False,
                )

        # ── Operations row ────────────────────────────────────────────
        with gr.Row(equal_height=True):
            with gr.Group():
                gr.Markdown("#### Resolution")
                resize_resolution = gr.Dropdown(
                    RESOLUTION_CHOICES, value="1280x720",
                    label="Target resolution", show_label=False,
                )
                btn_resize = gr.Button("Apply", variant="primary", size="sm")

            with gr.Group():
                gr.Markdown("#### Rotate / Flip")
                rotation_choice = gr.Dropdown(
                    ROTATION_CHOICES, value="90° Clockwise",
                    label="Transform", show_label=False,
                )
                btn_rotate = gr.Button("Apply", variant="primary", size="sm")

            with gr.Group():
                gr.Markdown("#### Change FPS")
                fps_value = gr.Slider(1, 120, value=30, step=1, label="FPS", show_label=False)
                btn_fps = gr.Button("Apply", variant="primary", size="sm")

        # ── Crop ──────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("#### Crop  *(trim from each edge as % of frame size)*")
            with gr.Row():
                crop_left   = gr.Slider(0, 49, value=0, step=1, label="Left %")
                crop_right  = gr.Slider(0, 49, value=0, step=1, label="Right %")
                crop_top    = gr.Slider(0, 49, value=0, step=1, label="Top %")
                crop_bottom = gr.Slider(0, 49, value=0, step=1, label="Bottom %")
            btn_crop = gr.Button("Apply Crop", variant="primary", size="sm")

        # ── Output ────────────────────────────────────────────────────
        with gr.Row():
            send_to_faceswap_btn = gr.Button(
                "↗ Send to Face Swap", size="sm",
                visible=bt_destfiles is not None,
            )
        with gr.Row():
            extra_files_output = gr.Files(label="Output", file_count="multiple")

    # ── Event wiring ──────────────────────────────────────────────────
    files_to_process.upload(
        fn=on_file_upload,
        inputs=[files_to_process],
        outputs=[preview_image, preview_video],
        show_progress="hidden",
    )

    btn_resize.click(fn=on_resize, inputs=[files_to_process, resize_resolution], outputs=[extra_files_output])
    btn_rotate.click(fn=on_rotate, inputs=[files_to_process, rotation_choice],   outputs=[extra_files_output])
    btn_fps.click(   fn=on_fps,    inputs=[files_to_process, fps_value],          outputs=[extra_files_output])
    btn_crop.click(  fn=on_crop,   inputs=[files_to_process, crop_left, crop_right, crop_top, crop_bottom], outputs=[extra_files_output])

    if bt_destfiles is not None:
        send_to_faceswap_btn.click(fn=on_send_to_faceswap, inputs=[extra_files_output], outputs=[bt_destfiles])


# ── Handlers ──────────────────────────────────────────────────────────

def on_file_upload(files):
    if not files:
        return gr.update(visible=False, value=None), gr.update(visible=False, value=None)
    path = files[0].name if hasattr(files[0], 'name') else str(files[0])
    if util.is_image(path):
        return gr.update(visible=True, value=path), gr.update(visible=False, value=None)
    if util.is_video(path):
        return gr.update(visible=False, value=None), gr.update(visible=True, value=path)
    return gr.update(visible=False), gr.update(visible=False)


def _paths(files):
    if not files:
        return []
    return [f.name if hasattr(f, 'name') else str(f) for f in files]


def on_resize(files, resolution):
    paths = _paths(files)
    if not paths:
        return None
    width, height = (int(x) for x in resolution.split('x'))
    out = []
    for f in paths:
        dest = util.get_destfilename_from_path(f, roop.globals.output_path, f'_{width}x{height}')
        if ffmpeg.resize_video(f, dest, width, height):
            out.append(dest)
        else:
            gr.Warning(f'Resize failed for {os.path.basename(f)}')
    return out or None


def on_rotate(files, transform):
    paths = _paths(files)
    if not paths:
        return None
    out = []
    for f in paths:
        dest = util.get_destfilename_from_path(f, roop.globals.output_path, '_rot')
        if ffmpeg.rotate_media(f, dest, transform):
            out.append(dest)
        else:
            gr.Warning(f'Rotate failed for {os.path.basename(f)}')
    return out or None


def on_fps(files, fps):
    paths = _paths(files)
    if not paths:
        return None
    out = []
    for f in paths:
        if not util.is_video(f):
            gr.Warning(f'{os.path.basename(f)} is not a video — FPS change skipped')
            continue
        dest = util.get_destfilename_from_path(f, roop.globals.output_path, f'_{int(fps)}fps')
        if ffmpeg.change_fps(f, dest, fps):
            out.append(dest)
        else:
            gr.Warning(f'FPS change failed for {os.path.basename(f)}')
    return out or None


def on_crop(files, left, right, top, bottom):
    paths = _paths(files)
    if not paths:
        return None
    out = []
    for f in paths:
        dest = util.get_destfilename_from_path(f, roop.globals.output_path, '_crop')
        if ffmpeg.crop_media(f, dest, left, right, top, bottom):
            out.append(dest)
        else:
            gr.Warning(f'Crop failed for {os.path.basename(f)}')
    return out or None


def on_send_to_faceswap(files):
    if files is None:
        return None
    return [f.name for f in files]
