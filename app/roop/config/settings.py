import yaml

from roop.face_analytics_models import (
    DEFAULT_FACE_DETECTOR_MODEL,
    DEFAULT_FACE_LANDMARKER_MODEL,
    DEFAULT_FACE_MASKER_MODEL,
    get_face_detector_model_key,
    get_face_landmarker_model_key,
    get_face_masker_model_key,
)
from roop.face_swap_models import (
    DEFAULT_FACE_SWAP_MODEL,
    get_face_swap_model_key,
    normalize_face_swap_upscale,
)


class Settings:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load()

    def default_get(_, data, name, default):
        value = default
        try:
            value = data.get(name, default)
        except Exception:
            pass
        return value

    def load(self):
        try:
            with open(self.config_file, "r", encoding="utf-8") as handle:
                data = yaml.load(handle, Loader=yaml.FullLoader)
        except Exception:
            data = None

        self.selected_theme = self.default_get(data, "selected_theme", "Default")
        self.server_name = self.default_get(data, "server_name", "")
        self.server_port = self.default_get(data, "server_port", 0)
        self.server_share = self.default_get(data, "server_share", False)
        self.output_image_format = self.default_get(data, "output_image_format", "png")
        self.output_video_format = self.default_get(data, "output_video_format", "mp4")
        self.output_video_codec = self.default_get(data, "output_video_codec", "libx264")
        self.video_quality = self.default_get(data, "video_quality", 14)
        self.clear_output = self.default_get(data, "clear_output", True)
        self.max_threads = self.default_get(data, "max_threads", 2)
        self.detect_pack_frame_count = self.default_get(data, "detect_pack_frame_count", 256)
        self.staged_chunk_size = self.default_get(data, "staged_chunk_size", 96)
        if not self.staged_chunk_size or self.staged_chunk_size <= 0:
            self.staged_chunk_size = 96
        self.prefetch_frames = self.default_get(data, "prefetch_frames", 24)
        if not self.prefetch_frames or self.prefetch_frames <= 0:
            self.prefetch_frames = 24
        self.detect_batch_size = self.default_get(data, "detect_batch_size", 8)
        if not self.detect_batch_size or self.detect_batch_size <= 0:
            self.detect_batch_size = 8
        self.detect_single_batch_workers = self.default_get(data, "detect_single_batch_workers", 1)
        if not self.detect_single_batch_workers or self.detect_single_batch_workers <= 0:
            self.detect_single_batch_workers = 1
        self.swap_batch_size = self.default_get(data, "swap_batch_size", 32)
        if not self.swap_batch_size or self.swap_batch_size <= 0:
            self.swap_batch_size = 32
        self.mask_batch_size = self.default_get(data, "mask_batch_size", 64)
        if not self.mask_batch_size or self.mask_batch_size <= 0:
            self.mask_batch_size = 64
        self.enhance_batch_size = self.default_get(data, "enhance_batch_size", 8)
        if not self.enhance_batch_size or self.enhance_batch_size <= 0:
            self.enhance_batch_size = 8
        self.single_batch_workers = self.default_get(data, "single_batch_workers", 1)
        if not self.single_batch_workers or self.single_batch_workers <= 0:
            self.single_batch_workers = 1
        self.provider = self.default_get(data, "provider", "cuda")
        self.force_cpu = self.default_get(data, "force_cpu", False)
        self.face_detector_model = get_face_detector_model_key(
            self.default_get(data, "face_detector_model", DEFAULT_FACE_DETECTOR_MODEL)
        )
        self.face_landmarker_model = get_face_landmarker_model_key(
            self.default_get(data, "face_landmarker_model", DEFAULT_FACE_LANDMARKER_MODEL)
        )
        self.face_masker_model = get_face_masker_model_key(
            self.default_get(data, "face_masker_model", DEFAULT_FACE_MASKER_MODEL)
        )
        self.output_template = self.default_get(data, "output_template", "{file}_{time}")
        self.use_os_temp_folder = self.default_get(data, "use_os_temp_folder", False)
        self.output_show_video = self.default_get(data, "output_show_video", True)
        self.launch_browser = self.default_get(data, "launch_browser", False)
        self.max_face_distance = self.default_get(data, "max_face_distance", 0.85)
        self.face_detection_mode = self.default_get(data, "face_detection_mode", "All faces")
        self.num_swap_steps = self.default_get(data, "num_swap_steps", 1)
        self.selected_enhancer = self.default_get(data, "selected_enhancer", "GPEN")
        self.face_swap_model = get_face_swap_model_key(
            self.default_get(data, "face_swap_model", DEFAULT_FACE_SWAP_MODEL)
        )
        self.subsample_upscale = normalize_face_swap_upscale(
            self.default_get(data, "subsample_upscale", "256px"),
            self.face_swap_model,
        )
        self.blend_ratio = self.default_get(data, "blend_ratio", 0.80)
        legacy_video_swapping_method = self.default_get(data, "video_swapping_method", "Smart staged processing")
        if legacy_video_swapping_method == "In-Memory processing":
            legacy_video_swapping_method = "Smart staged processing"
        elif legacy_video_swapping_method == "Extract Frames to media":
            legacy_video_swapping_method = "Legacy extract frames"
        self.video_swapping_method = legacy_video_swapping_method
        self.no_face_action = self.default_get(data, "no_face_action", "Retry rotated")
        self.vr_mode = self.default_get(data, "vr_mode", False)
        self.autorotate_faces = self.default_get(data, "autorotate_faces", True)
        self.skip_audio = self.default_get(data, "skip_audio", False)
        self.keep_frames = self.default_get(data, "keep_frames", False)
        self.wait_after_extraction = self.default_get(data, "wait_after_extraction", False)
        self.output_method = self.default_get(data, "output_method", "File")
        self.mask_engine = self.default_get(data, "mask_engine", "DFL XSeg")
        self.mask_clip_text = self.default_get(data, "mask_clip_text", "cup,hands,hair,banana")
        self.show_mask_offsets = self.default_get(data, "show_mask_offsets", False)
        self.restore_original_mouth = self.default_get(data, "restore_original_mouth", False)
        self.mask_top = self.default_get(data, "mask_top", 0.0)
        self.mask_bottom = self.default_get(data, "mask_bottom", 0.0)
        self.mask_left = self.default_get(data, "mask_left", 0.0)
        self.mask_right = self.default_get(data, "mask_right", 0.0)
        self.face_mask_blend = self.default_get(data, "face_mask_blend", 20.0)
        self.mouth_mask_blend = self.default_get(data, "mouth_mask_blend", 10.0)
        self.mouth_top_scale = self.default_get(data, "mouth_top_scale", 1.0)
        self.mouth_bottom_scale = self.default_get(data, "mouth_bottom_scale", 1.0)
        self.mouth_left_scale = self.default_get(data, "mouth_left_scale", 1.0)
        self.mouth_right_scale = self.default_get(data, "mouth_right_scale", 1.0)

    def save(self):
        data = {
            "selected_theme": self.selected_theme,
            "server_name": self.server_name,
            "server_port": self.server_port,
            "server_share": self.server_share,
            "output_image_format": self.output_image_format,
            "output_video_format": self.output_video_format,
            "output_video_codec": self.output_video_codec,
            "video_quality": self.video_quality,
            "clear_output": self.clear_output,
            "max_threads": self.max_threads,
            "detect_pack_frame_count": self.detect_pack_frame_count,
            "staged_chunk_size": self.staged_chunk_size,
            "prefetch_frames": self.prefetch_frames,
            "detect_batch_size": self.detect_batch_size,
            "detect_single_batch_workers": self.detect_single_batch_workers,
            "swap_batch_size": self.swap_batch_size,
            "mask_batch_size": self.mask_batch_size,
            "enhance_batch_size": self.enhance_batch_size,
            "single_batch_workers": self.single_batch_workers,
            "provider": self.provider,
            "force_cpu": self.force_cpu,
            "face_detector_model": self.face_detector_model,
            "face_landmarker_model": self.face_landmarker_model,
            "face_masker_model": self.face_masker_model,
            "output_template": self.output_template,
            "use_os_temp_folder": self.use_os_temp_folder,
            "output_show_video": self.output_show_video,
            "launch_browser": self.launch_browser,
            "max_face_distance": self.max_face_distance,
            "face_detection_mode": self.face_detection_mode,
            "num_swap_steps": self.num_swap_steps,
            "selected_enhancer": self.selected_enhancer,
            "face_swap_model": self.face_swap_model,
            "subsample_upscale": self.subsample_upscale,
            "blend_ratio": self.blend_ratio,
            "video_swapping_method": self.video_swapping_method,
            "no_face_action": self.no_face_action,
            "vr_mode": self.vr_mode,
            "autorotate_faces": self.autorotate_faces,
            "skip_audio": self.skip_audio,
            "keep_frames": self.keep_frames,
            "wait_after_extraction": self.wait_after_extraction,
            "output_method": self.output_method,
            "mask_engine": self.mask_engine,
            "mask_clip_text": self.mask_clip_text,
            "show_mask_offsets": self.show_mask_offsets,
            "restore_original_mouth": self.restore_original_mouth,
            "mask_top": self.mask_top,
            "mask_bottom": self.mask_bottom,
            "mask_left": self.mask_left,
            "mask_right": self.mask_right,
            "face_mask_blend": self.face_mask_blend,
            "mouth_mask_blend": self.mouth_mask_blend,
            "mouth_top_scale": self.mouth_top_scale,
            "mouth_bottom_scale": self.mouth_bottom_scale,
            "mouth_left_scale": self.mouth_left_scale,
            "mouth_right_scale": self.mouth_right_scale,
        }
        with open(self.config_file, "w", encoding="utf-8") as handle:
            yaml.dump(data, handle)
