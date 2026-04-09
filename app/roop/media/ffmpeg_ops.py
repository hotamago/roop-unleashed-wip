
import os
import subprocess
import tempfile
import roop.config.globals
import roop.utils as util

from typing import List, Any
from roop.media.video_io import resolve_video_writer_config


def _strip_preset_arg(ffmpeg_params: List[str]) -> List[str]:
    cleaned: List[str] = []
    skip_next = False
    for token in ffmpeg_params:
        if skip_next:
            skip_next = False
            continue
        if token == "-preset":
            skip_next = True
            continue
        cleaned.append(token)
    return cleaned


def _build_concat_reencode_args() -> List[str]:
    writer_config = resolve_video_writer_config(
        roop.config.globals.video_encoder,
        roop.config.globals.video_quality,
    )
    resolved_codec = writer_config["codec"]
    quality_args = list(writer_config["quality_args"])
    ffmpeg_params = _strip_preset_arg(list(writer_config["ffmpeg_params"]))
    return ["-c:v", resolved_codec, *quality_args, *ffmpeg_params]

def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-hwaccel', 'auto', '-y', '-loglevel', roop.config.globals.log_level]
    commands.extend(args)
    print ("Running ffmpeg")
    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception as e:
        print("Running ffmpeg failed! Commandline:")
        print (" ".join(commands))
    return False


def transcode_video_range(
    original_video: str,
    output_video: str,
    start_frame: int,
    end_frame: int,
    fps: float,
    include_audio: bool,
) -> bool:
    writer_config = resolve_video_writer_config(roop.config.globals.video_encoder, roop.config.globals.video_quality)
    start_frame = max(0, int(start_frame or 0))
    end_frame = max(start_frame + 1, int(end_frame or 0))
    fps = float(fps or util.detect_fps(original_video) or 1.0)
    start_time = start_frame / fps
    num_frames = max(1, end_frame - start_frame)
    commands = [
        "-ss",
        format(start_time, ".6f"),
        "-i",
        original_video,
        "-frames:v",
        str(num_frames),
        "-c:v",
        writer_config["codec"],
    ]
    commands.extend(writer_config["quality_args"])
    commands.extend(writer_config["ffmpeg_params"])
    if include_audio:
        commands.extend(["-c:a", "aac", "-shortest"])
    else:
        commands.append("-an")
    commands.append(output_video)
    return run_ffmpeg(commands)



def cut_video(original_video: str, cut_video: str, start_frame: int, end_frame: int, reencode: bool):
    fps = util.detect_fps(original_video)
    start_time = start_frame / fps
    num_frames = end_frame - start_frame

    if reencode:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video, '-c:v', roop.config.globals.video_encoder, '-c:a', 'aac', '-frames:v', str(num_frames), cut_video])
    else:
        run_ffmpeg(['-ss',  format(start_time, ".2f"), '-i', original_video,  '-frames:v', str(num_frames), '-c:v' ,'copy','-c:a' ,'copy', cut_video])

def join_videos(videos: List[str], dest_filename: str, simple: bool, reencode: bool = False):
    if simple:
        temp_root = util.resolve_relative_path('../temp')
        os.makedirs(temp_root, exist_ok=True)
        txtfilename = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                suffix=".txt",
                prefix="joinvids_",
                dir=temp_root,
                delete=False,
            ) as handle:
                txtfilename = handle.name
                for v in videos:
                    v = v.replace('\\', '/').replace("'", "'\\''")
                    handle.write(f"file '{v}'\n")

            commands = ['-f', 'concat', '-safe', '0', '-i', txtfilename]
            if reencode:
                commands.extend(_build_concat_reencode_args())
                commands.extend(['-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an'])
            else:
                commands.extend(['-vcodec', 'copy'])
            commands.append(dest_filename)
            return run_ffmpeg(commands)
        finally:
            if txtfilename and os.path.isfile(txtfilename):
                os.remove(txtfilename)

    else:
        inputs = []
        filter = ''
        for i,v in enumerate(videos):
            inputs.append('-i')
            inputs.append(v)
            filter += f'[{i}:v:0][{i}:a:0]'
        return run_ffmpeg([" ".join(inputs), '-filter_complex', f'"{filter}concat=n={len(videos)}:v=1:a=1[outv][outa]"', '-map', '"[outv]"', '-map', '"[outa]"', dest_filename])    

        #     filter += f'[{i}:v:0][{i}:a:0]'
        # run_ffmpeg([" ".join(inputs), '-filter_complex', f'"{filter}concat=n={len(videos)}:v=1:a=1[outv][outa]"', '-map', '"[outv]"', '-map', '"[outa]"', dest_filename])    



def extract_frames(target_path : str, trim_frame_start, trim_frame_end, fps : float, temp_directory_path: str = None, image_format: str = None) -> bool:
    if temp_directory_path is None:
        util.create_temp(target_path)
        temp_directory_path = util.get_temp_directory_path(target_path)
    else:
        os.makedirs(temp_directory_path, exist_ok=True)
    if image_format is None:
        image_format = roop.config.globals.CFG.output_image_format
    commands = ['-i', target_path, '-q:v', '1', '-pix_fmt', 'rgb24', ]
    if trim_frame_start is not None and trim_frame_end is not None:
        commands.extend([ '-vf', 'trim=start_frame=' + str(trim_frame_start) + ':end_frame=' + str(trim_frame_end) + ',fps=' + str(fps) ])
    commands.extend(['-vsync', '0', os.path.join(temp_directory_path, '%06d.' + image_format)])
    return run_ffmpeg(commands)


def create_video(target_path: str, dest_filename: str, fps: float = 24.0, temp_directory_path: str = None) -> None:
    if temp_directory_path is None:
        temp_directory_path = util.get_temp_directory_path(target_path)
    run_ffmpeg(['-r', str(fps), '-i', os.path.join(temp_directory_path, f'%06d.{roop.config.globals.CFG.output_image_format}'), '-c:v', roop.config.globals.video_encoder, '-crf', str(roop.config.globals.video_quality), '-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', dest_filename])
    return dest_filename


def create_gif_from_video(video_path: str, gif_path):
    from roop.media.capturer import get_video_frame, release_video

    fps = util.detect_fps(video_path)
    frame = get_video_frame(video_path)
    release_video()

    scalex = frame.shape[0]
    scaley = frame.shape[1]

    if scalex >= scaley:
        scaley = -1
    else:
        scalex = -1

    run_ffmpeg(['-i', video_path, '-vf', f'fps={fps},scale={int(scalex)}:{int(scaley)}:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse', '-loop', '0', gif_path])



def create_video_from_gif(gif_path: str, output_path):
    fps = util.detect_fps(gif_path)
    filter = """scale='trunc(in_w/2)*2':'trunc(in_h/2)*2',format=yuv420p,fps=10"""
    run_ffmpeg(['-i', gif_path, '-vf', f'"{filter}"', '-movflags', '+faststart', '-shortest', output_path])



def resize_video(input_path: str, output_path: str, width: int, height: int) -> bool:
    scale_filter = (
        f'scale={width}:{height}:force_original_aspect_ratio=decrease,'
        f'pad={width}:{height}:(ow-iw)/2:(oh-ih)/2'
    )
    return run_ffmpeg(['-i', input_path, '-vf', scale_filter,
                       '-c:v', roop.config.globals.video_encoder,
                       '-crf', str(roop.config.globals.video_quality),
                       '-c:a', 'copy', output_path])


def rotate_media(input_path: str, output_path: str, transform: str) -> bool:
    transform_map = {
        "90 deg Clockwise":        "transpose=1",
        "90 deg Counter-clockwise": "transpose=2",
        "180 deg":                  "transpose=1,transpose=1",
        "Flip Horizontal":       "hflip",
        "Flip Vertical":         "vflip",
    }
    vf = transform_map.get(transform, "transpose=1")
    return run_ffmpeg(['-i', input_path, '-vf', vf, '-c:a', 'copy', output_path])


def change_fps(input_path: str, output_path: str, fps: float) -> bool:
    return run_ffmpeg(['-i', input_path, '-vf', f'fps={fps}',
                       '-c:v', roop.config.globals.video_encoder,
                       '-crf', str(roop.config.globals.video_quality),
                       '-c:a', 'copy', output_path])


def crop_media(input_path: str, output_path: str,
               left_pct: float, right_pct: float,
               top_pct: float,  bottom_pct: float) -> bool:
    l, r, t, b = left_pct / 100, right_pct / 100, top_pct / 100, bottom_pct / 100
    crop_filter = (
        f"crop=in_w*(1-{l:.4f}-{r:.4f}):in_h*(1-{t:.4f}-{b:.4f})"
        f":in_w*{l:.4f}:in_h*{t:.4f}"
    )
    return run_ffmpeg(['-i', input_path, '-vf', crop_filter, '-c:a', 'copy', output_path])


def apply_media_transforms(input_path: str, output_path: str,
                           vf_filters: list, is_video: bool) -> bool:
    """Apply a list of -vf filters in a single ffmpeg pass."""
    if not vf_filters:
        return False
    vf = ','.join(vf_filters)
    args = ['-i', input_path, '-vf', vf]
    if is_video:
        args += ['-c:v', roop.config.globals.video_encoder,
                 '-crf', str(roop.config.globals.video_quality),
                 '-c:a', 'copy']
    args.append(output_path)
    return run_ffmpeg(args)


def restore_audio(intermediate_video: str, original_video: str, trim_frame_start, trim_frame_end, final_video : str) -> None:
	fps = util.detect_fps(original_video)
	commands = [ '-i', intermediate_video ]
	if trim_frame_start is None and trim_frame_end is None:
		commands.extend([ '-c:a', 'copy' ])
	else:
		# if trim_frame_start is not None:
		# 	start_time = trim_frame_start / fps
		# 	commands.extend([ '-ss', format(start_time, ".2f")])
		# else:
		# 	commands.extend([ '-ss', '0' ])
		# if trim_frame_end is not None:
		# 	end_time = trim_frame_end / fps
		# 	commands.extend([ '-to', format(end_time, ".2f")])
		# commands.extend([ '-c:a', 'aac' ])
		if trim_frame_start is not None:
			start_time = trim_frame_start / fps
			commands.extend([ '-ss', format(start_time, ".2f")])
		else:
			commands.extend([ '-ss', '0' ])
		if trim_frame_end is not None:
			end_time = trim_frame_end / fps
			commands.extend([ '-to', format(end_time, ".2f")])
		commands.extend([ '-i', original_video, "-c",  "copy" ])

	commands.extend([ '-map', '0:v:0', '-map', '1:a:0?', '-shortest', final_video ])
	run_ffmpeg(commands)

