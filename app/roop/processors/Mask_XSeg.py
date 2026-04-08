import numpy as np
import cv2
import threading
import roop.config.globals

from roop.onnx.runtime import resolve_model_path_for_processor
from roop.onnx.session import create_inference_session
from roop.processors.base import BaseProcessor
from roop.config.types import Frame
from roop.utils import resolve_relative_path

THREAD_LOCK_CLIP = threading.Lock()


class Mask_XSeg(BaseProcessor):
    plugin_options:dict = None

    model_xseg = None

    processorname = 'mask_xseg'
    type = 'mask'
    supports_batch = True


    def Initialize(self, plugin_options:dict):
        if self.plugin_options is not None:
            if self.plugin_options["devicename"] != plugin_options["devicename"]:
                self.Release()

        self.plugin_options = plugin_options
        if self.model_xseg is None:
            model_path = resolve_model_path_for_processor(resolve_relative_path('../models/xseg.onnx'), self.processorname)
            self.model_xseg = create_inference_session(model_path, self.processorname)
            self.model_inputs = self.model_xseg.get_inputs()
            self.model_outputs = self.model_xseg.get_outputs()
            self.input_name = self.model_inputs[0].name
            self.output_name = self.model_outputs[0].name

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')


    def Run(self, img1, keywords:str) -> Frame:
        temp_frame = self._preprocess_frame(img1)[None, ...]
        io_binding = self.model_xseg.io_binding()           
        io_binding.bind_cpu_input(getattr(self, "input_name", self.model_inputs[0].name), temp_frame)
        io_binding.bind_output(getattr(self, "output_name", self.model_outputs[0].name), self.devicename)
        self.model_xseg.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        return self._normalize_mask_output(ort_outs[0][0])


    def _preprocess_frame(self, image):
        temp_frame = cv2.resize(image, (256, 256), cv2.INTER_CUBIC)
        return np.ascontiguousarray(temp_frame.astype(np.float32) / 255.0)


    def _normalize_mask_output(self, result):
        if result.ndim == 4 and result.shape[0] == 1:
            result = result[0]
        result = np.clip(result, 0, 1.0)
        result[result < 0.1] = 0
        if result.ndim == 2:
            result = result[..., None]
        return 1.0 - result


    def RunBatch(self, images, keywords:str, batch_size=1):
        outputs = []
        effective_batch_size = max(1, int(batch_size))
        for batch_start in range(0, len(images), effective_batch_size):
            current_images = images[batch_start:batch_start + effective_batch_size]
            if not current_images:
                continue
            batch_input = self._get_batch_buffer("mask_input", (effective_batch_size, 256, 256, 3), np.float32)
            for index, image in enumerate(current_images):
                batch_input[index] = self._preprocess_frame(image)
            batch_outputs = self.model_xseg.run(
                None,
                {getattr(self, "input_name", self.model_inputs[0].name): np.ascontiguousarray(batch_input[: len(current_images)])},
            )[0]
            for result in batch_outputs:
                outputs.append(self._normalize_mask_output(result))
        return outputs


    def Release(self):
        del self.model_xseg
        self.model_xseg = None
        self._clear_batch_buffers()



