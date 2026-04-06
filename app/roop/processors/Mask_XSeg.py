import numpy as np
import cv2
import onnxruntime
import threading
import roop.globals

from roop.typing import Frame
from roop.utilities import resolve_relative_path

THREAD_LOCK_CLIP = threading.Lock()


class Mask_XSeg():
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
            model_path = resolve_relative_path('../models/xseg.onnx')
            onnxruntime.set_default_logger_severity(3)
            self.model_xseg = onnxruntime.InferenceSession(model_path, None, providers=roop.globals.execution_providers)
            self.model_inputs = self.model_xseg.get_inputs()
            self.model_outputs = self.model_xseg.get_outputs()

            # replace Mac mps with cpu for the moment
            self.devicename = self.plugin_options["devicename"].replace('mps', 'cpu')


    def Run(self, img1, keywords:str) -> Frame:
        temp_frame = cv2.resize(img1, (256, 256), cv2.INTER_CUBIC)
        temp_frame = temp_frame.astype('float32') / 255.0
        temp_frame = temp_frame[None, ...]
        io_binding = self.model_xseg.io_binding()           
        io_binding.bind_cpu_input(self.model_inputs[0].name, temp_frame)
        io_binding.bind_output(self.model_outputs[0].name, self.devicename)
        self.model_xseg.run_with_iobinding(io_binding)
        ort_outs = io_binding.copy_outputs_to_cpu()
        result = ort_outs[0][0]
        result = np.clip(result, 0, 1.0)
        result[result < 0.1] = 0
        # invert values to mask areas to keep
        result = 1.0 - result
        return result       


    def RunBatch(self, images, keywords:str, batch_size=1):
        outputs = []
        for batch_start in range(0, len(images), max(1, batch_size)):
            batch = []
            for img in images[batch_start:batch_start + max(1, batch_size)]:
                temp_frame = cv2.resize(img, (256, 256), cv2.INTER_CUBIC)
                temp_frame = temp_frame.astype('float32') / 255.0
                batch.append(temp_frame)
            batch_input = np.stack(batch, axis=0).astype(np.float32)
            batch_outputs = self.model_xseg.run(None, {self.model_inputs[0].name: batch_input})[0]
            for result in batch_outputs:
                result = result[0]
                result = np.clip(result, 0, 1.0)
                result[result < 0.1] = 0
                outputs.append(1.0 - result)
        return outputs


    def Release(self):
        del self.model_xseg
        self.model_xseg = None


