from djl_python import Input
from djl_python import Output

import fastdeploy as fd
import numpy as np
import os
import cv2
import yaml
import time
import base64
import logging

"""
    定义PP YoloE Plus 模型类
"""
class PPYoloEPlus(object):
    def __init__(self):
        self.model = None
        self.mapping = dict()
        self.initialized = False

    """
        模型初始化
    """
    def initialize(self, properties) -> Output:
        begin = time.time()
        # 配置Runtime
        option = fd.RuntimeOption()
        option.use_paddle_infer_backend()
        option.paddle_infer_option.enable_log_info = False
        if properties.__contains__('gpu'):
            option.use_gpu(int(properties['gpu']))

        # 加载模型
        model_path = properties['model']
        model_file = os.path.join(model_path, "inference.pdmodel")
        params_file = os.path.join(model_path, "inference.pdiparams")
        config_file = os.path.join(model_path, "inference.yml")
        self.model = fd.vision.detection.PaddleDetectionModel(
            model_file, params_file, config_file, runtime_option=option)

        # 加载目标识别的标签
        self.load_label_mapping(config_file)

        self.initialized = True

        outputs = Output()
        outputs.set_message("PPYoloE Plus Model Loaded, spend time {} second.".format(round(time.time() - begin)))
        return outputs

    """
        载入目标识别的标签
    """
    def load_label_mapping(self, config_file):
        yfile = open(config_file, "r", encoding='utf-8')
        ydata = yaml.safe_load(yfile)
        labels = ydata['label_list']
        for index, label in enumerate(labels):
            self.mapping[index] = label

    """
        将label_id翻译为Lable
    """
    @staticmethod
    def map_class_to_label(label_ids, mapping):
        labels = []
        for id in label_ids:
            labels.append(mapping.get(id))
        return labels

    """
        推理
    """
    def inference(self, inputs: Input) -> Output:
        outputs = Output()
        im = None
        if inputs.contains_key("image"):
            im = cv2.imread(inputs.get_as_string("image"))  # DJL环境用这个
            # im = cv2.imread(inputs.get_data("image"))       # Python环境用这个
        elif inputs.contains_key("image_base64"):
            img_data = base64.b64decode(inputs.get_as_string("image_base64"))    # DJL环境用这个
            # img_data = base64.b64decode(inputs.get_data("image_base64"))         # Python环境用这个
            img_nparr = np.frombuffer(img_data, np.uint8)
            im = cv2.imdecode(img_nparr, cv2.IMREAD_COLOR)

        result = self.model.predict(im)
        labels = self.map_class_to_label(result.label_ids, self.mapping)

        outputs.add(result.boxes, "boxes")
        outputs.add(labels, "labels")
        outputs.add(result.scores, "scores")
        return outputs


_service = PPYoloEPlus()


"""
    Python服务入口
"""
def handle(inputs: Input) -> Output:
    if not _service.initialized:
        properties = inputs.get_properties()
        if not properties.__contains__('model'):
            logging.warning("The service cannot be initialized, because parameter 'model' is empty. Please Check")
            return None
        else:
            outputs = _service.initialize(properties)
            logging.info(outputs.message)

    # 调用推理接口
    return _service.inference(inputs)


"""
    Python内测试

inputs = Input()
inputs.properties['model'] = "D:/opt/paddle_models/ppyoloE_plus"
# inputs.properties['gpu'] = 0
inputs.content.add("image", "C:/Users/Tommy/Desktop/ONext/images/p3.jpg")

# properties = inputs.get_properties()
# if properties.__contains__('gpu'):
#     print(properties.__contains__('gpu'))

result = handle(inputs)
print("result = ", result)

if result is not None:
    print(result.content.get("boxes"))
    print(result.content.get("labels"))
    print(result.content.get("scores"))

# print(type(_service.mapping), _service.mapping)
# print(type(_service.mapping[0]), _service.mapping[0])
"""
