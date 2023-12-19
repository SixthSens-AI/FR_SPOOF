import os  # 导入操作系统相关的模块
import cv2  # 导入OpenCV库
import math  # 导入数学库
import torch  # 导入PyTorch库
import numpy as np  # 导入NumPy库
import torch.nn.functional as F  # 导入PyTorch的函数模块

# 导入自定义模型类，这些模型用于活体检测
from src.model_lib.MiniFASNet import MiniFASNetV1, MiniFASNetV2, MiniFASNetV1SE, MiniFASNetV2SE

# 导入数据预处理和工具函数
from src.data_io import transform as trans
from src.utility import get_kernel, parse_model_name

# 模型类型的映射字典，将模型类型名称映射到相应的模型类
MODEL_MAPPING = {
    'MiniFASNetV1': MiniFASNetV1,
    'MiniFASNetV2': MiniFASNetV2,
    'MiniFASNetV1SE': MiniFASNetV1SE,
    'MiniFASNetV2SE': MiniFASNetV2SE
}


# 活体检测类，继承自人脸检测类
class AntiSpoofPredict():
    def __init__(self, device_id, model_dir):
        super(AntiSpoofPredict, self).__init__()  # 调用父类的初始化方法
        self.device = torch.device("cuda:{}".format(device_id)
                                   if torch.cuda.is_available() else "cpu")  # 检查是否有可用的GPU，如果有，就使用GPU，否则使用CPU
        self.models = self.load_models(model_dir)  # 加载活体检测模型

    # 加载活体检测模型
    def load_models(self, model_dir):
        models = {}
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            h_input, w_input, model_type, _ = parse_model_name(model_name)  # 解析模型文件名
            kernel_size = get_kernel(h_input, w_input)  # 获取卷积核大小
            model = MODEL_MAPPING[model_type](conv6_kernel=kernel_size).to(self.device)  # 创建模型实例并移至GPU或CPU
            state_dict = torch.load(model_path, map_location=self.device)  # 加载模型权重
            keys = iter(state_dict)
            first_layer_name = keys.__next__()
            if first_layer_name.find('module.') >= 0:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    name_key = key[7:]
                    new_state_dict[name_key] = value
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(state_dict)
            models[model_name] = model
        return models

    # 批量进行活体检测
    def predict_batch(self, imgs):
        test_transform = trans.Compose([trans.ToTensor()])  # 定义图像预处理方法
        img_batch = torch.stack([test_transform(img) for img in imgs]).to(self.device)  # 添加批处理维度并移至GPU或CPU
        result_batch = {}  # 存储批量结果的字典

        for model_name, model in self.models.items():
            model.eval()  # 将模型设置为评估模式
            with torch.no_grad():  # 禁用梯度计算
                result = model.forward(img_batch)  # 对图像进行前向传播
                result = F.softmax(result, dim=1).cpu().numpy()  # 对模型输出进行softmax归一化并转换为NumPy数组
                result_batch[model_name] = result

        return result_batch  # 返回批量结果