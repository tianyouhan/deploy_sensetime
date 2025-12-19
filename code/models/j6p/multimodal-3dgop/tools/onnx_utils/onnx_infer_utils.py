# coding: utf-8
import onnxruntime
import numpy as np


class ONNXModel():
    def __init__(self, onnx_path, input_name=None, output_name=None):
        """
        :param onnx_path:
        """
        self.onnx_session = onnxruntime.InferenceSession(onnx_path)
        if input_name is None:
            self.input_name = self.get_input_name(self.onnx_session)
        else:
            self.input_name = input_name
        if output_name is None:
            self.output_name = self.get_output_name(self.onnx_session)
        else:
            self.output_name = output_name
        print("input_name:{}".format(self.input_name))
        print("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        """
        output_name = onnx_session.get_outputs()[0].name
        :param onnx_session:
        :return:
        """
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        """
        input_name = onnx_session.get_inputs()[0].name
        :param onnx_session:
        :return:
        """
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, image_numpy):
        """
        input_feed={self.input_name: image_numpy}
        :param input_name:
        :param image_numpy:
        :return:
        """
        input_feed = {}
        for idx,name in enumerate(input_name):
            if type(image_numpy[name]) is not np.ndarray:
                image_numpy[name] = to_numpy(image_numpy[name])
            input_feed[name] = image_numpy[name]
        return input_feed

    def forward(self, image_numpy_dict):
        '''
        # image_numpy = image.transpose(2, 0, 1)
        # image_numpy = image_numpy[np.newaxis, :]
        # onnx_session.run([output_name], {input_name: x})
        # :param image_numpy:
        # :return:
        '''
        # 输入数据的类型必须与模型一致,以下三种写法都是可以的
        # scores, boxes = self.onnx_session.run(None, {self.input_name: image_numpy})
        # scores, boxes = self.onnx_session.run(self.output_name, input_feed={self.input_name: iimage_numpy})
        input_feed = self.get_input_feed(self.input_name, image_numpy_dict)
        # input_feed = self.get_input_feed(['feature', 'sampling_locations', 'attention_weights', 'bev_mask'], image_numpy_dict)
        result = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return result


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


