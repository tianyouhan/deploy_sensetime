import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from scipy.spatial.distance import cosine
import os


def load_bin(file_path, dtype=np.float32):
    """加载 bin 文件为 numpy 数组"""
    return np.fromfile(file_path, dtype=dtype)


def cosine_similarity(a, b):
    """计算 Cosine 相似度"""
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isinf(a)) or np.any(np.isinf(b)):
        return float('nan')
    return 1 - cosine(a.flatten(), b.flatten())


def check_overflow(arr):
    """检测 NaN / Inf"""
    return np.any(np.isnan(arr)) or np.any(np.isinf(arr))


# ====== ONNX 推理 ======
def run_onnx(onnx_model_path, inputs_dict):
    sess = ort.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
    return sess.run(None, inputs_dict)


# ====== TensorRT 推理 ======
def run_trt(engine_path, inputs_dict):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    io_tensors = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    bindings = {}
    device_buffers = {}

    # 分配输入输出 buffer
    for name in io_tensors:
        shape = tuple(context.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            # 取对应的 numpy 输入
            if name not in inputs_dict:
                raise ValueError(f"缺少输入: {name}, 可用输入: {list(inputs_dict.keys())}")
            host_buf = inputs_dict[name].astype(dtype).copy()
        else:
            host_buf = np.empty(shape, dtype=dtype)

        # 分配 device buffer
        device_buf = cuda.mem_alloc(host_buf.nbytes)
        device_buffers[name] = (host_buf, device_buf)
        bindings[name] = int(device_buf)

    # 拷贝所有输入到 GPU
    for name, (host_buf, device_buf) in device_buffers.items():
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
            cuda.memcpy_htod(device_buf, host_buf)

    # 设置 context 绑定
    for name, buf in bindings.items():
        context.set_tensor_address(name, buf)

    # 推理
    stream = cuda.Stream()
    context.execute_async_v3(stream_handle=stream.handle)
    stream.synchronize()

    # 从 GPU 拷贝输出
    outputs = {}
    for name, (host_buf, device_buf) in device_buffers.items():
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            cuda.memcpy_dtoh(host_buf, device_buf)
            outputs[name] = host_buf
    return outputs


# ====== 主函数 ======
if __name__ == "__main__":
    input_root = "/mnt/data/hantianyou/backbone_input"
    onnx_model = "/mnt/data/hantianyou/far3d_backbone/spetr.onnx"
    trt_engine = "/mnt/data/hantianyou/backbone_fp16.trt"

    # 读取输入
    inputs_dict = {}
    for cam_dir in sorted(os.listdir(input_root)):
        bin_file = os.path.join(input_root, cam_dir, "0.bin")
        if not os.path.exists(bin_file):
            continue
        data = load_bin(bin_file, dtype=np.float32).reshape(1, 3, 576, 1024)
        inputs_dict[cam_dir] = data

    print(f"=== Inputs ready: {list(inputs_dict.keys())} ===")

    # ONNX 推理
    onnx_outs = run_onnx(onnx_model, inputs_dict)

    # TRT 推理
    trt_outs = run_trt(trt_engine, inputs_dict)

    # 对比结果
    for i, o_onnx in enumerate(onnx_outs):
        # TRT 可能是 dict，取第 i 个输出
        trt_key = list(trt_outs.keys())[i]
        o_trt = trt_outs[trt_key]

        cos = cosine_similarity(o_onnx, o_trt)
        overflow = check_overflow(o_trt)
        print(f"Output {i} ({trt_key}): cosine={cos:.6f}, overflow={overflow}, "
              f"max={o_trt.max()}, min={o_trt.min()}")
