import os
import json
import argparse
import torch
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
trt_runtime = trt.Runtime(TRT_LOGGER)
# ========================================
# 工具函数
# ========================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    eps = 1e-8
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm < eps or b_norm < eps:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm + eps))

def compare_outputs(name: str, ref: np.ndarray, pred: np.ndarray):
    if ref.shape != pred.shape:
        print(f"{name:<20}: [WARN] Shape mismatch: Ref {ref.shape} vs Pred {pred.shape}")
        min_len = min(ref.size, pred.size)
        ref = ref.flatten()[:min_len]
        pred = pred.flatten()[:min_len]
    sim = cosine_similarity(ref, pred)
    mse = np.mean((ref.flatten() - pred.flatten()) ** 2)
    print(f"{name:<20}: Cosine = {sim:.6f}, MSE = {mse:.3e}")

# ========================================
# 自动加载或生成输入
# ========================================
def load_or_random_input(sess_or_engine, is_trt=False, npy_dir=None):
    input_map = {}
    os.makedirs(npy_dir, exist_ok=True)
    input_names = []

    if is_trt:
        for name in sess_or_engine:
            if sess_or_engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                input_names.append(name)
    else:
        input_names = [inp.name for inp in sess_or_engine.get_inputs()]

    print("\n[📦 构造输入信息]")

    for name in input_names:
        npy_path = os.path.join(npy_dir, f"{name}.npy")
        if os.path.exists(npy_path):
            arr = np.load(npy_path)
            print(f" ├─ {name:<30} ← 从文件加载 {arr.shape}")
        else:
            print(f" ├─ [⚠️ 缺失] {npy_path}，自动生成随机输入")
            if is_trt:
                shape = sess_or_engine.get_tensor_shape(name)
                shape = [1 if s <= 0 else s for s in shape]
            else:
                # ONNX 模型
                input_obj = next(inp for inp in sess_or_engine.get_inputs() if inp.name == name)
                shape = [1 if isinstance(s, str) or s is None else s for s in input_obj.shape]
            print(f" │   → 随机 shape = {shape}")
            arr = np.random.uniform(-1, 1, size=shape).astype(np.float32)
            np.save(npy_path, arr)
            print(f" │   ✅ 已保存随机输入 {npy_path}")
        input_map[name] = arr

    print(f"[✅ 输入共 {len(input_map)} 个]\n")
    return input_map

# ========================================
# ONNX 推理
# ========================================
def run_onnx_inference(model_path, input_map):
    sess = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    outputs = sess.run(None, input_map)
    return dict(zip([o.name for o in sess.get_outputs()], outputs))

# ========================================
# TRT 推理
# ========================================
def run_trt_inference(engine_path, input_map):

    with open(engine_path, 'rb') as f:
        engine = trt_runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    stream = cuda.Stream()

    # 设置动态 shape
    for name in engine:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT and name in input_map:
            arr = input_map[name]
            shp = tuple(int(x) for x in arr.shape)
            context.set_input_shape(name, shp)

    bindings = []
    io_desc = {}

    for name in engine:
        mode = engine.get_tensor_mode(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        shape = tuple(max(1, int(d)) for d in context.get_tensor_shape(name))
        size = int(np.prod(shape))
        host = cuda.pagelocked_empty(size, dtype)
        dev = cuda.mem_alloc(host.nbytes)
        io_desc[name] = {"host": host, "dev": dev, "shape": shape, "dtype": dtype, "mode": mode}
        bindings.append(int(dev))

    # 拷贝输入
    for name, desc in io_desc.items():
        if desc["mode"] == trt.TensorIOMode.INPUT:
            np.copyto(desc["host"], input_map[name].astype(desc["dtype"]).flatten())
            cuda.memcpy_htod_async(desc["dev"], desc["host"], stream)

    context.execute_v2(bindings)

    # 取输出
    trt_outputs = {}
    for name, desc in io_desc.items():
        if desc["mode"] == trt.TensorIOMode.OUTPUT:
            cuda.memcpy_dtoh_async(desc["host"], desc["dev"], stream)
    stream.synchronize()

    for name, desc in io_desc.items():
        if desc["mode"] == trt.TensorIOMode.OUTPUT:
            trt_outputs[name] = desc["host"].reshape(desc["shape"])
    return trt_outputs

# ========================================
# 主逻辑
# ========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--trt", required=True, help="Path to TensorRT engine")
    parser.add_argument("--inputs", default="./inputs_npy", help="Folder for npy inputs (auto-generate if missing)")
    args = parser.parse_args()

    # 加载 ONNX session
    onnx_sess = ort.InferenceSession(args.onnx, providers=['CUDAExecutionProvider'])
    # 加载/生成输入
    input_map = load_or_random_input(onnx_sess, is_trt=False, npy_dir=args.inputs)

    print("\n[🔹 Run ONNX]")
    onnx_out = run_onnx_inference(args.onnx, input_map)

    print("\n[🔹 Run TensorRT]")
    # trt_input = load_or_random_input(trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(open(args.trt, "rb").read()), is_trt=True, npy_dir=args.inputs)
    trt_out = run_trt_inference(args.trt, input_map)

    print("\n[🔍 Compare Outputs]")
    common = set(onnx_out.keys()) & set(trt_out.keys())
    for k in common:
        compare_outputs(k, onnx_out[k], trt_out[k])
