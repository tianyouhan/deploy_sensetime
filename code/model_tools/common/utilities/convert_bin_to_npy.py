import os
import numpy as np
import onnx
from onnx import TensorProto


# ================ 用户需要设置 ====================
DATA_DIR = "/data/work/mount/J6P/model/logistics_vehicle/road/6v_multi_task_rclane/quant484"
ONNX_PATH = "/data/work/mount/J6P/model/logistics_vehicle/road/6v_multi_task_rclane/gridsample_e2e_hnop_v3.10.33_guangxiong/model_gs.onnx"
# ==================================================


def onnx_dtype_to_numpy(elem_type):
    """将 ONNX dtype 转换成 numpy dtype（bfloat16 返回特殊标识）"""
    mapping = {
        TensorProto.FLOAT: np.float32,
        TensorProto.FLOAT16: np.float16,
        TensorProto.BFLOAT16: "bfloat16",
        TensorProto.INT8: np.int8,
        TensorProto.UINT8: np.uint8,
        TensorProto.INT16: np.int16,
        TensorProto.UINT16: np.uint16,
        TensorProto.INT32: np.int32,
        TensorProto.INT64: np.int64,
    }
    return mapping.get(elem_type, None)


def load_bfloat16(bin_path, shape):
    """读取 BF16 并转 float32"""
    raw = np.fromfile(bin_path, dtype=np.uint16)
    if raw.size != np.prod(shape):
        raise ValueError(f"BF16 数据大小不匹配: {raw.size} vs expected {np.prod(shape)}")

    out = np.zeros_like(raw, dtype=np.float32)
    out_view = out.view(np.uint32)
    out_view[:] = raw.astype(np.uint32) << 16
    return out.reshape(shape)


def read_tensor_from_bin(bin_path, elem_type, shape):
    """根据 dtype 自动读取 .bin 文件"""

    numpy_dtype = onnx_dtype_to_numpy(elem_type)

    # BF16 特殊处理
    if numpy_dtype == "bfloat16":
        return load_bfloat16(bin_path, shape)

    # 常规类型
    return np.fromfile(bin_path, dtype=numpy_dtype).reshape(shape)


def get_io_tensor_info(onnx_path):
    """从 ONNX 获取输入/输出 tensor 的 dtype 和 shape"""
    model = onnx.load(onnx_path)
    graph = model.graph

    tensors = {}

    print("\n================= 模型输入 =================")
    for inp in graph.input:
        name = inp.name
        elem_type = inp.type.tensor_type.elem_type
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        dtype_str = TensorProto.DataType.Name(elem_type)
        print(f"输入:  {name:<30} dtype={dtype_str:<10} shape={shape}")
        tensors[name] = (elem_type, tuple(shape))

    print("\n================= 模型输出 =================")
    for out in graph.output:
        name = out.name
        elem_type = out.type.tensor_type.elem_type
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        dtype_str = TensorProto.DataType.Name(elem_type)
        print(f"输出:  {name:<30} dtype={dtype_str:<10} shape={shape}")
        tensors[name] = (elem_type, tuple(shape))

    print("\n总计 I/O Tensor 数量：", len(tensors))
    return tensors


def get_dtype_size(elem_type):
    """返回 ONNX dtype 的字节数"""
    table = {
        TensorProto.FLOAT: 4,
        TensorProto.FLOAT16: 2,
        TensorProto.BFLOAT16: 2,
        TensorProto.INT8: 1,
        TensorProto.UINT8: 1,
        TensorProto.INT16: 2,
        TensorProto.UINT16: 2,
        TensorProto.INT32: 4,
        TensorProto.INT64: 8,
    }
    return table.get(elem_type, None)


def convert_bin_to_npy():
    tensors = get_io_tensor_info(ONNX_PATH)

    print("\n================= 开始转换 bin → npy =================")

    for tensor_name, (elem_type, shape) in tensors.items():
        tensor_dir = os.path.join(DATA_DIR, tensor_name)
        if not os.path.isdir(tensor_dir):
            print(f"[跳过] 未找到目录：{tensor_dir}")
            continue

        dtype_bytes = get_dtype_size(elem_type)
        expected_size_bytes = np.prod(shape) * dtype_bytes

        print(f"\n处理 Tensor: {tensor_name}, shape={shape}, dtype={TensorProto.DataType.Name(elem_type)}")

        for fname in os.listdir(tensor_dir):
            if not fname.endswith(".bin"):
                continue

            bin_path = os.path.join(tensor_dir, fname)
            npy_path = bin_path.replace(".bin", ".npy")

            file_size = os.path.getsize(bin_path)

            # ===== 校验 bin 文件大小 =====
            if file_size != expected_size_bytes:
                print(f"[错误] {fname} 文件大小不匹配: size={file_size} bytes, expected={expected_size_bytes} bytes")
                print("      已跳过该文件！")
                continue

            print(f"  读取文件: {fname}")

            # 读取 bin
            data = read_tensor_from_bin(bin_path, elem_type, shape)

            # 保存 npy
            np.save(npy_path, data)
            print(f"    [OK] 保存 npy: {os.path.basename(npy_path)}")

            # 删除 bin
            os.remove(bin_path)
            print(f"    [DEL] 删除 bin: {fname}")

    print("\n=========== 全部转换完成 ===========\n")


if __name__ == "__main__":
    convert_bin_to_npy()
