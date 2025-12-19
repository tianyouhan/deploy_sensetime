import os
import numpy as np
import onnx
from onnx.helper import tensor_dtype_to_np_dtype
import modelopt.onnx.quantization as moq
import argparse
import json


def load_file(file_path, shape, file_type="bin", dtype=np.float32):
    data = (
        np.fromfile(file_path, dtype=dtype)
        if file_type == "bin"
        else np.load(file_path)
    )
    print(file_path, "data.size:", data.size, " shape:", shape)

    # 自动推断 -1 或 0 的维度（只能有一个动态维度）
    inferred_shape = []
    known_volume = 1
    dynamic_dim_index = -1

    for idx, dim in enumerate(shape):
        if isinstance(dim, str):
            try:
                dim = int(dim)
            except:
                pass

        if isinstance(dim, int) and dim > 0:
            inferred_shape.append(dim)
            known_volume *= dim
        elif dim in [-1, 0, None, "None"]:
            if dynamic_dim_index != -1:
                raise ValueError(f"Shape {shape} 有多个动态维度，不能自动推断")
            dynamic_dim_index = idx
            inferred_shape.append(-1)
        else:
            raise ValueError(f"不支持的 shape 元素: {dim} (type: {type(dim)})")

    if dynamic_dim_index != -1:
        inferred_dim = data.size // known_volume
        if data.size % known_volume != 0:
            raise ValueError(
                f"数据大小与静态维度不匹配，data.size={data.size}, known_volume={known_volume}"
            )
        inferred_shape[dynamic_dim_index] = inferred_dim

    try:
        data = data.reshape(inferred_shape)
    except Exception as e:
        print(
            f"[ERROR] 自动 reshape 失败: shape={inferred_shape}, data.size={data.size}"
        )
        raise

    return data


def gen_calib_data(onnx_model_path, data_dir):
    model = onnx.load(onnx_model_path)
    model_inputs = {}

    # 解析模型输入信息
    for input_tensor in model.graph.input:
        tensor_name = input_tensor.name
        tensor_dtype = input_tensor.type.tensor_type.elem_type
        tensor_shape_proto = input_tensor.type.tensor_type.shape.dim

        tensor_shape = []
        for dim in tensor_shape_proto:
            if dim.HasField("dim_value") and dim.dim_value > 0:
                tensor_shape.append(dim.dim_value)
            else:
                tensor_shape.append(-1)  # 动态维度

        model_inputs[tensor_name] = {
            "dtype": tensor_dtype_to_np_dtype(tensor_dtype).name,
            "shape": tensor_shape,
        }

    print("模型输入信息：", model_inputs)

    # 用第一个输入路径中的 bin 文件名作为参考
    first_input_dir = os.path.join(data_dir, list(model_inputs)[0])
    timestamps = sorted(
        [f[:-4] for f in os.listdir(first_input_dir) if f.endswith(".bin")]
    )
    selected_timestamps = timestamps[:]  # 可限制数量

    print(f"Selected timestamps: {selected_timestamps}")

    # 每个输入名 -> list of arrays
    calib_data = {k: [] for k in model_inputs}

    for ts in selected_timestamps:
        for input_name, info in model_inputs.items():
            bin_path = os.path.join(data_dir, input_name, f"{ts}.bin")
            dtype = np.dtype(info["dtype"])
            shape = info["shape"]

            arr = load_file(bin_path, shape, dtype=dtype)
            calib_data[input_name].append(arr)

    # 合并 batch（智能维度匹配）
    for input_name in calib_data:
        single_shape = model_inputs[input_name]["shape"]
        elem_arrs = calib_data[input_name]
        elem_rank = elem_arrs[0].ndim
        model_rank = len(single_shape)

        # 自动 squeeze 多余的维度（如 [B, N, 1] -> [B, N]）
        for i in range(len(elem_arrs)):
            while elem_arrs[i].ndim > model_rank:
                elem_arrs[i] = np.squeeze(elem_arrs[i], axis=-1)

        elem_rank = elem_arrs[0].ndim  # 更新 rank
        if elem_rank == model_rank:
            # 模型包含 batch，直接 concat
            calib_data[input_name] = np.concatenate(elem_arrs, axis=0)
        elif elem_rank == model_rank - 1:
            # 模型不包含 batch，stack 出 batch 维
            calib_data[input_name] = np.stack(elem_arrs, axis=0)
        else:
            raise ValueError(
                f"[ERROR] 输入 {input_name} rank 不匹配，模型期望 {model_rank}，实际 {elem_rank}"
            )

    return calib_data


def ptq(onnx_path, calib_dir, quant_onnx, nodes_to_exclude, op_types_to_quantize):
    print("[INFO] 生成校准数据...")
    calib_data = gen_calib_data(onnx_path, calib_dir)

    print("[INFO] 量化参数:")
    print(f"  排除节点: {nodes_to_exclude}")
    print(f"  量化算子类型: {op_types_to_quantize}")
    print(f"  排除节点数量: {len(nodes_to_exclude) if nodes_to_exclude else 0}")

    print("[INFO] 开始量化...")

    model = onnx.load(onnx_path)
    existing_nodes = {node.name for node in model.graph.node}

    # # 自动排除 ConvTranspose，避免量化 deconv 时触发断言
    # convtranspose_nodes = [node.name for node in model.graph.node if node.op_type == "ConvTranspose"]
    # print(f"[DEBUG] ConvTranspose nodes detected: {convtranspose_nodes}")
    # if convtranspose_nodes:
    #     print(f"[INFO] 自动排除 ConvTranspose 节点: {convtranspose_nodes}")
    #     nodes_to_exclude = [name for name in nodes_to_exclude if name]
    #     for name in convtranspose_nodes:
    #         if name not in nodes_to_exclude:
    #             nodes_to_exclude.append(name)
    #     print(f"[INFO] 合并后的排除节点: {nodes_to_exclude}")

    missing = [name for name in nodes_to_exclude if name and name not in existing_nodes]
    if missing:
        raise ValueError(f"[ERROR] 节点不存在，请检查: {missing}")

    moq.quantize(
        onnx_path=onnx_path,
        calibration_data=calib_data,
        output_path=quant_onnx,
        quantize_mode="int8",
        high_precision_dtype="fp32",
        # high_precision_dtype="fp16",
        verbose=True,
        nodes_to_exclude=nodes_to_exclude,
        op_types_to_quantize=op_types_to_quantize,
    )

    print("[INFO] 量化完成，保存至：", quant_onnx)


# 示例调用：
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_path", type=str, required=True)
    parser.add_argument("-c", "--calib_dir", type=str, required=True)
    parser.add_argument("-q", "--quant_onnx", type=str, required=True)
    parser.add_argument(
        "-n", "--nodes_to_exclude", type=str, required=False, default=""
    )
    parser.add_argument(
        "-op", "--op_types_to_quantize", type=str, required=False, default=""
    )

    args = parser.parse_args()

    onnx_path = args.onnx_path
    calib_dir = args.calib_dir
    quant_onnx = args.quant_onnx
    # nodes_to_exclude = json.loads(args.nodes_to_exclude)
    # op_types_to_quantize = json.loads(args.op_types_to_quantize)
    # 处理节点排除参数，过滤空字符串
    if args.nodes_to_exclude and args.nodes_to_exclude.strip():
        nodes_to_exclude = [node.strip() for node in args.nodes_to_exclude.split(",") if node.strip()]
    else:
        nodes_to_exclude = []
    if args.op_types_to_quantize and args.op_types_to_quantize.strip():
        op_types_to_quantize = [
            op_type.strip()
            for op_type in args.op_types_to_quantize.split(",")
            if op_type.strip()
        ]
    else:
        op_types_to_quantize = [
            "Conv", "Gemm", "MatMul", "ConvTranspose",
            "Add", "Mul", "Sub", "Div", "Pow", 
            "Relu", "Sigmoid", "Tanh", "LeakyRelu",
            "BatchNormalization", "LayerNorm",
            "AveragePool", "MaxPool", "GlobalAveragePool",
            "Concat", "Reshape", "Transpose"
        ]

    ptq(onnx_path, calib_dir, quant_onnx, nodes_to_exclude, op_types_to_quantize)

