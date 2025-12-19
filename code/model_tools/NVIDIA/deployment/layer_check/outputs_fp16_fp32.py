import tensorrt as trt
import numpy as np
import os
import json
import io
import base64
import pycuda.driver as cuda
import pycuda.autoinit
import argparse

# -----------------------------
# 命令行参数 / 配置开关
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--markAllOutputs", action="store_true", help="标记所有中间层输出")
args = parser.parse_args()

MARK_ALL_OUTPUTS = args.markAllOutputs  # 也可以直接改成 True/False 测试
# -----------------------------
# 配置
# -----------------------------
onnx_path = "/mnt/data/hantianyou/occ_pred/spetr_occ_1212.onnx"
engine_path = "/mnt/data/hantianyou/occ_pred/spetr_occ_1212_fp16_select.trt"
output_json_path = "/mnt/data/hantianyou/road_compare_tool/layer_check/save_outputs/outputs_fp16_select.json"
input_dir = "/mnt/data/hantianyou/occ_pred/spetr_occ_input_1212_10000"
# fp32_layers_set = [
#     # "/ScatterND",
#     # "/ScatterND_1",
#     # 以下为手动指定的 LayerNorm 与 Attention 层，强制使用 FP32 精度
#     "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/output_proj/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/output_proj/Add",
#     "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/Add_12",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/output_proj/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/output_proj/Add",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/Add_12",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/output_proj/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/output_proj/Add",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/Add_12",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/output_proj/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/output_proj/Add",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/Add_12",
#     "/model/pts_bbox_head/transformer/decoder/layers.0/ffns.0/layers/layers.0/layers.0.0/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/ffns.0/layers/layers.0/layers.0.0/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/ffns.0/layers/layers.0/layers.0.0/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/ffns.0/layers/layers.0/layers.0.0/MatMul",
#     "/model/pts_bbox_head/ego_pose_memory/ln/Pow",
#     "/model/pts_bbox_head/ego_pose_memory/ln/ReduceMean_1",
#     "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.4/Pow",
#     "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.4/ReduceMean_1",
#     "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.4/Add",
#     "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.4/Sqrt",
#     "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.4/Add_1",
#     "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/cam_embed/cam_embed.4/Pow",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/cam_embed/cam_embed.4/ReduceMean_1",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/cam_embed/cam_embed.4/Add",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/cam_embed/cam_embed.4/Sqrt",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/cam_embed/cam_embed.4/Add_1",
#     "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/cam_embed/cam_embed.4/Pow",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/cam_embed/cam_embed.4/ReduceMean_1",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/cam_embed/cam_embed.4/Add",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/cam_embed/cam_embed.4/Sqrt",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/cam_embed/cam_embed.4/Add_1",
#     "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/MatMul",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/cam_embed/cam_embed.4/Pow",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/cam_embed/cam_embed.4/ReduceMean_1",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/cam_embed/cam_embed.4/Add",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/cam_embed/cam_embed.4/Sqrt",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/cam_embed/cam_embed.4/Add_1",
#     "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/MatMul",
# ]

fp32_layers_set = [
    # "/model/occ3d_head/predicter/predicter.0/Add",
    # "/model/occ3d_head/occ_head/occ_head.0/Add",
    "/model/occ3d_head/predicter/predicter.1/Softplus",
    "/model/occ3d_head/occ_head/occ_head.1/Softplus",
]

# -----------------------------
# 初始化 TRT
# -----------------------------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)

parser = trt.OnnxParser(network, TRT_LOGGER)
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

# -----------------------------
# 根据开关决定是否导出中间层
# -----------------------------
if MARK_ALL_OUTPUTS:
    print("🔹 标记所有中间层输出为 network output ...")
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            tensor = layer.get_output(j)
            if tensor and not tensor.is_network_output:
                network.mark_output(tensor)
    print(f"✅ 已标记 {network.num_outputs} 个输出张量")
else:
    print("⚙️ 仅保留最终输出层，不导出中间层。")

# -----------------------------
# BuilderConfig
# -----------------------------
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
if hasattr(trt.BuilderFlag, "TF32"):
    config.clear_flag(trt.BuilderFlag.TF32)

# -----------------------------
# 设置敏感层 FP32
# -----------------------------
for i in range(network.num_layers):
    layer = network.get_layer(i)
    set_fp32 = False

    # 手动列表匹配
    normalized_layer_name = layer.name.replace('/', '.').replace('', '').strip()
    for name in fp32_layers_set:
        normalized_name = name.replace('/', '.').replace('', '').strip()
        if normalized_name in normalized_layer_name:
            set_fp32 = True
            break

    if set_fp32:
        for j in range(layer.num_outputs):
            try:
                layer.precision = trt.DataType.FLOAT
                layer.set_output_type(j, trt.DataType.FLOAT)
                print(f"Found node {layer.name} ({layer.type}) to FP32")
            except Exception as e:
                print(f"⚠️ Cannot set FP32 for layer {layer.name}: {e}")
scatter_keywords = ["Scatter", "scatter", "ScatterND", "scatter_nd", "ScatterElements"]

# print("\n=== Network layers (name | type | precision) ===")
# for i in range(network.num_layers):
#     layer = network.get_layer(i)
#     try:
#         print(f"{i:03d}: name='{layer.name}' type='{str(layer.type)}' precision='{layer.precision}' outputs={layer.num_outputs}")
#     except Exception:
#         print(f"{i:03d}: name='{layer.name}' type='{str(layer.type)}' (error reading precision)")

# # -----------------------------
# # 配置动态shape优化profile
# # -----------------------------
# print("🔧 配置动态shape优化profile...")
# profile = builder.create_optimization_profile()

# 首先加载输入数据来获取实际shape
inputs = {}
input_shapes = {}

for name in os.listdir(input_dir):
    subdir = os.path.join(input_dir, name)
    if not os.path.isdir(subdir):
        continue

    npy_path = os.path.join(subdir, "0.npy")
    bin_path = os.path.join(subdir, "0.bin")
    shape_path = os.path.join(subdir, "shape.json")

    if os.path.exists(npy_path):
        # ---------- npy ----------
        arr = np.load(npy_path)
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        inputs[name] = arr
        input_shapes[name] = arr.shape
        print(f"✅ 加载 npy 输入: {name}, shape={arr.shape}")

    elif os.path.exists(bin_path):
        # ---------- bin ----------
        if not os.path.exists(shape_path):
            print(f"❌ {subdir} 中存在 0.bin 但缺少 shape.json，跳过")
            continue

        with open(shape_path, "r") as f:
            meta = json.load(f)

        dtype = np.dtype(meta["dtype"])
        shape = tuple(meta["shape"])

        arr = np.fromfile(bin_path, dtype=dtype).reshape(shape)

        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        inputs[name] = arr
        input_shapes[name] = arr.shape
        print(f"✅ 加载 bin 输入: {name}, shape={arr.shape}")

    else:
        print(f"⚠️ 跳过 {subdir}，既没有 0.npy 也没有 0.bin")
# input_data = {}
# input_shapes = {}
# for fname in os.listdir(input_dir):
#     if fname.endswith(".npy"):
#         key = fname.replace(".npy", "")
#         arr = np.load(os.path.join(input_dir, fname)).astype(np.float32)
#         # if arr.ndim == 2:
#         #     arr = np.expand_dims(arr, axis=0)
#         input_data[key] = arr
#         input_shapes[key] = arr.shape
#         print(f"输入 {key}: shape={arr.shape}")

# # 为每个输入设置动态范围
# for i in range(network.num_inputs):
#     input_tensor = network.get_input(i)
#     input_name = input_tensor.name
    
#     if input_name in input_shapes:
#         actual_shape = input_shapes[input_name]
        
#         # 处理动态维度（-1）
#         min_shape = []
#         opt_shape = []
#         max_shape = []
        
#         for dim in actual_shape:
#             if dim == -1 or dim is None:
#                 # 动态维度，设置合理范围
#                 min_shape.append(1)      # 最小
#                 opt_shape.append(8)      # 最优（可根据实际情况调整）
#                 max_shape.append(32)     # 最大（可根据实际情况调整）
#             else:
#                 # 固定维度
#                 min_shape.append(dim)
#                 opt_shape.append(dim)
#                 max_shape.append(dim)
        
#         print(f"设置输入 '{input_name}' 动态范围:")
#         print(f"  min: {min_shape}")
#         print(f"  opt: {opt_shape}")
#         print(f"  max: {max_shape}")
        
#         profile.set_shape(input_name, min_shape, opt_shape, max_shape)
#     else:
#         print(f"⚠️ 警告: 输入 '{input_name}' 未在输入数据中找到，使用默认shape")

# config.add_optimization_profile(profile)

# -----------------------------
# 构建 engine
# -----------------------------
print("🔨 开始构建 engine...")
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build engine!")

with open(engine_path, "wb") as f:
    f.write(serialized_engine)
print(f"\n✅ 已生成支持动态shape的混合精度 engine: {engine_path}")

# -----------------------------
# 加载 engine 并进行推理
# -----------------------------
runtime = trt.Runtime(TRT_LOGGER)
with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# -----------------------------
# 分配 CUDA buffer - 支持动态shape
# -----------------------------
bindings = []
cuda_buffers = {}

print("=== 开始分配 CUDA 内存（动态shape） ===")

# 方法1: 使用新的 TensorRT API (推荐)
if hasattr(engine, 'num_io_tensors'):
    # TensorRT 8.5+ 新 API
    num_io_tensors = engine.num_io_tensors
    print(f"使用新 API，共有 {num_io_tensors} 个 IO tensors")
    
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io_tensors)]
    
    for i, name in enumerate(tensor_names):
        try:
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            
            # 对于动态shape，先设置输入的实际shape
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name in input_shapes:
                    actual_shape = input_shapes[name]
                    context.set_input_shape(name, actual_shape)
                    print(f"✅ 设置输入 '{name}' 的shape为: {actual_shape}")
            
            # 获取设置后的实际shape
            shape = context.get_tensor_shape(name)
            
            print(f"\n--- Tensor {i}: {name} ---")
            print(f"  Shape: {shape}")
            print(f"  Dtype: {dtype}")
            
            # 检查形状中是否有无效值
            if any(dim is not None and dim <= 0 for dim in shape):
                print(f"  ⚠️ 警告: 形状包含无效维度: {shape}")
                # 对于无效形状，设置默认形状 [1]
                shape = [1]
                print(f"  使用默认形状: {shape}")
            
            # 计算大小
            volume = trt.volume(shape) if shape else 1
            dtype_size = np.dtype(dtype).itemsize
            size = volume * dtype_size
            
            print(f"  Volume: {volume}")
            print(f"  Dtype size: {dtype_size}")
            print(f"  Total size: {size} bytes")
            
            # 检查大小是否有效
            if size <= 0:
                print(f"  ❌ 错误: 计算的大小无效: {size}")
                # 分配最小内存
                size = 1024  # 1KB
                print(f"  分配最小内存: {size} bytes")
            
            print(f"  正在分配 {size} bytes 内存...")
            buffer = cuda.mem_alloc(size)
            print(f"  ✅ 内存分配成功")
            
            cuda_buffers[name] = buffer
            bindings.append(int(buffer))
            
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                print(f"  Input: {name}, shape: {shape}, dtype: {dtype}")
            else:
                print(f"  Output: {name}, shape: {shape}, dtype: {dtype}")
                
        except Exception as e:
            print(f"  ❌ 处理tensor {name}时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

else:
    # TensorRT 8.4及以下旧 API
    print(f"使用旧 API，共有 {engine.num_bindings} 个 bindings")
    
    for i in range(engine.num_bindings):
        try:
            name = engine.get_binding_name(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            
            # 对于动态shape，先设置输入的实际shape
            if engine.binding_is_input(i):
                if name in input_shapes:
                    actual_shape = input_shapes[name]
                    context.set_binding_shape(i, actual_shape)
                    print(f"✅ 设置输入 '{name}' 的shape为: {actual_shape}")
            
            # 获取设置后的实际shape
            shape = context.get_binding_shape(i)
            
            print(f"\n--- Binding {i}: {name} ---")
            print(f"  Shape: {shape}")
            print(f"  Dtype: {dtype}")
            
            # 检查形状中是否有无效值
            if any(dim is not None and dim <= 0 for dim in shape):
                print(f"  ⚠️ 警告: 形状包含无效维度: {shape}")
                # 对于无效形状，设置默认形状 [1]
                shape = [1]
                print(f"  使用默认形状: {shape}")
            
            # 计算大小
            volume = trt.volume(shape) if shape else 1
            dtype_size = np.dtype(dtype).itemsize
            size = volume * dtype_size
            
            print(f"  Volume: {volume}")
            print(f"  Dtype size: {dtype_size}")
            print(f"  Total size: {size} bytes")
            
            # 检查大小是否有效
            if size <= 0:
                print(f"  ❌ 错误: 计算的大小无效: {size}")
                # 分配最小内存
                size = 1024  # 1KB
                print(f"  分配最小内存: {size} bytes")
            
            print(f"  正在分配 {size} bytes 内存...")
            buffer = cuda.mem_alloc(size)
            print(f"  ✅ 内存分配成功")
            
            cuda_buffers[name] = buffer
            bindings.append(int(buffer))
            
            if engine.binding_is_input(i):
                print(f"  Input: {name}, shape: {shape}, dtype: {dtype}")
            else:
                print(f"  Output: {name}, shape: {shape}, dtype: {dtype}")
                
        except Exception as e:
            print(f"  ❌ 处理binding {i} ({name})时出错: {e}")
            import traceback
            traceback.print_exc()
            continue

print(f"\n=== 内存分配完成 ===")
print(f"成功分配了 {len(cuda_buffers)} 个 buffers")

# -----------------------------
# 载入输入
# -----------------------------
print("\n=== 加载输入数据 ===")
for key, arr in inputs.items():
    if key not in cuda_buffers:
        print(f"⚠️ 跳过未在engine中找到的输入: {key}")
        continue
    
    try:
        cuda.memcpy_htod(cuda_buffers[key], arr)
        print(f"✅ 已加载输入: {key}, shape: {arr.shape}")
    except Exception as e:
        print(f"❌ 加载输入 {key} 失败: {e}")

# -----------------------------
# 执行推理
# -----------------------------
print("\n=== 开始推理 ===")
try:
    # 检查所有输入的形状是否有效
    if hasattr(engine, 'num_io_tensors'):
        # 新API：检查所有输入tensor的形状
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                shape = context.get_tensor_shape(name)
                print(f"输入 '{name}' 推理前shape: {shape}")
    else:
        # 旧API：检查所有binding的形状
        for i in range(engine.num_bindings):
            if engine.binding_is_input(i):
                name = engine.get_binding_name(i)
                shape = context.get_binding_shape(i)
                print(f"输入 '{name}' 推理前shape: {shape}")
    
    # 执行推理
    if hasattr(context, 'execute_v2'):
        success = context.execute_v2(bindings)
    else:
        success = context.execute_v1(bindings)
    
    print("✅ 推理完成")
except Exception as e:
    print(f"❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# -----------------------------
# 拷贝输出 - 支持动态shape
# -----------------------------
outputs = {}
missing_outputs = []

print("\n=== 开始拷贝输出 ===")

if hasattr(engine, 'num_io_tensors'):
    # 新 API
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for name in tensor_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            if name not in cuda_buffers:
                print(f"❌ 输出tensor '{name}' 不在cuda_buffers中，跳过")
                missing_outputs.append(name)
                continue
                
            try:
                dtype = trt.nptype(engine.get_tensor_dtype(name))
                shape = context.get_tensor_shape(name)  # 获取动态推理后的实际shape
                print(f"拷贝输出: {name}, shape: {shape}, dtype: {dtype}")
                
                host_arr = np.empty(shape, dtype=dtype)
                cuda.memcpy_dtoh(host_arr, cuda_buffers[name])
                outputs[name] = host_arr
                print(f"✅ 成功拷贝输出: {name}")
            except Exception as e:
                print(f"❌ 拷贝输出 {name} 失败: {e}")
else:
    # 旧 API
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if not engine.binding_is_input(i):
            if name not in cuda_buffers:
                print(f"❌ 输出tensor '{name}' 不在cuda_buffers中，跳过")
                missing_outputs.append(name)
                continue
                
            try:
                dtype = trt.nptype(engine.get_binding_dtype(i))
                shape = context.get_binding_shape(i)  # 获取动态推理后的实际shape
                print(f"拷贝输出: {name}, shape: {shape}, dtype: {dtype}")
                
                host_arr = np.empty(shape, dtype=dtype)
                cuda.memcpy_dtoh(host_arr, cuda_buffers[name])
                outputs[name] = host_arr
                print(f"✅ 成功拷贝输出: {name}")
            except Exception as e:
                print(f"❌ 拷贝输出 {name} 失败: {e}")

if missing_outputs:
    print(f"\n⚠️ 警告: 跳过了 {len(missing_outputs)} 个缺失的输出tensor")
    for name in missing_outputs[:10]:  # 只显示前10个
        print(f"  - {name}")

# -----------------------------
# 输出检查 + 超出FP16范围写log
# -----------------------------
log_path = "/mnt/data/hantianyou/occ_pred/fp16_abnormal.log"
list_path = "/mnt/data/hantianyou/occ_pred/fp16_abnormal_layers.txt"

naninf_layers = set()

with open(log_path, "w", encoding="utf-8") as logf:
    print(f"\n=== 推理输出检查 ===")
    print(f"⚠️ 仅记录 NaN/Inf 或超出 FP16 范围的层到: {log_path}")

    for name, arr in outputs.items():
        if arr.size == 0:
            continue

        max_val = arr.max() if arr.size > 0 else 0
        min_val = arr.min() if arr.size > 0 else 0

        msg = f"[{name}] dtype={arr.dtype} min={min_val:.6e} max={max_val:.6e}"
        print(msg)

        # # 检查 FP16 溢出
        # if arr.dtype == np.float16 and (max_val > 65504 or min_val < -65504):
        #     msg = f"⚠️ [{name}] 超出 FP16 范围: max={max_val:.6f}, min={min_val:.6f}"
        #     print(msg)
        #     logf.write(msg + "\n")

        # 检查 NaN / Inf
        if np.isnan(arr).any() or np.isinf(arr).any():
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            msg = f"⚠️ [{name}] 检测到 NaN/Inf: NaN={nan_count}, Inf={inf_count}"
            print(msg)
            logf.write(msg + "\n")
            naninf_layers.add(name)

print(f"\n✅ 异常层日志已保存到: {log_path}")

# 只保存含 NaN/Inf 的层名为字符串列表
if naninf_layers:
    list_str = "fp32_layers_set = [\n"
    for nm in sorted(naninf_layers):
        list_str += f'    "{nm}",\n'
    list_str += "]\n"

    with open(list_path, "w", encoding="utf-8") as lf:
        lf.write(list_str)

    print(f"✅ 含 NaN/Inf 的层列表已保存到: {list_path}")

# -----------------------------
# 保存 JSON + base64
# -----------------------------
json_dict = {"lst": [[None, [{"outputs": {}}]]]}
for name, arr in outputs.items():
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=True)
    b64str = base64.b64encode(bio.getvalue()).decode("ascii")
    json_dict["lst"][0][1][0]["outputs"][name] = {
        "dtype": str(arr.dtype),
        "values": {"array": b64str}
    }

with open(output_json_path, "w") as f:
    json.dump(json_dict, f)

print(f"\n✅ 已保存混合精度输出到 {output_json_path}")
print(f"✅ 总共保存了 {len(outputs)} 个输出张量")
print(f"⚠️ 跳过了 {len(missing_outputs)} 个缺失的输出张量")