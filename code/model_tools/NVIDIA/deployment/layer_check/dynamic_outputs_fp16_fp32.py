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

MARK_ALL_OUTPUTS = args.markAllOutputs

# -----------------------------
# 配置 - 动态shape设置
# -----------------------------
onnx_path = "/mnt/data/hantianyou/fsdv2/pts_backbone/spetr.onnx"   # ONNX 模型路径
engine_path = "/mnt/data/hantianyou/road_compare_tool/layer_check/save_trt/pts_fp32_all_layers.trt"                                # 保存的 TensorRT engine 路径
output_json_path = "/mnt/data/hantianyou/road_compare_tool/layer_check/save_outputs/pts_outputs_fp16_select.json"                         # 输出 JSON 路径
input_dir = "/mnt/data/hantianyou/2025_10_14_test/save_tensors/pts_backbone_align_npy"  # 输入数据文件夹（.npy）

# 动态shape配置
DYNAMIC_SHAPES = {
    'voxel_coords': {
        'min': [1, 2],
        'opt': [20000, 2], 
        'max': [20000, 2]
    },
    'vfe_input': {
        'min': [1, 9, 32, 1],
        'opt': [20000, 9, 32, 1],
        'max': [20000, 9, 32, 1]
    }
}

fp32_layers_set = [
    "/model/Mul",
    "/model/Concat",
    "/model/Unsqueeze_1",
    "/model/Expand_1",
    "/model/Cast",
    "/model/Add",
    "/model/Add_1",
    "/model/ScatterND",
]

# -----------------------------
# 初始化 TRT
# -----------------------------
TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)  # 改为VERBOSE以便查看更多信息
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

print(f"✅ ONNX解析成功，网络有 {network.num_inputs} 个输入，{network.num_outputs} 个输出")

# 打印输入信息
for i in range(network.num_inputs):
    input_tensor = network.get_input(i)
    print(f"输入 {i}: {input_tensor.name}, shape: {input_tensor.shape}")

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
# BuilderConfig - 添加动态shape支持
# -----------------------------
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)
if hasattr(trt.BuilderFlag, "TF32"):
    config.clear_flag(trt.BuilderFlag.TF32)

# 设置动态shape优化配置文件
print("=== 设置动态shape优化配置文件 ===")
profile = builder.create_optimization_profile()

# 为每个输入设置动态shape范围
for i in range(network.num_inputs):
    input_tensor = network.get_input(i)
    input_name = input_tensor.name
    
    if input_name in DYNAMIC_SHAPES:
        shape_config = DYNAMIC_SHAPES[input_name]
        min_shape = shape_config['min']
        opt_shape = shape_config['opt'] 
        max_shape = shape_config['max']
        
        print(f"设置输入 '{input_name}' 动态shape:")
        print(f"  min: {min_shape}")
        print(f"  opt: {opt_shape}")
        print(f"  max: {max_shape}")
        
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    else:
        # 对于非动态输入，使用固定shape
        shape = list(input_tensor.shape)
        print(f"设置输入 '{input_name}' 固定shape: {shape}")
        profile.set_shape(input_name, shape, shape, shape)

config.add_optimization_profile(profile)

# -----------------------------
# 设置敏感层 FP32
# -----------------------------
for i in range(network.num_layers):
    layer = network.get_layer(i)
    set_fp32 = False

    # # 自动匹配 MatMul
    # if layer.type == trt.LayerType.MATRIX_MULTIPLY:
    #     set_fp32 = True

    # elif layer.type == trt.LayerType.ELEMENTWISE:
    #     # 直接通过 getattr 安全访问 operation 属性
    #     op = getattr(layer, "operation", None)
    #     if op == trt.ElementWiseOperation.DIV:
    #         print(f"[DIV match] {i:4d} | {layer.name}")
    #         set_fp32 = True
            
    # if layer.type == trt.LayerType.SOFTMAX:
    #     set_fp32 = True

    # 手动列表匹配
    normalized_layer_name = layer.name.replace('/', '.').replace('_output_0', '').strip()
    for name in fp32_layers_set:
        normalized_name = name.replace('/', '.').replace('_output_0', '').strip()
        if normalized_name in normalized_layer_name:
            set_fp32 = True
            break

    # 3️⃣ 设置 FP32，但跳过常量权重类型不允许的
    if set_fp32:
        for j in range(layer.num_outputs):
            out_tensor = layer.get_output(j)
            # 仅对非整数常量输出设置 FP32
            if out_tensor.dtype != trt.DataType.INT32 and out_tensor.dtype != trt.DataType.INT64:
                try:
                    layer.precision = trt.DataType.FLOAT
                    layer.set_output_type(j, trt.DataType.FLOAT)
                    print(f"Found node {layer.name} ({layer.type}) to FP32")
                except Exception as e:
                    print(f"⚠️ Cannot set FP32 for layer {layer.name}: {e}")
            else:
                print(f"⚠️ Skip FP32 for {layer.name} output {j} because dtype={out_tensor.dtype}")
# import pdb;pdb.set_trace()
# -----------------------------
# 构建 engine
# -----------------------------
print("\n=== 开始构建 engine ===")
try:
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build engine!")
    
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ 已生成混合精度 engine: {engine_path}")
    
except Exception as e:
    print(f"❌ 构建失败: {e}")
    # 尝试不使用FP16
    print("尝试使用FP32构建...")
    config.clear_flag(trt.BuilderFlag.FP16)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine:
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"✅ 使用FP32生成 engine: {engine_path}")
    else:
        raise RuntimeError("FP32构建也失败了!")

# -----------------------------
# 加载 engine 并进行推理
# -----------------------------
runtime = trt.Runtime(TRT_LOGGER)
with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# -----------------------------
# 分配 CUDA buffer - 简化版本，避免动态shape的复杂性
# -----------------------------
print("=== 开始分配 CUDA 内存 ===")

# 首先收集输入数据
input_data = {}
input_shapes = {}
print("\n=== 加载输入数据并确定shape ===")
for fname in os.listdir(input_dir):
    if fname.endswith(".npy"):
        key = fname.replace(".npy", "")
        arr = np.load(os.path.join(input_dir, fname)).astype(np.float32)
        input_data[key] = arr
        input_shapes[key] = arr.shape
        print(f"输入 {key}: shape={arr.shape}, min={arr.min():.6f}, max={arr.max():.6f}")

# 分配内存 - 使用新API
bindings = []
cuda_buffers = {}

if hasattr(engine, 'num_io_tensors'):
    # TensorRT 8.5+ 新 API
    num_io_tensors = engine.num_io_tensors
    tensor_names = [engine.get_tensor_name(i) for i in range(num_io_tensors)]
    
    print(f"\n使用新API，共有 {num_io_tensors} 个IO tensors")
    
    for name in tensor_names:
        try:
            # 对于输入，使用实际数据的shape
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                if name in input_shapes:
                    shape = input_shapes[name]
                    # 设置输入shape
                    context.set_input_shape(name, shape)
                    print(f"设置输入shape: {name} -> {shape}")
                else:
                    print(f"⚠️ 警告: 输入 {name} 没有对应的数据文件")
                    continue
            else:
                # 对于输出，获取当前shape
                shape = context.get_tensor_shape(name)
            
            dtype = trt.nptype(engine.get_tensor_dtype(name))
            volume = trt.volume(shape)
            size = volume * np.dtype(dtype).itemsize
            
            print(f"分配: {name}, shape: {shape}, dtype: {dtype}, size: {size} bytes")
            
            buffer = cuda.mem_alloc(size)
            cuda_buffers[name] = buffer
            bindings.append(int(buffer))
            
        except Exception as e:
            print(f"❌ 处理tensor {name}时出错: {e}")
            import traceback
            traceback.print_exc()

else:
    # 旧 API
    print(f"使用旧API，共有 {engine.num_bindings} 个bindings")
    
    for i in range(engine.num_bindings):
        try:
            name = engine.get_binding_name(i)
            dtype = trt.nptype(engine.get_binding_dtype(i))
            
            # 对于输入，使用实际数据的shape
            if engine.binding_is_input(i):
                if name in input_shapes:
                    shape = input_shapes[name]
                    # 设置输入shape
                    context.set_binding_shape(i, shape)
                    print(f"设置输入shape: {name} -> {shape}")
                else:
                    print(f"⚠️ 警告: 输入 {name} 没有对应的数据文件")
                    continue
            else:
                # 对于输出，获取当前shape
                shape = context.get_binding_shape(i)
            
            volume = trt.volume(shape)
            size = volume * np.dtype(dtype).itemsize
            
            print(f"分配: {name}, shape: {shape}, dtype: {dtype}, size: {size} bytes")
            
            buffer = cuda.mem_alloc(size)
            cuda_buffers[name] = buffer
            bindings.append(int(buffer))
            
        except Exception as e:
            print(f"❌ 处理binding {i} ({name})时出错: {e}")
            import traceback
            traceback.print_exc()

print(f"\n=== 内存分配完成 ===")
print(f"成功分配了 {len(cuda_buffers)} 个 buffers")

# -----------------------------
# 载入输入数据到GPU
# -----------------------------
print("\n=== 拷贝输入数据到GPU ===")
for name, arr in input_data.items():
    if name in cuda_buffers:
        try:
            # 确保数据是连续的并且类型正确
            arr_contiguous = np.ascontiguousarray(arr.astype(np.float32))
            cuda.memcpy_htod(cuda_buffers[name], arr_contiguous)
            print(f"✅ 已加载输入: {name}, shape: {arr.shape}")
        except Exception as e:
            print(f"❌ 加载输入 {name} 失败: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠️ 输入 {name} 没有对应的CUDA buffer")

# -----------------------------
# 执行推理
# -----------------------------
print("\n=== 开始推理 ===")
try:
    # 检查所有绑定形状是否有效
    if hasattr(context, 'all_binding_shapes_specified'):
        if not context.all_binding_shapes_specified:
            print("❌ 错误: 不是所有绑定的形状都已指定")
    
    if hasattr(context, 'all_shape_inputs_specified'):  
        if not context.all_shape_inputs_specified:
            print("❌ 错误: 不是所有形状输入都已指定")
    
    if hasattr(context, 'execute_v2'):
        success = context.execute_v2(bindings)
    else:
        success = context.execute_v1(bindings)
        
    if success:
        print("✅ 推理完成")
    else:
        print("❌ 推理执行返回失败")
        
except Exception as e:
    print(f"❌ 推理失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# -----------------------------
# 拷贝输出
# -----------------------------
outputs = {}
print("\n=== 开始拷贝输出 ===")

if hasattr(engine, 'num_io_tensors'):
    # 新 API
    tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    for name in tensor_names:
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            if name in cuda_buffers:
                try:
                    dtype = trt.nptype(engine.get_tensor_dtype(name))
                    shape = context.get_tensor_shape(name)
                    
                    print(f"拷贝输出: {name}, shape: {shape}, dtype: {dtype}")
                    
                    host_arr = np.empty(shape, dtype=dtype)
                    cuda.memcpy_dtoh(host_arr, cuda_buffers[name])
                    outputs[name] = host_arr
                    
                    # 立即检查输出是否全零
                    if np.all(host_arr == 0):
                        print(f"⚠️ 警告: 输出 {name} 全部为0!")
                    else:
                        print(f"✅ 成功拷贝输出: {name}, 非零值范围: [{host_arr.min():.6f}, {host_arr.max():.6f}]")
                        
                except Exception as e:
                    print(f"❌ 拷贝输出 {name} 失败: {e}")
            else:
                print(f"❌ 输出tensor '{name}' 不在cuda_buffers中")
else:
    # 旧 API
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        if not engine.binding_is_input(i):
            if name in cuda_buffers:
                try:
                    dtype = trt.nptype(engine.get_binding_dtype(i))
                    shape = context.get_binding_shape(i)
                    
                    print(f"拷贝输出: {name}, shape: {shape}, dtype: {dtype}")
                    
                    host_arr = np.empty(shape, dtype=dtype)
                    cuda.memcpy_dtoh(host_arr, cuda_buffers[name])
                    outputs[name] = host_arr
                    
                    # 立即检查输出是否全零
                    if np.all(host_arr == 0):
                        print(f"⚠️ 警告: 输出 {name} 全部为0!")
                    else:
                        print(f"✅ 成功拷贝输出: {name}, 非零值范围: [{host_arr.min():.6f}, {host_arr.max():.6f}]")
                        
                except Exception as e:
                    print(f"❌ 拷贝输出 {name} 失败: {e}")
            else:
                print(f"❌ 输出tensor '{name}' 不在cuda_buffers中")

# -----------------------------
# 输出检查
# -----------------------------
print("\n=== 推理输出检查 ===")
all_zeros = True
for name, arr in outputs.items():
    if arr.size == 0:
        print(f"Layer {name}: dtype={arr.dtype}, empty array!")
        continue
    
    max_val = arr.max() if arr.size > 0 else 0
    min_val = arr.min() if arr.size > 0 else 0
    mean_val = arr.mean() if arr.size > 0 else 0
    
    print(f"Layer {name}: dtype={arr.dtype}, shape={arr.shape}")
    print(f"  max={max_val:.6f}, min={min_val:.6f}, mean={mean_val:.6f}")
    
    if not np.all(arr == 0):
        all_zeros = False
        
    if arr.dtype == np.float16:
        if max_val > 65504 or min_val < -65504:
            print(f"⚠️ Layer {name} 超出 FP16 范围！")
    elif arr.dtype == np.float32:
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"⚠️ Layer {name} 包含 NaN 或 Inf！")

if all_zeros:
    print("\n❌ 所有输出都是0！可能的问题：")
    print("1. 输入数据没有正确传递")
    print("2. 模型构建有问题")
    print("3. 动态shape设置不正确")
    print("4. FP16精度损失太大，尝试使用FP32")

# -----------------------------
# 保存 JSON + base64
# -----------------------------
if outputs:
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

    print(f"\n✅ 已保存输出到 {output_json_path}")
    print(f"✅ 总共保存了 {len(outputs)} 个输出张量")
else:
    print("\n❌ 没有输出数据可以保存")