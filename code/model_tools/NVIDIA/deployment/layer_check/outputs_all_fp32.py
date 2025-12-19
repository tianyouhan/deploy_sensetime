import tensorrt as trt
import numpy as np
import os
import json
import io
import base64
from polygraphy.backend.trt import TrtRunner

# -----------------------------
# 配置
# -----------------------------
onnx_path = "/mnt/data/hantianyou/occ_pred/spetr_occ_1212.onnx"
engine_path = "/mnt/data/hantianyou/occ_pred/spetr_occ_1212_fp32.trt"
output_json_path = "/mnt/data/hantianyou/road_compare_tool/layer_check/save_outputs/outputs_fp32_all.json"
input_dir = "/mnt/data/hantianyou/occ_pred/spetr_occ_input_1212_10000"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")

# =============================
# ==== ONNX -> TRT engine 构建 ====
# =============================
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)

# 解析 ONNX
parser = trt.OnnxParser(network, TRT_LOGGER)
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise RuntimeError("ONNX parse failed")

# -----------------------------
# 标记所有中间层输出
# -----------------------------
for i in range(network.num_layers):
    layer = network.get_layer(i)
    for j in range(layer.num_outputs):
        tensor = layer.get_output(j)
        if tensor is not None and not tensor.is_network_output:
            network.mark_output(tensor)

# -----------------------------
# 配置动态shape
# -----------------------------
config = builder.create_builder_config()
# config.set_flag(trt.BuilderFlag.STRONGLY_TYPED)

# 创建优化profile处理动态shape
profile = builder.create_optimization_profile()

# 获取输入名称和shape
# inputs = {}
# input_shapes = {}
# for fname in os.listdir(input_dir):
#     if fname.endswith(".npy"):
#         key = fname.replace(".npy", "")
#         arr = np.load(os.path.join(input_dir, fname)).astype(np.float32)
#         inputs[key] = arr
#         input_shapes[key] = arr.shape
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

print("✅ 已加载输入：", list(inputs.keys()))

# 为每个输入设置动态范围
for i in range(network.num_inputs):
    input_tensor = network.get_input(i)
    input_name = input_tensor.name
    
    if input_name in input_shapes:
        shape = input_shapes[input_name]
        # 设置最小、最优、最大shape
        # 这里使用相同的shape，您可以根据需要调整
        min_shape = shape
        opt_shape = shape  
        max_shape = shape
        
        print(f"设置输入 '{input_name}' 的shape范围: min{min_shape}, opt{opt_shape}, max{max_shape}")
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
    else:
        print(f"警告: 输入 '{input_name}' 未在输入数据中找到")

config.add_optimization_profile(profile)
# print("🔍 检查网络层精度设置:")
# for i in range(network.num_layers):
#     layer = network.get_layer(i)
#     layer_type_str = str(layer.type)
#     print(f"[{i:4d}] {layer.name:60s} | type={layer_type_str:25s} | precision={layer.precision} | "
#         f"input_types={[layer.get_input(i).dtype for i in range(layer.num_inputs)]} "
#         f"-> output_types={[layer.get_output(i).dtype for i in range(layer.num_outputs)]}")
# -----------------------------
# 构建 engine
# -----------------------------
serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Failed to build engine!")

with open(engine_path, "wb") as f:
    f.write(serialized_engine)

print(f"✅ 已生成支持动态shape的FP32 engine: {engine_path}")

# =============================
# ==== 推理 ====
# =============================
runtime = trt.Runtime(TRT_LOGGER)
with open(engine_path, "rb") as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# 使用 Polygraphy TrtRunner 推理
runner = TrtRunner(engine)
runner.activate()

# 设置执行context的shape
context = runner.context
for input_name, arr in inputs.items():
    # TensorRT 10+ 使用 set_input_shape
    if hasattr(context, "set_input_shape"):
        context.set_input_shape(input_name, arr.shape)
    else:
        context.set_binding_shape(engine.get_binding_index(input_name), arr.shape)

outputs = runner.infer(feed_dict=inputs)
runner.deactivate()

# -----------------------------
# 保存输出
# -----------------------------
json_dict = {"lst": [[None, [{"outputs": {}}]]]}
for name, arr in outputs.items():
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    b64str = base64.b64encode(bio.getvalue()).decode("ascii")
    json_dict["lst"][0][1][0]["outputs"][name] = {"values": {"array": b64str}}

with open(output_json_path, "w") as f:
    json.dump(json_dict, f)

print(f"✅ 已保存所有中间层输出到 {output_json_path}")