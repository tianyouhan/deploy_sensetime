import tensorrt as trt
import numpy as np
import os
import json
import io
import base64

# -----------------------------
# 配置
# -----------------------------
onnx_path = "/mnt/data/hantianyou/iter_38072/deploy/head/spetr.onnx"
engine_path = "/mnt/data/hantianyou/iter_38072/deploy/head/spetr.trt"
output_json_path = "/mnt/data/hantianyou/iter_45692/deploy/head/outputs_fp16_select.json"
input_dir = "/mnt/data/hantianyou/far3d_head_align_npy"

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(TRT_LOGGER, "")
# =============================
# ==== ONNX -> TRT engine 构建 ====
# =============================
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
# 多层 mark 输出
# -----------------------------
target_names = [
    # "/model/pts_bbox_head/transformer/decoder/layers.0/norms.0/Sub_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/norms.0/Sqrt_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.0/attn/Div_2_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.0/attn/Transpose_5_output_0",
    # "/model/pts_bbox_head/Concat_42_output_0",
    # "/model/pts_bbox_head/Concat_43_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/Concat_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/Concat_1_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/Add_2_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.4/Add_1_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.4/Sqrt_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.4/Sub_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.0/attn/Softmax_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/Reshape_12_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/weights_fc_img/Add_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.0/MatMul_output_0",
    # "/model/pts_query_generator/pre_bev_embed/pre_bev_embed.1/Sub_output_0",
    "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.0/Add_output_0",
    "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.2/MatMul_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.3/Clip_output_0",
    "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/cam_embed/cam_embed.0/Add_output_0",
    "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/cam_embed/cam_embed.2/MatMul_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.1/attentions.1/cam_embed/cam_embed.3/Clip_output_0",
    "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/cam_embed/cam_embed.0/Add_output_0",
    "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/cam_embed/cam_embed.2/MatMul_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.2/attentions.1/cam_embed/cam_embed.3/Clip_output_0",
    "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/cam_embed/cam_embed.0/Add_output_0",
    "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/cam_embed/cam_embed.2/MatMul_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.3/attentions.1/cam_embed/cam_embed.3/Clip_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/cam_embed/cam_embed.4/ReduceMean_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/Softmax_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/Tile_1_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/Reshape_13_output_0",
    # "/model/pts_bbox_head/transformer/decoder/layers.0/attentions.1/MultiscaleDeformableAttnPlugin_TRT_output_0",
]

found_layers = set()
for i in range(network.num_layers):
    layer = network.get_layer(i)
    for j in range(layer.num_outputs):
        tensor = layer.get_output(j)
        if tensor.name in target_names and tensor.name not in found_layers:
            network.mark_output(tensor)
            found_layers.add(tensor.name)
            print(f"✅ 已 mark 输出层: {tensor.name}")

# 检查未找到的层
for name in target_names:
    if name not in found_layers:
        print(f"⚠️ 未找到指定层: {name}, 将不会 mark 输出")

# -----------------------------
# BuilderConfig 强制 FP16
# -----------------------------
config = builder.create_builder_config()
# config.max_workspace_size = 4 << 30  # 4GB
config.set_flag(trt.BuilderFlag.FP16)
if hasattr(trt.BuilderFlag, "TF32"):
    config.clear_flag(trt.BuilderFlag.TF32)

serialized_engine = builder.build_serialized_network(network, config)
if serialized_engine is None:
    raise RuntimeError("Engine 构建失败")
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(serialized_engine)
with open(engine_path, "wb") as f:
    f.write(engine.serialize())
print(f"✅ 已生成 FP32 engine 并 mark 指定中间层: {engine_path}")

# =============================
# ==== 推理部分（改造） ====
# =============================
from polygraphy.backend.trt import TrtRunner

runner = TrtRunner(engine)
runner.activate()
# -----------------------------
# 准备输入
# -----------------------------
inputs = {}
for fname in os.listdir(input_dir):
    if fname.endswith(".npy"):
        key = fname.replace(".npy", "")
        arr = np.load(os.path.join(input_dir, fname)).astype(np.float32)
        inputs[key] = arr

# 自动检查输入 shape，并加 batch 维度
feed_dict_fixed = {}
num_io_tensors = engine.num_io_tensors

# for i in range(num_io_tensors):
#     tensor_name = engine.get_tensor_name(i)
#     if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
#         if tensor_name not in inputs:
#             print(f"⚠️ 输入 {tensor_name} 未在输入文件中找到，跳过")
#             continue

#         arr = inputs[tensor_name]
#         binding_shape = engine.get_tensor_shape(tensor_name)

#         # 显式 batch 模式：shape[0] 通常是 batch
#         if arr.ndim == len(binding_shape) - 1:
#             arr = np.expand_dims(arr, 0)

#         # 调整 dtype
#         dtype = engine.get_tensor_dtype(tensor_name)
#         if dtype == trt.float16:
#             arr = arr.astype(np.float16)
#         else:
#             arr = arr.astype(np.float32)

#         feed_dict_fixed[tensor_name] = arr
for i in range(num_io_tensors):
    tensor_name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
        binding_shape = engine.get_tensor_shape(tensor_name)

        # ==============================
        # 如果没找到输入文件 -> 自动生成随机 npy
        # ==============================
        if tensor_name not in inputs:
            print(f"⚠️ 输入 {tensor_name} 未找到，自动生成随机输入 npy")
            dtype = engine.get_tensor_dtype(tensor_name)
            np_dtype = np.float16 if dtype == trt.float16 else np.float32

            # 将 -1（动态维）替换为默认值，例如 batch=1
            shape = [1 if s == -1 else s for s in binding_shape]
            arr = np.random.randn(*shape).astype(np_dtype)

            # 保存 .npy 文件
            save_path = os.path.join(input_dir, f"{tensor_name}.npy")
            os.makedirs(input_dir, exist_ok=True)
            np.save(save_path, arr)
            print(f"✅ 已生成随机输入文件: {save_path}")

            inputs[tensor_name] = arr
        else:
            arr = inputs[tensor_name]

        # ==============================
        # 显式 batch 维度调整
        # ==============================
        if arr.ndim == len(binding_shape) - 1:
            arr = np.expand_dims(arr, 0)

        dtype = engine.get_tensor_dtype(tensor_name)
        if dtype == trt.float16:
            arr = arr.astype(np.float16)
        else:
            arr = arr.astype(np.float32)

        feed_dict_fixed[tensor_name] = arr

# 推理
outputs = runner.infer(feed_dict=feed_dict_fixed)
runner.deactivate()
# -----------------------------
# 保存 JSON + base64
# -----------------------------
json_dict = {"lst": [[None, [{"outputs": {}}]]]}
for name, arr in outputs.items():
    bio = io.BytesIO()
    np.save(bio, arr.astype(np.float16), allow_pickle=True)
    b64str = base64.b64encode(bio.getvalue()).decode("ascii")
    json_dict["lst"][0][1][0]["outputs"][name] = {"values": {"array": b64str}}

with open(output_json_path, "w") as f:
    json.dump(json_dict, f)

print(f"✅ 已保存 FP16 指定中间层输出到 {output_json_path}")
