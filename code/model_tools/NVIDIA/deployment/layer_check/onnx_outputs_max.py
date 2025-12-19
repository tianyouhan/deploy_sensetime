import onnx
import onnxruntime as ort
import numpy as np
import os
import json
from datetime import datetime

FP16_MAX = 65504.0
onnx_path = "/mnt/data/hantianyou/road_compare_tool/cast_onnx/jc/lidar-branch.onnx"
input_dir = "/mnt/data/hantianyou/lidar-branch/inputs"
output_json_path = f"onnx_max_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
top_n = 20

# -----------------------------
# 加载 ONNX
# -----------------------------
import onnx
from onnx import shape_inference, numpy_helper

# ===== 先做类型/形状推断 =====
model = onnx.load(onnx_path)
inferred = shape_inference.infer_shapes(model, strict_mode=True)

# 建立 name -> (elem_type, shape) 索引
def make_vi_map(g):
    vi_map = {}
    def put_vi(vi):
        if vi and vi.name and vi.type and vi.type.tensor_type:
            tt = vi.type.tensor_type
            elem = tt.elem_type  # onnx.TensorProto.DataType 枚举
            # 形状可选；我们只关心类型，形状给 None 也行
            vi_map[vi.name] = (elem, None)

    for vi in list(g.value_info) + list(g.input) + list(g.output):
        put_vi(vi)

    # 常量初始值（initializer）也能提供类型
    for init in g.initializer:
        vi_map[init.name] = (init.data_type, None)

    return vi_map

vi_map = make_vi_map(inferred.graph)

# 已有的 graph outputs 名集合，避免重复添加
existing_outs = {o.name for o in model.graph.output}

# ===== 逐节点输出，按真实类型追加为 graph output =====
added = 0
for node in model.graph.node:
    for out in node.output:
        if out in existing_outs:
            continue
        # 从推断/已有信息里取真实类型；取不到就跳过（或按需默认 FLOAT）
        dtype_shape = vi_map.get(out, None)
        if dtype_shape is None:
            # 尝试已知特殊算子兜底（可选）
            if node.op_type in ("Shape", "ArgMax", "ArgMin", "NonZero", "TopK"):
                elem_type = onnx.TensorProto.INT64
            elif node.op_type in ("Greater", "Less", "Equal", "And", "Or", "Not"):
                elem_type = onnx.TensorProto.BOOL
            else:
                # 如果你确实想“全都拿到”，可以勉强默认 FLOAT，但**可能再次触发类型不匹配**
                # elem_type = onnx.TensorProto.FLOAT
                # 为了稳妥，这里选择跳过，或者记录一下名字日志
                # print(f"Skip add output (unknown dtype): {out} from {node.op_type}")
                continue
            shape = None
        else:
            elem_type, shape = dtype_shape

        model.graph.output.append(
            onnx.helper.make_tensor_value_info(out, elem_type, shape)  # shape 可以 None
        )
        existing_outs.add(out)
        added += 1

print(f"Added {added} intermediate outputs with correct dtypes.")

tmp_onnx_path = "tmp_all_outputs.onnx"
onnx.save(model, tmp_onnx_path)
print("✅ Done saving tmp ONNX with correct dtypes.")

# -----------------------------
# 创建 ORT session
# -----------------------------
sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3
sess = ort.InferenceSession(tmp_onnx_path, sess_options)

# 准备输入
# inputs = {}
# for fname in os.listdir(input_dir):
#     if fname.endswith(".npy"):
#         key = fname.replace(".npy", "")
#         arr = np.load(os.path.join(input_dir, fname)).astype(np.float32)
#         # if arr.ndim == 2:
#         #     arr = np.expand_dims(arr, axis=0)
#         inputs[key] = arr
inputs = {}

for subdir in os.listdir(input_dir):
    subdir_path = os.path.join(input_dir, subdir)
    if not os.path.isdir(subdir_path):
        continue

    # 取 0.npy，若不存在，取最小编号的帧
    target_npy = os.path.join(subdir_path, "0.npy")

    if not os.path.exists(target_npy):
        # 找最小编号 .npy
        npy_files = sorted([f for f in os.listdir(subdir_path) if f.endswith(".npy")])
        if len(npy_files) == 0:
            print(f"⚠️ 跳过 {subdir}（无 .npy 文件）")
            continue
        target_npy = os.path.join(subdir_path, npy_files[0])

    arr = np.load(target_npy).astype(np.float32)
    inputs[subdir] = arr

    print(f"Loaded {subdir}: {os.path.basename(target_npy)}, shape={arr.shape}")

# -----------------------------
# 遍历输出并计算最大绝对值
# -----------------------------
max_abs_list = []
output_names = [o.name for o in sess.get_outputs()]
total_outputs = len(output_names)

for idx, out_name in enumerate(output_names, 1):
    try:
        out = sess.run([out_name], inputs)[0]
        max_abs = float(np.max(np.abs(out)))
        max_abs_list.append((out_name, max_abs))
        if idx % 10 == 0 or idx == total_outputs:
            print(f"[{idx}/{total_outputs}] processed output: {out_name}, max_abs={max_abs:.6f}")
    except Exception as e:
        print(f"⚠️ 节点 {out_name} 推理失败: {e}")

# 排序 top N
max_abs_list.sort(key=lambda x: x[1], reverse=True)
print(f"\n🔝 Top-{top_n} 最大绝对值输出节点:")
for rank, (name, val) in enumerate(max_abs_list[:top_n], 1):
    overflow_flag = " ⚠️ FP16 溢出" if val > FP16_MAX else ""
    print(f"{rank:>2}. {name:<60} max_abs={val:.6f}{overflow_flag}")

# -----------------------------
# 保存 JSON
# -----------------------------
json_data = []
for idx, (name, val) in enumerate(max_abs_list, 1):
    if idx % 50 == 0 or idx == total_outputs:
        print(f"[{idx}/{total_outputs}] preparing JSON entry for: {name}")
    json_data.append({"name": name, "max_abs": val})

with open(output_json_path, "w") as f:
    json.dump(json_data, f, indent=2)

print(f"✅ 已保存到 {output_json_path}")
