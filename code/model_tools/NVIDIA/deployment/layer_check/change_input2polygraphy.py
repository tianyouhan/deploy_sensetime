# import numpy as np
# import base64
# import io
# import json

# def encode_array(arr):
#     buffer = io.BytesIO()
#     np.save(buffer, arr, allow_pickle=False)
#     buffer.seek(0)
#     return base64.b64encode(buffer.read()).decode("utf-8")

# # 路径
# tensor_dir = "/mnt/data/hantianyou/save_tensors/far3d_head_align_bin_in"
# out_json = "/mnt/data/hantianyou/road_compare_tool/layer_check/save_tensors_inputs_list.json"

# # 定义输入
# inputs = {
#     "local_map": {
#         "file": f"{tensor_dir}/local_maps.bin",
#         "shape": (1, 32, 40, 3),
#         "dtype": np.float32,
#     },
#     "agent_trajs": {
#         "file": f"{tensor_dir}/agent_traj.bin",
#         "shape": (1, 32, 20, 5),
#         "dtype": np.float32,
#     },
# }

# sample = {}

# for name, meta in inputs.items():
#     arr = np.fromfile(meta["file"], dtype=meta["dtype"]).reshape(meta["shape"])
#     sample[name] = {
#         "array": encode_array(arr),
#         "polygraphy_class": "ndarray",
#     }

# # 保存
# with open(out_json, "w") as f:
#     json.dump([sample], f)

# print(f"✅ 已保存到 {out_json}")

import numpy as np
import base64
import io
import json
import os

def encode_array(arr):
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

# 输入目录
tensor_dir = "/mnt/data/hantianyou/far3d_head_align_npy"
# 输出 JSON
out_json = "/mnt/data/hantianyou/road_compare_tool/layer_check/save_tensors_inputs_list_pts.json"

sample = {}

for fname in os.listdir(tensor_dir):
    if fname.endswith(".npy"):
        name = os.path.splitext(fname)[0]  # 去掉扩展名作为 key
        fpath = os.path.join(tensor_dir, fname)
        arr = np.load(fpath, allow_pickle=False)
        sample[name] = {
            "array": encode_array(arr),
            "polygraphy_class": "ndarray",
        }

# 保存
with open(out_json, "w") as f:
    json.dump([sample], f)

print(f"✅ 已保存到 {out_json}，共 {len(sample)} 个输入")
