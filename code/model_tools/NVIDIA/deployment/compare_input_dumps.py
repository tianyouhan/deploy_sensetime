import os 
import json
import argparse
import torch
import numpy as np
import onnxruntime as ort
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
#gop
def load_gop_data_camera_branch(folder_path="/mnt/data/hantianyou/gop_dump_data_v1_11_2/dump_data_0717/cam-branch/inputs"):
    names = [
        'gridsample_indexes_0', 'gridsample_indexes_1','center_camera_fov120', 'center_camera_fov30','ref_points_valid_num','gridsample_ref_points'
    ]
    
    input_data = {}
    for name in names:
        save_dir = os.path.join(folder_path, name)
        path = os.path.join(save_dir, f"0.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {path}")
        input_data[name] = np.load(path)
    
    return input_data

def load_gop_data_fusion_head(folder_path="/mnt/data/hantianyou/gop_dump_data_v1_11_2/dump_data_0717/fuser-fusion-head/inputs"):
    names = [
        'spatial_features_2d_cam', 'spatial_features_2d_lidar'
    ]
    
    input_data = {}
    for name in names:
        save_dir = os.path.join(folder_path, name)
        path = os.path.join(save_dir, f"0.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {path}")
        input_data[name] = np.load(path)
    
    return input_data

def load_gop_data_lidar_branch(folder_path="/mnt/data/hantianyou/gop_dump_data_v1_11_2/dump_data_0717/lidar-branch/inputs"):
    names = [
        'vfe_input', 'voxel_coords'
    ]
    
    input_data = {}
    for name in names:
        save_dir = os.path.join(folder_path, name)
        path = os.path.join(save_dir, f"0.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {path}")
        input_data[name] = np.load(path)
    
    return input_data

def load_bin_with_shape(bin_path: str, shape):
    """
    从 .bin 文件加载 float32 数据并 reshape
    """
    data = np.fromfile(bin_path, dtype=np.float32)
    return data.reshape(shape)

def load_bin_and_shape(bin_path: str):
    return np.fromfile(bin_path, dtype=np.float32)

def load_tensor_or_bin(path):
    if path.endswith('.npy'):
        return np.load(path).astype(np.float32)

    elif path.endswith('.bin'):
        if os.path.exists(path):
            return load_bin_and_shape(path)
        else:
            raise FileNotFoundError(f"Bin file not found: {path}")

    elif path.endswith('.pt'):
        data = torch.load(path, map_location='cpu')
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy().astype(np.float32)
        elif isinstance(data, (list, tuple)):
            flat = []
            for item in data:
                if isinstance(item, torch.Tensor):
                    flat.append(item.detach().cpu().numpy().flatten())
                elif isinstance(item, dict):
                    for v in item.values():
                        if isinstance(v, torch.Tensor):
                            flat.append(v.detach().cpu().numpy().flatten())
            return np.concatenate(flat).astype(np.float32)
        elif isinstance(data, dict):
            flat = []
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    flat.append(v.detach().cpu().numpy().flatten())
            return np.concatenate(flat).astype(np.float32)
        else:
            raise TypeError(f"Unsupported data type in .pt file: {type(data)}")

    else:
        raise ValueError(f"Unsupported file format: {path}")

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()

    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    eps = 1e-8
    if a_norm < eps or b_norm < eps:
        print(f"[⚠️] Norm too small → a_norm={a_norm:.2e}, b_norm={b_norm:.2e}")
        return 0.0

    cosine = float(np.dot(a, b) / (a_norm * b_norm + eps))

    if np.isnan(cosine):
        print(f"[❌] Got NaN cosine! a.max={a.max():.4f}, min={a.min():.4f}, b.max={b.max():.4f}, min={b.min():.4f}")
        return 0.0

    return cosine

# def load_input_data_fsd_pts_backbone(folder_path="/mnt/data/hantianyou/bin"):
#     """
#     加载 backbone 输入（支持直接指定 bin 文件 shape）
#     """
#     names = ['vfe_input', 'voxel_coords']
#     input_data = {}

#     # 手动写死每个输入的 shape
#     shapes = {
#         'vfe_input': (10295, 9,32,1),   # 这里换成你自己的 shape
#         'voxel_coords': (10295,2) # 这里换成你自己的 shape
#     }

#     input_data = {}
#     for name in names:
#         bin_path = os.path.join(folder_path, name, "0.bin")
#         if not os.path.exists(bin_path):
#             raise FileNotFoundError(f"Missing file: {bin_path}")
#         input_data[name] = load_bin_with_shape(bin_path, shapes[name])

#     return input_data

def load_input_data():
    feats = load_tensor_or_bin("/mnt/data/hantianyou/jiaming_road/dump_datas/laneline_3d/bin/img_feats_2/0.bin")
    reference_points_cam = load_tensor_or_bin("/mnt/data/hantianyou/jiaming_road/dump_datas/laneline_3d/bin/reference_points_cam/0.bin")
    # bev_embed_in = load_tensor_or_bin("/mnt/data/hantianyou/bev_neck/pt/bev_embed_in.pt")
    return {
        "img_feats_2": feats,
        "reference_points_cam": reference_points_cam 
        # "lidar2img": lidar2img
    }

    # bev_embed = load_tensor_or_bin("/mnt/data/hantianyou/dump_datas/bev_occ/pt/bev_embed.pt")
    # return {
    #     "bev_embed_in": bev_embed_in
    # }

def load_input_data_fsd_pts_backbone(folder_path="/mnt/data/hantianyou/pts_backbone_align_npy"):
    names = [
        'vfe_input','voxel_coords'
    ]
    
    input_data = {}
    for name in names:
        path = os.path.join(folder_path, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {path}")
        input_data[name] = np.load(path)
    
    return input_data

def load_input_data_fsd_head_npy(folder_path="/mnt/data/hantianyou/wzw/far3d_head_align_npy"):
    names = [
        'img_feats_0', 'img_feats_1', 'img_feats_2', 'img_feats_3','intrinsics', 'extrinsics', 'lidar2img',
        'memory_timestamp', 'memory_egopose',
        'memory_embedding_0', 'memory_embedding_1', 'memory_embedding_2', 'memory_embedding_3',
        'memory_reference_point_0', 'memory_reference_point_1',
        'memory_reference_point_2', 'memory_reference_point_3',
        'memory_velo_0', 'memory_velo_1', 'memory_velo_2', 'memory_velo_3',
        'query_feats','query_xyz','query_pred'
        # ,'voxel_feats'
    ]
    
    input_data = {}
    for name in names:
        path = os.path.join(folder_path, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {path}")
        input_data[name] = np.load(path)
    
    return input_data

def load_input_data_gop_head_npy(folder_path="/mnt/data/hantianyou/fsdv2_gop/save_tensors/gop_head_npy"):
    names = [
        'rpn_cls_preds', 'rpn_box_preds','rpn_dir_cls_preds','spatial_features_2d_pvb','spatial_features_2d_lidar'
    ]
    
    input_data = {}
    for name in names:
        path = os.path.join(folder_path, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {path}")
        input_data[name] = np.load(path)
    
    return input_data

def load_input_data_camera_branch_npy(folder_path="/mnt/data/hantianyou/fsdv2_gop/save_tensors/gop_cam_branch_npy"):
    names = [
        'gridsample_indexes_0', 'gridsample_indexes_1','gridsample_input', 'ref_points_valid_num','gridsample_ref_points'
    ]
    
    input_data = {}
    for name in names:
        path = os.path.join(folder_path, f"{name}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required input file: {path}")
        input_data[name] = np.load(path)
    
    return input_data

def load_reference_outputs():
    keys = ['spatial_features_2d_lidar']
    names = ['spatial_features_2d_lidar']
    # keys = ['det_pred_dicts_fusion_gop_driving_cls', 'det_pred_dicts_fusion_gop_driving_box','det_pred_dicts_fusion_gop_driving_dir_cls']
    # names = ['det_pred_dicts_fusion_gop_driving_cls', 'det_pred_dicts_fusion_gop_driving_box','det_pred_dicts_fusion_gop_driving_dir_cls']
    # keys = ['score', 'bbox','embedding']
    # names = ['score', 'bbox','embedding']
    # keys = ['final_box_dicts','rpn_cls_preds', 'rpn_box_preds','rpn_dir_cls_preds','spatial_features_2d_pvb','spatial_features_2d_lidar']
    # names = ['final_box_dicts','rpn_cls_preds', 'rpn_box_preds','rpn_dir_cls_preds','spatial_features_2d_pvb','spatial_features_2d_lidar']
    data = {}
    for key, name in zip(keys, names):
        path = f"/mnt/data/hantianyou/gop_dump_data_v1_11_2/dump_data_0717/lidar-branch/outputs/{key}/0.npy"
        # path = f"/mnt/data/hantianyou/pts_backbone_align_npy/{key}.npy"
        data[name] = load_tensor_or_bin(path)
    return data

# def load_reference_outputs(folder_path="/mnt/data/hantianyou/bin"):
#     """
#     加载参考输出，可以同时支持 .npy 和 .bin 文件；
#     若使用 .bin，则在代码中直接指定 shape。
#     """
#     keys = ['rpn_cls_preds', 'rpn_box_preds', 'rpn_dir_cls_preds', 'spatial_features_2d_pvb', 'spatial_features_2d_lidar']
#     names = ['rpn_cls_preds', 'rpn_box_preds', 'rpn_dir_cls_preds', 'spatial_features_2d_pvb', 'spatial_features_2d_lidar']

#     # ✅ 在这里写死每个 tensor 的 shape
#     shapes = {
#         'rpn_cls_preds': (1,1,573440,7),
#         'rpn_box_preds': (1,1,573440,7),
#         'rpn_dir_cls_preds': (1,1,573440,2),
#         'spatial_features_2d_pvb': (1,128,320,96),
#         'spatial_features_2d_lidar': (1, 96, 128, 320)
#     }

#     data = {}

#     for key, name in zip(keys, names):
#         # 目录结构：folder_path/key/0.bin
#         bin_path = os.path.join(folder_path, key, "0.bin")

#         if not os.path.exists(bin_path):
#             raise FileNotFoundError(f"Missing file: {bin_path}")
#         if name not in shapes:
#             raise ValueError(f"Shape for {name} not specified.")

#         data[name] = load_bin_with_shape(bin_path, shapes[name])

#     return data

def run_trt_inference(input_map):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, "")
    trt_runtime = trt.Runtime(TRT_LOGGER)

    engine_path = "/mnt/data/hantianyou/road_compare_tool/cast_onnx/lidar-branch-sim_fp16.trt"
    with open(engine_path, 'rb') as f:
        engine = trt_runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    stream = cuda.Stream()

    # 你关心的最终输出名字（确保和 engine 输出 tensor 名称一致）
    wanted_outputs = ['spatial_features_2d_lidar']
    # wanted_outputs =['final_box_dicts','rpn_cls_preds', 'rpn_box_preds','rpn_dir_cls_preds','spatial_features_2d_pvb','spatial_features_2d_lidar']
    # ==== 1) 如果有动态输入，设置 shape ====
    # 1) 先设置所有动态输入的 shape（并保证补上 batch dim）
    for tensor_name in engine:
        if engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
            if tensor_name in input_map:
                arr = input_map[tensor_name]
                # 目标：拿到 engine/context 期望的维度数量 (nbDims)
                try:
                    # context.get_tensor_shape 在未 set 时可能返回 dims with -1,
                    # 但它的 rank (len) 仍然是 engine 要求的 nbDims
                    expected_dims = len(context.get_tensor_shape(tensor_name))
                except Exception:
                    # fallback：如果 context.get_tensor_shape 抛异常，尝试 engine 层面的获取
                    expected_dims = len(engine.get_tensor_shape(tensor_name))

                # 如果输入缺少 batch 维（常见），在前面补 1
                if arr.ndim == expected_dims - 1:
                    arr = np.expand_dims(arr, 0)
                    print(f"[AUTO] Prepend batch dim for {tensor_name}: new shape {arr.shape}")
                    # 更新 input_map so downstream copy uses the corrected arr
                    input_map[tensor_name] = arr
                elif arr.ndim != expected_dims:
                    # 严格检查：如果仍然不匹配，报错提示
                    raise RuntimeError(f"[ERROR] Input '{tensor_name}' ndim {arr.ndim} != expected {expected_dims}. "
                                       f"Shape: {arr.shape}. You may need to reshape/pad the input.")

                # set input shape (tuple of ints)
                shp = tuple(int(x) for x in arr.shape)
                try:
                    context.set_input_shape(tensor_name, shp)
                    print(f"[INFO] Set dynamic shape for {tensor_name}: {shp}")
                except Exception as e:
                    # 打印更详细的错误，便于诊断
                    print(f"[ERROR] Failed to set_input_shape for {tensor_name} -> {shp}: {e}")
            else:
                print(f"[WARN] Input {tensor_name} not provided in input_map")

    # 2) 再按 engine 的 IO 顺序分配 buffers（一定要用 context.get_tensor_shape()）
    bindings = []
    inputs = []
    outputs = []
    all_io_names = []

    for tensor_name in engine:
        all_io_names.append(tensor_name)
        mode = engine.get_tensor_mode(tensor_name)
        dtype = trt.nptype(engine.get_tensor_dtype(tensor_name))

        # ✅ 关键：使用 context.get_tensor_shape()，这是已解析的实际 shape（包含 batch）
        try:
            shape_dims = context.get_tensor_shape(tensor_name)
            # shape_dims 可能包含 non-positive values for unknown dims, 替换为 1
            shape = tuple(int(d) if int(d) > 0 else 1 for d in shape_dims)
        except Exception:
            # fallback（一般不会走到）
            shape = (1,)

        size = int(np.prod(shape))

        # 分配 host/device
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))

        if mode == trt.TensorIOMode.INPUT:
            inputs.append({
                "name": tensor_name,
                "host": host_mem,
                "device": device_mem,
                "shape": shape,
                "dtype": dtype
            })
        else:
            outputs.append({
                "name": tensor_name,
                "host": host_mem,
                "device": device_mem,
                "shape": shape,
                "dtype": dtype
            })

    print(f"[INFO] Engine IO names (binding order): {all_io_names}")

    # ==== 3) 拷贝输入数据到 host/device ====
    for inp in inputs:
        name = inp["name"]
        if name not in input_map:
            raise AssertionError(f"[ERROR] Missing input: {name}")
        data = input_map[name].astype(inp["dtype"])
        # 注意 flatten 到 host buffer
        np.copyto(inp["host"], data.flatten())
        cuda.memcpy_htod_async(inp["device"], inp["host"], stream)

    # ==== 4) 执行推理（execute_v2 要求 bindings 列表长度等于 engine io 数） ====
    context.execute_v2(bindings)

    # ==== 5) 拷贝所有 outputs 回主机并 reshape ====
    trt_outputs = {}
    for out in outputs:
        cuda.memcpy_dtoh_async(out["host"], out["device"], stream)
    stream.synchronize()

    for out in outputs:
        trt_outputs[out["name"]] = out["host"].reshape(out["shape"])

    # 如果你只关心部分输出，可以从 trt_outputs 中筛选
    filtered = {k: v for k, v in trt_outputs.items() if k in wanted_outputs}
    # 如果 wanted_outputs 名单里的名字未命中，打印提示
    for w in wanted_outputs:
        if w not in trt_outputs:
            print(f"[WARN] wanted output '{w}' not in engine outputs")

    return trt_outputs


def run_onnx_inference(input_map):
    sess = ort.InferenceSession("/mnt/data/hantianyou/road_compare_tool/cast_onnx/cam-branch-sim_fp16_y.onnx", providers=['CUDAExecutionProvider'])
    input_feed = {}

    for inp in sess.get_inputs():
        name = inp.name
        if name not in input_map:
            raise KeyError(f"[ERROR] ONNX expects input '{name}', but it's missing in input_map.")
        arr = input_map[name]

        # 自动转换 dtype
        if arr.dtype != np.float32:
            print(f"[INFO] Auto casting input '{name}' from {arr.dtype} to float32")
            arr = arr.astype(np.float32)

        # 自动补维到期望 rank
        expected_rank = len([d for d in inp.shape if d != 'None' and d != 'dynamic'])
        actual_rank = len(arr.shape)
        if actual_rank < expected_rank:
            # 在前面补 1
            new_shape = (1,) * (expected_rank - actual_rank) + arr.shape
            arr = arr.reshape(new_shape)
            print(f"[AUTO] Expanded input '{name}' shape from {actual_rank}D {input_map[name].shape} to {expected_rank}D {arr.shape}")
        elif actual_rank > expected_rank:
            # 自动 squeeze
            arr = arr.reshape(arr.shape[:expected_rank])
            print(f"[AUTO] Squeezed input '{name}' shape from {actual_rank}D to {expected_rank}D {arr.shape}")

        input_feed[name] = arr

    output_names = [out.name for out in sess.get_outputs()]
    outputs = sess.run(output_names, input_feed)
    return dict(zip(output_names, outputs))


def compare_outputs(name: str, ref: np.ndarray, pred: np.ndarray, verbose=True):
    if ref.shape != pred.shape:
        print(f"{name:<12}: [WARN] Shape mismatch: Ref {ref.shape} vs Pred {pred.shape}")
        min_len = min(ref.size, pred.size)
        ref = ref.flatten()[:min_len]
        pred = pred.flatten()[:min_len]
    else:
        ref = ref.flatten()
        pred = pred.flatten()

    sim = cosine_similarity(ref, pred)
    mse = ((ref - pred) ** 2).mean()

    print(f"{name:<12}: Cosine similarity = {sim:.6f}, MSE = {mse:.6e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["trt", "onnx", "both"], default="both")
    args = parser.parse_args()

    input_map = load_gop_data_lidar_branch()
    ref_outputs = load_reference_outputs()
    if args.backend in ("trt", "both"):
        print("\n[TensorRT Cosine Similarity]")
        trt_out = run_trt_inference(input_map)
        for key, ref in ref_outputs.items():
            if key in trt_out:
                compare_outputs(key, ref, trt_out[key])
            else:
                print(f"{key:<12}: [ERROR] Not found in TRT output")

    if args.backend in ("onnx", "both"):
        print("\n[ONNX Runtime Cosine Similarity]")
        onnx_out = run_onnx_inference(input_map)
        for key, ref in ref_outputs.items():
            if key in onnx_out:
                compare_outputs(key, ref, onnx_out[key])
            else:
                print(f"{key:<12}: [ERROR] Not found in ONNX output")


if __name__ == "__main__":
    main()
