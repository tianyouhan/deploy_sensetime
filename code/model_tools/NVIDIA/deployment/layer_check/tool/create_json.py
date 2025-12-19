#!/usr/bin/env python3
# usage:
#   python bins_from_onnx_inputs_to_json.py model.onnx /path/to/bin_dir --out merged.json \
#       --dim N=1 H=224 W=224  # 为符号/未知维度赋值
#   python bins_from_onnx_inputs_to_json.py model.onnx /path/to/bin_dir --out merged.npy \
#       --dim N=1 H=224 W=224  # 输出npy格式（推荐用于modelopt）
import argparse, json
from pathlib import Path
import numpy as np
import onnx
from onnx import shape_inference

# 映射 ONNX elem_type -> numpy dtype
ONNX_TO_NP = {
    1: np.float32,   # FLOAT
    2: np.uint8,
    3: np.int8,
    4: np.uint16,
    5: np.int16,
    6: np.int32,
    7: np.int64,
    9: np.bool_,
    10: np.float16,  # FLOAT16
    11: np.double,   # FLOAT64
    12: np.uint32,
    13: np.uint64,
}

def parse_dim_overrides(pairs):
    # e.g. ["N=1","H=224","W=224"] -> {"N":1,"H":224,"W":224}
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = int(v)
    return out

def get_input_specs(model, dim_overrides):
    # 优先做一次 shape 推断，补全可能缺失的信息
    try:
        model = shape_inference.infer_shapes(model, strict_mode=False, data_prop=False)
    except Exception:
        pass  # 推断失败也继续用原模型 [6][5]
    specs = []
    for inp in model.graph.input:
        ttype = inp.type.tensor_type
        elem_type = ttype.elem_type
        np_dtype = ONNX_TO_NP.get(elem_type)
        if np_dtype is None:
            raise ValueError(f"Unsupported ONNX elem_type {elem_type} for input {inp.name}")
        shape = []
        if ttype.HasField("shape"):
            for d in ttype.shape.dim:
                if d.HasField("dim_value"):
                    shape.append(int(d.dim_value))
                elif d.HasField("dim_param"):
                    sym = d.dim_param
                    if sym in dim_overrides:
                        shape.append(int(dim_overrides[sym]))
                    else:
                        raise ValueError(f"Input {inp.name} has symbolic dim '{sym}' without override. Use --dim {sym}=<value>")
                else:
                    raise ValueError(f"Input {inp.name} has unknown dimension; provide override via --dim")
        else:
            raise ValueError(f"Input {inp.name} has unknown rank; cannot reconstruct")
        specs.append({"name": inp.name, "dtype": np_dtype, "shape": tuple(shape), "onnx_elem_type": elem_type})
    return specs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx_model", type=Path)
    ap.add_argument("bin_dir", type=Path)
    ap.add_argument("--out", type=Path, default=Path("merged.json"))
    ap.add_argument("--dim", nargs="*", help="Override dynamic/unknown dims, e.g. N=1 H=224 W=224")
    args = ap.parse_args()

    dim_overrides = parse_dim_overrides(args.dim)
    model = onnx.load(str(args.onnx_model))  # 载入 ONNX 模型 [3]
    specs = get_input_specs(model, dim_overrides)  # 读取输入 dtype/shape [1][6]

    # 读取所有输入数据
    inputs_data = {}
    for spec in specs:
        bin_path = args.bin_dir / f"{spec['name']}/1.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f"Missing bin for input '{spec['name']}': {bin_path}")
        # 读取原始数据并重构形状 [11][12]
        arr = np.fromfile(bin_path, dtype=spec["dtype"])
        need_elems = int(np.prod(spec["shape"]))
        if arr.size != need_elems:
            raise ValueError(f"Size mismatch for {spec['name']}: file has {arr.size} elements, "
                             f"but shape {spec['shape']} needs {need_elems}")
        arr = arr.reshape(spec["shape"])
        inputs_data[spec["name"]] = arr
    
    # 根据输出文件扩展名选择输出格式
    if args.out.suffix.lower() == '.npy':
        # 输出为npy格式（推荐用于modelopt）
        # modelopt期望的是字典格式，键为输入名称，值为numpy数组
        np.save(args.out, inputs_data)
        print(f"Wrote {args.out} with {len(inputs_data)} inputs in .npy format.")
        print("Note: This format is recommended for modelopt calibration.")
    elif args.out.suffix.lower() == '.npz':
        # 输出为npz格式（推荐用于modelopt）
        # modelopt期望的是字典格式，键为输入名称，值为numpy数组
        np.savez(args.out, **inputs_data)
        print(f"Wrote {args.out} with {len(inputs_data)} inputs in .npz format.")
        print("Note: This format is recommended for modelopt calibration.")
    elif args.out.suffix.lower() == '.json':
        # 输出为JSON格式（主要输出格式）
        # 使用Polygraphy的save_json确保格式完全正确
        try:
            from polygraphy.json import save_json
            # Polygraphy期望的格式：List[Dict[str, np.ndarray]]，外层需要列表包装
            json_data = [inputs_data]  # 包装在列表中
            save_json(json_data, str(args.out), description="calibration data")
            print(f"Wrote {args.out} with {len(inputs_data)} inputs in JSON format using Polygraphy.")
            print("Note: JSON format follows Polygraphy convention: List[Dict[str, np.ndarray]]")
        except ImportError:
            # 如果Polygraphy不可用，回退到标准JSON（可能不完全兼容）
            print("Warning: Polygraphy not available, using standard JSON (may not be fully compatible)")
            json_data = [inputs_data]  # 包装在列表中
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            print(f"Wrote {args.out} with {len(inputs_data)} inputs in JSON format.")
    else:
        # 默认输出为JSON格式
        default_out = args.out.with_suffix('.json')
        try:
            from polygraphy.json import save_json
            json_data = [inputs_data]  # 包装在列表中
            save_json(json_data, str(default_out), description="calibration data")
            print(f"Unknown output format, wrote {default_out} with {len(inputs_data)} inputs in JSON format using Polygraphy.")
        except ImportError:
            json_data = [inputs_data]  # 包装在列表中
            with open(default_out, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
            print(f"Unknown output format, wrote {default_out} with {len(inputs_data)} inputs in JSON format.")

if __name__ == "__main__":
    main()
