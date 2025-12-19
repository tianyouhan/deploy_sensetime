import onnx
from onnx import helper, TensorProto


def set_output_dtype(model_path, save_path, node_names, new_dtype=TensorProto.FLOAT):
    model = onnx.load(model_path)
    graph = model.graph

    # 收集所有可能的 ValueInfo：inputs、outputs、value_info
    all_vi = {}

    for vi in graph.value_info:
        all_vi[vi.name] = vi
    for vi in graph.input:
        all_vi[vi.name] = vi
    for vi in graph.output:
        all_vi[vi.name] = vi

    # 建立 node map
    node_map = {n.name: n for n in graph.node}

    for node_name in node_names:
        if node_name not in node_map:
            print(f"[WARN] node {node_name} 不存在")
            continue

        node = node_map[node_name]

        for out_name in node.output:
            if out_name == "":
                continue

            # 该输出是否已经有 value_info 描述
            if out_name not in all_vi:
                # 创建 value_info
                vi = helper.make_tensor_value_info(out_name, new_dtype, None)
                graph.value_info.append(vi)
                print(f"[ADD] 为 {out_name} 新建 value_info，并设置 dtype={new_dtype}")
            else:
                # 直接修改 dtype
                vi = all_vi[out_name]
                vi.type.tensor_type.elem_type = new_dtype
                print(f"[OK] 修改 {out_name} dtype → {new_dtype}")

    onnx.save(model, save_path)
    print(f"[DONE] 保存到 {save_path}")


if __name__ == "__main__":
    set_output_dtype(
        model_path="/mnt/data/hantianyou/road_compare_tool/cast_onnx/jc/lidar_branch_fp16_cast.onnx",
        save_path="/mnt/data/hantianyou/road_compare_tool/cast_onnx/jc/lidar_branch_fp16_cast_1.onnx",
        node_names=["Mul_98"],
        new_dtype=TensorProto.FLOAT,  # FP32
    )