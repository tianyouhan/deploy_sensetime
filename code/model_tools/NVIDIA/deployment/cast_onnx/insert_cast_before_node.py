import onnx
from onnx import helper, TensorProto


def insert_cast_before_input(model_path, save_path, rules):
    """
    在指定节点的指定输入前插入 Cast 到 float32
    rules 是 list, 每个元素是:
        {
            "node_name": "Mul_123",
            "input_index": 0,   # 对第 0 个输入插入
            "to": "float32"
        }
    """
    model = onnx.load(model_path)
    graph = model.graph

    node_map = {n.name: n for n in graph.node}

    for r in rules:
        node_name = r["node_name"]
        input_idx  = r["input_index"]
        to_dtype   = r.get("to", "float32")

        if node_name not in node_map:
            print(f"[WARN] node {node_name} 不存在")
            continue

        node = node_map[node_name]

        if input_idx >= len(node.input):
            print(f"[WARN] node {node_name} 没有 input[{input_idx}]")
            continue

        original_input = node.input[input_idx]

        cast_output = original_input + f"_cast_fp32_for_{node_name}"

        # 创建 Cast 节点
        cast_node = helper.make_node(
            "Cast",
            inputs=[original_input],
            outputs=[cast_output],
            name=f"{node_name}_input{input_idx}_CastFP32",
            to=TensorProto.FLOAT
        )

        # 替换该输入
        node.input[input_idx] = cast_output

        # 插入 cast node
        graph.node.append(cast_node)

        print(f"[OK] 在 {node_name}.input[{input_idx}] 前插入 Cast → FP32")

    onnx.save(model, save_path)
    print(f"[DONE] 保存到 {save_path}")


if __name__ == "__main__":
    insert_cast_before_input(
        model_path="/mnt/data/hantianyou/road_compare_tool/cast_onnx/jc/lidar_branch_fp16.onnx",
        save_path="/mnt/data/hantianyou/road_compare_tool/cast_onnx/jc/lidar_branch_fp16_cast.onnx",
        rules=[
            {"node_name": "Mul_98", "input_index": 0},
            {"node_name": "Mul_98", "input_index": 1},
        ]
    )
