import onnx
from onnx import helper
from collections import deque
import argparse


def build_output_to_node(model):
    mp = {}
    for node in model.graph.node:
        for out in node.output:
            mp[out] = node
    return mp


def find_upstream_nodes(model, input_names, output_names):
    """
    output_names <- ... <- input_names
    反向追踪依赖节点
    """
    output_to_node = build_output_to_node(model)

    required = {}  # key = tensor_name, value = node
    queue = deque(output_names)
    visited_tensors = set()

    while queue:
        tensor = queue.popleft()
        if tensor in visited_tensors:
            continue
        visited_tensors.add(tensor)

        node = output_to_node.get(tensor, None)
        if node is None:
            continue

        # 以 node 第一个输出作为 key（node.output 是唯一标识）
        required[node.output[0]] = node

        # 继续向上追输入
        for inp in node.input:
            if inp not in input_names:  # 输入节点不再继续回溯
                queue.append(inp)

    return required


def extract_subgraph(model, input_names, output_names, out_path):
    required_nodes = find_upstream_nodes(model, input_names, output_names)

    keep_nodes = list(required_nodes.values())

    # 找到所有涉及的张量
    used_tensors = set()
    for node in keep_nodes:
        used_tensors.update(node.input)
        used_tensors.update(node.output)

    # subgraph inputs
    sg_inputs = []
    for inp in model.graph.input:
        if inp.name in used_tensors or inp.name in input_names:
            sg_inputs.append(inp)

    # subgraph outputs
    sg_outputs = []
    for out in model.graph.output:
        if out.name in output_names:
            sg_outputs.append(out)

    # initializer
    sg_inits = []
    for init in model.graph.initializer:
        if init.name in used_tensors:
            sg_inits.append(init)

    # 保留节点顺序
    sg_nodes = [n for n in model.graph.node if n in keep_nodes]

    new_graph = helper.make_graph(
        sg_nodes,
        "subgraph",
        sg_inputs,
        sg_outputs,
        sg_inits,
    )

    new_model = helper.make_model(new_graph)
    onnx.save(new_model, out_path)

    print(f"Saved to {out_path}, nodes kept = {len(sg_nodes)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--opath", required=True)
    args = parser.parse_args()

    model = onnx.load(args.path)
    inputs = [x.strip() for x in args.inputs.split(",")]
    outputs = [x.strip() for x in args.outputs.split(",")]

    extract_subgraph(model, inputs, outputs, args.opath)


if __name__ == "__main__":
    main()
