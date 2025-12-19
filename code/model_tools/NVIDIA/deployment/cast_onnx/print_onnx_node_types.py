import onnx
from onnx import numpy_helper, TensorProto

def get_initializer_dtype(initializers, name):
    for init in initializers:
        if init.name == name:
            return TensorProto.DataType.Name(init.data_type)
    return None

def main(path):
    model = onnx.load(path)
    graph = model.graph

    print(f"Loaded model: {path}")
    print(f"Total nodes: {len(graph.node)}\n")

    # build dtype map from value_info
    dtype_map = {}

    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.type.tensor_type.elem_type != 0:  # NON-UNDEFINED
            dtype_map[vi.name] = TensorProto.DataType.Name(
                vi.type.tensor_type.elem_type
            )

    # mapping initializers dtype
    init_map = {init.name: TensorProto.DataType.Name(init.data_type) for init in graph.initializer}

    for i, node in enumerate(graph.node):
        print(f"Node {i}: {node.name}   op={node.op_type}")

        # inputs
        for inp in node.input:
            dt = dtype_map.get(inp, init_map.get(inp, "Unknown"))
            print(f"    input:  {inp:40s} dtype={dt}")

        # outputs
        for out in node.output:
            dt = dtype_map.get(out, "Unknown")
            print(f"    output: {out:40s} dtype={dt}")

        print("-" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print ONNX node type and tensor dtype")
    parser.add_argument("--path", required=True, help="Path to ONNX model")
    args = parser.parse_args()

    main(args.path)
