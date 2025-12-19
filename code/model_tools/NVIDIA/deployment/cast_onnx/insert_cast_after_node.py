#!/usr/bin/env python3
"""
Insert an ONNX Cast-to-FP32 node immediately after a specified node by renaming outputs.
This approach renames the original node's output, then casts back to the original name,
so downstream connections remain unchanged without iterating over all consumers.
Usage:
    python3 insert_cast_after_node.py \
        --model input.onnx \
        --node Unsqueeze_237 \
        --output output.onnx
"""
import argparse
import onnx
from onnx import helper, TensorProto

def insert_cast_after_node(model_path: str, node_name: str, output_path: str):
    # Load the ONNX model
    model = onnx.load(model_path)
    graph = model.graph

    # Find the target node by name
    target_node = next((n for n in graph.node if n.name == node_name), None)
    if target_node is None:
        raise ValueError(f"Node '{node_name}' not found in graph.")

    # Ensure the node has exactly one output
    if len(target_node.output) != 1:
        raise ValueError(f"Node '{node_name}' has {len(target_node.output)} outputs; expected exactly one.")

    # Original output name
    orig_output = target_node.output[0]
    # Rename the original output to an intermediate name
    intermediate = orig_output + "_orig"
    target_node.output[0] = intermediate

    # Create Cast node that casts intermediate back to original name as FP32
    cast_node = helper.make_node(
        op_type='Cast',
        inputs=[intermediate],
        outputs=[orig_output],
        name=node_name + '_CastToF32',
        to=TensorProto.FLOAT
    )

    # Insert Cast node right after the target node
    idx = list(graph.node).index(target_node)
    graph.node.insert(idx + 1, cast_node)

    # Save the modified model
    onnx.save(model, output_path)
    print(f"Inserted Cast after '{node_name}' by renaming output '{orig_output}'->'{intermediate}', saved to '{output_path}'")

def parse_args():
    parser = argparse.ArgumentParser(description="Insert Cast-to-FP32 node after specified ONNX node by output renaming.")
    parser.add_argument('--model', required=True, help='Path to input ONNX model')
    parser.add_argument('--node', required=True, help='Name of node after which to insert Cast')
    parser.add_argument('--output', required=True, help='Path to output ONNX model')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    insert_cast_after_node(args.model, args.node, args.output)
