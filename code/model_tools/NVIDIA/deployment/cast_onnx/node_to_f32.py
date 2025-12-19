#!/usr/bin/env python3
"""
force_tensor_to_fp32.py

把 ONNX 中指定 tensor 名称（例如: Mul_388_output_cast0）从 FP16 改为 FP32。

用法:
python force_tensor_to_fp32.py --onnx input.onnx --tensor Mul_388_output_cast0 --out output.onnx
"""
import onnx
from onnx import helper, TensorProto, shape_inference, numpy_helper
import argparse
import os
import sys
import uuid
import copy
import numpy as np

def log(msg):
    print(msg)

def find_initializer(graph, name):
    for init in graph.initializer:
        if init.name == name:
            return init
    return None

def find_value_info(graph, name):
    for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
        if vi.name == name:
            return vi
    return None

def find_producers(graph, tensor_name):
    producers = []
    for node in graph.node:
        if tensor_name in node.output:
            producers.append(node)
    return producers

def find_consumers(graph, tensor_name):
    consumers = []
    for node in graph.node:
        for inp in node.input:
            if inp == tensor_name:
                consumers.append(node)
                break
    return consumers

def replace_consumers_inputs(graph, old_name, new_name):
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == old_name:
                node.input[i] = new_name

def insert_node_after(graph, index, node_to_insert):
    # insert after index (0-based)
    graph.node.insert(index + 1, node_to_insert)

def make_cast_node(input_name, output_name, to_dtype):
    node_name = f"Cast_{uuid.uuid4().hex[:8]}"
    node = helper.make_node(
        "Cast",
        inputs=[input_name],
        outputs=[output_name],
        name=node_name,
        to=to_dtype
    )
    return node

def convert_initializer_fp16_to_fp32(graph, init):
    arr = numpy_helper.to_array(init)
    if arr.dtype == np.float16:
        arr32 = arr.astype(np.float32)
        new_init = numpy_helper.from_array(arr32, init.name)
        # preserve other fields (doc_string etc) by copying
        new_init.doc_string = init.doc_string
        # replace in graph.initializer
        for i, it in enumerate(graph.initializer):
            if it.name == init.name:
                graph.initializer[i] = new_init
                return True
    return False

def set_value_info_dtype_to_float(graph, name):
    vi = find_value_info(graph, name)
    if vi is not None:
        try:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
            return True
        except Exception:
            return False
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="Input ONNX model path")
    parser.add_argument("--tensor", required=True, help="Tensor name to convert to FP32 (e.g. Mul_388_output_cast0)")
    parser.add_argument("--out", required=True, help="Output ONNX model path")
    args = parser.parse_args()

    if not os.path.exists(args.onnx):
        print("Input ONNX not found:", args.onnx)
        sys.exit(1)

    model = onnx.load(args.onnx)
    graph = model.graph

    # try shape inference to fill value_info
    try:
        model = shape_inference.infer_shapes(model)
        graph = model.graph
        log("Ran shape_inference.")
    except Exception as e:
        log("shape_inference failed or raised warning: " + str(e))

    tensor_name = args.tensor
    log(f"Target tensor: {tensor_name}")

    # 1) If initializer exists with this name and is float16 -> convert to float32
    init = find_initializer(graph, tensor_name)
    if init is not None:
        # check dtype
        if init.data_type == TensorProto.FLOAT16:
            log(f"Initializer '{tensor_name}' found and is FLOAT16 -> converting to FLOAT32.")
            ok = convert_initializer_fp16_to_fp32(graph, init)
            if ok:
                set_value_info_dtype_to_float(graph, tensor_name)
                onnx.save(model, args.out)
                log(f"Saved modified model (initializer converted) to {args.out}")
                return
            else:
                log("Initializer conversion failed.")
        else:
            log(f"Initializer '{tensor_name}' exists but not FLOAT16 (data_type={init.data_type}).")

    # 2) Find producers that output this tensor
    producers = find_producers(graph, tensor_name)
    if producers:
        log(f"Found {len(producers)} producer node(s) for '{tensor_name}'.")
    else:
        log(f"No producer node found for '{tensor_name}' (could be graph input or produced in subgraph).")

    # Try to find if there is a Cast node producing it with to=FLOAT16 -> modify it to FLOAT
    modified_cast = False
    for node in producers:
        if node.op_type == "Cast":
            # find 'to' attribute
            for attr in node.attribute:
                if attr.name == "to":
                    if attr.i == TensorProto.FLOAT16:
                        log(f"Producer is Cast to FLOAT16 -> changing attribute to FLOAT32 (node: {node.name}).")
                        attr.i = TensorProto.FLOAT
                        # also update value_info if present
                        set_value_info_dtype_to_float(graph, tensor_name)
                        modified_cast = True
                    else:
                        log(f"Producer Cast's to attr is {attr.i} (not FLOAT16).")
                    break
    if modified_cast:
        # done, save
        try:
            model = shape_inference.infer_shapes(model)
        except Exception:
            pass
        onnx.save(model, args.out)
        log(f"Saved modified model (cast attr changed) to {args.out}")
        return

    # 3) If not a direct cast we will insert a Cast(FP16->FP32) after producer(s)
    # Create a new tensor name for cast output
    new_tensor = tensor_name + "_to_fp32_" + uuid.uuid4().hex[:8]
    # Choose insertion index: after the last producer node if any; otherwise append at end
    insert_index = -1
    if producers:
        # find index of last producer node in graph.node (the last occurrence)
        last_idx = -1
        for i, node in enumerate(graph.node):
            if node in producers:
                last_idx = i
        insert_index = last_idx
    else:
        insert_index = len(graph.node) - 1

    cast_node = make_cast_node(tensor_name, new_tensor, TensorProto.FLOAT)
    # insert cast right after insert_index
    if insert_index >= 0 and insert_index < len(graph.node):
        insert_node_after(graph, insert_index, cast_node)
        log(f"Inserted Cast node after node index {insert_index}.")
    else:
        # append
        graph.node.append(cast_node)
        log("Appended Cast node at the end of graph.node.")

    # Replace all consumers of original tensor to use new tensor
    replace_consumers_inputs(graph, tensor_name, new_tensor)
    log(f"Replaced consumers inputs: {tensor_name} -> {new_tensor}")

    # If graph.output refers to original tensor, update it to new tensor and set type FLOAT
    for out in graph.output:
        if out.name == tensor_name:
            out.name = new_tensor
            try:
                out.type.tensor_type.elem_type = TensorProto.FLOAT
            except Exception:
                pass
            log(f"Updated graph.output name to {new_tensor} and set type to FLOAT.")

    # Update or add value_info for new tensor (attempt to copy shape info)
    orig_vi = find_value_info(graph, tensor_name)
    if orig_vi is not None:
        new_vi = copy.deepcopy(orig_vi)
        new_vi.name = new_tensor
        try:
            new_vi.type.tensor_type.elem_type = TensorProto.FLOAT
        except Exception:
            pass
        graph.value_info.append(new_vi)
        log("Added value_info for new tensor (copied shape info).")
    else:
        # no shape info available; skip
        log("No value_info for original tensor; skipping adding shaped value_info for new tensor.")

    # Also, if any value_info exists for original, set it to FLOAT (so declared types consistent)
    if orig_vi is not None:
        try:
            orig_vi.type.tensor_type.elem_type = TensorProto.FLOAT
            log("Set original value_info elem_type to FLOAT (if existed).")
        except Exception:
            pass

    # final shape_inference and save
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:
        log("shape_inference after modification warning: " + str(e))

    onnx.save(model, args.out)
    log(f"Saved modified model to {args.out}")
    log("Done.")

if __name__ == "__main__":
    main()
