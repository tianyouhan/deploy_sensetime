# import onnx
# from onnxconverter_common import float16
# import argparse
# import os
# import numpy as np


# ops_to_skip = float16.DEFAULT_OP_BLOCK_LIST + ["Clip","Expand","Shape", "Slice", "Concat", "TopK", "ArgMax", "NonMaxSuppression", "Where", "ScatterElements"]

# def extract_node_names_from_subgraph(subgraph):
#     """
#     提取子图中所有节点的名称，用于 float16 转换时保留这些节点为 FP32。
#     """
#     fp32_nodes = []
#     for node in subgraph.graph.node:
#         if node.name or (node.op_type in ops_to_skip):
#             fp32_nodes.append(node.name)
#             print(f'block {node.name} {node.op_type}')
#     return fp32_nodes

# def main(ipath, ipath2, trto, opath):
#     assert os.path.exists(ipath), f"Input model not found: {ipath}"
#     assert os.path.exists(ipath2), f"Subgraph model not found: {ipath2}"
#     # assert os.path.exists(trto), f"Trt outputs not found: {trto}"

#     print(f"Loading main model: {ipath}")
#     model = onnx.load(ipath)

#     print(f"Loading subgraph model: {ipath2}")
#     subgraph = onnx.load(ipath2)

#     print("Extracting FP32 node names from subgraph...")
#     fp32_nodes = extract_node_names_from_subgraph(subgraph)
#     print(f"Found {len(fp32_nodes)} nodes to keep in FP32.")
#     for node in model.graph.node:
#         if (node.op_type in ops_to_skip):
#             fp32_nodes.append(node.name)
#             print(f'block {node.name} {node.op_type}')
#     # fp32_nodes.append('Conv_125')
#     # fp32_nodes.remove('Gather_242')
#     # outputs = load_json(trto)[0][1][0].dct
#     # other_nodes = []
#     # for name, arr in outputs.items():
#     #     if arr.arr.dtype != np.float16:
#     #         other_nodes.append(name)
#     #         print('other:', name, ' ', arr.arr.dtype)
#     #     else:
#     #         print('fp16:', name)
#     # quit()
#     print("Converting to FP16 (with selected FP32 nodes)...")
    
#     fp16_model = float16.convert_float_to_float16(
#         model,
#         max_finite_val=65505,
#         keep_io_types=True,
#         disable_shape_infer=True,
#         op_block_list=ops_to_skip,
#         node_block_list=fp32_nodes
#     )

#     print(f"Saving converted model to: {opath}")
#     onnx.save(fp16_model, opath)
#     print("Conversion completed successfully.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Convert ONNX model to FP16, keeping a subgraph in FP32.")
#     parser.add_argument("--path", required=True, help="Path to input ONNX model")
#     parser.add_argument("--path2", required=True, help="Path to subgraph ONNX model whose nodes should stay in FP32")
#     parser.add_argument("--trt_outs", required=False, help="Path to FP16 trt engine outputs")
#     parser.add_argument("--opath", required=True, help="Output path for the converted FP16 model")
    
#     args = parser.parse_args()
#     main(args.path, args.path2, args.trt_outs, args.opath)

import onnx
from onnxconverter_common import float16
from onnx import helper, TensorProto, TensorShapeProto
import argparse
import os

ops_to_skip = float16.DEFAULT_OP_BLOCK_LIST + [
    "Clip","Expand","Shape","Slice","Concat","TopK",
    "ArgMax","NonMaxSuppression","Where","ScatterElements"
]

def extract_node_names_from_subgraph(subgraph):
    """
    提取子图中的所有节点名称，用于保持这些节点为 FP32。
    """
    fp32_nodes = []
    for node in subgraph.graph.node:
        if node.name or (node.op_type in ops_to_skip):
            fp32_nodes.append(node.name)
            print(f'[subgraph FP32] block {node.name} ({node.op_type})')
    return fp32_nodes


def main(ipath, ipath2, opath, skip_node_names, manual_io_cast):
    assert os.path.exists(ipath), f"Input model not found: {ipath}"
    print(f"Loading main model: {ipath}")
    model = onnx.load(ipath)

    # -------------------------------
    # 1. Load optional subgraph
    # -------------------------------
    fp32_nodes = []
    if ipath2 is not None:
        print(f"Loading subgraph model: {ipath2}")
        subgraph = onnx.load(ipath2)
        print("Extracting FP32 node names from subgraph...")
        fp32_nodes.extend(extract_node_names_from_subgraph(subgraph))
        print(f"Found {len(fp32_nodes)} nodes from subgraph")
    else:
        print("No subgraph (--path2 not provided), skipping FP32 extraction.")

    # -------------------------------
    # 2. Skip built-in ops
    # -------------------------------
    print("Adding built-in FP32 operators...")
    for node in model.graph.node:
        if node.op_type in ops_to_skip:
            fp32_nodes.append(node.name)
            print(f'[builtin FP32] block {node.name} ({node.op_type})')

    # -------------------------------
    # 3. Add user skip-node-names
    # -------------------------------
    if skip_node_names:
        print("Adding user --skip-node-names ...")
        user_nodes = [x.strip() for x in skip_node_names.split(",") if x.strip()]
        for name in user_nodes:
            fp32_nodes.append(name)
            print(f'[user FP32] block {name}')
        print(f"Total user-skip nodes: {len(user_nodes)}")

    print(f"Total FP32 nodes collected: {len(fp32_nodes)}")

    # -------------------------------
    # 4. Convert to FP16
    # -------------------------------
    print("Converting to FP16 ...")

    if manual_io_cast:
        print(" ⚠️ manual_io_cast=True → keep_io_types=False")
        # 让 onnx 先把 input/output 全部转 FP16
        fp16_model = float16.convert_float_to_float16(
            model,
            max_finite_val=65505,
            keep_io_types=False,
            disable_shape_infer=True,
            op_block_list=ops_to_skip,
            node_block_list=fp32_nodes
        )
    else:
        print(" manual_io_cast=False → keep_io_types=True（保持输入输出原类型）")
        # 让 onnx 保护输入输出（它们会保持 FP32）
        fp16_model = float16.convert_float_to_float16(
            model,
            max_finite_val=65505,
            keep_io_types=True,
            disable_shape_infer=True,
            op_block_list=ops_to_skip,
            node_block_list=fp32_nodes
        )
    # ======================================================================
    # 5 + 6. Only when manual_io_cast=True, do custom IO handling
    # ======================================================================
    if manual_io_cast:
        print("manual_io_cast=True → manually patching inputs & outputs")

        # -------------------------------
        # 5. Force all inputs to FP32
        # -------------------------------
        print("Force all model inputs to FP32...")
        for inp in fp16_model.graph.input:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT
            print(f"Set input `{inp.name}` type → FP32")

        # -------------------------------
        # 6. Add safe Cast(FP16->FP32) to outputs
        # -------------------------------
        print("Adding Cast(FP16 -> FP32) to all outputs (keep original names)...")

        new_output_cast_nodes = []

        # Build producer map
        output_producer = {}
        for node in fp16_model.graph.node:
            for o in node.output:
                output_producer[o] = node

        for out in list(fp16_model.graph.output):
            orig_name = out.name
            internal_name = orig_name + "_fp16_internal"

            producer = output_producer.get(orig_name, None)
            if producer is not None:
                # rename producer output
                for i, o in enumerate(producer.output):
                    if o == orig_name:
                        producer.output[i] = internal_name
                output_producer.pop(orig_name, None)
                output_producer[internal_name] = producer

            # update all node inputs
            for node in fp16_model.graph.node:
                for i, n in enumerate(node.input):
                    if n == orig_name:
                        node.input[i] = internal_name

            # Cast node
            cast_node = helper.make_node(
                "Cast",
                inputs=[internal_name],
                outputs=[orig_name],
                name=f"Cast_{orig_name}_to_fp32",
                to=TensorProto.FLOAT,
            )
            new_output_cast_nodes.append(cast_node)

            # Set output dtype
            out.type.tensor_type.elem_type = TensorProto.FLOAT

            print(f"Patched output '{orig_name}'")

        # append cast nodes
        for n in new_output_cast_nodes:
            fp16_model.graph.node.append(n)

    else:
        print("manual_io_cast=False → using keep_io_types=True behavior (auto FP32 IO).")

    # -------------------------------
    # Save final model
    # -------------------------------
    onnx.save(fp16_model, opath)
    print("Conversion completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX to FP16 while keeping selected nodes FP32")

    parser.add_argument("--path", required=True, help="Path to input ONNX model")
    parser.add_argument("--path2", required=False, default=None,
                        help="Optional subgraph ONNX model; nodes inside will stay in FP32")
    parser.add_argument("--skip-node-names", required=False, default="",
                        help="Comma-separated node names to keep in FP32")
    parser.add_argument("--opath", required=True, help="Output FP16 model path")
    # 关键开关：是否手动控制输入输出
    parser.add_argument("--manual_io_cast", action="store_true",
                        help="If set, manually force inputs FP32 and add output Casts. Otherwise use keep_io_types=True.")
    args = parser.parse_args()

    main(args.path, args.path2, args.opath, args.skip_node_names,args.manual_io_cast)