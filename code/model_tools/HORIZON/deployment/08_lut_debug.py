import os
import copy
import argparse
import numpy as np
from hmct.common import modify_model_by_cpp_func
from hmct.ir import load_model, save_model
from hmct.ir.horizon_onnx import quantizer, global_attributes, quant_attributes
from horizon_tc_ui.hb_runtime import HBRuntime

default_lut_kinds = [
    "Sigmoid", "Exp","Reciprocal",
    "HardSigmoid", "Tanh", "Swish", "Softplus", 
    "HzMish", "Mish", "HardSwish", 
    "Gelu", "LeakyRelu", "Cos", "Sin", "Log", 
    "Clip", "Elu", "Pow", "Rsqrt", "Sqrt", "HzRsqrt",
    "Abs", "Atan", "Acos", "Acosh", "ThresholdedRelu", 
    "Asin", "Asinh", "Atanh", "Cosh", "Erf", "Selu", 
    "Sinh", "Tan", "Celu", "Round", "Sign", "Softsign"
]

def modify_model_kind(args, calibrated_model):
    node_dict = {}
    if args.cpu_kind:
        cpu_kind = args.cpu_kind
    else:
        cpu_kind = default_lut_kinds
    print(cpu_kind)
    for node_kind in cpu_kind:
        for node in calibrated_model.graph.type2nodes[node_kind]:
            node_dict[node.name] = {"InputType": "float32"}
    print(node_dict)
    quant_attributes.set({}, {}, node_dict)
    ptq_modify_model = modify_model_by_cpp_func(calibrated_model, quantizer.convert_for_hbdk4)
    return ptq_modify_model

def modify_model_node(calibrated_model, lut_ops, node_name):
    node_dict = {}
    for node in lut_ops:
        node_dict[node] = {"InputType": "float32"}
    removed_node_dict = copy.deepcopy(node_dict)
    calibrated_model_copy = copy.deepcopy(calibrated_model)
    removed_node_dict.pop(node_name, None) # pop单个节点 该节点不设置fp32则只有该节点转为查表
    quant_attributes.set({}, {}, removed_node_dict)
    ptq_modify_model = modify_model_by_cpp_func(calibrated_model_copy, quantizer.convert_for_hbdk4)
    return ptq_modify_model


def main(args, lut_ops=None):
    global_attributes.set_march(args.march)
    calibrated_model = load_model(os.path.join(os.getcwd(), args.modelDir, args.modelPrefix + "_calibrated_model.onnx"))
 
    if args.cpu_kind:
        ptq_modify_model = modify_model_kind(args, calibrated_model)
        name = "_".join(args.cpu_kind)
        save_model(ptq_modify_model, os.path.join(os.getcwd(), args.modelDir, args.modelPrefix + f"_ptq_remove_{name}_model.onnx"))
    else:
        for node_name in lut_ops:
            ptq_modify_model = modify_model_node(calibrated_model, lut_ops, node_name)
            save_model(os.path.join(os.getcwd(), args.modelDir, args.modelPrefix + f"_ptq_remove_{node_name}_model.onnx"))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--march", type=str, default="nash", help="Target BPU micro architecture or other backends.")
    parser.add_argument("--cpu_kind", "-ck", type=str, action="append", help="Input quantized model(.onnx) file.")
    parser.add_argument("--dataDir", "-d", type=str,  help="Calibration data path.")
    parser.add_argument("--modelDir", "-md", type=str, help="Working dir.")
    parser.add_argument("--modelPrefix", "-mp", type=str, help="Output model file prefix.")

    return parser.parse_args()           

if __name__ == "__main__":
    lut_ops = []
    args = get_args()
    main(args, lut_ops)

    