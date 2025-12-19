from hmct.common import find_input_calibration
from hmct.ir import load_model, save_model
from hmct.quantizer.debugger import get_sensitivity_of_nodes
import argparse
import os


def parse_model(args):
    model = load_model(os.path.join(args.rootDir, args.modelDir, args.modelPrefix + "_calibrated_model.onnx"))
    interested_nodes = []
    matmul_nodes = model.graph.type2nodes[args.nodeKind]
    for node in matmul_nodes:
        feature0 = find_input_calibration(node, 0)
        feature1 = find_input_calibration(node, 1)
        if feature0 is not None:
            feature0.qtype = "int8"
            interested_nodes.append(feature0.name)
        if feature1 is not None:
            feature1.qtype = "int8"
            interested_nodes.append(feature1.name)
    if args.save:
        save_model(model, os.path.join(args.rootDir, args.modelDir, args.modelPrefix + "_modify_calibrated_model.onnx"))
    return model, interested_nodes, matmul_nodes


def print_calib_info(args):
    cali_data = os.path.join(args.calibDataDir)
    model, interested_nodes, matmul_nodes = parse_model(args)

    # calculate node sensitivity
    node_message = get_sensitivity_of_nodes(
        model_or_file=model.proto,
        metrics=['cosine-similarity', 'mse', 'mre', 'sqnr', 'chebyshev'],
        calibrated_data=cali_data,
        data_num=args.numData,
        interested_nodes=interested_nodes,
        output_node=args.outputNode,
        verbose=True,
    )

    for node in matmul_nodes:
        feature0 = find_input_calibration(node, 0)
        feature1 = find_input_calibration(node, 1)
        if feature0 and feature1:
            print("{}  {}  {} {}".format(
                node.name,
                node_message[feature0.name].get("cosine-similarity"),
                node_message[feature1.name].get("cosine-similarity"),
                1 if float(node_message[feature1.name].get("cosine-similarity")) < float(node_message[feature0.name].get("cosine-similarity")) else 0
            ))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--rootDir", type=str, required=True, help='model.onnx dir path')
    parser.add_argument("-c", "--calibDataDir", type=str, required=True, help='Calibration data path')
    parser.add_argument("-md", "--modelDir", type=str, required=True, help='Calibration model path')
    parser.add_argument("-mp", "--modelPrefix", type=str, required=True, help='Calibration model prefix')
    parser.add_argument('-nk', '--nodeKind', type=str, required=True, default="MatMul", help='Node kind to analyze')
    parser.add_argument("-n", "--numData", type=int, default=5, help='Calibration data number')
    parser.add_argument('-s', '--save', action='store_true', help='Save modified model')
    parser.add_argument('-o', '--outputNode', type=str, action="append", default=None, )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print_calib_info(args)