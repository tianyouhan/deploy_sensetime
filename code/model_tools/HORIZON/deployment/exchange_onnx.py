from onnx import helper
import onnx
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python exchange_onnx.py arg1 [arg2 ...]")
        sys.exit(1)
    onnx_path=sys.argv[1]
    model_horizon = onnx.load(onnx_path)
    opset_id = helper.make_operatorsetid("horizon", 1)
    model_horizon.opset_import.append(opset_id)
    for node in model_horizon.graph.node:
        if (node.op_type == 'GridSample'):
            node.domain = 'horizon'
            model_horizon.opset_import[0].version = 11
            model_horizon.ir_version = 7
            align_corners = False
            for  attr_id, attr in enumerate(node.attribute):
                if attr.name == "align_corners":
                    align_corners = True
            if not align_corners:
                node.attribute.extend([helper.make_attribute("align_corners", 0)])
    onnx.save(model_horizon, onnx_path.replace('.onnx', '_gs.onnx'))
