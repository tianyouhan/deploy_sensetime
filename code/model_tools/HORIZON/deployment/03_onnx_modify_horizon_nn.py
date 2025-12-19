from horizon_nn.ir import load_model,save_model,extract_submodel
from horizon_nn.common import find_input_calibration

def extract_model():
    model = load_model('customer_model.onnx')
    sub_model = extract_submodel(model, ['input_node_name'], ['output_node_name'])
    save_model(sub_model, 'customer_model_modify.onnx')

def conv_weight():
    model = load_model('customer_model.onnx')
    conv_names = ['Conv_324']
    for conv_name in conv_names:
        print(conv_name)
        conv_node = model.graph.node_mappings[conv_name]
        conv_weight = find_input_calibration(conv_node, 1)
        conv_weight.qtype = "int16"  
    save_model(model, 'customer_model_modify.onnx')

def real_int16():
    model = load_model('/home/mnt/zengyuqian/code/gridsample_e2e/gridsample_e2e_v0325_AD_regnetx400_bevg8mk7_bevneck64exp2fixfreeze_e8/quant_386_lite_0321/torch-jit-export_subnet0_ptq_model.onnx')
    calibration_nodes = model.graph.type2nodes["HzCalibration"]
    # 配置所有激活节点采用int16
    for node in calibration_nodes:
        if node.tensor_type=="feature":
            node.qtype="int16"  
    # 配置所有权重节点采用int16
    for node in calibration_nodes:
        if node.tensor_type=="weight":
            node.qtype="int16"
    save_model(model, 'customer_ptq_model.onnx')

if __name__ == '__main__':
    # extract_model(model)
    # conv_weight()
    real_int16()