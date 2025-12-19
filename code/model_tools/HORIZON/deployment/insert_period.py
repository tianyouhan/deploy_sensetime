import numpy as np
from hmct.ir import load_model, save_model


def insert_period(model, mul_node):
    div_val = model.graph.create_variable(
        is_param=True,
        value=np.array([6.2831854820251465], dtype=np.float32)
    )
    div_node = model.graph.create_node(
        op_type="Div",
        num_outputs=1).insert_after(mul_node)
    add_val = model.graph.create_variable(
        is_param=True,
        value=np.array([0.5], dtype=np.float32)
    )
    add_node = model.graph.create_node(
        op_type="Add",
        inputs=[div_node.outputs[0], add_val],
        num_outputs=1).insert_after(div_node)
    floor_node = model.graph.create_node(
        op_type="Floor",
        inputs=[add_node.outputs[0]],
        num_outputs=1).insert_after(add_node)
    mul2_val = model.graph.create_variable(
        is_param=True,
        value=np.array([6.2831854820251465], dtype=np.float32)
    )
    mul2_node = model.graph.create_node(
        op_type="Mul",
        inputs=[floor_node.outputs[0], mul2_val],
        num_outputs=1).insert_after(floor_node)
    sub_node = model.graph.create_node(
        op_type="Sub",
        num_outputs=1
    ).insert_after(mul2_node)
    mul_node.replace_all_uses_with(sub_node)
    div_node.append_input(mul_node.outputs[0])
    div_node.append_input(div_val)
    sub_node.append_input(mul_node.outputs[0])
    sub_node.append_input(mul2_node.outputs[0])
    model.infer_shapes()
    model.check_validity()
    return model


if __name__ == "__main__":
    model = load_model("customer_model.onnx")
    mul_node = model.graph.node_mappings["node_name"]
    insert_period(model, mul_node)
    save_model(model, "customer_model_period.onnx")
