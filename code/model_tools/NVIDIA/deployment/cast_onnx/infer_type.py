import onnx
from onnx import shape_inference

model = onnx.load("/mnt/data/hantianyou/road_compare_tool/cast_onnx/jc/gop_driving_cam_branch_fp16.onnx")
model = shape_inference.infer_shapes(model)
onnx.save(model, "/mnt/data/hantianyou/road_compare_tool/cast_onnx/jc/gop_driving_cam_branch_fp16.onnx")