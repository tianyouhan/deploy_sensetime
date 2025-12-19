#!/usr/bin/env python3
"""
将 ONNX 模型转换到更高的 opset 版本以支持 GridSample 操作
Usage: python3 convert_opset.py input.onnx output.onnx [--target-opset 17]
"""
import argparse
import onnx
from onnx import version_converter
from pathlib import Path


def convert_opset(input_path, output_path, target_opset=17):
    """
    将 ONNX 模型转换到指定的 opset 版本
    
    Args:
        input_path: 输入模型路径
        output_path: 输出模型路径
        target_opset: 目标 opset 版本（GridSample 在 opset 16 加入，推荐 17）
    """
    print(f"Loading model from {input_path}")
    model = onnx.load(str(input_path))
    
    # 显示当前模型的 opset 版本
    current_opset = model.opset_import[0].version
    print(f"Current opset version: {current_opset}")
    
    if current_opset >= target_opset:
        print(f"Model already uses opset {current_opset}, no conversion needed")
        onnx.save(model, str(output_path))
        return
    
    print(f"Converting to opset {target_opset}...")
    try:
        # 转换 opset 版本
        converted_model = version_converter.convert_version(model, target_opset)
        
        # 检查模型
        print("Checking converted model...")
        onnx.checker.check_model(converted_model)
        
        # 保存转换后的模型
        print(f"Saving converted model to {output_path}")
        onnx.save(converted_model, str(output_path))
        
        print(f"✓ Successfully converted model to opset {target_opset}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        print("\nAlternative approach: Try exporting the model again with a higher opset version")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Convert ONNX model to higher opset version to support GridSample"
    )
    parser.add_argument("input", type=Path, help="Input ONNX model path")
    parser.add_argument("output", type=Path, help="Output ONNX model path")
    parser.add_argument(
        "--target-opset",
        type=int,
        default=17,
        help="Target opset version (default: 17, minimum 16 for GridSample)"
    )
    
    args = parser.parse_args()
    
    if args.target_opset < 16:
        print("Warning: GridSample operation requires opset >= 16")
        print(f"Setting target opset to 16 (you specified {args.target_opset})")
        args.target_opset = 16
    
    convert_opset(args.input, args.output, args.target_opset)


if __name__ == "__main__":
    main()

