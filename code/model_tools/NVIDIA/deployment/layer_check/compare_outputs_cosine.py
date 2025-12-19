import os
import base64
import json
import numpy as np
import io
import logging
from datetime import datetime

FP16_MAX = 65504.0  # FP16 最大可表示数

def setup_logger(log_path):
    logger = logging.getLogger("compare_outputs")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def decode_array(b64str):
    raw = base64.b64decode(b64str)
    return np.load(io.BytesIO(raw), allow_pickle=True)

def load_outputs(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if "lst" in data:
        return data["lst"][0][1][0]["outputs"]
    elif "outputs" in data:
        return data["outputs"]
    else:
        raise ValueError(f"未知的 JSON 结构: {json_path}")

def cosine_similarity(u, v, eps=1e-8):
    u = u.astype(np.float32)
    v = v.astype(np.float32)
    dot = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u < eps or norm_v < eps:
        return 0.0
    return dot / (norm_u * norm_v)

def compare_outputs(fp32_path, fp16_path, logger, cosine_threshold=0.99, top_n=20):
    outputs_fp32 = load_outputs(fp32_path)
    outputs_fp16 = load_outputs(fp16_path)

    # 保留原始顺序（Python 3.7+ 字典是有序的）
    ordered_keys_fp32 = list(outputs_fp32.keys())
    common_keys = [k for k in ordered_keys_fp32 if k in outputs_fp16]

    logger.info(f"🔍 开始比对，共有 {len(common_keys)} 个共同层输出（按网络顺序）")

    bad_layers = []
    abs_list = []

    for name in common_keys:  # ✅ 不再用 sorted()
        try:
            arr_fp32 = decode_array(outputs_fp32[name]["values"]["array"]).flatten()
            arr_fp16 = decode_array(outputs_fp16[name]["values"]["array"]).flatten()
        except Exception as e:
            logger.error(f"❌ 解码失败: {name} → {e}")
            continue

        if arr_fp32.size == 0 or arr_fp16.size == 0:
            logger.warning(f"⚠️ 跳过空 tensor: {name}")
            continue

        max_abs = float(np.max(np.abs(arr_fp32)))
        abs_list.append((name, max_abs))

        if max_abs > FP16_MAX:
            logger.warning(f"⚠️ 层 {name} FP32 max_abs={max_abs:.2f} 超出 FP16 范围 {FP16_MAX}")
            bad_layers.append((name, -3))

        try:
            cos_sim = cosine_similarity(arr_fp32, arr_fp16)
        except Exception as e:
            logger.error(f"[异常] {name:<40} 计算 cosine 出错: {e}")
            continue

        logger.info(f"{name:<40} cosine={cos_sim:.6f}")
        logger.info(f"    FP32 → max={np.max(arr_fp32):.6f}, min={np.min(arr_fp32):.6f}")
        logger.info(f"    FP16 → max={np.max(arr_fp16):.6f}, min={np.min(arr_fp16):.6f}")

        if cos_sim < cosine_threshold:
            bad_layers.append((name, cos_sim))

    abs_list.sort(key=lambda x: x[1], reverse=True)
    logger.info(f"🔝 Top-{top_n} 层 max_abs 排序:")
    for name, val in abs_list[:top_n]:
        logger.info(f"    {name:<50} max_abs={val:.6f}")

    logger.info("✅ 比对完成")
    if bad_layers:
        logger.warning("🚨 出现问题的输出层：")
        for name, sim in bad_layers:
            logger.warning(f"    {name:<40} 标记={sim}")
    else:
        logger.info("🎉 所有输出相似度良好！")
        
if __name__ == "__main__":
    fp16_path = "/mnt/data/hantianyou/road_compare_tool/layer_check/save_outputs/outputs_fp32_all.json"
    fp32_path = "/mnt/data/hantianyou/road_compare_tool/layer_check/save_outputs/outputs_fp16_select.json"
    log_path = f"compare_cosine_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_path)

    compare_outputs(fp32_path, fp16_path, logger)
