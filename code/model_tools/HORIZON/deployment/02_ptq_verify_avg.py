import os
import cv2
import argparse
import numpy as np
from horizon_tc_ui.hb_runtime import HBRuntime


def cosine_similarity(a, b):
    a = a.flatten()
    b = b.flatten()
    similarity = 0.0
    similarity = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    similarity = "{:.8f}".format(similarity)
    return similarity

def inverse_normal(image, mean, scale, Mode_bgr=True):
    assert mean is not None and scale is not None, 'please check out mean and scale'
    if image.ndim != 3:
        image = np.squeeze(image)
        assert image.ndim == 3, "image is not BGR format"
    image_inv = np.zeros(image.shape)
    for i in range(3):
        image_inv[i] = (image[i] / scale[i] + mean[i])
    image_inv = image_inv.transpose(1, 2, 0).astype(np.uint8)
    if not Mode_bgr:
        image_inv = cv2.cvtColor(image_inv, cv2.COLOR_RGB2BGR)
    return image_inv

def brg2nv12_opencv(image, mean, scale,  Mode_bgr=True):
    # Remember, when configuration file "input_type_train: nv12" , convert image nv12 to use.
    image = inverse_normal(image, mean, scale,  Mode_bgr)
    image = image.astype(np.uint8)
    height, width = image.shape[0], image.shape[1]
    yuv420p = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).reshape((width * height * 3 // 2, )) 
    y = yuv420p[:width * height].reshape(1, width, height, 1)
    uv_planar = yuv420p[width * height:].reshape((2, width * height // 4)) 
    uv = uv_planar.transpose((1, 0)).reshape(1, width//2, height//2, 2)
    return y.astype(np.uint8), uv.astype(np.uint8)

def bgr2nv12_calc(image, mean, scale,  Mode_bgr=True):
    image = inverse_normal(image, mean, scale,  Mode_bgr)
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    y = (0.299 * r + 0.587 * g + 0.114 * b)
    u = (-0.169 * r - 0.331 * g + 0.5 * b + 128)[::2, ::2]
    v = (0.5 * r - 0.419 * g - 0.081 * b + 128)[::2, ::2]
    assert u.shape == v.shape, "size of Channel U is different with Channel V"
    uv = np.zeros(shape=(u.shape[0], u.shape[1] * 2))
    for i in range(0, u.shape[0]):
        for j in range(0, u.shape[1]):
            uv[i, 2 * j] = u[i, j]
            uv[i, 2 * j + 1] = v[i, j]
    return y.astype(np.uint8), uv.astype(np.uint8)

def loadNpyData(dataDir, inputName, index):
    dataPath = os.path.join(dataDir, inputName, f"{index}.npy")
    data = np.load(dataPath)
    return data

def loadBinData(dataDir, inputName, index):
    dataPath = os.path.join(dataDir, inputName, f"{index}.bin")
    assert os.path.exists(dataPath), f"Not found data file: {dataPath}"
    with open(dataPath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.float32)
    return data

def loadInputs_fp(dataDir, index, sess):
    input_names = sess.input_names
    datas_fp = []

    for idx in range(len(input_names)):
        data = loadNpyData(dataDir, input_names[idx], index)
        datas_fp.append(data)

    input_shapes = sess.input_shapes
    assert len(datas_fp) == len(input_shapes)
    for i in range(len(datas_fp)):
        datas_fp[i] = datas_fp[i].reshape(input_shapes[i])
    return datas_fp

def loadInputs_hbir(datas_fp, start_idx, end_idx, sess, mean, scale):
    assert end_idx > start_idx, "incorrect list index parameter"
    datas_hbir = []

    if start_idx != 0:
        datas_hbir.extend(datas_fp[0:start_idx])
    for i in range(end_idx - start_idx + 1 ):
        # data_y, data_uv = bgr2nv12_calc(datas_fp[start_idx + i], mean, scale,  Mode_bgr=False)
        data_y, data_uv = bgr2nv12_calc(datas_fp[start_idx + i], mean, scale,  Mode_bgr=True)  #BGR数据
        datas_hbir.append(data_y)
        datas_hbir.append(data_uv)
    if end_idx != ( len(datas_fp) - 1 ):
        datas_hbir.extend(datas_fp[end_idx + 1 :])

    input_shapes = sess.input_shapes
    assert len(datas_hbir) == len(input_shapes)
    for i in range(len(datas_hbir)):
        datas_hbir[i] = datas_hbir[i].reshape(input_shapes[i])
    return datas_hbir

def loadFeed(sess, datas):
    # import ipdb; ipdb.set_trace()
    # print(f'input_name: {sess.input_names}')
    # print(f'datas len: {len(datas)}')
    input_names = sess.input_names
    assert len(input_names) == len(datas)
    input_feed = {}
    for i in range(len(input_names)):
        assert input_names[i] not in input_feed
        input_feed[input_names[i]] = datas[i]
    return input_feed


def compare(args):
    print("=================================================================== start")
    # sess_onnx = HBRuntime(os.path.join(args.rootDir, "model_gs.onnx")) #3.0.22支持
    sess_onnx = HBRuntime(os.path.join(args.rootDir, "model.onnx")) 
    sess_original = HBRuntime(os.path.join(args.rootDir, args.modelDir, args.modelPrefix + "_original_float_model.onnx"))
    sess_optimized = HBRuntime(os.path.join(args.rootDir, args.modelDir, args.modelPrefix + "_optimized_float_model.onnx"))
    sess_calib = HBRuntime(os.path.join(args.rootDir, args.modelDir, args.modelPrefix + "_calibrated_model.onnx"))
    sess_ptq = HBRuntime(os.path.join(args.rootDir, args.modelDir, args.modelPrefix + "_ptq_model.onnx"))
    sess_quanti = HBRuntime(os.path.join(args.rootDir, args.modelDir, args.modelPrefix + "_quantized_model.bc"))
    
    output_names = sess_onnx.output_names
    assert args.length > 1
    
    layer_results_onnx = {name: [] for name in output_names}
    layer_results_original = {name: [] for name in output_names}
    layer_results_optimized = {name: [] for name in output_names}
    layer_results_calib = {name: [] for name in output_names}
    layer_results_ptq = {name: [] for name in output_names}
    layer_results_quanti = {name: [] for name in output_names}

    for i in range(args.length):
        index = args.indexOri + i
        datas_fp = loadInputs_fp(args.dataDir, index, sess_onnx)

        input_feed_onnx = loadFeed(sess_onnx, datas_fp)
        input_feed_original = loadFeed(sess_original, datas_fp) 
        input_feed_optimized = loadFeed(sess_optimized, datas_fp)
        input_feed_calib = loadFeed(sess_calib, datas_fp) 
        input_feed_ptq = loadFeed(sess_ptq, datas_fp) 

        ## 当模型部署时的输入类型为NV12的时候，使用下边的代码，车端需要输入类型为NV12
        # input_feed_quanti = loadFeed(sess_quanti, loadInputs_hbir(datas_fp, 0, 6, sess_quanti, args.mean, args.scale)) 
        if args.cameraNum != 0:
            input_feed_quanti = loadFeed(sess_quanti, loadInputs_hbir(datas_fp, 0, args.cameraNum - 1, sess_quanti, args.mean, args.scale))
        else:
            input_feed_quanti = loadFeed(sess_quanti, datas_fp)
        
        ## 当模型部署时的输入类型为featuremap的时候，使用下边的代码
        # input_feed_quanti = loadFeed(sess_quanti, datas_fp)
        #print("=================================================================== data_" + str(index) + " onnx begin run ==================================================================================")
        output_onnx = sess_onnx.run(sess_onnx.output_names, input_feed_onnx)
        #print("=================================================================== data_" + str(index) + " original onnx begin run ===================================================================")
        gts = output_onnx
        output_original = sess_original.run(sess_original.output_names, input_feed_original)
        #print("=================================================================== data_" + str(index) + " optimized onnx begin run ===================================================================")
        output_optimized = sess_optimized.run(sess_optimized.output_names, input_feed_optimized)
        #print("=================================================================== data_" + str(index) + " calib onnx begin run ===================================================================")
        output_calib = sess_calib.run(sess_calib.output_names, input_feed_calib)
        #print("=================================================================== data_" + str(index) + " ptq onnx begin run ===================================================================")
        output_ptq = sess_ptq.run(sess_ptq.output_names, input_feed_ptq)
        output_quanti = sess_quanti.run(sess_quanti.output_names, input_feed_quanti)

        print("=================================================================== data_" + str(index) + " =======================================================================================")
        header = f"{'output_name':<25} {'gt vs onnx':<25} {'gt vs original':<25} {'gt vs optimized':<25} {'gt vs calib':<25} {'gt vs ptq':<25} {'gt vs quanti':<25}"
        print(header)
        print('-' * len(header))
        
        for j in range(len(output_onnx)):
            sim_onnx = float(cosine_similarity(gts[j], output_onnx[j]))
            sim_original = float(cosine_similarity(gts[j], output_original[j]))
            sim_optimized = float(cosine_similarity(gts[j], output_optimized[j]))
            sim_calib = float(cosine_similarity(gts[j], output_calib[j]))
            sim_ptq = float(cosine_similarity(gts[j], output_ptq[j]))
            sim_quanti = float(cosine_similarity(gts[j], output_quanti[j]))

            print(f"{output_names[j]:<25} "
                  f"{sim_onnx:<25.8f} "
                  f"{sim_original:<25.8f} "
                  f"{sim_optimized:<25.8f} "
                  f"{sim_calib:<25.8f} "
                  f"{sim_ptq:<25.8f}"
                  f"{sim_quanti:<25.8f}"
                )

            layer_results_onnx[output_names[j]].append(sim_onnx)
            layer_results_original[output_names[j]].append(sim_original)
            layer_results_optimized[output_names[j]].append(sim_optimized)
            layer_results_calib[output_names[j]].append(sim_calib)
            layer_results_ptq[output_names[j]].append(sim_ptq) 
            layer_results_quanti[output_names[j]].append(sim_quanti) 
        print("\n")

    print("=================================================================== Summary for gt vs onnx ================================================================================")
    for layer_name, results in layer_results_onnx.items():
        min_value = min(results)
        max_value = max(results)
        avg_value = sum(results) / len(results)
        print(f"{layer_name:<25} Min: {min_value:<25.8f} Max: {max_value:<25.8f} Avg: {avg_value:<25.8f}")

    print("\n=================================================================== Summary for gt vs original ================================================================================")
    for layer_name, results in layer_results_original.items():
        min_value = min(results)
        max_value = max(results)
        avg_value = sum(results) / len(results)
        print(f"{layer_name:<25} Min: {min_value:<25.8f} Max: {max_value:<25.8f} Avg: {avg_value:<25.8f}")

    print("\n=================================================================== Summary for gt vs optimized ================================================================================")
    for layer_name, results in layer_results_optimized.items():
        min_value = min(results)
        max_value = max(results)
        avg_value = sum(results) / len(results)
        print(f"{layer_name:<25} Min: {min_value:<25.8f} Max: {max_value:<25.8f} Avg: {avg_value:<25.8f}")

    print("\n=================================================================== Summary for gt vs calib  ================================================================================")
    for layer_name, results in layer_results_calib.items():
        min_value = min(results)
        max_value = max(results)
        avg_value = sum(results) / len(results)
        print(f"{layer_name:<25} Min: {min_value:<25.8f} Max: {max_value:<25.8f} Avg: {avg_value:<25.8f}")
    
    print("\n=================================================================== Summary for gt vs ptq    ================================================================================")
    for layer_name, results in layer_results_ptq.items():
        min_value = min(results)
        max_value = max(results)
        avg_value = sum(results) / len(results)
        print(f"{layer_name:<25} Min: {min_value:<25.8f} Max: {max_value:<25.8f} Avg: {avg_value:<25.8f}")
    
    print("\n=================================================================== Summary for gt vs quanti    =============================================================================")
    for layer_name, results in layer_results_quanti.items():
        min_value = min(results)
        max_value = max(results)
        avg_value = sum(results) / len(results)
        print(f"{layer_name:<25} Min: {min_value:<25.8f} Max: {max_value:<25.8f} Avg: {avg_value:<25.8f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataDir", type=str, required=False)
    parser.add_argument("-g", "--gtDir", type=str, required=False)
    parser.add_argument("-r", "--rootDir", type=str, required=False)
    parser.add_argument("-md", "--modelDir", type=str, required=False)
    parser.add_argument("-cn", "--cameraNum", type=int, required=False)
    parser.add_argument("-mp", "--modelPrefix", type=str, required=False)
    parser.add_argument("-i", "--indexOri", type=int, default=0, required=False)
    parser.add_argument("-l", "--length", type=int, default=9, required=False)
    parser.add_argument("-m", "--mean", type=float, nargs='+', required=False)
    parser.add_argument("-s", "--scale", type=float, nargs='+', required=False) 
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    compare(args)
    
