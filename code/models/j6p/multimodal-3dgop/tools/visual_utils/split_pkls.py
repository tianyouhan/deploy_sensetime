import os
import pickle as pkl
from pcdet.utils import pose_utils, ceph_utils
from petrel_client.client import Client
import argparse
from tqdm import tqdm

def ceph_write(data, path, use_ceph=False, client=None, update_cache=True):
    postfix = os.path.splitext(path)[1].lower()
    assert postfix == '.pkl', 'Only .pkl saving is supported currently'

    file_bytes = pkl.dumps(data)

    if use_ceph:
        assert client is not None, 'client should not be None'
        client.put(path, file_bytes, update_cache=update_cache)
    else:
        with open(path, 'wb') as f:
            f.write(file_bytes)
    



if __name__ == '__main__':
    # import debugpy
    # debugpy.listen(("10.5.37.44", 38758))
    # print('Waitting for debuger attach')
    # # 等待debug工具连接
    # debugpy.wait_for_client()
    parser = argparse.ArgumentParser(description='multi')
    parser.add_argument('-i', '--input', type=str, default='', help='pkl path')
    parser.add_argument('-o', '--output', type=str, default='aoss-zhc-v2:s3://zhc-v2/lidargop/dataset', help='save path')
    parser.add_argument('-s', '--data_source', type=str, default='atx', help='data_source')
    args = parser.parse_args()
    data_source = args.data_source
    print(data_source)
    conf_path = '/iag_ad_01/ad/zhanghongcheng/code/multimodal-3dgop/petreloss.conf'
    if not os.path.exists(conf_path):
        conf_path = '~/petreloss.conf'
    client = Client(conf_path)
    print('client: ', client)

    pkl_dir = args.input
    if os.path.isfile(pkl_dir):
        pkl_paths = [pkl_dir]
    else:
        pkl_paths = [os.path.join(pkl_dir, f) for f in os.listdir(pkl_dir)]
    print(pkl_paths)
    save_dir = args.output 
    new_pkls = []
    for pkl_path in pkl_paths:
        use_ceph = 's3' in pkl_path
        res = ceph_utils.ceph_read(pkl_path, dtype=None, use_ceph=use_ceph, client=client)
        # import ipdb; ipdb.set_trace()
        basename = pkl_path.split('/')[-1][:-4]
        new_pkl_name = basename + '_S.pkl'
        new_pkl = dict()
        for seq_name in tqdm(list(res.keys())):
            cur_seq = res[seq_name]
            new_pkl[seq_name] = []
            for i, frame in tqdm(enumerate(cur_seq)):
                # print(frame.keys())
                frame['data_source'] = data_source
                frame_path = os.path.join(save_dir, basename, seq_name, "%s.pkl" % i)
                new_pkl[seq_name].append(frame_path)
                ceph_write(frame, frame_path, True, client)
        new_pkl_savepath = os.path.join(save_dir, basename, new_pkl_name)
        print(new_pkl_savepath)
        new_pkls.append(new_pkl_savepath)
        ceph_write(new_pkl, new_pkl_savepath, True, client)
    print(new_pkls)            