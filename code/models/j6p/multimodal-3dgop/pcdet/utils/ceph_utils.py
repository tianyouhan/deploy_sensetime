import os
import cv2
import json
import pickle
import numpy as np
from petrel_client.client import Client
import pickle

def ceph_init(conf_path=None):
    if conf_path is None:
        conf_path = '../petreloss.conf'
        if not os.path.exists(conf_path):
            conf_path = '~/petreloss.conf'
    client = Client(conf_path) # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件
    return client


def ceph_url(prefix=None, filename=None):
    # prefix:'1424nvme:s3://data/unknown/cloud_bin'
    ceph_url = os.path.join(prefix, filename)
    return ceph_url


def ceph_read(path, dtype, use_ceph, client=None):
    # data = client.get(url)                      # 默认情况由配置文件决定是否使用 MC
    # data = client.get(url, no_cache=True)       # 本次 get 直接从 ceph 读取
    # data = client.get(url, update_cache=True)   # 本次 get 直接从 ceph 读取，并将数据缓存至 MC
    # client.put(url, data)                       # 默认 put 不会更新 MC
    # client.put(url, data, update_cache=True)    # 本次 put 将数据存入 ceph 之后并更新 MC
    postfix = os.path.splitext(path)[1]
    if use_ceph:
        assert client is not None, 'client should not be None'
        file_byte = client.get(path, update_cache=True)
        if file_byte is None:
            print(path)
        if postfix in ['.txt']:
            file = str(file_byte, encoding='utf-8').split('\n')
        elif postfix in ['.json']:
            file = json.loads(file_byte)
        elif postfix == '.pkl':
            file = pickle.loads(file_byte)
        elif postfix in ['.png', '.jpg', '.jepg']:
            file = cv2.imdecode(np.frombuffer(file_byte, dtype=dtype), cv2.IMREAD_UNCHANGED)
        else:
            file = np.frombuffer(file_byte, dtype=dtype).copy()
    else:
        if postfix in ['.txt']:
            file = open(path, 'r', encoding='utf-8').readlines()
        elif postfix in ['.json']:
            file = json.load(open(path, 'r', encoding='utf-8'))
        elif postfix == '.pkl':
            with open(path, 'rb') as f:
                file = pickle.load(f)
        elif postfix in ['.png', '.jpg', '.jepg']:
            file = cv2.imdecode(np.fromfile(path, dtype=dtype), cv2.IMREAD_UNCHANGED)
        else:
            file = np.fromfile(path, dtype=dtype)
    return file
