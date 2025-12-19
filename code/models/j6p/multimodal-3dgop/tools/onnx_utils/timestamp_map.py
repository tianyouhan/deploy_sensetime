import os
import numpy as np
import json
from datetime import datetime


def gettimestamp(string_timestamp):
	dt = datetime.strptime(string_timestamp,"%Y-%m-%d-%H-%M-%S-%f")
	d = dt.timestamp()
	return d

def getdatetime(timestamp):
	dt = datetime.fromtimestamp(timestamp)
	dt = datetime.strftime(dt, "%Y-%m-%d-%H-%M-%S-%f")[:-3]
	return dt

def find_nearest_idx(array,value):
    dis = np.abs(array-value)
    idx = dis.argmin()
    return idx, dis[idx]

def get_src_image_timestamp():
    img_path = '/mnt/lustrenew/share_data/xuzhiyong/THOR_data/batch_test/2024_05_09/2024_05_09_08_38_51_AutoCollect/image_undistort/'
    ref_txt = '/mnt/lustrenew/share_data/xuzhiyong/THOR_data/batch_test/image_list/segment_list_concat_M1_lidar/2024_05_09_08_38_51_AutoCollect_0.txt'

    cam_time_arr = {}
    cam_line = {}
    cams = ['center_camera_fov30','center_camera_fov120']
    for cam in cams:
        lines = open(os.path.join(img_path, cam + '.txt')).readlines()
        array_time = []
        line_list = []
        for line in lines:
            if len(line.strip()) > 0:
                ori_time = float(line.split()[-1]) / 1e9
                array_time.append(ori_time)
                line_list.append(line.split()[-1])
        array_time = np.array(array_time)
        cam_time_arr[cam] = array_time
        cam_line[cam] = line_list

    
    lines = open(ref_txt, 'r')
    count = -1
    map_dict = {}
    for line in lines:
        if len(line) > 10:
            count += 1
            curtime = line.strip().split('/')[-1][:-4]
            query_time = gettimestamp(curtime)
            cam_time = {}
            for key, array_time in cam_time_arr.items():
                idx, dis = find_nearest_idx(array_time, query_time)
                cam_time[key] = cam_line[key][idx]
            map_dict[count] = cam_time
    json_file = open('image_timestamp_map.json', 'w')
    json.dump(map_dict, json_file, indent=4)

def get_src_lidar_timestamp():
    lidar_txt = '/mnt/lustrenew/share_data/xuzhiyong/THOR_data/batch_test/2024_05_09/2024_05_09_08_38_51_AutoCollect/lidar/concat.txt'
    ref_txt = '/mnt/lustrenew/share_data/xuzhiyong/THOR_data/batch_test/image_list/segment_list_concat_M1_lidar/2024_05_09_08_38_51_AutoCollect_0.txt'

    cam_time_arr = {}
    cam_line = {}
    cams = ['concat']
    for cam in cams:
        lines = open(lidar_txt).readlines()
        array_time = []
        line_list = []
        for line in lines:
            if len(line.strip()) > 0:
                ori_time = float(line.strip()[:-4]) / 1e9
                array_time.append(ori_time)
                line_list.append(line.strip()[:-4])
        array_time = np.array(array_time)
        cam_time_arr[cam] = array_time
        cam_line[cam] = line_list

    
    # img_txt = 'tmp.txt'
    lines = open(ref_txt, 'r')
    count = -1
    map_dict = {}
    for line in lines:
        if len(line) > 10:
            count += 1
            curtime = line.strip().split('/')[-1][:-4]
            query_time = gettimestamp(curtime)
            cam_time = {}
            for key, array_time in cam_time_arr.items():
                idx, dis = find_nearest_idx(array_time, query_time)
                cam_time[key] = cam_line[key][idx]
            map_dict[count] = cam_time
    json_file = open('lidar_timestamp_map.json', 'w')
    json.dump(map_dict, json_file, indent=4)

if __name__ == '__main__':
     get_src_image_timestamp()
     get_src_lidar_timestamp()