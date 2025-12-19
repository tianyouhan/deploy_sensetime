import os
import cv2
import numpy as np
import glob
from subprocess import call
from multiprocessing import Process
from tqdm import tqdm


def mk_video(img_path, save_path, frame, rm_avi=True):
    if img_path[-1] != '/':
        img_path += '/'
    file_list = sorted(glob.glob(img_path + '*.png'))
    if not os.path.exists(file_list[0]):
        file_list = sorted(glob.glob(img_path + '*.jpg'))
    
    img = cv2.imread(file_list[0])
    height, width = img.shape[0:2]
    size = (width, height)
    
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'), frame, size)
    for filename in file_list:
        img = cv2.imread(filename)
        if img.shape[0:2] != size[::-1]:
            img = cv2.resize(img, size)
        out.write(img)
    out.release()
    new_save_path = save_path[:-4] + '.mp4'
    command = "ffmpeg -y -i %s -b:v 3000k %s" % (save_path, new_save_path)
    call(command.split())
    if rm_avi:
        os.system(f'rm {save_path}')


def base_video(seq_path_list):
    if type(seq_path_list) is not list:
        seq_path_list = [seq_path_list]
    for seq_path in tqdm(seq_path_list):
        seq_name = os.path.split(seq_path)[-1]
        print(f'process {seq_path}')
        img_path = os.path.join(seq_path, 'merge')
        save_path = os.path.join(seq_path, 'video')
        os.makedirs(save_path, exist_ok=True)
        save_video_file = os.path.join(save_path, seq_name + '.avi')
        mk_video(img_path, save_video_file, frame=10, rm_avi=True)
    return


def split_list_n_list(origin_list, n):
    n_list = []
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1
    for i in range(0, n):
        n_list.append(origin_list[i*cnt:(i+1)*cnt])
    return n_list

def main(one_seq, thread_num=20):
    if one_seq:
        base_video(seq_path)
    else:
        seq_path_list = [os.path.join(seq_path, seq) for seq in sorted(os.listdir(seq_path))]
        if thread_num > 1:
            n_list = split_list_n_list(seq_path_list, thread_num)
            process_list = []
            for cur_list in n_list:
                t = Process(target=base_video, args=(cur_list, ))
                t.start()
                process_list.append(t)
            for t in process_list: # 等待所有进程运行完毕
                t.join()
        else:
            base_video(seq_path_list)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='multi')
    parser.add_argument('--id', type=int, default=1, help='epoch id')
    args = parser.parse_args()
    args.id = 25

    seq_path = f'unknown_vis_lustrenew/fusion_A02_3dgop_seg_det_2V_V1.9_No1_rainy_batch_10/det_vis/fusion'

    main(one_seq=False, thread_num=20)
    
