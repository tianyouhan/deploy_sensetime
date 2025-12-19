import cv2
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Process


def merge_img(bev_path, img_path_list, save_path):
    for file in tqdm(sorted(os.listdir(bev_path))):
        frame_name = os.path.splitext(file)[0]
        bev_map = cv2.imread(os.path.join(bev_path, file))
        bev_h, bev_w = bev_map.shape[0:2]
        if max(bev_h, bev_w) < 1920:
            bev_map = cv2.resize(bev_map, (1920, int(bev_h * (1920 / bev_w))), interpolation=cv2.INTER_LINEAR)
        imgs = [cv2.imread(os.path.join(path, frame_name + '.png')) for path in img_path_list]

        bev_h, bev_w = bev_map.shape[0:2]
        img_h, img_w = imgs[0].shape[0:2]
        if len(imgs) == 1:
            resize_w = bev_w
            resize_h = int(img_h * (resize_w / img_w))
            imgs = [cv2.resize(img, (resize_w, resize_h)) for img in imgs]
            merge_photo = np.concatenate((imgs[0], bev_map), axis=0)
        elif len(imgs) == 2:
            """
                    ego_car  ego_car  center_fov120
                    ego_car  ego_car  center fov30
            """
            resize_w = bev_w // 2
            resize_h = int(img_h * (resize_w / img_w))
            imgs = [cv2.resize(img, (resize_w, resize_h)) for img in imgs]
            merge_photo = np.concatenate([np.concatenate(imgs, axis=1), bev_map], axis=0)  # center front fov120, fov30
        elif len(imgs) == 7:
            """
                    left_rear left_front
                rear ego_car  ego_car  center_fov120
                        ego_car  ego_car  center fov30
                    right_rear right_front
            """
            resize_w = bev_w // 2
            resize_h = int(img_h * (resize_w / img_w))
            imgs = [cv2.resize(img, (resize_w, resize_h)) for img in imgs]
            board_h, board_w = 2 * resize_h + bev_h, 4 * resize_w
            merge_photo = np.zeros((board_h, board_w, 3), dtype=np.uint8)
            merge_photo[0:resize_h, board_w//2-resize_w:board_w//2+resize_w] = np.concatenate([imgs[3], imgs[2]], axis=1)  # left rear, left front
            merge_photo[board_h//2-resize_h:board_h//2+resize_h, board_w-resize_w:] = np.concatenate(imgs[0:2], axis=0)  # center front fov120, fov30
            merge_photo[board_h-resize_h:, board_w//2-resize_w:board_w//2+resize_w] = np.concatenate([imgs[6], imgs[5]], axis=1)  # right rear, right front
            merge_photo[board_h//2-resize_h//2:board_h//2+resize_h//2, 0:resize_w] = imgs[4]  # rear
            merge_photo[board_h//2-bev_h//2:board_h//2+bev_h//2, board_w//2-bev_w//2:board_w//2+bev_w//2] = bev_map
        cv2.imwrite(os.path.join(save_path,'{}.png'.format(frame_name)), merge_photo)


def base_merge(seq_path_list):
    if type(seq_path_list) is not list:
        seq_path_list = [seq_path_list]
    for seq_path in seq_path_list:
        print(f'process {seq_path}')
        bev_path = os.path.join(seq_path, 'bev')
        img_path_list = [os.path.join(seq_path, f'{cam}_cam') for cam in cam_list]
        save_path = os.path.join(seq_path, 'merge')
        os.makedirs(save_path, exist_ok=True)
        merge_img(bev_path, img_path_list, save_path)
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
        base_merge(seq_path)
    else:
        seq_path_list = [os.path.join(seq_path, seq) for seq in sorted(os.listdir(seq_path))]
        if thread_num > 1:
            n_list = split_list_n_list(seq_path_list, thread_num)
            process_list = []
            for cur_list in n_list:
                t = Process(target=base_merge, args=(cur_list, ))
                t.start()
                process_list.append(t)
            for t in process_list: # 等待所有进程运行完毕
                t.join()
        else:
            base_merge(seq_path_list)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='multi')
    parser.add_argument('--id', type=int, default=1, help='epoch id')
    args = parser.parse_args()
    args.id = 25

    seq_path = f'unknown_vis_lustrenew/fusion_A02_3dgop_seg_det_2V_V1.9_No1_rainy_batch_10/det_vis/fusion'
    cam_list = ['center_camera_fov120', 'center_camera_fov30']
    
    main(one_seq=False, thread_num=20)

