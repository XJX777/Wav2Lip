import torch
import numpy as np
import json
import random
import cv2

from torch.utils.data import Dataset
import pdb

def get_data(json_name,augment_num):
    print('start loading data')
    with open(json_name,'r') as f:
        data_dic = json.load(f)
    data_dic_name_list = []
    for augment_index in range(augment_num):
        for video_name in data_dic.keys():
            data_dic_name_list.append(video_name)
    random.shuffle(data_dic_name_list)
    data_dic_name_list = data_dic_name_list[:4000]
    print('finish loading')
    return data_dic_name_list,data_dic


class DINetDataset(Dataset):
    def __init__(self,path_json,augment_num,mouth_region_size):
        super(DINetDataset, self).__init__()
        self.data_dic_name_list,self.data_dic = get_data(path_json,augment_num)
        self.mouth_region_size = mouth_region_size
        self.radius = mouth_region_size//2
        self.radius_1_4 = self.radius//4
        self.img_h = self.radius * 3 + self.radius_1_4
        self.img_w = self.radius * 2 + self.radius_1_4 * 2
        self.length = len(self.data_dic_name_list)

    def __getitem__(self, index):
        video_name = self.data_dic_name_list[index]
        video_clip_num = len(self.data_dic[video_name]['clip_data_list'])
        # pdb.set_trace()
        source_anchor, wrong_anchor = random.sample(range(video_clip_num), 2)[0], random.sample(range(video_clip_num), 2)[1]
        source_image_path_list = self.data_dic[video_name]['clip_data_list'][source_anchor]['frame_path_list']
        source_clip_list = []
        for source_frame_index in range(2, 2 + 5):
            ## load source clip
            # print("source img_path :   ", source_image_path_list[source_frame_index])
            source_image_data = cv2.imread('.' + source_image_path_list[source_frame_index])[:, :, ::-1]
            source_image_data = cv2.resize(source_image_data, (self.img_w, self.img_h)) / 255.0
            source_clip_list.append(source_image_data)
            
        wrong_image_path_list = self.data_dic[video_name]['clip_data_list'][wrong_anchor]['frame_path_list']
        wrong_clip_list = []
        for wrong_frame_index in range(2, 2 + 5):
            ## load source clip
            # print("wrong img_path :   ", wrong_image_path_list[wrong_frame_index])
            wrong_image_data = cv2.imread('.' + wrong_image_path_list[wrong_frame_index])[:, :, ::-1]
            wrong_image_data = cv2.resize(wrong_image_data, (self.img_w, self.img_h)) / 255.0
            wrong_clip_list.append(wrong_image_data)

        source_clip = np.stack(source_clip_list, 0)
        wrong_clip = np.stack(wrong_clip_list, 0)
        deep_speech_full = np.array(self.data_dic[video_name]['clip_data_list'][source_anchor]['deep_speech_list'])

        # # display data
        # display_source = np.concatenate(source_clip_list,1)
        # display_source_mask = np.concatenate(source_clip_mask_list,1)
        # display_reference0 = np.concatenate([reference_clip_list[0][:,:,:3],reference_clip_list[0][:,:,3:6],reference_clip_list[0][:,:,6:9],
        #                                 reference_clip_list[0][:,:,9:12],reference_clip_list[0][:,:,12:15]],1)
        # display_reference1 = np.concatenate([reference_clip_list[1][:, :, :3], reference_clip_list[1][:, :, 3:6],
        #                                 reference_clip_list[1][:, :, 6:9],
        #                                 reference_clip_list[1][:, :, 9:12], reference_clip_list[1][:, :, 12:15]],1)
        # display_reference2 = np.concatenate([reference_clip_list[2][:, :, :3], reference_clip_list[2][:, :, 3:6],
        #                                 reference_clip_list[2][:, :, 6:9],
        #                                 reference_clip_list[2][:, :, 9:12], reference_clip_list[2][:, :, 12:15]],1)
        # display_reference3 = np.concatenate([reference_clip_list[3][:, :, :3], reference_clip_list[3][:, :, 3:6],
        #                                 reference_clip_list[3][:, :, 6:9],
        #                                 reference_clip_list[3][:, :, 9:12], reference_clip_list[3][:, :, 12:15]],1)
        # display_reference4 = np.concatenate([reference_clip_list[4][:, :, :3], reference_clip_list[4][:, :, 3:6],
        #                                 reference_clip_list[4][:, :, 6:9],
        #                                 reference_clip_list[4][:, :, 9:12], reference_clip_list[4][:, :, 12:15]],1)
        # merge_img = np.concatenate([display_source,display_source_mask,
        #                             display_reference0,display_reference1,display_reference2,display_reference3,
        #                             display_reference4],0)
        # cv2.imshow('test',(merge_img[:,:,::-1] * 255).astype(np.uint8))
        # cv2.waitKey(-1)

        # # 2 tensor
        source_clip = torch.from_numpy(source_clip).float().permute(0, 3, 1, 2)
        wrong_clip = torch.from_numpy(wrong_clip).float().permute(0, 3, 1, 2)
        deep_speech_full = torch.from_numpy(deep_speech_full).permute(1, 0)
        # print("source_clip.shape: ", source_clip.shape)
        # print("wrong_clip.shape: ", wrong_clip.shape)
        # print("deep_speech_full.shape: ", deep_speech_full.shape)
        if random.choice([True, False]):
            return source_clip, deep_speech_full, torch.ones(1, 8, 8, dtype=torch.float)
        else:
            return wrong_clip, deep_speech_full, torch.zeros(1, 8, 8, dtype=torch.float)

    def __len__(self):
        return self.length
