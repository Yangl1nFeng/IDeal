"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
# from imageio import imread
import random
import json

import h5py
import nltk
import csv
import io
import pandas as pd

import logging

logger = logging.getLogger(__name__)

def process_image(data_split, image):
        
        if data_split == 'train':  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image[np.where(rand_list < 0.12)] = 1e-8
            return image
        else:
            return image
        
class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train):
        self.tokenizer = tokenizer
        self.opt = opt
        self.data_split = data_split
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        loc_cap = ''
        loc_image = ''

        self.is_transfering = opt.is_transfering

        # Captions
        self.captions = []
        self.memory = []
        self.scene_ids = []
                    
        if 'scanrefer_train' == opt.data_name:
            self.pos = None
            self.sample_list = []

            if data_split == 'train':
                self.images = np.load('./lib/data/pt2vec_200_random_train.npy')  
                pos_temp = np.load('./lib/data/pt2vec_200_random_pos_train.npy')
            else:
                self.images = np.load('./lib/data/pt2vec_200_random_val.npy')
                pos_temp = np.load('./lib/data/pt2vec_200_random_pos_val.npy')
            if data_split == 'train':
                self.txt = open("./lib/data/split/ScanRefer_filtered_train.txt", "r", encoding="utf-8")
                for line in self.txt.readlines():
                    name = line.replace('\n', '')
                    self.sample_list.append(name)
                if self.is_transfering:
                    with open("./lib/data/text/scanrefer/scanrefer_merged_data_8_9.jsonl", "r", encoding="utf-8") as file:
                        json_content = [json.loads(line) for line in file]
                else:
                    with open("./lib/data/text/scanrefer/ori_data.jsonl", "r", encoding="utf-8") as file:
                        json_content = [json.loads(line) for line in file]    
                for i in range(len(json_content)):
                    self.captions.append(json_content[i]['description'])
                    self.memory.append(json_content[i]['memory'])
                    self.scene_ids.append(json_content[i]['scene_id'])
            else:
                self.txt = open("./lib/data/split/ScanRefer_filtered_val.txt", "r", encoding="utf-8")
                for line in self.txt.readlines():
                    name = line.replace('\n', '')
                    self.sample_list.append(name)
                json_file = open("./lib/data/text/scanrefer/ScanRefer_filtered_val_with_memory.json", "r", encoding="utf-8")
                self.pt_div = 10
                json_content = json.load(json_file)
                content_len = []
                for i in range(len(json_content)):
                    content_len.append(len(json_content[i]['token']))
                
                content_len = np.array(content_len)
                index = (-content_len).argsort()

                new_json_content = []

                for i in range(len(json_content)):
                    new_json_content.append(json_content[index[i]])
                
                for i in range(len(self.sample_list)):
                    count = 0
                    k = 0
                    txt = ''
                    memo = ''
                    for j in range(len(new_json_content)):
                        if count < self.pt_div: 
                            if new_json_content[j]['scene_id'] == self.sample_list[i]:
                                if k < 1:
                                    txt = txt + new_json_content[j]['description']
                                    memo = memo + new_json_content[j]['memory']
                                    k = k + 1
                                if k >= 1:
                                    self.captions.append(txt.strip())
                                    self.memory.append(memo.strip())
                                    self.scene_ids.append(new_json_content[j]['scene_id'])
                                    count = count + 1
                                    k = 0
                                    txt = ''
                                    memo = ''
                        if j >= len(new_json_content)-1 and count < self.pt_div:
                            for i in range(self.pt_div-count):
                                self.captions.append(self.captions[-1])
                                self.memory.append(self.memory[-1])
                                self.scene_ids.append(self.scene_ids[-1])
            
            x_coords = pos_temp[:, :, 0]
            y_coords = pos_temp[:, :, 1]

            x_min = x_coords.min(axis=1, keepdims=True)
            x_max = x_coords.max(axis=1, keepdims=True)
            x_normalized = (x_coords - x_min) / (x_max - x_min)

            y_min = y_coords.min(axis=1, keepdims=True)
            y_max = y_coords.max(axis=1, keepdims=True)
            y_normalized = (y_coords - y_min) / (y_max - y_min)

            self.pos = pos_temp.copy()
            self.pos[:, :, 0] = x_normalized
            self.pos[:, :, 1] = y_normalized
        
        elif 'scanrefer_memory' == opt.data_name:
            self.pos = None
            self.sample_list = []
            if data_split == 'train':
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_train.npy')   # bs * imagedim
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_pos_train.npy')
            else:
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_val.npy')
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_pos_val.npy')
            if data_split == 'train':
                self.txt = open("../../vsepp-python3/data/ScanNet/ScanRefer_filtered_train.txt", "r", encoding="utf-8")
                json_file = open("/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/ScanRefer_filtered_train_updated.json", "r", encoding="utf-8")
            else:
                self.txt = open("../../vsepp-python3/data/ScanNet/ScanRefer_filtered_val.txt", "r", encoding="utf-8")
                json_file = open("/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/ScanRefer_filtered_val_with_memory.json", "r", encoding="utf-8")

            for line in self.txt.readlines():
                name = line.replace('\n', '')
                self.sample_list.append(name)
            
            x_coords = pos_temp[:, :, 0]
            y_coords = pos_temp[:, :, 1]

            x_min = x_coords.min(axis=1, keepdims=True)
            x_max = x_coords.max(axis=1, keepdims=True)
            x_normalized = (x_coords - x_min) / (x_max - x_min)

            y_min = y_coords.min(axis=1, keepdims=True)
            y_max = y_coords.max(axis=1, keepdims=True)
            y_normalized = (y_coords - y_min) / (y_max - y_min)

            self.pos = pos_temp.copy()
            self.pos[:, :, 0] = x_normalized
            self.pos[:, :, 1] = y_normalized

            json_content = json.load(json_file)
            self.pt_div = 10

            content_len = []
            for i in range(len(json_content)):
                content_len.append(len(json_content[i]['token']))
            
            content_len = np.array(content_len)
            index = (-content_len).argsort()

            new_json_content = []

            for i in range(len(json_content)):
                new_json_content.append(json_content[index[i]])
            
            for i in range(len(self.sample_list)):
                count = 0
                k = 0
                txt = ''
                memo = ''
                for j in range(len(new_json_content)):
                    if count < self.pt_div: 
                        if new_json_content[j]['scene_id'] == self.sample_list[i]:
                            if k < 1:
                                txt = txt + new_json_content[j]['description']
                                memo = memo + new_json_content[j]['memory']
                                k = k + 1
                            elif k >= 1:
                                self.captions.append(memo.strip())
                                self.memory.append(memo.strip())
                                self.scene_ids.append(new_json_content[j]['scene_id'])
                                count = count + 1
                                k = 0
                                txt = ''
                                memo = ''
                    if j >= len(new_json_content)-1 and count < self.pt_div:
                        for i in range(self.pt_div-count):
                            self.captions.append(self.memory[-1])
                            self.memory.append(self.memory[-1])
                            self.scene_ids.append(self.scene_ids[-1])

        elif 'nr3d' == opt.data_name:
            self.pos = None
            if data_split == 'train':
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_train.npy')   # bs * imagedim
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_pos_train.npy')
            else:
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_val.npy')
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_pos_val.npy')

            self.sample_list = []
            print("Dealing Nr3D csv file!")
            df = pd.read_csv('../../vsepp-python3/data/ScanNet/nr3d.csv')

            if data_split == 'train':
                self.txt = open("../../vsepp-python3/data/ScanNet/ScanRefer_filtered_train.txt", "r", encoding="utf-8")
                json_file = open("../../vsepp-python3/data/ScanNet/ScanRefer_filtered_train.json", "r", encoding="utf-8")
            else:
                self.txt = open("../../vsepp-python3/data/ScanNet/ScanRefer_filtered_val.txt", "r", encoding="utf-8")
                json_file = open("../../vsepp-python3/data/ScanNet/ScanRefer_filtered_val.json", "r", encoding="utf-8")

            for line in self.txt.readlines():
                name = line.replace('\n', '')
                self.sample_list.append(name)

            x_coords = pos_temp[:, :, 0]
            y_coords = pos_temp[:, :, 1]

            x_min = x_coords.min(axis=1, keepdims=True)
            x_max = x_coords.max(axis=1, keepdims=True)
            x_normalized = (x_coords - x_min) / (x_max - x_min)

            y_min = y_coords.min(axis=1, keepdims=True)
            y_max = y_coords.max(axis=1, keepdims=True)
            y_normalized = (y_coords - y_min) / (y_max - y_min)

            self.pos = pos_temp.copy()
            self.pos[:, :, 0] = x_normalized
            self.pos[:, :, 1] = y_normalized

            self.pt_div = 10

            print("Making Nr3D txt!")
            for i in range(len(self.sample_list)):
                temp = df.loc[df["scan_id"] == self.sample_list[i],:]
                if temp.shape[0] < (1*self.pt_div):
                    itr = temp.shape[0] // 1
                    for juzi_count in range(int(itr)):
                        txt = ''
                        memo = ''
                        for k in range(1):
                            if temp.iloc[juzi_count * 1 + k]['utterance'].strip()[-1] !='.' and k != 0:
                                txt = txt + '. ' + temp.iloc[juzi_count * 1 + k]['utterance'].strip()
                            else:
                                txt = txt + ' ' + temp.iloc[juzi_count * 1 + k]['utterance'].strip()

                        self.captions.append(txt)
                        self.memory.append(memo)
                        self.scene_ids.append(temp.iloc[juzi_count * 1 + k]['scan_id'])
                    for juzi_count in range(self.pt_div-int(itr)):
                        self.captions.append(self.captions[-1].strip())
                        self.memory.append(self.memory[-1])
                        self.scene_ids.append(self.scene_ids[-1])
                else:
                    for juzi_count in range(self.pt_div):
                        txt = ''
                        memo = ''
                        for k in range(1):
                            if temp.iloc[juzi_count * 1 + k]['utterance'].strip()[-1] != '.' and k != 0:
                                txt = txt + '. ' + temp.iloc[juzi_count * 1 + k]['utterance'].strip()
                            else:
                                txt = txt + ' ' + temp.iloc[juzi_count * 1 + k]['utterance'].strip()
                        self.captions.append(txt)
                        self.memory.append(memo)
                        self.scene_ids.append(temp.iloc[juzi_count * 1 + k]['scan_id'])
        elif 'nr3d_memory' == opt.data_name:
            self.pos = None
            if data_split == 'train':
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_train.npy')   # bs * imagedim
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_pos_train.npy')
            else:
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_val.npy')
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_200_random_pos_val.npy')

            self.sample_list = []
            print("Dealing Nr3D csv file!")
            df = pd.read_csv('/home/fengyanglin/ESA-main/ESA_BIGRU/nr3d_with_memory.csv')

            if data_split == 'train':
                self.txt = open("../../vsepp-python3/data/ScanNet/ScanRefer_filtered_train.txt", "r", encoding="utf-8")
            else:
                self.txt = open("../../vsepp-python3/data/ScanNet/ScanRefer_filtered_val.txt", "r", encoding="utf-8")
            
            if data_split == 'train':
                self.txt = open("/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/nr3d_train_ids.txt", "r", encoding="utf-8")
                # with open("/home/fengyanglin/ESA-main/ESA_BIGRU/nr3d_ori_data.jsonl", "r", encoding="utf-8") as file:
                with open("/home/fengyanglin/ESA-main/ESA_BIGRU/nr3d_combined.jsonl", "r", encoding="utf-8") as file:
                    json_content = [json.loads(line) for line in file]    
                for i in range(len(json_content)):
                    self.captions.append(json_content[i]['description'])
                    self.memory.append(json_content[i]['memory'])
                    self.scene_ids.append(json_content[i]['scene_id'])
                    
            for line in self.txt.readlines():
                name = line.replace('\n', '')
                self.sample_list.append(name)

            x_coords = pos_temp[:, :, 0]
            y_coords = pos_temp[:, :, 1]

            x_min = x_coords.min(axis=1, keepdims=True)
            x_max = x_coords.max(axis=1, keepdims=True)
            x_normalized = (x_coords - x_min) / (x_max - x_min)

            y_min = y_coords.min(axis=1, keepdims=True)
            y_max = y_coords.max(axis=1, keepdims=True)
            y_normalized = (y_coords - y_min) / (y_max - y_min)

            self.pos = pos_temp.copy()
            self.pos[:, :, 0] = x_normalized
            self.pos[:, :, 1] = y_normalized

            self.pt_div = 10

            print("Making Nr3D txt!")
            if data_split != 'train':
                for i in range(len(self.sample_list)):
                    temp = df.loc[df["scan_id"] == self.sample_list[i],:]
                    if temp.shape[0] < (1*self.pt_div):
                        itr = temp.shape[0] // 1
                        for juzi_count in range(int(itr)):
                            txt = ''
                            memo = ''
                            for k in range(1):
                                if temp.iloc[juzi_count * 1 + k]['utterance'].strip()[-1] !='.' and k != 0:
                                    txt = txt + '. ' + temp.iloc[juzi_count * 1 + k]['utterance'].strip()
                                else:
                                    txt = txt + ' ' + temp.iloc[juzi_count * 1 + k]['utterance'].strip()
                            self.memory.append(temp.iloc[juzi_count * 1 + k]['memory'].strip())
                            self.captions.append(temp.iloc[juzi_count * 1 + k]['memory'].strip())
                            self.scene_ids.append(temp.iloc[juzi_count * 1 + k]['scan_id'])
                        for juzi_count in range(self.pt_div-int(itr)):
                            self.memory.append(self.memory[-1].strip())
                            self.captions.append(self.memory[-1].strip())
                            self.scene_ids.append(self.scene_ids[-1])
                    else:
                        for juzi_count in range(self.pt_div):
                            txt = ''
                            for k in range(1):
                                if temp.iloc[juzi_count * 1 + k]['utterance'].strip()[-1] != '.' and k != 0:
                                    txt = txt + '. ' + temp.iloc[juzi_count * 1 + k]['utterance'].strip()
                                else:
                                    txt = txt + ' ' + temp.iloc[juzi_count * 1 + k]['utterance'].strip()
                            self.memory.append(temp.iloc[juzi_count * 1 + k]['memory'].strip())
                            self.captions.append(temp.iloc[juzi_count * 1 + k]['memory'].strip())
                            self.scene_ids.append(temp.iloc[juzi_count * 1 + k]['scan_id'])

        elif 'nr3d_train' == opt.data_name:
            self.pos = None
            if data_split == 'train':
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_nr3d_random_train.npy')   # bs * imagedim
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_nr3d_random_pos_train.npy')
            else:
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_nr3d_random_val.npy')
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_nr3d_random_pos_val.npy')

            self.sample_list = []
            print("Dealing Nr3D csv file!")
            df = pd.read_csv('../../vsepp-python3/data/ScanNet/nr3d.csv')
            self.pt_div = 10

            if data_split == 'train':
                self.txt = open("/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/nr3d_train_ids.txt", "r", encoding="utf-8")
                # with open("/home/fengyanglin/ESA-main/ESA_BIGRU/nr3d_ori_data.jsonl", "r", encoding="utf-8") as file:
                with open("/home/fengyanglin/ESA-main/ESA_BIGRU/nr3d_combined.jsonl", "r", encoding="utf-8") as file:
                    json_content = [json.loads(line) for line in file]    
                for i in range(len(json_content)):
                    self.captions.append(json_content[i]['description'])
                    self.memory.append(json_content[i]['memory'])
                    self.scene_ids.append(json_content[i]['scene_id'])
            else:
                self.txt = open("/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/nr3d_val_ids.txt", "r", encoding="utf-8")
                json_file = open("/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/nr3d_dataset/nr3d_val_extracted.json", "r", encoding="utf-8")

                for line in self.txt.readlines():
                    name = line.replace('\n', '')
                    self.sample_list.append(name)

                json_content = json.load(json_file)
                for scene_name in self.sample_list:
                    for item in json_content:
                        if item.get('scene_id') == scene_name:
                            self.captions.append(item.get('description'))
                            self.memory.append(item.get('memory'))
                            self.scene_ids.append(item.get('scene_id'))

            x_coords = pos_temp[:, :, 0]
            y_coords = pos_temp[:, :, 1]

            x_min = x_coords.min(axis=1, keepdims=True)
            x_max = x_coords.max(axis=1, keepdims=True)
            x_normalized = (x_coords - x_min) / (x_max - x_min)

            y_min = y_coords.min(axis=1, keepdims=True)
            y_max = y_coords.max(axis=1, keepdims=True)
            y_normalized = (y_coords - y_min) / (y_max - y_min)

            self.pos = pos_temp.copy()
            self.pos[:, :, 0] = x_normalized
            self.pos[:, :, 1] = y_normalized


        elif 'sr3d_train' == opt.data_name:
            self.pos = None
            if data_split == 'train':
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_sr3d_random_train.npy')   # bs * imagedim
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_sr3d_random_pos_train.npy')
            else:
                self.images = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_sr3d_random_val.npy')
                pos_temp = np.load('/home/fengyanglin/dgcnn.pytorch-master/our_ijcv/pt2vec_sr3d_random_pos_val.npy')

            self.sample_list = []
            print("Dealing Sr3D csv file!")
            # df = pd.read_csv('/home/fengyanglin/ESA-main/ESA_BIGRU/nr3d_with_memory.csv')
            # json_file = pd.read_csv('/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/nr3d_data_self_memory_tokenized.json')

            if data_split == 'train':
                self.txt = open("/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/sr3d_train_ids.txt", "r", encoding="utf-8")
                # with open("/home/fengyanglin/ESA-main/ESA_BIGRU/sr3d_ori_data.jsonl", "r", encoding="utf-8") as file:
                with open("/home/fengyanglin/ESA-main/ESA_BIGRU/sr3d_combined.jsonl", "r", encoding="utf-8") as file:
                    json_content = [json.loads(line) for line in file]    
                for i in range(len(json_content)):
                    self.captions.append(json_content[i]['description'])
                    self.memory.append(json_content[i]['memory'])
                    self.scene_ids.append(json_content[i]['scene_id'])
            else:
                self.txt = open("/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/sr3d_val_ids.txt", "r", encoding="utf-8")
                json_file = open("/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/sr3d_dataset/sr3d_val_extracted.json", "r", encoding="utf-8")

                for line in self.txt.readlines():
                    name = line.replace('\n', '')
                    self.sample_list.append(name)

                json_content = json.load(json_file)
                for scene_name in self.sample_list:
                    for item in json_content:
                        if item.get('scene_id') == scene_name:
                            self.captions.append(item.get('description'))
                            self.memory.append(item.get('memory'))
                            self.scene_ids.append(item.get('scene_id'))

            x_coords = pos_temp[:, :, 0]
            y_coords = pos_temp[:, :, 1]

            x_min = x_coords.min(axis=1, keepdims=True)
            x_max = x_coords.max(axis=1, keepdims=True)
            x_normalized = (x_coords - x_min) / (x_max - x_min)

            y_min = y_coords.min(axis=1, keepdims=True)
            y_max = y_coords.max(axis=1, keepdims=True)
            y_normalized = (y_coords - y_min) / (y_max - y_min)

            self.pos = pos_temp.copy()
            self.pos[:, :, 0] = x_normalized
            self.pos[:, :, 1] = y_normalized


        self.length = len(self.captions)
        print('txt len:' + str(self.length))
        num_images = len(self.images)
        print('image len:' + str(num_images))


        if '3DLLM' == opt.data_name:
            self.im_div = 2
        else:
            self.im_div = 10

    def __getitem__(self, index):
        
        caption = self.captions[index]
        memory = self.memory[index]
        # memory = self.captions[index]
        if self.data_split == 'train':
            if self.is_transfering:
                img_id = index // 50
            else:
                img_id = index // 10
        else:
            img_id = index // 10
        scene_id = self.scene_ids[index]
        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)
        
        target = process_caption(self.tokenizer, caption_tokens, train=False)
        images = self.images[img_id]
        pos = self.pos[img_id]
        images = torch.Tensor(images)
        pos = torch.Tensor(pos)
        return images, target, index, img_id, pos, caption, memory, scene_id
        
    def __len__(self):
        return self.length

def process_caption(tokenizer, tokens, train=True, prob_i=0.2, tfidf_weights=None):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        # 计算 TF-IDF 权重，低权重 token 更容易被 mask
        if tfidf_weights:
            importance = tfidf_weights.get(token, 1.0)
            prob_i = prob_i * (1.5 - importance)  # 低权重词更容易被替换

        if prob < prob_i and train:
            prob /= prob_i
            
            if prob < 0.5:  # 50% 概率 mask
                output_tokens.extend(["[MASK]"] * len(sub_tokens))
            elif prob < 0.7:  # 20% 概率随机替换
                output_tokens.extend([random.choice(list(tokenizer.vocab.keys())) for _ in sub_tokens])
            else:  # 30% 概率保留原词
                output_tokens.extend(sub_tokens)
                deleted_idx.append(len(output_tokens) - 1)
        else:
            output_tokens.extend(sub_tokens)

    if deleted_idx:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    return torch.tensor(target)

def collate_fn_test(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions, ids, img_ids, pos, raw_captions, memory, scene_id = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Merge images
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)
        # Merget captions
        lengths = [len(cap) for cap in captions]
        # targets = torch.zeros(len(captions), max(lengths)).long()
        targets = torch.zeros(len(captions), 500).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            if end<500:
                targets[i, :end] = cap[:end]
            else:
                targets[i, :500] = cap[:500]

        all_pos = torch.zeros(len(pos),len(pos[0]),3)
        for i, p in enumerate(pos):
            all_pos[i] = torch.Tensor(pos[i])
        
        return all_images, img_lengths, targets, torch.Tensor(lengths), ids, all_pos, raw_captions, memory, scene_id

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions, img_lens, cap_lens, ids, img_ids, pos, raw_captions, memory = zip(*data)
    if len(images[0].shape) == 2:  # region feature
        # Merge images
        img_lengths = [len(image) for image in images]
        all_images = torch.zeros(len(images), max(img_lengths), images[0].size(-1))
        for i, image in enumerate(images):
            end = img_lengths[i]
            all_images[i, :end] = image[:end]
        img_lengths = torch.Tensor(img_lengths)
        # Merget captions
        cap_len = torch.Tensor(cap_lens).long()
        # all_captions = torch.zeros(len(captions),len(cap_len[0]),cap_len.max()).long()
        all_captions = torch.zeros(len(captions),len(cap_len[0]),500).long()
        for i,cap in enumerate(captions):
            for j,index in enumerate(cap_lens[i]):
                end = index
                all_captions[i,j,:end] = cap[j][:end]
        cap_lens = torch.Tensor(cap_lens)
        
        all_pos = torch.zeros(len(pos),len(pos[0]),3)
        for i, p in enumerate(pos):
            all_pos[i] = torch.Tensor(pos[i])
    
        return all_images, img_lengths, all_captions, cap_lens, ids, all_pos, raw_captions, memory

def get_loader(data_path, data_name, data_split, vocab, opt, batch_size=100,
               shuffle=True, num_workers=0, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        dset = PrecompRegionDataset(data_path, data_name, data_split, vocab, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn_test,
                                                num_workers=0,
                                                drop_last=True)
    else:
        dset = PrecompRegionDataset(data_path, data_name, data_split, vocab, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                pin_memory=True,
                                                collate_fn=collate_fn_test,
                                                num_workers=0,
                                                drop_last=True)
        
    return data_loader

def get_loaders(data_path, data_name, vocab, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', vocab, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, data_name, 'dev', vocab, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader

def get_test_loader(split_name, data_name, vocab, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, vocab, opt,
                             batch_size, False, workers, train=False)
    return test_loader
