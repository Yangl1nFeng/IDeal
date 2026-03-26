import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
# from imageio import imread
import random
import json
import csv
import io
import pandas as pd

import logging

logger = logging.getLogger(__name__)

def process_image(data_split, image):
        
        if data_split == 'train':  # Size augmentation on region features.
            num_features = image.shape[0]
            rand_list = np.random.rand(num_features)
            image[np.where(rand_list < 0.2)] = 1e-8
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

        # Captions
        self.captions = []
        self.memory = []
        self.scene_ids = []
    
        if opt.data_name == 'cc152k':
            cap_p = osp.join(loc_cap, '%s_caps.tsv' % data_split)
            with io.open(cap_p) as f:
                tsvreader = csv.reader(f, delimiter='\t')
                for line in tsvreader:
                    self.captions.append(line[1].strip())
                    
        elif 'scanrefer' == opt.data_name:
            self.pos = None
            self.sample_list = []
            if data_split == 'train':
                self.images = np.load('./lib/data/pt2vec_200_random_pos_train.npy')   # bs * imagedim
                pos_temp = np.load('./lib/data/pt2vec_200_random_pos_train.npy')
            else:
                self.images = np.load('./lib/data/pt2vec_200_random_val.npy')
                pos_temp = np.load('./lib/data/pt2vec_200_random_pos_val.npy')
            if data_split == 'train':
                self.txt = open("./lib/data/split/ScanRefer_filtered_train.txt", "r", encoding="utf-8")
                json_file = open("./lib/data/text/scanrefer/ori_data.jsonl", "r", encoding="utf-8")
            else:
                self.txt = open("./lib/data/split/ScanRefer_filtered_val.txt", "r", encoding="utf-8")
                json_file = open("./lib/data/split/ScanRefer_filtered_val_with_memory.json", "r", encoding="utf-8")

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

        elif 'scanrefer_ori' == opt.data_name:
            self.pos = None
            self.sample_list = []
            if data_split == 'train':
                self.images = np.load('./lib/data/pt2vec_200_random_pos_train.npy')   # bs * imagedim
                pos_temp = np.load('./lib/data/pt2vec_200_random_pos_train.npy')
            else:
                self.images = np.load('./lib/data/pt2vec_200_random_val.npy')
                pos_temp = np.load('./lib/data/pt2vec_200_random_pos_val.npy')
            if data_split == 'train':
                self.txt = open("./lib/data/split/ScanRefer_filtered_train.txt", "r", encoding="utf-8")
                json_file = open("./lib/data/text/scanrefer/ori_data.jsonl", "r", encoding="utf-8")
            else:
                self.txt = open("./lib/data/split/ScanRefer_filtered_val.txt", "r", encoding="utf-8")
                json_file = open("./lib/data/text/scanrefer/ScanRefer_val_self_memory_8_9.json", "r", encoding="utf-8")

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

            new_json_content = json.load(json_file)
            self.pt_div = 10
            
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
            with open("captions.txt", "w", encoding="utf-8") as f:
                for caption in self.captions:
                    f.write(caption + "\n") 
            
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
        img_id = index // 10
        scene_id = self.scene_ids[index]

        caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption)

        target = process_caption(self.tokenizer, caption_tokens, False)
        images = self.images[img_id]
        pos = self.pos[img_id]
        images = torch.Tensor(images)
        pos = torch.Tensor(pos)
        return images, target, index, img_id, pos, caption, memory, scene_id
        
    def __len__(self):
        return self.length

def process_caption(tokenizer, tokens, train=True,prob_i=0.2):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob <prob_i and train:  # mask/remove the tokens only during training
            prob /= prob_i
            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    return target

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
    images, captions,  ids, img_ids, pos, raw_captions, memory, scene_id = zip(*data)
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
            targets[i, :end] = cap[:end]

        all_pos = torch.zeros(len(pos),len(pos[0]),3)
        for i, p in enumerate(pos):
            all_pos[i] = torch.Tensor(pos[i])
        
        return all_images, img_lengths, targets, lengths, ids, all_pos, raw_captions, memory, scene_id

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
                                                shuffle=False,
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
