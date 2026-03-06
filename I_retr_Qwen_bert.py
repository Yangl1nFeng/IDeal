"""Training script"""
import os
import torch
from transformers import BertTokenizer
from lib.datasets import I_image_caption_bert
from lib.vse_bert import VSEModel
from lib.evaluation import i2t, t2i, encode_data, compute_sim

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from tqdm import tqdm
import logging
import json
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

import arguments

print(torch.__version__)
print(transformers.__version__)

TOKENIZERS_PARALLELISM=False
MODEL_ID = "Qwen/Qwen-7B-Instruct"
MODEL_NAME = MODEL_ID.split("/")[-1]
CONTEXT_LENGTH = 3000  
MODEL_DIR = "./lib/LLMs/qwen_7b"


def process_caption(tokenizer, tokens, train=True,prob_i=0.2):
    output_tokens = []
    deleted_idx = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            # no masking token (will be ignored by loss function later)
            output_tokens.append(sub_token)

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)

    if len(target) < 500:
        target = target
    else:
        target = target[:500]
    return target


import numpy as np
from scipy.spatial.distance import cosine
from itertools import combinations

def spherical_mean(vectors):
    vec = np.mean(vectors, axis=0)
    return vec / np.linalg.norm(vec)

def meb_cosine_fusion(answer_embs, embs_stat, lambda_=0.5):
    fused_embs = answer_embs.copy()
    n, num_views, dim = answer_embs.shape

    for i in range(num_views - 1):
        mask = embs_stat[:, i] == 0
        fused_embs[mask, i + 1] = lambda_ * fused_embs[mask, i] + (1 - lambda_) * fused_embs[mask, i + 1]

    final_embs = np.zeros((n, dim))

    for i in range(n):
        valid_embs = fused_embs[i][embs_stat[i] == 1]

        if len(valid_embs) > 2:
            valid_embs = valid_embs / np.linalg.norm(valid_embs, axis=1, keepdims=True)
            pair_dists = []
            for idx1, idx2 in combinations(range(len(valid_embs)), 2):
                dist = cosine(valid_embs[idx1], valid_embs[idx2])
                pair_dists.append((dist, idx1, idx2))

            top3_pairs = sorted(pair_dists, key=lambda x: -x[0])[:1]

            centers = []
            non_circular_points = []

            for dist, idx1, idx2 in top3_pairs:
                v1, v2 = valid_embs[idx1], valid_embs[idx2]
                center = spherical_mean(np.array([v1, v2]))
                centers.append(center)
                radius = dist 

                for j, v in enumerate(valid_embs):
                    d = cosine(center, v)
                    if d < 0.8 * radius:
                        non_circular_points.append(v)

            fusion_parts = centers.copy()
            if non_circular_points:
                other_avg = spherical_mean(non_circular_points)
                fusion_parts.append(other_avg)

            final_embs[i] = spherical_mean(fusion_parts)

        elif len(valid_embs) > 0:
            final_embs[i] = spherical_mean(valid_embs)
        else:
            final_embs[i] = np.zeros(dim)

    return final_embs

def compute_sim(a, b):
    return a @ b.T


def I_t2i(img_embs, cap_embs, answer_embs, etps, threshold=2.75):

    ori_sims = compute_sim(img_embs, cap_embs)
    tags = np.load('./temp_results/8_9/self_scanrefer/bert_tags_5.npy')
    sum_emb = np.load('./temp_results/8_9/self_scanrefer/summarized_features.npy')
    valid_mask = tags.astype(bool)  
    answer_emb = meb_cosine_fusion(answer_embs[:,1:3,:], valid_mask[:,1:3], lambda_=0.5)

    new_sims = compute_sim(img_embs, answer_emb)
    sum_sims = compute_sim(img_embs, sum_emb[:,-2,:])

    sims = 0.55  * ori_sims +  0.35 * new_sims + 0.1 * sum_sims 

    (r1, r5, r10, medr, meanr) = t2i(img_embs.shape[0], sims)
    
    print("Interactive Text retrieve img: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    
    return r1, r5, r10



def compute_density(img_embs, k=20):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(img_embs)
    distances, _ = nbrs.kneighbors(img_embs)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    densities = 1.0 / (avg_distances + 1e-9)
    return densities


def interative_validate():
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    logger = logging.getLogger(__name__)

    # Load Vocabulary
    bert_path = opt.bert_path
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = bert_tokenizer.vocab
    opt.vocab_size = len(vocab)
    
    # Load LLM
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        trust_remote_code=True
    )

    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to(device)
    llm.eval()

    # Load existing retriever
    model_path = "./runs/scanrefer_train_butd_ESAregion_bigru/transfer.pth" 
    checkpoint = torch.load(model_path)

    model = VSEModel(opt)
    model.load_state_dict(checkpoint['model'])
    # Load Text encoder
    txt_encoder = model.txt_enc.cuda()
    txt_encoder.eval()

    model.val_start()
    logger.info(opt)

    _, val_loader = I_image_caption_bert.get_loaders(
        opt.data_path, opt.data_name, bert_tokenizer, opt.batch_size, 0, opt)

    with torch.no_grad():
        img_embs, cap_embs, text, memory, scene_id = encode_data(
            model, val_loader, opt.log_step, logging.info)

    data_div = 10
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), data_div)])

    sims = compute_sim(img_embs[:,1,:], cap_embs[:,1,:])
       
    response = [] 
    response_embs = np.zeros((cap_embs.shape[0], 6 , cap_embs.shape[-1]))

    # settings
    top_k = 20
    theta = 2.7

    tags = np.ones((cap_embs.shape[0], 6))
    etp_list = np.zeros((cap_embs.shape[0], 6))

    system_prompt = "You are a helpful assistant."

    ''' Based on the saved interaction text and summary text, features are encoded. '''
    data_path = './temp_results/8_9/self_scanrefer/bert_interactive_text_with_responses_5.jsonl'
    features, sum_feats = process_dataset(data_path, bert_tokenizer, txt_encoder)

    np.save('./temp_results/8_9/self_scanrefer/bert_interactive_features_5.npy', features)
    np.save('./temp_results/8_9/self_scanrefer/summarized_features.npy', sum_feats)
   
    ''' If the features have already been saved, you can comment out the steps above and directly read the features from the .npy file. '''
    response_embs = np.load('./temp_results/8_9/self_scanrefer/bert_interactive_features_5.npy')
    etps = np.load('./temp_results/8_9/self_scanrefer/bert_etp_5.npy')
    
    r_1, r_5, r_10 = I_t2i(img_embs[:,1,:], cap_embs[:,1,:], response_embs, etps) # response_embs

    # caption retrieval
    npts = img_embs.shape[0]
    (r1, r5, r10, medr, meanr) = i2t(npts, sims)
    print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    (r1i, r5i, r10i, medri, meanr) = t2i(npts, sims)
    print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1i, r5i, r10i, medri, meanr))
    return 

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def process_dataset(file_path, vocab, txt_encoder):
    data = load_jsonl(file_path)
    features = np.zeros((len(data), 4, 1024))  
    sum_features = np.zeros((len(data), 3, 1024)) 
    
    for i, item in enumerate(tqdm(data, desc="Processing")):
        desc_feat = encode_text(item["description"], vocab, txt_encoder)
        ans_feats = np.array([encode_text(ans, vocab, txt_encoder) for ans in item["answers"]])
        sum_feats = np.array([encode_text(ans, vocab, txt_encoder) for ans in item["response"]])
        
        features[i, 0, :] = desc_feat
        features[i, 1:4, :] = ans_feats
        sum_features[i, :, :] = sum_feats
    
    return features, sum_features

def encode_text(text, tokenizer, txt_encoder):
    caption_tokens = tokenizer.basic_tokenizer.tokenize(text)
    te = process_caption(tokenizer, caption_tokens).cuda()
    te_length = torch.tensor([te.size(0)]).cuda()
    te_tensor = torch.zeros((1, 500)).cuda()
    te_tensor[0, :te_length] = te
    
    _, te_feat = txt_encoder(te_tensor.long(), te_length, None)
    return te_feat.cpu().detach().numpy().squeeze()

if __name__ == '__main__':
    interative_validate()