"""Training script"""
import os
import time
import numpy as np
import torch
import json

from lib.vocab import deserialize_vocab
from lib.datasets import image_caption
from lib.vse import VSEModel
from lib.evaluation import i2t, t2i, encode_data, compute_sim

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from tqdm import tqdm
from lib.pos_encoder import EncoderText
import nltk
import logging
from scipy.special import comb

import numpy as np

import arguments

TOKENIZERS_PARALLELISM=False
MODEL_ID = "Qwen/Qwen-7B-Instruct"  
MODEL_NAME = MODEL_ID.split("/")[-1]
CONTEXT_LENGTH = 2048  # Qwen-7B-Instruct 的上下文长度较大
MODEL_DIR = "/home/fengyanglin/ESA-main/ESA_BIGRU/lib/LLMs/qwen_7b"

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X using numpy"""
    norm = np.sqrt(np.sum(np.power(X, 2), axis=dim, keepdims=True)) + eps
    X = X / norm
    return X

def process_caption(vocab, caption, max_length=500):
    # Tokenize the caption and convert to lowercase
    tokens = nltk.tokenize.word_tokenize(caption.lower())
    
    # Initialize the caption list with the <start> token
    caption = [vocab('<start>')]
    
    # Add the tokens from the caption to the list
    caption.extend([vocab(token) for token in tokens])
    
    # Add the <end> token
    caption.append(vocab('<end>'))
    
    # Create a tensor for the caption, ensuring it's a LongTensor
    target = torch.tensor(caption, dtype=torch.long)
    if len(target) < 500:
        target = target
    else:
        target = target[:500]
    return target

def bezier_curve_interpolation(p0, p1, p2, t):
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

import numpy as np

def slerp_interpolation(p0, p1, t):
    dot_product = np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1))  
    dot_product = np.clip(dot_product, -1.0, 1.0)  
    
    theta = np.arccos(dot_product)  
    sin_theta = np.sin(theta)

    if sin_theta == 0:
        return p0  
    
    w0 = np.sin((1 - t) * theta) / sin_theta
    w1 = np.sin(t * theta) / sin_theta

    return w0 * p0 + w1 * p1

def fuse_answer_embs(answer_embs, embs_stat, lambda_=0.5, use_slerp=True):
    fused_embs = answer_embs.copy()
    n, num_views, dim = answer_embs.shape

    for i in range(num_views - 1):  
        mask = embs_stat[:, i] == 0  
        
        fused_embs[mask, i + 1] = lambda_ * fused_embs[mask, i] + (1 - lambda_) * fused_embs[mask, i + 1]

    final_embs = np.zeros((n, dim))  
    
    for i in range(n):  
        sample_embs = fused_embs[i]
        final_result = np.zeros(dim)
        bezier_points = []
        
        for j in range(num_views):
            if embs_stat[i, j] == 1:
                bezier_points.append(sample_embs[j])
        
        if len(bezier_points) >= 2:
            if use_slerp:
                # 使用SLERP插值
                slerp_result = np.zeros(dim)
                for t in np.linspace(0, 1, num=10):  # 在插值范围内做多个插值
                    interpolated_emb = slerp_interpolation(bezier_points[0], bezier_points[1], t)
                    slerp_result += interpolated_emb
                final_result = slerp_result / len(np.linspace(0, 1, num=10))  # 除以采样次数
            else:
                # 使用Bézier曲线插值
                bezier_result = np.zeros(dim)
                for t in np.linspace(0, 1, num=10):  # 在 Bézier 曲线范围内做多个插值
                    interpolated_emb = bezier_curve_interpolation(bezier_points[0], bezier_points[1], bezier_points[2], t)
                    bezier_result += interpolated_emb
                final_result = bezier_result / len(np.linspace(0, 1, num=10))  # 除以采样次数

        final_embs[i] = final_result

    return final_embs



def I_t2i(img_embs, cap_embs, answer_embs, question_types):
    ori_sims = compute_sim(img_embs, cap_embs)
    question_types[:, 7] = 1
    valid_mask = question_types.astype(bool)  

    fused_embs = fuse_answer_embs(answer_embs, valid_mask, lambda_=0.7)
    sims = compute_sim(img_embs, fused_embs)

    (r1, r5, r10, medr, meanr) = t2i(img_embs.shape[0], sims)
    print("Interactive Text retrieve img: %.1f, %.1f, %.1f, %.1f, %.1f" %
                 (r1, r5, r10, medr, meanr))
    
    return fused_embs

def predict( 
    model,
    tokenizer,
    device,
    message,
    history,
    system_prompt,
    max_new_tokens=512,
):
    stop_tokens = ["<|im_end|>"]  # Qwen 终止 token
    usr = "<|im_start|>user\n"
    asi = "<|im_start|>assistant\n"
    
    instruction = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
    
    for user, assistant in history:
        instruction += f"{usr}{user}<|im_end|>\n{asi}{assistant}<|im_end|>\n"
    
    instruction += f"{usr}{message}<|im_end|>\n{asi}"

    enc = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc.input_ids.to(device)

    generate_kwargs = dict(
        input_ids=input_ids,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        temperature=1.0
    )

    try:
        generated_ids = model.generate(**generate_kwargs)
        response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        start_idx = response.rfind('assistant')
        if start_idx != -1:
            response = response[start_idx + len('assistant'):].strip()
        return response.split("<|im_end|>")[0].strip()

    except Exception as e:
        return f"Error: {e}"

def prompt_route(count, img_embs, cap_emb, top_k, theta):
    cosine_similarities = img_embs @ cap_emb  
    
  
    top_k_indices = np.argpartition(-cosine_similarities, top_k)[:top_k]
    top_k_similarities = cosine_similarities[top_k_indices]

    T = 0.05
    exp_similarities = np.exp((top_k_similarities - np.max(top_k_similarities)) / T)
    probs = exp_similarities / np.sum(exp_similarities)
    # print(probs)
    
    # 计算信息熵 H = -sum(p * log(p))
    entropy = -np.sum(probs * np.log(probs + 1e-9)) 
    print("Etp: "+str(entropy))

    if count > 1:
        return 1, entropy
    else:
        return int(entropy < theta), entropy


def interative_validate():
     
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR,torch_dtype=torch.float16).to(device)
    llm.eval()


    text = []
    with open('/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/scanrefer/ori_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                text.append(json.loads(line.strip()))

    memory = []
    with open('/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/scanrefer/train_data.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip(): 
                memory.append(json.loads(line.strip()))
                
    
    output_file = 'scanrefer_train_data_8_7.jsonl'  
    output_file_raw = 'scanrefer_ori_data_8_7.jsonl'

    for i in tqdm(range(len(memory)), desc="Processing", unit="step"):
        system_prompt = "You are a helpful assistant."
        print('Randomly spliced memory：-----------------------------------------------------')
        print(memory[i]['description'])
        print('Original query：-----------------------------------------------------')
        print(text[i]['description'])
        
        elem = {
                'scene_id': text[i]['scene_id'],
                'description': text[i]['description'],
                'memory': ''
                }
        
        with open(output_file_raw, 'a', encoding='utf-8') as f:
            f.write(json.dumps(elem, ensure_ascii=False) + '\n')

        for j in range(4):
            prompt = [
                'Requirement: Continue describing and enriching in the same language style with the original sentence only based information from your memory passage with 4-5 additional sentences. Summarize all sentences to form new sentences in the original style.',
                'Requirement: Try to randomly describe other different areas and objects within the passage from your memory that have not been covered in previous conversations. Only answer with about 5-7 sentences. Avoiding a repeat of before conversations.'
            ]
            if j == 0: 
                message = f"""Assume this is the passage in your memory: 
                {memory[i]['description']}
                
                Here is an original sentence describing a scene: 
                {text[i]['description']}

                Template for beginning of every sentence in your answer:
                1. this is ...
                2. it is ...
                3. there are ...
                4. there is ...
                5. xxx is ...
                6. xxx are ...

                {prompt[0]}

                Important rules: No in Chinese!
                1. Answer 5-6 English sentences at most. The returned answer is a paragraph, without blank lines. 2. Only use details that are available in your memory passage. Do not repeat the previous sentences. Do not fabricate new details! 3. If no new details about the object are present in the passage, do not fabricate new ones. 4. The sentence format should imitate the original sentence, for example, every letter in the words should not be capitalized, there should be a space before the period and comma, and they should all be simple sentences without clauses. 5. Be brief and don't answer with any unnecessary extra descriptions beyond the spacial relationships of objects.
                """
                history = []
                response = predict(llm, tokenizer, device, message, history, system_prompt)
                history.append((message, response))
            else:
                message = prompt[1] + ' Follow the Important rules. Do not similar to the previous answers. Some descriptive phrases or words can be replaced with synonyms with a certain probability.'
                response = predict(llm, tokenizer, device, message, history, system_prompt)
                history.append((message, response))
            
            elem = {
                'scene_id': text[i]['scene_id'],
                'description': response,
                'memory': ''
            }
            
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(elem, ensure_ascii=False) + '\n')

            print(str(j + 1) + '-th sentence：-----------------------------------------------------')
            print(response)

            
    return 

if __name__ == '__main__':
    interative_validate()






''' SR3D '''
# """Training script"""
# import os
# import time
# import numpy as np
# import torch
# import json

# from lib.vocab import deserialize_vocab
# from lib.datasets import image_caption
# from lib.vse import VSEModel
# from lib.evaluation import i2t, t2i, encode_data, compute_sim

# from threading import Thread
# from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# from tqdm import tqdm
# from lib.pos_encoder import EncoderText
# import nltk
# import logging
# from scipy.special import comb

# import numpy as np

# import arguments
# TOKENIZERS_PARALLELISM=False
# MODEL_ID = "Qwen/Qwen-7B-Instruct"  
# MODEL_NAME = MODEL_ID.split("/")[-1]
# CONTEXT_LENGTH = 2048  # Qwen-7B-Instruct 的上下文长度较大
# MODEL_DIR = "/home/fengyanglin/ESA-main/ESA_BIGRU/lib/LLMs/qwen_7b"

# def l2norm(X, dim, eps=1e-8):
#     """L2-normalize columns of X using numpy"""
#     norm = np.sqrt(np.sum(np.power(X, 2), axis=dim, keepdims=True)) + eps
#     X = X / norm
#     return X

# def process_caption(vocab, caption, max_length=500):
#     # Tokenize the caption and convert to lowercase
#     tokens = nltk.tokenize.word_tokenize(caption.lower())
    
#     # Initialize the caption list with the <start> token
#     caption = [vocab('<start>')]
    
#     # Add the tokens from the caption to the list
#     caption.extend([vocab(token) for token in tokens])
    
#     # Add the <end> token
#     caption.append(vocab('<end>'))
    
#     # Create a tensor for the caption, ensuring it's a LongTensor
#     target = torch.tensor(caption, dtype=torch.long)
    
#     # Create a tensor of zeros with the specified max_length (500)
#     # padded_target = torch.zeros(max_length, dtype=torch.long)
    
#     # Place the valid tokens into the beginning of the tensor
#     if len(target) < 500:
#         target = target
#     else:
#         target = target[:500]
#     return target

# def bezier_curve_interpolation(p0, p1, p2, t):
#     """
#     计算二次Bézier曲线的插值。
    
#     参数：
#     p0, p1, p2 (numpy.ndarray): 控制点。
#     t (float): 参数，表示曲线位置，0 <= t <= 1。
    
#     返回：
#     numpy.ndarray: 插值结果。
#     """
#     return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

# import numpy as np

# def slerp_interpolation(p0, p1, t):
#     """
#     计算球面线性插值（SLERP）。

#     参数：
#     p0, p1 (numpy.ndarray): 起始点和终止点。
#     t (float): 参数，表示插值位置，0 <= t <= 1。

#     返回：
#     numpy.ndarray: 插值结果。
#     """
#     # 计算两点的夹角
#     dot_product = np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1))  # 归一化
#     dot_product = np.clip(dot_product, -1.0, 1.0)  # 防止由于浮点数误差超过[-1, 1]
    
#     theta = np.arccos(dot_product)  # 夹角
#     sin_theta = np.sin(theta)

#     if sin_theta == 0:
#         return p0  # 如果两点完全相同，返回其中一个点
    
#     # 计算权重
#     w0 = np.sin((1 - t) * theta) / sin_theta
#     w1 = np.sin(t * theta) / sin_theta

#     # 计算插值
#     return w0 * p0 + w1 * p1

# def fuse_answer_embs(answer_embs, embs_stat, lambda_=0.5, use_slerp=True):
#     """
#     对于 `answer_embs` 中 `embs_stat` 为 0 的特征（且不是最后一个视角），
#     使用线性融合方法与下一个视角进行融合。

#     参数：
#     answer_embs (numpy.ndarray): 形状为 (n, 6, 1024) 的特征矩阵。
#     embs_stat (numpy.ndarray): 形状为 (n, 6) 的状态矩阵。
#     lambda_ (float): 线性融合的权重系数，默认 0.5。
#     use_slerp (bool): 是否使用球面线性插值（SLERP），默认为 False 使用 Bézier 曲线。

#     返回：
#     numpy.ndarray: 融合后的 answer_embs。
#     """
#     fused_embs = answer_embs.copy()
#     n, num_views, dim = answer_embs.shape

#     # 线性融合部分
#     for i in range(num_views - 1):  # 遍历前 5 个视角
#         mask = embs_stat[:, i] == 0  # 找到当前视角状态为 0 的样本
        
#         # 仅对满足条件的样本进行线性融合
#         fused_embs[mask, i + 1] = lambda_ * fused_embs[mask, i] + (1 - lambda_) * fused_embs[mask, i + 1]

#     # 融合部分
#     final_embs = np.zeros((n, dim))  # 用于存储最终的融合特征
    
#     for i in range(n):  # 对每个样本进行处理
#         # 获取每个样本所有视角的特征
#         sample_embs = fused_embs[i]
        
#         # 针对 `embs_stat = 1` 的视角特征进行融合
#         final_result = np.zeros(dim)
#         bezier_points = []
        
#         # 获取所有 `embs_stat == 1` 的视角
#         for j in range(num_views):
#             if embs_stat[i, j] == 1:
#                 bezier_points.append(sample_embs[j])
        
#         # 如果至少有 2 个控制点，可以进行插值
#         if len(bezier_points) >= 2:
#             if use_slerp:
#                 # 使用SLERP插值
#                 slerp_result = np.zeros(dim)
#                 for t in np.linspace(0, 1, num=10):  # 在插值范围内做多个插值
#                     interpolated_emb = slerp_interpolation(bezier_points[0], bezier_points[1], t)
#                     slerp_result += interpolated_emb
#                 final_result = slerp_result / len(np.linspace(0, 1, num=10))  # 除以采样次数
#             else:
#                 # 使用Bézier曲线插值
#                 bezier_result = np.zeros(dim)
#                 for t in np.linspace(0, 1, num=10):  # 在 Bézier 曲线范围内做多个插值
#                     interpolated_emb = bezier_curve_interpolation(bezier_points[0], bezier_points[1], bezier_points[2], t)
#                     bezier_result += interpolated_emb
#                 final_result = bezier_result / len(np.linspace(0, 1, num=10))  # 除以采样次数

#         final_embs[i] = final_result

#     return final_embs



# def I_t2i(img_embs, cap_embs, answer_embs, question_types):
#     # m * emb_size    n * emb_size   =   m * n 
#     ori_sims = compute_sim(img_embs, cap_embs)

#     # new_sims_1 = compute_sim(img_embs, answer_embs[:,1,:])
#     # new_sims_2 = compute_sim(img_embs, answer_embs[:,2,:])
#     # new_sims_3 = compute_sim(img_embs, answer_embs[:,3,:])
#     # new_sims_4 = compute_sim(img_embs, answer_embs[:,4,:])
#     # new_sims_5 = compute_sim(img_embs, answer_embs[:,5,:])
#     # new_sims_6 = compute_sim(img_embs, answer_embs[:,6,:])
#     # new_sims_7 = compute_sim(img_embs, answer_embs[:,7,:])
#     # # m * emb_size    n * 3 * emb_size   =   m * n * 3
#     # sims =  (ori_sims + new_sims_1 + new_sims_2 + new_sims_3 + new_sims_4 + new_sims_5 + new_sims_6 + new_sims_7)/7

#     # 设置第一列 tags 为 0，确保第一列不参与计算
#     # tags = np.ones((answer_embs.shape[0], 8))  # 生成标签
#     question_types[:, 7] = 1
#     # valid_mask = tags.astype(bool)  
#     valid_mask = question_types.astype(bool)  
#     # valid_mask[:, 0] = False 

#     fused_embs = fuse_answer_embs(answer_embs, valid_mask, lambda_=0.7)
#     sims = compute_sim(img_embs, fused_embs)

#     (r1, r5, r10, medr, meanr) = t2i(img_embs.shape[0], sims)
#     print("Interactive Text retrieve img: %.1f, %.1f, %.1f, %.1f, %.1f" %
#                  (r1, r5, r10, medr, meanr))
    
#     return fused_embs

# def predict( 
#     model,
#     tokenizer,
#     device,
#     message,
#     history,
#     system_prompt,
#     max_new_tokens=512,
# ):
#     stop_tokens = ["<|im_end|>"]  # Qwen 终止 token
#     usr = "<|im_start|>user\n"
#     asi = "<|im_start|>assistant\n"
    
#     # 构建系统提示和对话历史
#     instruction = f"<|im_start|>system\n{system_prompt}<|im_end|>\n" if system_prompt else ""
    
#     for user, assistant in history:
#         instruction += f"{usr}{user}<|im_end|>\n{asi}{assistant}<|im_end|>\n"
    
#     instruction += f"{usr}{message}<|im_end|>\n{asi}"

#     # Tokenize the input text
#     enc = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
#     input_ids = enc.input_ids.to(device)

#     # 生成参数
#     generate_kwargs = dict(
#         input_ids=input_ids,
#         do_sample=True,
#         max_new_tokens=max_new_tokens,
#         eos_token_id=tokenizer.eos_token_id,
#         repetition_penalty=1.2,
#         temperature=1.0
#     )

#     try:
#         generated_ids = model.generate(**generate_kwargs)
#         response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

#         # 找到最后一个 asi 的位置
#         start_idx = response.rfind('assistant')
#         if start_idx != -1:
#             response = response[start_idx + len('assistant'):].strip()

#         # 清除可能存在的任何终止标记或额外字符
#         return response.split("<|im_end|>")[0].strip()

#     except Exception as e:
#         return f"Error: {e}"

# def prompt_route(count, img_embs, cap_emb, top_k, theta):
#     """
#     计算cap_emb与img_embs的余弦相似度，选择top_k个最相似的图片嵌入，
#     计算基于softmax的概率分布，并计算信息熵。
#     若信息熵大于theta，返回0，否则返回1。
    
#     :param img_embs: numpy.ndarray, 形状 (n, 1024) 的图片嵌入矩阵
#     :param cap_emb: numpy.ndarray, 形状 (1024,) 的文本嵌入向量
#     :param top_k: int, 选择的最近邻数量
#     :param theta: float, 阈值
#     :return: int, 0 或 1
#     """
#     # 计算余弦相似度
#     cosine_similarities = img_embs @ cap_emb  # 计算余弦相似度
    
#     # 选取 top_k 个最高相似度的索引
#     top_k_indices = np.argpartition(-cosine_similarities, top_k)[:top_k]
#     top_k_similarities = cosine_similarities[top_k_indices]

#     # 计算 softmax（加温度参数）
#     T = 0.05
#     exp_similarities = np.exp((top_k_similarities - np.max(top_k_similarities)) / T)
#     probs = exp_similarities / np.sum(exp_similarities)
#     # print(probs)
    
#     # 计算信息熵 H = -sum(p * log(p))
#     entropy = -np.sum(probs * np.log(probs + 1e-9)) 
#     print("这一句 得到的熵为："+str(entropy))

#     if count > 1:
#         return 1, entropy
#     else:
#         return int(entropy < theta), entropy


# def interative_validate():
     
#     # Load Text encoder Old
#     # txt_encoder = EncoderText(opt, use_bi_gru=True, no_txtnorm=False,is_init=False)

#     # Load LLM
#     device = torch.device("cuda:1")
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#     llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR,torch_dtype=torch.float16).to(device)
#     llm.eval()


#     text = []
#     with open('/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/sr3d_raw_data.jsonl', 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():  # 忽略空行
#                 text.append(json.loads(line.strip()))

#     memory = []
#     with open('/home/fengyanglin/ESA-main/ESA_BIGRU/lib/data/sr3d_train_memory.jsonl', 'r', encoding='utf-8') as f:
#         for line in f:
#             if line.strip():  # 忽略空行
#                 memory.append(json.loads(line.strip()))
                
    
#     output_file = 'sr3d_train_data.jsonl'  # 使用 jsonl 格式
#     output_file_raw = 'sr3d_ori_data.jsonl'


#     for i in tqdm(range(len(memory)), desc="Processing", unit="step"):
#         system_prompt = "You are a helpful assistant."
#         print('记忆：-----------------------------------------------------')
#         print(memory[i]['description'])
#         print('原话：-----------------------------------------------------')
#         print(text[i]['description'])
        
#         elem = {
#                 'scene_id': text[i]['scene_id'],
#                 'description': text[i]['description'],
#                 'memory': ''
#                 }
        
#         with open(output_file_raw, 'a', encoding='utf-8') as f:
#             f.write(json.dumps(elem, ensure_ascii=False) + '\n')

#         # 每个batch 有 3 轮对话
#         for j in range(4):
#             prompt = [
#                 'Requirement: Continue describing and enriching in the same language style with the original sentence only based information from your memory passage with 4-5 additional sentences. Summarize all sentences to form new sentences in the original style.',
#                 'Requirement: Try to randomly describe other different areas and objects within the passage from your memory that have not been covered in previous conversations. Only answer with about 5-7 sentences. Avoiding a repeat of before conversations.'
#             ]
#             if j == 0:  # 第 1 轮对话
#                 message = f"""Assume this is the passage in your memory: 
#                 {memory[i]['description']}
                
#                 Here is an original sentence describing a scene: 
#                 {text[i]['description']}

#                 Template for beginning of sentence such as:
#                 1. The xxx . 2. This xxx is . 3. Choose xxx . 4. Select xxx . 5. Find xxx .

#                 {prompt[0]}

#                 Important rules:
#                 1. Answer 8 sentences at most. The returned answer is a paragraph without blank lines. 2. Only use details that are available in your memory passage. Do not repeat the previous sentences. I conducted seven rounds of dialogue in total, and at the end all answers should include as many objects as possible in the passage. 3. If no new details about the object are present in the passage, do not fabricate new ones. 4. The sentence format should imitate the original sentence, for example, every letter in the words should not be capitalized and they should all be simple sentences without clauses. 5. Short Answer in English! Do not start with "in my room" / "in the room", describe directly.
#                 """
#                 history = []
#                 response = predict(llm, tokenizer, device, message, history, system_prompt)
#                 history.append((message, response))
#             else:
#                 message = prompt[1] + ' Follow the Important rules. Do not similar to the previous answers. Some descriptive phrases or words can be replaced with synonyms with a certain probability.'
#                 response = predict(llm, tokenizer, device, message, history, system_prompt)
#                 history.append((message, response))
            
#             elem = {
#                 'scene_id': text[i]['scene_id'],
#                 'description': response,
#                 'memory': ''
#             }
            
#             # 追加写入 JSONL 文件
#             with open(output_file, 'a', encoding='utf-8') as f:
#                 f.write(json.dumps(elem, ensure_ascii=False) + '\n')

#             print('第' + str(j + 1) + '句话：-----------------------------------------------------')
#             print(response)

            
#     return 

# if __name__ == '__main__':
#     interative_validate()