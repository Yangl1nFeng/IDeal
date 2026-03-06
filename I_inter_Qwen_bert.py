"""Training script"""
import os
import time
import numpy as np
import torch
from transformers import BertTokenizer
from lib.vocab import deserialize_vocab
from lib.datasets import I_image_caption_bert
from lib.vse_bert import VSEModel
from lib.evaluation import i2t, t2i, encode_data, compute_sim

from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
from tqdm import tqdm
from lib.pos_encoder import EncoderText
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

def concatenate_responses(response_history):
    return "".join(response_history)

def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X using numpy"""
    norm = np.sqrt(np.sum(np.power(X, 2), axis=dim, keepdims=True)) + eps
    X = X / norm
    return X

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

    # Place the valid tokens into the beginning of the tensor
    if len(target) < 500:
        target = target
    else:
        target = target[:500]
    return target

import numpy as np
from scipy.spatial.distance import cosine


def compute_sim(a, b):
    """
    dot(a, b) / (||a|| * ||b||)
    """
    return a @ b.T


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

    # Tokenize the input text
    enc = tokenizer(instruction, return_tensors="pt", padding=True, truncation=True)
    input_ids = enc.input_ids.to(device)

    generate_kwargs = dict(
        input_ids=input_ids,
        do_sample=True,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        temperature=0.7,
        top_p=0.8
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

def compute_density(img_embs, k=20):
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(img_embs)
    distances, _ = nbrs.kneighbors(img_embs)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    densities = 1.0 / (avg_distances + 1e-9)
    return densities

def prompt_route_density(count, img_embs, cap_emb, top_k, theta, density_values=None):
    if density_values is None:
        density_values = compute_density(img_embs)

    cosine_similarities = img_embs @ cap_emb

    top_k_indices = np.argpartition(-cosine_similarities, top_k)[:top_k]
    top_k_similarities = cosine_similarities[top_k_indices]
    top_k_densities = density_values[top_k_indices]

    alpha = 0.5  
    adjusted_similarities = top_k_similarities / (top_k_densities ** alpha + 1e-9)

    T = 0.05
    exp_similarities = np.exp((adjusted_similarities - np.max(adjusted_similarities)) / T)
    probs = exp_similarities / np.sum(exp_similarities)

    entropy = -np.sum(probs * np.log(probs + 1e-9))
    print("Density-normalized entropy: " + str(entropy))

    if count > 1:
        return 1, entropy
    else:
        return int(entropy < theta), entropy


def interative_validate():
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    logger = logging.getLogger(__name__)

    # Load Vocabulary
    bert_path = opt.bert_path
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
    vocab = bert_tokenizer.vocab
    opt.vocab_size = len(vocab)
    
    # Load LLM
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR,torch_dtype=torch.float16).to(device)
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

    density_values = compute_density(img_embs[:,1,:])

    for i in tqdm(range(cap_embs.shape[0]), desc="Processing", unit="step"):
        print('Memory: -----------------------------------------------------')
        print(memory[i])
        print('Original query: -----------------------------------------------------')
        print(text[i])

        caption_tokens = bert_tokenizer.basic_tokenizer.tokenize(text[i])
        te = process_caption(bert_tokenizer, caption_tokens).cuda()
        te_length = torch.tensor([te.size(0)]).cuda()
        te_tensor = torch.zeros((1,500)).cuda()
        te_tensor[0,:te_length] = te

        _,te_feat = txt_encoder(te_tensor.long(), te_length, None)
        response_embs[i,0,:] = te_feat.detach().cpu().numpy()

        prompt_1 = f"""
        Continue to ask for detailed relationships between the objects in the current description and other objects.

        Template:
        Continue describing and enriching the last description sentences about xxx,yyy only based information from your memory.
        
        """
        prompt_2 = f"""
        You need to strictly follow the following Template ask question.
        
        Template:
        Try to describe different furnitures and the relationships from your memory.
        
        """

        for j in range(3):
            question_prompt = [prompt_1, 
                               prompt_2]
            if j == 0:
                
                prompt_index, entropy = prompt_route_density(0, img_embs[:,1,:], cap_embs[i,1,:], top_k=10, theta=1.5, density_values=density_values)
                print('Selected prompt template: '+str(prompt_index))
                count = 0
                ques_his = []
                response_history = []
                history = []
                answers = []
                tags_list = []
                etps_list = []
                
                question_message = f"""Assume you are an expert in asking questions, and you need to constantly interact with users and ask questions to continuously understand the indoor room.

                User's description content in the current round:
                {text[i]}

                Requirement:
                {question_prompt[0]}
                
                Important rules:
                1. In English! 
                2. Don't ask simple closed-ended questions; you want to ask more open-ended ones.
                3. Don't specify object categories, ask more broadly. 
                4. Just return to the question you want to ask, without any extra content or blank lines.
                """

                question = predict(llm, tokenizer, device, question_message, ques_his, system_prompt)
                print('----------------------Question------------------------')
                print(question)

                ques_his.append((question_message, question))

                message = f"""Assume this is the passage in your memory: 
                {memory[i]}
                
                Here is an original sentence describing a scene: 
                {text[i]}

                Template for beginning of every sentence in your answer:
                1. this is ...
                2. it is ...
                3. there are ...
                4. there is ...
                5. xxx is ...
                6. xxx are ...

                Requirement: {question} Supplement with 1-3 additional sentences with the original sentence. Summarize all sentences to form new sentences in the original style. Follow the Important rules. Must use the original language style.

                Important rules:
                1. Answer 3-5 English sentences at most. The returned answer is a paragraph, without blank lines.
                2. Only use details that are available in your memory passage. Do not repeat the previous sentences. Do not fabricate new details!
                3. There is a lot of repetitive content in memory that needs to be integrated and the relationship between objects needs to be clarified.
                4. The sentence format should imitate the original sentence, for example, every letter in the words should not be capitalized, there should be a space before the period and comma, and they should all be simple sentences without clauses.
                5. Be brief and don't answer with any unnecessary extra descriptions beyond the spacial relationships of objects.
                """

                response = predict(llm, tokenizer, device, message, history, system_prompt)
                response_history.append(response)
                history.append((message,response))
                answers.append(response)
                tags_list.append(0)
                etps_list.append(entropy)
                tags[i,j] = 0
            else:
                prompt_index, entropy = prompt_route_density(count, img_embs[:,1,:], response_embs[i,j,:], top_k, theta)
                print('Selected prompt template: '+str(prompt_index))
                res_his = concatenate_responses(response_history)

                question_message = f"""Assume you are an expert in asking questions, and you need to constantly interact with users and ask questions to continuously understand the indoor room.

                User's description content in the current round:
                {text[i]}

                Requirement:
                {question_prompt[prompt_index]}
                
                Important rules:
                1. In English! Follow the Template! 
                2. Don't ask simple closed-ended questions; you want to ask more open-ended ones.
                3. Just return to the question you want to ask, without any extra content or blank lines.
                4. There is a lot of repetitive content in memory that needs to be integrated and the relationship between objects needs to be clarified.
                5. Objects are based on context and cannot be made up.
                """

                question = predict(llm, tokenizer, device, question_message, ques_his, system_prompt)
                print('----------------------Question------------------------')
                print(question)
                ques_his.append((question_message, question))

                if prompt_index==0:
                    message = question + ' Supplement with 1-2 additional sentences with the last user descriptions. Summarize all sentences to form new sentences in the original style. Follow the Important rules. Must use the original language style. Don not describe the purpose of the object, just focus on describing its placement in the room! Do not answer with any unnecessary extra descriptions beyond the spacial relationships of objects.' 
                else:
                    message = ' Try to describe different objects within the passage from your memory that have not been covered in previous conversations. Only answer with about 2-5 sentences. Avoiding a repeat of before conversations. When answering, start with a focus object that is different from the previous ones. Just focus on describing spacial relationships of objects! Do not answer with any unnecessary extra descriptions beyond the spacial relationships of objects.'
                response = predict(llm, tokenizer, device, message, history, system_prompt)
                response_history.append(response)
                history.append((message,response))
                answers.append(response)
                tags_list.append(prompt_index)
                etps_list.append(entropy)
                tags[i,j] = prompt_index
        
            if entropy >= theta:
                count = count + 1
            else:
                count = 0
            
            etp_list[i,j] = entropy

            print(str(j+1)+'-th interaction: -----------------------------------------------------')
            print(response)
            
            response_tokens = bert_tokenizer.basic_tokenizer.tokenize(response)
            cap = process_caption(bert_tokenizer, response_tokens)
            caption_length = torch.tensor([cap.size(0)]).cuda()
            caption_tensor = torch.zeros((1,500)).cuda()
            caption_tensor[0,:caption_length] = cap

            _, feature_cap = txt_encoder(caption_tensor.long(), caption_length, None)
            response_embs[i,j+1,:] = feature_cap.detach().cpu().numpy()

        elem = {
            'scene_id': scene_id[i],
            'description': text[i],
            'answers': answers,
            'tag': tags_list,
            'etps': etps_list
            }
    
        with open('./temp_results/8_9/self_scanrefer/bert_interactive_text.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(elem, ensure_ascii=False) + '\n')

    np.save('./temp_results/8_9/self_scanrefer/bert_interactive_features_5.npy', response_embs)
    np.save('./temp_results/8_9/self_scanrefer/bert_tags_5.npy', tags)
    np.save('./temp_results/8_9/self_scanrefer/bert_etp_5.npy', etp_list)

    return 

if __name__ == '__main__':
    interative_validate()