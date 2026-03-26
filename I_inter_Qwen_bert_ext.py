"""Training script"""
import numpy as np
import torch
from transformers import BertTokenizer
from lib.datasets import I_image_caption_bert
from lib.vse_bert import VSEModel
from lib.evaluation import encode_data

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import logging
import json
import arguments

TOKENIZERS_PARALLELISM=False
MODEL_ID = "Qwen/Qwen-7B-Instruct"
MODEL_NAME = MODEL_ID.split("/")[-1]
CONTEXT_LENGTH = 3000 
MODEL_DIR = "./lib/LLMs/qwen_7b"

def concatenate_responses(response_history):
    return "".join(response_history)

def l2norm(X, dim, eps=1e-8):
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

def predict( 
    model,
    tokenizer,
    device,
    message,
    history,
    system_prompt,
    max_new_tokens=512,
):
    stop_tokens = ["<|im_end|>"] 
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

def prompt_route(count, img_embs, cap_emb, top_k, theta):
    cosine_similarities = img_embs @ cap_emb  
    
    top_k_indices = np.argpartition(-cosine_similarities, top_k)[:top_k]
    top_k_similarities = cosine_similarities[top_k_indices]

    T = 0.05
    exp_similarities = np.exp((top_k_similarities - np.max(top_k_similarities)) / T)
    probs = exp_similarities / np.sum(exp_similarities)
    entropy = -np.sum(probs * np.log(probs + 1e-9)) 

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

    # opt = checkpoint['opt']
    model.val_start()
    logger.info(opt)

    _, val_loader = I_image_caption_bert.get_loaders(
        opt.data_path, opt.data_name, bert_tokenizer, opt.batch_size, 0, opt)

    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        img_embs, cap_embs, text, memory, scene_id = encode_data(
            model, val_loader, opt.log_step, logging.info)

    data_div = 10
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), data_div)])
       
    response = [] 
    response_embs = np.zeros((cap_embs.shape[0], 6 , cap_embs.shape[-1]))

    # settings
    top_k = 20
    theta = 1.9

    tags = np.ones((cap_embs.shape[0], 6))
    etp_list = np.zeros((cap_embs.shape[0], 6))

    system_prompt = "You are a helpful assistant."

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

        for j in range(5):
            prompt = ['Requirement: Continue describing and enriching the last description sentences only based information from your memory passage with 4-5 additional sentences. Summarize all sentences to form new sentences in the original style.',
                       'Requirement: Try to describe different areas and objects within the passage from your memory that have not been covered in previous conversations. Only answer with about 5-7 sentences. Avoiding a repeat of before conversations. When answering, start with a focus object that is different from the previous ones.']
            question_prompt = ['Continue to ask for details placement of the related objects in the current description, and ask about the surrounding environment or objects related to these objects currently described.', 
                               'Ask if there is any additional spatial arrangement of objects that does not overlap with the conversation history and current descriptions. Instead of asking whether there is xxx, you should ask what objects are there.']
            if j == 0:
                prompt_index, entropy = prompt_route(0, img_embs[:,1,:], cap_embs[i,1,:], top_k, theta)
                print('Selected prompt template: '+str(prompt_index))
                count = 0
                ques_his = []
                response_history = []
                history = []
                answers = []
                tags_list = []
                etps_list = []
                
                question_message = f"""Assume you are an expert in asking questions, and you need to constantly interact with users and ask questions to continuously understand the various details in a 3D indoor room. I will give you a user's description history and the description content in the current round, and you need to ask questions about the room according to a requirement.
                User's description history:
                None

                User's description content in the current round:
                {text[i]}

                Requirement:
                {question_prompt[0]}
                
                Important rules:
                1. In English!
                2. Don't ask simple closed-ended questions; you want to ask more open-ended ones about multiple objects.
                3. Just return to the question you want to ask, without any extra content or blank lines.
                4. Do not repeat the previous history questions.
                """

                question = predict(llm, tokenizer, device, question_message, ques_his, system_prompt)
                print('----------------------Question------------------------')
                print(question)

                ques_his.append((question_message, question))

                message = f"""Assume this is the passage in your memory: 
                {memory[i]}
                
                Here is an original sentence describing a scene: 
                {text[i]}

                Template for beginning of sentence:
                1. this is ...
                2. it is ...
                3. there are ...
                4. there is ...
                5. xxx is ...
                6. xxx are ...

                Requirement: {question} Supplement with 4-5 additional sentences with the original sentence. Summarize all sentences to form new sentences in the original style. Follow the Important rules. Must use the original language style.

                Important rules:
                1. Answer 5-7 sentences at most. The returned answer is a paragraph, without blank lines and without Chinese.
                2. Only use details that are available in your memory passage. Do not repeat the previous sentences. Do not fabricate new details!
                3. If no new details about the object are present in the passage, do not fabricate new ones.
                4. The sentence format should imitate the original sentence, for example, every letter in the words should not be capitalized, there should be a space before the period and comma, and they should all be simple sentences without clauses.
                5. Answer in English! Do not fabricate new details!
                """

                response = predict(llm, tokenizer, device, message, history, system_prompt)
                response_history.append(response)
                history.append((message,response))
                answers.append(response)
                tags_list.append(0)
                etps_list.append(entropy)
                tags[i,j] = 0
            else:
                prompt_index, entropy = prompt_route(count, img_embs[:,1,:], response_embs[i,j,:], top_k, theta)
                print('Selected prompt template: '+str(prompt_index))
                res_his = concatenate_responses(response_history)

                question_message = f"""Assume you are an expert in asking questions, and you need to constantly interact with users and ask questions to continuously understand the various details in a indoor room. I will give you a user's description history and the description content in the current round, and you need to ask questions about the room according to a requirement.
                User's description history:
                {res_his}

                User's description content in the current round:
                {text[i]}

                Requirement:
                {question_prompt[prompt_index]}
                
                Important rules:
                1. In English!
                2. Don't ask simple closed-ended questions; you want to ask more open-ended ones about multiple objects.
                3. Just return to the question you want to ask, without any extra content or blank lines.
                4. Do not repeat the previous history questions.
                """

                question = predict(llm, tokenizer, device, question_message, ques_his, system_prompt)
                print('----------------------Question------------------------')
                print(question)
                ques_his.append((question_message, question))

                if prompt_index==0:
                    message = question + ' Supplement with 4-5 additional sentences with the last sentences. Summarize all sentences to form new sentences in the original style. Follow the Important rules. Must use the original language style.'
                else:
                    message = question + ' Try to describe different areas and objects within the passage from your memory that have not been covered in previous conversations. Only answer with about 5-7 sentences. Avoiding a repeat of before conversations. When answering, start with a focus object that is different from the previous ones.'
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

    np.save('./temp_results/bert_interactive_features_5.npy', response_embs)
    np.save('./temp_results/bert_tags_5.npy', tags)
    np.save('./temp_results/bert_etp_5.npy', etp_list)

    return 

if __name__ == '__main__':
    interative_validate()