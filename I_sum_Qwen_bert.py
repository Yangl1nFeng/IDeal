"""Training script"""
import os
import time
import numpy as np
import torch

from lib.vocab import deserialize_vocab
from lib.datasets import image_caption
from lib.vse_bert import VSEModel
from lib.evaluation import i2t, t2i, encode_data, compute_sim
from transformers import BertTokenizer
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from tqdm import tqdm
from lib.pos_encoder import EncoderText
import nltk
import logging
from scipy.special import comb
import json
import torch.nn.functional as F
import numpy as np

from I_inter_Qwen_bert import predict, process_caption
import arguments
TOKENIZERS_PARALLELISM=False
MODEL_ID = "Qwen/Qwen-7B-Instruct"  
MODEL_NAME = MODEL_ID.split("/")[-1]
CONTEXT_LENGTH = 2048  
MODEL_DIR = "./lib/LLMs/qwen_7b"


def summery_answers():
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()
    logger = logging.getLogger(__name__)

    bert_path = opt.bert_path
    bert_tokenizer = BertTokenizer.from_pretrained(bert_path)

    # Load Vocabulary
   
    # Load LLM
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    llm = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).to(device)
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

    # Load jsonl data
    jsonl_file = "./temp_results/8_9/self_scanrefer/bert_interactive_text.jsonl"
    output_jsonl_file = "./temp_results/8_9/self_scanrefer/bert_interactive_text_with_responses_5.jsonl"
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    
    sum_feat = np.zeros((len(data), 1024))
    system_prompt = "You are a helpful assistant."

    for i, item in enumerate(data):
        description = item.get("description", "")
        answers_list = item.get("answers", [])  
        all_responses = [] 

        merged_answers = ""  
        for j in range(len(answers_list)):
            history = []
            merged_answers += " " + answers_list[j]  
            text = f"{description} {merged_answers.strip()}"

            message_1 = f"""
            There are many sentences here, and they may contain some repeated information. First, help me identify the objects mentioned in the sentence and list them in the form of keywords. Requirements: According to your understanding, put the indoor objects with discrimination in front, and the common objects such as doors, chairs, and tables in the back. Just list the objects, no more nonsense.
            Sentences:
            {text}
            """

            message_2 = f"""
            Now you are asked to reconstruct the scene, describing the placement, color, and other properties of the objects according to the original language styles you mentioned. You should include all the information in the given sentences and describe the objects in the order you previously provided.
            Sentences:
            {text}

            Template for beginning of sentence: 
            1. this is ... 
            2. it is ... 
            3. there are ... 
            4. there is ... 
            5. xxx is ... 
            6. xxx are ...

            Important rules:
            0. Try to delete as much information as you think is redundant or uninformative, such as describing the same object multiple times.
            1. Answer 2-5 sentences at most. The returned answer is a paragraph, without blank lines and without Chinese.
            2. Only use details that are available in original sentences. Be as detailed as possible
            3. If no new details about the object are present in the passage, do not fabricate new ones.
            4. The sentence format should imitate the original sentence, for example, every letter in the words should not be capitalized, there should be a space before the period and comma, and they should all be simple sentences without clauses.
            5. Answer in English! Do not start with "in my room" / "in the room", describe directly.
            """

            response_1 = predict(llm, tokenizer, device, message_1, history, system_prompt)
            history.append((message_1, response_1)) 

            response_2 = predict(llm, tokenizer, device, message_2, history, system_prompt)
            all_responses.append(response_2)

            print(f'------ Iteration {j+1} ------')
            print(response_2)

        item["response"] = all_responses

        with open(output_jsonl_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

        logger.info(f"Appended item {i} to {output_jsonl_file}")

if __name__ == "__main__":
    summery_answers()
