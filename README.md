# IDeal
The original implementation version of NeurIPS 2025 paper [**Interactive Cross-modal Learning for Text-3D Scene Retrieval**](https://openreview.net/pdf?id=fohuurA03P). 😀

## Abstract
Text-3D Scene Retrieval (T3SR) aims to retrieve relevant scenes using linguistic queries. Although traditional T3SR methods have made significant progress in
capturing fine-grained associations, they implicitly assume that query descriptions are information-complete. In practical deployments, however, limited by the capabilities of users and models, it is difficult or even impossible to directly obtain a perfect textual query suiting the entire scene and model, thereby leading to performance degradation. To address this issue, we propose a novel Interactive Text-3D Scene Retrieval Method (IDeal), which promotes the enhancement of the alignment between texts and 3D scenes through continuous interaction. To achieve this, we present an Interactive Retrieval Refinement framework (IRR), which employs a questioner to pose contextually relevant questions to an answerer in successive rounds that either promote detailed probing or encourage exploratory divergence within scenes. Upon the iterative responses received from the answerer, IRR adopts a retriever to perform both feature-level and semantic-level information fusion, facilitating scene-level interaction and understanding for more precise re-rankings. To bridge the domain gap between queries and interactive texts, we propose an Interaction Adaptation Tuning strategy (IAT). IAT mitigates the discriminability and diversity risks among augmented text features that approximate the interaction text domain, achieving contrastive domain adaptation for our retriever. Extensive experimental results on three datasets demonstrate the superiority of IDeal. 

## Challenge 🖊
Extending existing static interactive methods to Text–3D Scene Retrieval faces two major challenges:
![challenge](./imgs/challenge.png)

## Method
![method](./imgs/method.png)

## Requirements ⚙️
following [RoMa](https://github.com/Yangl1nFeng/RoMa), the complex training and testing sets of this work have already been fully preprocessed by us, so the requirements for external libraries are not complex:
- python 3.8.16
- open3d 0.17.0
- pyTorch 1.12.1
- torchvision 0.13.1
- numpy 1.24.3
- transformers
- tensorboard_logger

Due to resource limitations, we use Qwen-Instruct 7B as the base model. The model is provided in our [Google Drive](https://drive.google.com/drive/folders/1ol7bwABpCJfb_xmqN7WmkF2Ci_l8smCJ?usp=sharing). Please download it and place it under `./lib/LLMs/qwen_7b`.

You are also encouraged to try newer or larger models, which may lead to better performance. Our experiments only represent a basic initial attempt.

## Data 📕
We follow RoMa for the use of query data to ensure fair and comprehensive evaluation.  

Please refer to RoMa to obtain [point cloud data](https://drive.google.com/drive/folders/19lox3eRF0EAVjz6TcQDb7Ns9vjI9qkXN?usp=drive_link), and place it in `./lib/data`. 

Please place the data from [Google Drive](https://drive.google.com/drive/folders/1ol7bwABpCJfb_xmqN7WmkF2Ci_l8smCJ?usp=sharing) into the corresponding directory under `./lib/data`.

The final directory structure should be:
```
data/
├── split/
│   ├── ScanRefer_filtered_train.txt
│   └── ScanRefer_filtered_val.txt
└── text/
    └── scanrefer/
        ├── ori_data.jsonl
        ├── ScanRefer_filtered_val_with_m...
        ├── scanrefer_merged_data_8_9.jsonl
        ├── ScanRefer_val_self_memory_8_9.json
        ├── train_memory_generate.py
├── pt2vec_200_random_pos_train.npy
├── pt2vec_200_random_pos_val.npy
├── pt2vec_200_random_train.npy
└── pt2vec_200_random_val.npy
```
To make the process clearer, we've broken the project down into several steps, rather than making it too complicated with an end-to-end approach.

## Interaction Adaptation Tuning 
First, you need to obtain a retrieval model trained on the original data without any interaction. After completing the above steps, run:
`./sh/train_scanrefer.sh`

Then set the obtained last or optimal checkpoints as the base model for the transfering and run:
`./sh/train_transfer_scanrefer.sh`

Then your model can be migrated to the interaction domain. Checkpoints please refer our [Google Drive](https://drive.google.com/drive/folders/1ol7bwABpCJfb_xmqN7WmkF2Ci_l8smCJ?usp=sharing), put it into `./runs/scanrefer_train_butd_ESAregion_bigru`.

(You can refer to `./lib/data/text/scanrefer/train_memory_generate.py` for the approach to obtaining interactive domain training data. In fact, we have already provided you with the generated data: `./lib/data/text/scanrefer/scanrefer_merged_data_8_9.jsonl`.)

## Interactive Retrieval
Rename the best tuned model and using the `./runs/scanrefer_train_butd_ESAregion_bigru/transfer.pth` that have been transfered and obtained in the previous step, then perform next interactive retrieval.

### Coarse-grained description memory
1. Interaction
Run `./sh/I_interaction.sh`.
Obtaining interactive texts, text features, and some intermediate metrics. 
2. Summary
Run `./sh/I_summary.sh`.
Obtaining summary texts.
3. Retrieval
Run `./sh/I_retrieval.sh`.
Using multi-round text to retrieve 3D scenes。
All intermediate interactive texts, summaries, their encoded features, and intermediate metrics are provided as **checkpoint cases**. Please refer to our [Google Drive](https://drive.google.com/drive/folders/1ol7bwABpCJfb_xmqN7WmkF2Ci_l8smCJ?usp=sharing).

R@1, R@5, R@10 Result: Interactive Text retrieve img: 16.5, 43.0, 59.4.

### Fine-grained description memory
(Updating in the next following day...)

## Reference 🤗
If this paper is helpful for your research, please cite:
```bibtex
@inproceedings{fenginteractive,
  title={Interactive Cross-modal Learning for Text-3D Scene Retrieval},
  author={Feng, Yanglin and Li, Yongxiang and Sun, Yuan and Qin, Yang and Peng, Dezhong and Hu, Peng},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
```bibtex
@article{feng2025pointcloud,
  title={Pointcloud-text matching: Benchmark dataset and baseline},
  author={Feng, Yanglin and Qin, Yang and Peng, Dezhong and Zhu, Hongyuan and Peng, Xi and Hu, Peng},
  journal={IEEE Transactions on Multimedia},
  year={2025},
  publisher={IEEE}
}
```
Feel free to reach out for discussion or collaboration: fcyzfyl@163.com