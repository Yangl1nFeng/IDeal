DATASET_NAME='scanrefer_ori'
DATA_PATH='../../data/'${DATASET_NAME}
VOCAB_PATH='./lib/vocab/'
MODEL_NAME='runs/'${DATASET_NAME}'_butd_ESAregion_bert'

CUDA_VISIBLE_DEVICES=0 python3 ./I_retr_Qwen_bert.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
  --logger_name ${MODEL_NAME}/log --model_name ${MODEL_NAME} \
  --num_epochs=50 --lr_update=24 --learning_rate=.0003  --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 3 --batch_size 1 --hardnum 2 --max_violation