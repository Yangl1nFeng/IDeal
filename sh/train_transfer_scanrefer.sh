DATASET_NAME='scanrefer_train'
DATA_PATH='../../data/'${DATASET_NAME}
VOCAB_PATH='./lib/vocab/'
MODEL_NAME='runs/'${DATASET_NAME}'_butd_ESAregion_bigru'

CUDA_VISIBLE_DEVICES=0 python3 ./train_bert.py \
  --data_path ${DATA_PATH} --data_name ${DATASET_NAME} --vocab_path ${VOCAB_PATH}\
  --logger_name ${MODEL_NAME}/log --model_name ${MODEL_NAME} \
  --is_transfering --original_ckpt ./runs/scanrefer_train_butd_ESAregion_bigru/source.pth \
  --num_epochs=12 --lr_update=5 --learning_rate=.0003  --workers 10 \
  --log_step 200 --embed_size 1024 --vse_mean_warmup_epochs 3 --batch_size 5 --hardnum 2 --max_violation
 
