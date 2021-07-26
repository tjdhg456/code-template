## train
python ./train.py \
  --config-file ./configs/faster_rcnn_KDN.yaml \
  --output_dir checkpoint \
  --num-gpus 1 SOLVER.IMS_PER_BATCH 16 \
