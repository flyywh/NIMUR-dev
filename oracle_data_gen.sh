CUDA_VISIBLE_DEVICES=0,1,2,3 python ./scripts/ORACLE_CNN_dataset_v2.py \
    --label ./data/oracle_cnn_inputs/label.txt \
    --train ./data/oracle_cnn_inputs/train.fasta \
    --valid ./data/oracle_cnn_inputs/valid.fasta \
    --outdir data/oracle_cnn/
