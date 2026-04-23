python scripts/prepare_oracle_cnn_inputs.py \
  --in_fasta data/MiBiG_formatted_BGC.fasta \
  --out_dir data/oracle_cnn_inputs \
  --train_ratio 0.8 \
  --add_neg \
  --neg_ratio 0.2

