
 bash scripts_train/08_full_tune_pipeline_lora.sh \
    --env evo2 --gpus 7 \
    --train-pos data/MiBiG_train_pos_esm2_1022.fasta \
    --train-neg data/MiBiG_train_neg_esm2_1022.fasta \
    --valid-pos data/MiBiG_valid_pos_esm2_1022.fasta \
    --valid-neg data/MiBiG_valid_neg_esm2_1022.fasta \
    --test-pos data/MiBiG_test_pos_esm2_1022.fasta \
    --test-neg data/MiBiG_test_neg_esm2_1022.fasta \
    --evo-train-pos data/MiBiG_train_pos.fasta \
    --evo-train-neg data/MiBiG_train_neg.fasta \
    --evo-test-pos data/MiBiG_test_pos.fasta \
    --evo-test-neg data/MiBiG_test_neg.fasta \
    --kg-tsv data/MiBiG_formatted_BGC_2.tsv
