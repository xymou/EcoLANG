for i in {1..3}
do
    output_dir="../output/pheme_base_${i}/"
    config_output_dir="../scen_configs/pheme_base_${i}/"
    score_ckpt_dir="../score_ckpt/pheme_base_${i}/"
    
    OUTPUT_DIR="$output_dir" CONFIG_OUTPUT_DIR="$config_output_dir" SCORE_CKPT_DIR="$score_ckpt_dir" ./scripts/pheme/eval_pheme.sh
done