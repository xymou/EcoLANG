for i in {1..3}
do
    output_dir="../output/hisim_base_${i}/"
    config_output_dir="../scen_configs/hisim_base_${i}/"
    score_ckpt_dir="../score_ckpt/hisim_base_${i}/"
    
    OUTPUT_DIR="$output_dir" CONFIG_OUTPUT_DIR="$config_output_dir" SCORE_CKPT_DIR="$score_ckpt_dir" ./scripts/hisim/eval_hisim.sh
done