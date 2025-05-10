for i in {1..3}
do
    output_dir="../output/pheme_vocab_topdown_rule_${i}/"
    config_output_dir="../scen_configs/pheme_vocab_topdown_rule_${i}/"
    score_ckpt_dir="../score_ckpt/pheme_vocab_topdown_rule_${i}/"
    
    OUTPUT_DIR="$output_dir" CONFIG_OUTPUT_DIR="$config_output_dir" SCORE_CKPT_DIR="$score_ckpt_dir" ./scripts/pheme/eval_pheme_vocab_topdown_rule.sh
done