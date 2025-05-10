dataset="hisim"
data_path="../data/hisim/"
sample_size=1
simu_config_path="../configs/hisim_oasis.json"
config_output_dir=${CONFIG_OUTPUT_DIR}
output_dir=${OUTPUT_DIR}
task_workers=1
score_ckpt_dir=${SCORE_CKPT_DIR}
judge_llm="gpt-4o-mini"
judge_base_url="YOUR_BASE_URL"
judge_api_key="YOUR_API_KEY"

python run_oasis_hisim_eval.py \
    --dataset $dataset \
    --data_path $data_path \
    --simu_config_path $simu_config_path \
    --config_output_dir $config_output_dir \
    --output_dir $output_dir \
    --sample_size $sample_size \
    --task_workers $task_workers \
    --score_ckpt_dir $score_ckpt_dir \
    --judge_llm $judge_llm \
    --judge_base_url $judge_base_url \
    --judge_api_key $judge_api_key
