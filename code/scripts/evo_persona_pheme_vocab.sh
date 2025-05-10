init_rule_pool="../rules/init_rules.txt"
dataset="persona"
data_path="../data/syn_persona_chat/Synthetic-Persona-Chat_valid.csv"
simu_config_path="../configs/syn_persona_pheme_vocab.json"
config_output_dir="../scen_configs/persona_pheme_vocab/"
output_dir="../output/persona_pheme_vocab/"
iter_output_dir="../iter_output/persona_pheme_vocab/"
sample_size=5
judge_llm="gpt-4o"
judge_base_url="YOUR_BASE_URL"
judge_api_key="YOUR_API_KEY"
w1=1
w2=0.6
w3=0.6
topk=0.7
num_child=5
reserve_parent=5
iters=5
task_workers=12

python run_evolution.py \
    --init_rule_pool $init_rule_pool \
    --dataset $dataset \
    --data_path $data_path \
    --simu_config_path $simu_config_path \
    --config_output_dir $config_output_dir \
    --output_dir $output_dir \
    --iter_output_dir $iter_output_dir \
    --sample_size $sample_size \
    --judge_llm $judge_llm \
    --judge_base_url $judge_base_url \
    --judge_api_key $judge_api_key \
    --w1 $w1 \
    --w2 $w2 \
    --w3 $w3 \
    --topk $topk \
    --num_child $num_child \
    --reserve_parent $reserve_parent \
    --iters $iters \
    --task_workers $task_workers
