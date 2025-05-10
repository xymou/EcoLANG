"""
Evolve the language through communication
"""

import os
import argparse
import json
import os
import numpy as np
import pickle
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from prepare_config import prepare_config
from diversification import crossover, mutation
from myselect import select
from collections import defaultdict
from chat_simulation import ChatSimulation
from transformers import AutoTokenizer
from utils.logger import setup_logger
from collect_simu_res import collect_simu_data
import random


def init_rules(path):
    try:
        rules = open(path, 'r').readlines()
        rules = [t.strip() for t in rules]
        logger.info(f"# of init rules: {len(rules)}")
    except:
        raise ValueError(f"Cannot read {path}")
    return rules


def prepare_tasks(
    dataset,
    rule_pool,
    config_output_dir,
    output_dir,
    sample_size,
    config_path,
    data_path,
    iters,
    evo
):
    """Prepare simulation configs given current rules and task data
    """
    return prepare_config(dataset,rule_pool,config_output_dir,output_dir,sample_size,config_path, data_path, iters, evo)
    
    
    
def chat_simulation(args, iters=1):
    output_dir = args.output_dir+f'/{iters}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    def process_task(config_file, args):
        task = ChatSimulation.from_task(os.path.join(args.config_output_dir+f'{iters}', config_file), output_dir)
        scene_id = task.scene['scene_id']
        
        output_path = os.path.join(output_dir, f'{scene_id}.json')
        
        if os.path.exists(output_path):
            res = json.load(open(output_path, 'r'))['chat_history']
            token_count = json.load(open(output_path, 'r'))['token_count']
        else:
            # simulation and evaluation
            task.run()
            res = task.groupchat_result.chat_history
            token_count = {'prompt_tokens':task.prompt_tokens,'completion_tokens':task.completion_tokens,
                    'total_tokens':task.total_tokens}
        
        return scene_id, res, token_count
    
    all_res = {}
    all_token_count = {}
    configs = [f for f in os.listdir(args.config_output_dir+f'{iters}') if os.path.isfile(os.path.join(args.config_output_dir+f'{iters}', f)) and f.endswith('.yaml')]
    logger.info("***** Runing Simulation at iter %d*****", iters)
    logger.info("  Num tasks = %d", len(configs))    
    with ThreadPoolExecutor(max_workers=args.task_workers) as executor:
        futures = {executor.submit(process_task, config_file, args): config_file for config_file in configs}    
        for future in tqdm(as_completed(futures), total=len(configs), desc="simulating"):
            config_file = futures[future]
            try:
                scene_id, res, token_count = future.result()
                all_res[scene_id] = res    
                all_token_count[scene_id] = token_count
                logger.info(f'Task {scene_id} completed!')
            except Exception as e:
                logger.error(f"Error in simulating {config_file}: {e}")
    return all_res, all_token_count
    
def twitter_simulation(args, iters=1):
    output_dir = args.output_dir+f'/{iters}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    def process_task(config_file, args):
        scene_id = '_'.join(config_file.split('_')[-2:]).replace('.yaml','')  # i_k
        output_path = os.path.join(output_dir, f'{scene_id}.db')
        
        if not os.path.exists(output_path):
            # simulation        
            cmd = f"""python twitter_simulation.py --config_path {config_file}"""
            print(cmd)
            process = subprocess.Popen(
                cmd, shell=True
            )
            process.wait()
        # load the generated posts
        print(output_path)
        conn = sqlite3.connect(output_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM post")
        rows = cursor.fetchall()     
        res = defaultdict(list)  #user2posts
        for row in rows: 
            user_id = row[1]
            content = row[2]
            res[user_id].append(content)
        token_count = json.load(open(os.path.join(output_dir, f'{scene_id}.json')))
        return scene_id, res, token_count
    all_res = {}
    all_token_count = {}
    configs = [os.path.join(args.config_output_dir, f) for f in os.listdir(args.config_output_dir) if os.path.isfile(os.path.join(args.config_output_dir, f)) and f.endswith('.yaml')]
    logger.info("***** Runing Simulation at iter %d*****", iters)
    logger.info("  Num tasks = %d", len(configs))   
    for config in tqdm(configs, desc="simulating"):
        scene_id, res = process_task(config, args)
        all_res[scene_id] = res 
        all_token_count[scene_id] = token_count
        logger.info(f'Task {scene_id} completed!')  
    return all_res, all_token_count


def diverse_rules(
    rule2score,
    num_child,
    model, 
    base_url,
    api_key
    ):
    """ Crossover & mutation
    select parents using Roulette Wheel - the prob of being chosen is p_i = s_i /\sum_{j=1}{s_j}
    """
    child_rules = []
    rules, scores = [key for key in rule2score],[rule2score[key] for key in rule2score]
    # normalization of the scores
    weights = [scores[i]/sum(scores) for i in range(len(scores))]
    for i in range(num_child):
        parents = random.choices(rules, weights = weights, k=2)
        new_rule = crossover(parents[0],parents[1],model,base_url,api_key)
        new_rule = mutation(new_rule,model,base_url,api_key)
        child_rules.append(new_rule)
    return child_rules


def single_iter(
    args,
    rule_pool,
    iters
):
    """Single interaction of language evolution
    - input: current rule pool, communication tasks
    - output: updated rule pool, communication records
    """
    # prepare task configs
    ref_data, ref_persona = prepare_tasks(args.dataset, rule_pool, args.config_output_dir, args.output_dir, args.sample_size, args.simu_config_path, args.data_path, iters, True)
    # simulation
    if args.dataset=='persona':
        chat_simulation(args,iters=iters)
    simu_data, simu_token_count = collect_simu_data(args.dataset, args.output_dir+f'/{iters}/')
    
    # selection
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    iter_output_dir = args.iter_output_dir+f'/{iters}/'
    if len(simu_data):
        rule2score, records, all_score_detail, all_detail = select(args.dataset, simu_data, simu_token_count, ref_data, ref_persona, rule_pool, args.judge_llm, args.judge_base_url, args.judge_api_key, tokenizer, args.w1, args.w2, args.w3, args.topk, iter_output_dir)    

    with open(iter_output_dir+'rule2score.pkl','wb') as f:
        pickle.dump(rule2score, f)
    new_rules = diverse_rules(rule2score, args.num_child, args.judge_llm, args.judge_base_url, args.judge_api_key)
    
    # updation
    # replace the worst indis with the newly generated ones
    rules = sorted(rule2score.items(), key=lambda x: x[1], reverse=True)
    rules = rules[:args.reserve_parent]
    rules = [r[0] for r in rules]
    rule_pool = rules + new_rules
    return rule_pool, records, all_score_detail, all_detail
    
    


def evolution(
    args,
    rule_pool,
):
    """Language evolution through natural selection
    > step1: init rules
    > step2: language using through communication (simulation)
    > step3: selection
    > step4: crossover & mutation
    > step5: update rules
    """
    rules = rule_pool
    for i in range(args.iters):
        logger.info(f"Evo iter{i} starts!")
        if os.path.exists(args.iter_output_dir+f'/{i}/rules.pkl'):
            new_rules = pickle.load(open(args.iter_output_dir+f'/{i}/rules.pkl','rb'))
            records = pickle.load(open(args.iter_output_dir+f'/{i}/records.pkl','rb'))
            all_score_detail = pickle.load(open(args.iter_output_dir+f'/{i}/all_score_detail.pkl','rb'))
            all_detail = pickle.load(open(args.iter_output_dir+f'/{i}/all_detail.pkl','rb'))
        else:
            new_rules, records, all_score_detail, all_detail = single_iter(
                args,
                rules,
                iters=i
            )
            # save the rules and records of current iteration
            if not os.path.exists(args.iter_output_dir+f'/{i}/'):
                os.makedirs(args.iter_output_dir+f'/{i}/')
            with open(args.iter_output_dir+f'/{i}/rules.pkl','wb') as f:
                pickle.dump(new_rules, f)
            with open(args.iter_output_dir+f'/{i}/records.pkl','wb') as f:
                pickle.dump(records, f)                            
        rules = new_rules
        logger.info(f"Evo iter{i} ends!")



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_rule_pool', type=str, default="../rules/init_rules.txt", help='path of initial rules')
    parser.add_argument('--dataset', type=str, default="persona", help='dataset for simulation')
    parser.add_argument('--data_path', type=str, default="..data/syn_persona_chat/Synthetic-Persona-Chat_valid.csv", help='dataset for simulation')
    parser.add_argument('--sample_size', type=int, default=5, help='number of samples for each scenario')
    parser.add_argument('--simu_config_path', type=str, default="../configs/syn_persona.json", help='path of overall simulation cofig')
    parser.add_argument('--config_output_dir', type=str, default="../scen_configs/persona/", help='dir of generated configs for each scenario')
    parser.add_argument('--output_dir', type=str, default="../output/persona", help='dir of simulation output')
    parser.add_argument('--iter_output_dir', type=str, default="iter_output/persona", help='dir of evaluation output')
    parser.add_argument('--judge_llm', type=str, default="gpt-4o", help='judge model name')
    parser.add_argument('--judge_base_url', type=str, default="", help='url of API interface')
    parser.add_argument('--judge_api_key', type=str, default="", help='key of API interface')
    parser.add_argument('--w1', type=float, default=0.4, help='weight of simulation acc')
    parser.add_argument('--w2', type=float, default=0.3, help='weight of token len')
    parser.add_argument('--w3', type=float, default=0.3, help='weight of expresiveness')
    parser.add_argument('--topk', type=float, default=0.7, help='the ratio of reserved traj')
    parser.add_argument('--num_child', type=int, default=5, help='the # of childs for each iteration')
    parser.add_argument('--reserve_parent', type=int, default=5, help='the # of reserved parent rules')
    parser.add_argument('--iters', type=int, default=10, help='the number of iteration')
    parser.add_argument('--tokenizer_path', type=str, default="/remote-home/share/LLM_CKPT/huggingface_models/Llama-3.1-8B-Instruct", help='path of tokenizer')
    parser.add_argument('--task_workers', type=int, default=4, help='num of parallel workers for simulation')
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logger('Simulation', args.output_dir, 0)
    logger.info('Simulating: {}'.format(args.dataset))
    
    # prepare task configs
    rule_pool = open(args.init_rule_pool,'r').readlines()
    rule_pool = [r.strip() for r in rule_pool]
    evolution(args, rule_pool)
    
    

if __name__=="__main__":
    main()