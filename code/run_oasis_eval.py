"""
Evaluate the simulation in certain settings
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
from collections import defaultdict
from transformers import AutoTokenizer
from utils.logger import setup_logger
from collect_simu_res import collect_simu_data
import random
import subprocess
from evaluation_prl import eval_dataset, cal_token_chat, cal_token_hisim, cal_token_oasis, semantic_sim_pheme
import sqlite3
from sentence_transformers import SentenceTransformer
import torch

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

def twitter_simulation(args):
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    def process_task(config_file, args):
        # scene_id = '_'.join(config_file.split('_')[-2:]).replace('.yaml','')  # i_k
        scene_id = config_file.split('/')[-1].replace('.yaml','')
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
    logger.info("  Num tasks = %d", len(configs))   
    for config in tqdm(configs, desc="simulating"):
        scene_id, res, token_count = process_task(config, args)
        all_res[scene_id] = res 
        all_token_count[scene_id] = token_count
        logger.info(f'Task {scene_id} completed!')  
    return all_res, all_token_count



def hisim_simulation(args):
    def process_task(config_file, args):
        scene_id = config_file.replace('.yaml','')
        output_path = os.path.join(args.output_dir, f'{scene_id}.pkl')
        config = json.load(open(args.simu_config_path,'r'))
        if not os.path.exists(output_path):
            # simulation        
            task = Simulation.from_task(config_file, args.config_output_dir, None, os.path.join(args.config_output_dir, config_file))
            task.run()
        df = pickle.load(open(output_path,'rb'))
        res = defaultdict(dict)
        for user in df:
            if user in ['opinion_results','token_count']:continue
            for step in df[user]:
                content = df[user][step]['response'].content
                res[user][step] = [content]
        token_count = {"prompt_tokens":df["token_count"]["prompt_tokens"],"completion_tokens":df["token_count"]["completion_tokens"]}
        return scene_id, res, token_count
    all_res = {}
    all_token_count = {}
    # configs = [f for f in os.listdir(args.config_output_dir+f'{iters}') if os.path.isfile(os.path.join(args.config_output_dir+f'{iters}', f)) and f.endswith('.yaml')]
    configs = [f for f in os.listdir(args.config_output_dir) if os.path.isfile(os.path.join(args.config_output_dir, f)) and f.endswith('.yaml')]
    logger.info("***** Runing Simulation*****")
    logger.info("  Num tasks = %d", len(configs))  
    with ThreadPoolExecutor(max_workers=args.task_workers) as executor:
        futures = {executor.submit(process_task, config_file, args): config_file for config_file in configs}
        for future in tqdm(as_completed(futures), total=len(configs), desc="simulating"):
            config_file = futures[future]
            scene_id, res, token_count = future.result()
            all_res[scene_id] = res  
            all_token_count[scene_id] = token_count
            logger.info(f'Task {scene_id} completed!')

    return all_res, all_token_count            




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_rule_pool', type=str, default=None, help='path of initial rules')
    parser.add_argument('--dataset', type=str, default="pheme", help='dataset for simulation')
    parser.add_argument('--data_path', type=str, default="../data/pheme/processed/", help='dataset for simulation')
    parser.add_argument('--sample_size', type=int, default=1, help='number of samples for each scenario')
    parser.add_argument('--simu_config_path', type=str, default="../configs/pheme.json", help='path of overall simulation cofig')
    parser.add_argument('--config_output_dir', type=str, default="../scen_configs/pheme_oasis/", help='dir of generated cofigs')
    parser.add_argument('--output_dir', type=str, default="../output/pheme_oasis/", help='dir of simulation output')
    parser.add_argument('--judge_llm', type=str, default="gpt-4o-mini", help='judge model name')
    parser.add_argument('--judge_base_url', type=str, default="", help='url of API interface')
    parser.add_argument('--judge_api_key', type=str, default="", help='key of API interface')
    parser.add_argument('--task_workers', type=int, default=2, help='num of parallel workers for simulation')
    parser.add_argument('--tokenizer', type=str, default="/remote-home/share/LLM_CKPT/huggingface_models/Llama-3.1-8B-Instruct")
    parser.add_argument('--score_ckpt_dir', type=str, default="score_ckpt/pheme_base/")
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logger('Simulation', args.output_dir, 0)
    logger.info('Simulating: {}'.format(args.dataset))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # prepare task configs
    if args.init_rule_pool:
        rule_pool = open(args.init_rule_pool,'r').readlines()
        rule_pool = [r.strip() for r in rule_pool]        
    else:
        rule_pool=[]
        
    # initialization
    ref_data, ref_persona = prepare_tasks(args.dataset, rule_pool, args.config_output_dir, args.output_dir,args.sample_size, args.simu_config_path, args.data_path, 0, False)   
    # simulation
    all_res, all_token_count = twitter_simulation(args) 
    
    simu_data, act_simu_token_count = collect_simu_data(args.dataset, args.output_dir)
    simu_token_count = cal_token_oasis(simu_data, tokenizer)
    # evaluation
    res = eval_dataset(args.dataset, simu_data, act_simu_token_count, ref_data, args.judge_llm, args.judge_base_url, args.judge_api_key, args.score_ckpt_dir)
    print(res)
    print('simu_token_cnt: ', simu_token_count)
    
    with open(args.score_ckpt_dir+'/final_res.json','w') as f:
        json.dump({"metrics":res,"token":simu_token_count}, f)
 
    
if __name__=="__main__":
    main()