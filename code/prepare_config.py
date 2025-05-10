"""
Prepare simulation scenarios
"""

import os
import json
import re
import pickle
from tqdm import tqdm
import random
import yaml
from sentence_transformers import SentenceTransformer,util
from collections import defaultdict
import pandas as pd
import torch
from datetime import datetime, timedelta
from multiprocessing import Pool            
   
def write_file(args):
    data, file_path = args
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    progress_bar.update(1)


def prepare_config(
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
    if dataset =='persona':
        func = prepare_persona
    elif dataset =='pheme':
        func = prepare_pheme
    elif dataset == 'hisim':
        func = prepare_hisim_oasis
    else:
        raise NotImplementedError(f"Missing preparation function for {dataset}")
    return func(
        rule_pool,
        config_output_dir,
        output_dir,
        sample_size,
        config_path,
        data_path,
        iters,
        evo        
    )


def clean_src_tweet(text):
    text = re.sub(r'http[s]?://\S+', '', text).strip()
    return text



def prepare_pheme(
    rule_pool,
    config_output_dir,
    output_dir,
    sample_size,
    config_path,
    data_path,
    iters,
    evo=False
):
    """
    prepare configs for simulation and reference data
    """
    data =  os.listdir(data_path)
    data = [d for d in data if d.endswith('csv')]
    config = json.load(open(config_path))
    tasks = []
    ref_data = {}
    PROMPT_TEMPATE = open(config['base_prompt_path'],'r').readlines()
    twt2time = json.load(open(config["start_time_dict"],'r'))
    if len(rule_pool)>1:
        config_output_dir = config_output_dir+f'/{iters}/'
    if not os.path.exists(config_output_dir):
        os.makedirs(config_output_dir)
    if len(os.listdir(config_output_dir)):
        ref_data = pickle.load(open(config_output_dir+'/ref_data.pkl','rb'))
        return ref_data, {}           
        
    max_depth = json.load(open(data_path+'/max_depth.json','r'))
    src_tweet = json.load(open(data_path+'/source_tweet.json','r'))
    
    if not rule_pool:
        for i in range(len(data)):
            key = data[i][:-4]
            user_path = data_path+'/'+data[i]
            ref_data[f"{key}"] = json.load(open(config["ref_data_path"]+key+'.json','r'))
            src_news = clean_src_tweet(src_tweet[key]['text'])   
            
            mydata={
                "data":{
                    "db_path":output_dir+f'/{key}.db', # config["output_dir"]+f'/{key}.db',
                    "csv_path":user_path,
                    "token_path":output_dir+f'/{key}.json', # config["output_dir"]+f'/{key}.json',
                    },
                "simulation":{
                    "num_timesteps":max_depth[key],
                    "clock_factor":config["clock_factor"],
                    "recsys_type":config["recsys_type"],
                    "source_post_time":twt2time[key][11:]
                },
                "model":{
                    "num_agents":len(pd.read_csv(user_path)),
                    "model_random_seed":42,
                    "cfgs":[{
                        "model_type":config["llm"]["model"],
                        "num":len(pd.read_csv(user_path)),
                        "server_url":config["llm"]["base_url"],
                        "model_path":config["llm"]["model_path"],
                        "stop_tokens":["<|eot_id|>", "<|end_of_text|>"],
                        "temperature":config["llm"]["temperature"],
                        "max_tokens":config["llm"]["max_tokens"],  
                    }]
                },
                "inference":{
                    "model_type":config["llm"]["model"],
                    "model_path":config["llm"]["model_path"],
                    "temperature":config["llm"]["temperature"],
                    "max_tokens":config["llm"]["max_tokens"], 
                    "vocab_path":config["llm"]['vocab_path']  if "vocab_path" in config["llm"] else "",
                    "stop_tokens":["<|eot_id|>", "<|end_of_text|>"],
                    "server_url":[{
                        "host":config["server_url"],
                        "ports":config["ports"]
                    }] 
                },
                    "prompt_path":config["base_prompt_path"]
                }
            with open(config_output_dir+f'{key}.yaml','w') as f:
                yaml.dump(mydata, f, default_flow_style=False, sort_keys=False)        
    else:    
        for i in range(len(data)):
            key = data[i][:-4]
            user_path = data_path+'/'+data[i]
            sampled_rule = random.sample(rule_pool, k=sample_size)
            for k in range(sample_size):
                idx = rule_pool.index(sampled_rule[k])
                # ref_data[f"{key}_{idx}"] = json.load(open(config["ref_data_path"]+key+'.json','r'))
                ref_data[f"{key}"] = json.load(open(config["ref_data_path"]+key+'.json','r'))
                src_news = clean_src_tweet(src_tweet[key]['text'])
                # prompt = "".join(PROMPT_TEMPATE).format(news = src_news).split("\n")+ ["When you create new post or comment, "+sampled_rule[k]]
                prompt = PROMPT_TEMPATE+ ["When you quote, "+sampled_rule[k]]
                with open(config_output_dir+f'prompt_{key}_{idx}.txt','w') as f:
                    for line in prompt:
                        f.write(line)
             
                mydata={
                    "data":{
                        "db_path":output_dir+f'/{iters}/{key}_{idx}.db' if evo else output_dir+f'/{key}_{idx}.db',# config["output_dir"]+f'/{iters}/{key}_{idx}.db',
                        "csv_path":user_path,
                        "token_path":output_dir+f'/{iters}/{key}_{idx}.json' if evo else output_dir+f'/{key}_{idx}.json',# config["output_dir"]+f'/{iters}/{key}_{idx}.json',
                    },
                    "simulation":{
                        "num_timesteps":max_depth[key],
                        "clock_factor":config["clock_factor"],
                        "recsys_type":config["recsys_type"],
                        "source_post_time":twt2time[key]
                    },
                    "model":{
                        "num_agents":len(pd.read_csv(user_path)),
                        "model_random_seed":42,
                        "cfgs":[{
                            "model_type":config["llm"]["model"],
                            "num":len(pd.read_csv(user_path)),
                            "server_url":config["llm"]["base_url"],
                            "model_path":config["llm"]["model_path"],
                            "stop_tokens":["<|eot_id|>", "<|end_of_text|>"],
                            "temperature":config["llm"]["temperature"],
                            "max_tokens":config["llm"]["max_tokens"],  
                        }]
                    },
                    "inference":{
                        "model_type":config["llm"]["model"],
                        "model_path":config["llm"]["model_path"],
                        "temperature":config["llm"]["temperature"],
                        "max_tokens":config["llm"]["max_tokens"], 
                        "vocab_path":config["llm"]['vocab_path']  if "vocab_path" in config["llm"] else "",
                        "stop_tokens":["<|eot_id|>", "<|end_of_text|>"],
                        "server_url":[{
                            "host":config["server_url"],
                            "ports":config["ports"]
                        }] 
                    },
                    "prompt_path":config_output_dir+f'prompt_{key}_{idx}.txt'
                }
                with open(config_output_dir+f'{key}_{idx}.yaml','w') as f:
                    yaml.dump(mydata, f, default_flow_style=False, sort_keys=False)
    with open(config_output_dir+'/ref_data.pkl','wb') as f:
        pickle.dump(ref_data, f)    
    return ref_data, {}
        
        
def transform_source_tweet(src_tweet):
    author = src_tweet["user"]["screen_name"]
    content = src_tweet["text"]
    return f'{author} posts a tweet: {content}'      
        


def prepare_hisim_oasis(
    rule_pool,
    config_output_dir,
    output_dir,
    sample_size,
    config_path,
    data_path,
    iters,
    evo=False
):
    """
    prepare configs for simulation and reference data
    """
    data = ['metoo.csv','roe.csv']
    config = json.load(open(config_path))
    ref_data = defaultdict(dict)

    PROMPT_TEMPATE = open(config['base_prompt_path'],'r').readlines()
    twt2time = {
        "roe":"22:00:00",
        "metoo":"20:00:00",
        "blm":"22:00:00"
    }
    trigger_news=config["trigger_news"]
    for key in trigger_news:
        trigger_news[key] = {int(k):v for k,v in trigger_news[key].items()}
    
    if len(rule_pool)>1:
        config_output_dir = config_output_dir+f'/{iters}/'
    if not os.path.exists(config_output_dir):
        os.makedirs(config_output_dir)

    
    if not rule_pool:
        for i in range(len(data)):
            key = data[i][:-4]
            user_path = data_path+'/processed/'+data[i]
            user2id={}
            df = pd.read_csv(user_path)
            for j in range(len(df)):
                user2id[df['name'][j]] = df['user_id'][j]
            
            gt_data = pickle.load(open(data_path+f'/{key}/macro_e2.pkl','rb'))
            for user in gt_data:
                ref_data[key][str(user2id[user])]={}
                for step in gt_data[user]:
                    ref_data[key][str(user2id[user])][step]=[d['rawContent'] for d in gt_data[user][step]]
            
            mydata={
                "data":{
                    "db_path":output_dir+f'/{key}.db',# config["output_dir"]+f'/{key}.db',
                    "csv_path":user_path,
                    "token_path":output_dir+f'/{key}.json',# config["output_dir"]+f'/{key}.json',
                    },
                "simulation":{
                    "num_timesteps":config["num_timesteps"],
                    "clock_factor":config["clock_factor"],
                    "recsys_type":config["recsys_type"],
                    "source_post_time":twt2time[key]
                },
                "model":{
                    "num_agents":len(pd.read_csv(user_path)),
                    "model_random_seed":42,
                    "cfgs":[{
                        "model_type":config["llm"]["model"],
                        "num":len(pd.read_csv(user_path)),
                        "server_url":config["llm"]["base_url"],
                        "model_path":config["llm"]["model_path"],
                        "stop_tokens":["<|eot_id|>", "<|end_of_text|>"],
                        "temperature":config["llm"]["temperature"],
                        "max_tokens":config["llm"]["max_tokens"],  
                    }]
                },
                "inference":{
                    "model_type":config["llm"]["model"],
                    "model_path":config["llm"]["model_path"],
                    "temperature":config["llm"]["temperature"],
                    "max_tokens":config["llm"]["max_tokens"], 
                    "vocab_path":config["llm"]['vocab_path']  if "vocab_path" in config["llm"] else "",
                    "stop_tokens":["<|eot_id|>", "<|end_of_text|>"],
                    "server_url":[{
                        "host":config["server_url"],
                        "ports":config["ports"]
                    }] 
                },
                    "prompt_path":config["base_prompt_path"],
                    "trigger_news":trigger_news[key]
                }
            with open(config_output_dir+f'{key}.yaml','w') as f:
                yaml.dump(mydata, f, default_flow_style=False, sort_keys=False)        
    else:    
        for i in range(len(data)):
            key = data[i][:-4]
            user_path = data_path+'/processed/'+data[i]
            user2id={}
            df = pd.read_csv(user_path)
            for j in range(len(df)):
                user2id[df['name'][j]] = df['user_id'][j]
                
            sampled_rule = random.sample(rule_pool, k=sample_size)
            gt_data = pickle.load(open(data_path+f'/{key}/macro_e2.pkl','rb'))
            for user in gt_data:
                ref_data[key][str(user2id[user])]={}
                for step in gt_data[user]:
                    ref_data[key][str(user2id[user])][step]=[d['rawContent'] for d in gt_data[user][step]]
                                      
            for k in range(sample_size):
                idx = rule_pool.index(sampled_rule[k])
                # prompt = "".join(PROMPT_TEMPATE).format(news = src_news).split("\n")+ ["When you create new post or comment, "+sampled_rule[k]]
                prompt = PROMPT_TEMPATE+ ["When you create post, quote or comment, "+sampled_rule[k]]
                with open(config_output_dir+f'prompt_{key}_{idx}.txt','w') as f:
                    for line in prompt:
                        f.write(line)
             
                mydata={
                    "data":{
                        "db_path":output_dir+f'/{iters}/{key}_{idx}.db' if evo else output_dir+f'/{key}_{idx}.db', # config["output_dir"]+f'/{iters}/{key}_{idx}.db',
                        "csv_path":user_path,
                        "token_path":output_dir+f'/{iters}/{key}_{idx}.json' if evo else output_dir+f'/{key}_{idx}.json',# config["output_dir"]+f'/{iters}/{key}_{idx}.json',
                    },
                    "simulation":{
                        "num_timesteps":config["num_timesteps"],
                        "clock_factor":config["clock_factor"],
                        "recsys_type":config["recsys_type"],
                        "source_post_time":twt2time[key]
                    },
                    "model":{
                        "num_agents":len(pd.read_csv(user_path)),
                        "model_random_seed":42,
                        "cfgs":[{
                            "model_type":config["llm"]["model"],
                            "num":len(pd.read_csv(user_path)),
                            "server_url":config["llm"]["base_url"],
                            "model_path":config["llm"]["model_path"],
                            "stop_tokens":["<|eot_id|>", "<|end_of_text|>"],
                            "temperature":config["llm"]["temperature"],
                            "max_tokens":config["llm"]["max_tokens"],  
                        }]
                    },
                    "inference":{
                        "model_type":config["llm"]["model"],
                        "model_path":config["llm"]["model_path"],
                        "temperature":config["llm"]["temperature"],
                        "max_tokens":config["llm"]["max_tokens"], 
                        "vocab_path":config["llm"]['vocab_path']  if "vocab_path" in config["llm"] else "",
                        "stop_tokens":["<|eot_id|>", "<|end_of_text|>"],
                        "server_url":[{
                            "host":config["server_url"],
                            "ports":config["ports"]
                        }] 
                    },
                    "prompt_path":config_output_dir+f'prompt_{key}_{idx}.txt',
                    "trigger_news":trigger_news[key]
                }
                with open(config_output_dir+f'{key}_{idx}.yaml','w') as f:
                    yaml.dump(mydata, f, default_flow_style=False, sort_keys=False)
    return ref_data, {}


def prepare_persona(
    rule_pool,
    config_output_dir,
    output_dir,
    sample_size = 5,
    config_path = '/xymou/AgentCom_local/evo/configs/syn_persona.json',
    data_path='/xymou/AgentCom_local/task/syn_persona_chat/Synthetic-Persona-Chat_valid.csv',
    iters=0,
    evo=True
    ):
    """
    prepare configs for simulation and reference data
    """
    def change_person(text):
        """change to the second person"""
        text = text.replace('I ','You ')
        text = text.replace(' i ','you')
        text = text.replace('My','Your')
        text = text.replace('my ','your')
        text = text.replace(' me ',' you ')
        text = text.replace('\n',' ')
        text = text.replace(' am ',' are ')
        text = text.replace(' was ',' were ')
        
        return text
    
    def parse_chat_history(chat_history):
        chat_history = chat_history[:15]
        data = []
        for d in chat_history:
            d = d.strip().split(': ')
            if len(d)<2:continue
            d[0] = d[0].replace(" ","")
            data.append({"name":d[0],"content":d[1]})
        return data
    
    config_output_dir = config_output_dir+f'/{iters}/'
    if not os.path.exists(config_output_dir):
        os.makedirs(config_output_dir)
    if len(os.listdir(config_output_dir)):
        ref_data = pickle.load(open(config_output_dir+'/ref_data.pkl','rb'))
        ref_persona = pickle.load(open(config_output_dir+'/ref_persona.pkl','rb'))  
        return ref_data, ref_persona    
    

    PROMPT_TEMPATE = """You are {agent_name}. {desc}\n{few_shot_diag}\nWhat will you, {agent_name}, speak next?"""
    df = pd.read_csv(data_path)
    config = json.load(open(config_path))
    tasks = []
    ref_data, ref_persona = {}, {}
    for i in tqdm(range(len(df))):
        persona1 = change_person(df['user 1 personas'][i])
        persona2 = change_person(df['user 2 personas'][i])
        personas = [persona1, persona2]
        conv_history = df['Best Generated Conversation'][i].split('\n')
        sampled_rule = random.sample(rule_pool, k=sample_size)
        for k in range(sample_size):
            idx = rule_pool.index(sampled_rule[k])
            ref_persona[f"{i}_{idx}"]=[{"name":"User1","profile":persona1}, {"name":"User2","profile":persona2}]
            ref_data[f"{i}_{idx}"]=parse_chat_history(conv_history[2:])
            
            data = []
            start_msg = conv_history[2]
            for j in range(len(personas)):
                name = f"User{j+1}"
                data.append({
                    "name":name,
                    "system_prompt":PROMPT_TEMPATE.format(agent_name=name, desc=personas[j], few_shot_diag="\n".join(conv_history[:2]))+'\n'+sampled_rule[k]
                })
            scene = {"scene_id":f"{i}_{idx}",
                     "start_msg":start_msg}
            groupchat = {
                "messages":[{"role":"User1","content":conv_history[0].replace("User 1: ","").replace("(name)","User 1").replace("(user 1's name)","User 1")},
                            {"role":"User2","content":conv_history[1].replace("User 2: ","").replace("(name)","User 2").replace("(user 2's name)","User 2")}],
                "max_round":config["max_round"],
                "speaker_selection_method":config["speaker_selection_method"],
                "allow_repeat_speaker":False
            }
            port = random.choice(config["llm"]["ports"])
            if "vocab_path" in config["llm"]:
                agents = [
                    {
                        "name":agent["name"],
                        "system_prompt":agent["system_prompt"],
                        "llm":{
                            "model":config["llm"]["model"],
                            "base_url":"http://"+config["llm"]["base_url"]+f":{port}/v1", # config["llm"]["base_url"],
                            "api_key":config["llm"]["api_key"],
                            "api_type":config["llm"]["api_type"],
                            "temperature":config["llm"]["temperature"],
                            "max_tokens":config["llm"]["max_tokens"],
                            "vocab_path":config["llm"]['vocab_path']        
                        }
                    }
                    for agent in data
                ]
            else:
                agents = [
                    {
                        "name":agent["name"],
                        "system_prompt":agent["system_prompt"],
                        "llm":{
                            "model":config["llm"]["model"],
                            "base_url": "http://"+config["llm"]["base_url"]+f":{port}/v1", # config["llm"]["base_url"],
                            "api_key":config["llm"]["api_key"],
                            "api_type":config["llm"]["api_type"],
                            "temperature":config["llm"]["temperature"],
                            "max_tokens":config["llm"]["max_tokens"],     
                        }
                    }
                    for agent in data
                ]                
            
            data = {
                "scene":scene,
                "groupchat":groupchat,
                "agents":agents,
            }
            
            if not os.path.exists(config_output_dir):
                os.makedirs(config_output_dir)
            with open(config_output_dir+f'persona_chat_{i}_{idx}.yaml','w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    with open(config_output_dir+'/ref_data.pkl','wb') as f:
        pickle.dump(ref_data, f)
    with open(config_output_dir+'/ref_persona.pkl','wb') as f:
        pickle.dump(ref_persona, f)        
    return ref_data, ref_persona