"""assign a score to a conversation trajectory
consider `expressiveness' and `simplicity'
- expressiveness: is the response clear and consistent with the corresponding persona? (measured by judge LLMs)
- simplicity: is the response concise? (measured by # of tokens)
- persona consistency: is the response consistent with the given persona? (measured by judge LLMs)
"""
from openai import OpenAI
import numpy as np
import os
import pickle
import json
from collections import defaultdict
from tqdm import tqdm

# expressiveness
EXP_PROMPT="""Please evaluate whether the agents' language is clear and easy to understand.\n\nAgents' language: {response}\n\n"""+\
    """Please rate on a scale of 1 to 5, with 1 being most unclear and 5 being most clear.\n"""+\
    """Please write a short reason and strictly follow the JSON format for your response:\n"""+\
    """{{"reason": <str>, "score": <int>}}"""

ALIGN_PROMPT="""Please evaluate whether the agent's response align with the persona reflected in the reference response. """+\
    """Please focus on the aspects of content, emotion and atttude, and ignore differences in language structure, e.g., word choice, sentence length, emoji usage and syntax.\n\nAgent's response: {response}\nReference response: {reference}\n\n"""+\
    """Please rate on a scale of 1 to 5, with 1 being most inconsistent and 5 being most like the persona.\n"""+\
    """Please write a short reason and strictly follow the JSON format for your response:\n"""+\
    """{{"reason": <str>, "score": <int>}}"""


def cal_token_chat(simu_data, tokenizer):
    # for persona-chat
    cnt = 0
    for key in simu_data:
        for d in simu_data[key]:
            content =  d['content']
            tokens = tokenizer.tokenize(content)
            cnt+=len(tokens)
    return cnt


def cal_token(simu_data, tokenizer):
    # for hisim and pheme
    cnt = 0
    for key in simu_data:
        for user in simu_data[key]:
            for d in simu_data[key][user]:
                content = d['content']
                tokens = tokenizer.tokenize(content)
                cnt+=len(tokens)
    return cnt



def calculate_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

def calculate_percentile_score(value, cdf_data):
    index = np.searchsorted(cdf_data[0], value)
    return cdf_data[1][index] if index < len(cdf_data[1]) else 1.0



def judge_exp(
    response,
    judge_llm,
    base_url,
    api_key,
    max_try=5,    
):
    client = OpenAI(api_key= api_key,
            base_url=base_url)

    prompt = EXP_PROMPT.format(response=response)
    retry=0
    score = -1
    max_tokens=128
    while retry<max_try:
        try:
            response = client.chat.completions.create(
                    model=judge_llm,
                    messages = [{"role":"user","content":prompt}],
                    temperature=0,
                    max_tokens=max_tokens
            )
            response = response.choices[0].message.content
            if "```json" in response: response = response.replace("```json","").replace("```","")
            response = json.loads(response)
            score = response["score"]
            break
        except:
            retry+=1
            max_tokens+=20
    if score==-1:
        raise ValueError("Didn't get valid score!")
    return score, response


def judge_persona(
    response,
    reference,
    persona,
    judge_llm,
    base_url,
    api_key,
    max_try=3,
):
    client = OpenAI(api_key= api_key,
            base_url=base_url)
    prompt =ALIGN_PROMPT.format(response=response, reference=reference)
    retry=0
    score = -1
    max_tokens=128
    while retry<max_try:
        try:
            response = client.chat.completions.create(
                model=judge_llm,
                messages = [{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=max_tokens
            )
            response = response.choices[0].message.content
            if "```json" in response: response = response.replace("```json","").replace("```","")
            response = json.loads(response)
            score = response["score"]
            break
        except:
            retry+=1
            max_tokens+=20
    if score==-1:
        raise ValueError("Didn't get valid score!")
    return score, response


def judge_opinion(
    response,
    reference,
    judge_llm,
    base_url,
    api_key,
    max_try=3,    
):
    client = OpenAI(api_key= api_key,
            base_url=base_url)

    prompt = OPINION_PROMPT.format(response=response, reference=reference)

    retry=0
    score = -1
    while retry<max_try:
        try:
            response = client.chat.completions.create(
                model=judge_llm,
                messages = [{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=128
            )
            response = response.choices[0].message.content
            if "```json" in response: response = response.replace("```json","").replace("```","")
            print(333, response)
            response = json.loads(response)
            score = response["score"]
            break
        except:
            retry+=1
    if score==-1:
        raise ValueError("Didn't get valid score!")
    return score, response    




def chat_history_tostring(chat_history):
    content = [d["content"] for d in chat_history]
    return " ".join(content)

def score_simp(
    simu_traj,
    tokenizer,
):
    """score the simplicity of the response
    """
    scores = {}
    simu_data = [chat_history_tostring(simu_traj[key]) for key in simu_traj]
    token_lens = [len(tokenizer(d)['input_ids']) for d in simu_data]
    for key in simu_traj:
        d = chat_history_tostring(simu_traj[key])
        token_len = len(tokenizer(d)['input_ids'])
        token_score = token_len/max(token_lens)
        scores[key]=token_score
    return scores      


def score_simp_pheme(
    simu_traj,
    tokenizer,
):
    """score the simplicity of the response
    """
    scores = {}
    token_lens = []
    simu_data_lens={}
    for key in simu_traj:
        lens = []
        for user in simu_traj[key]:
            content = [d["content"] if d["content"] else d["original_post_content"] for d in simu_traj[key][user]]
            lens+=[len(tokenizer(d)['input_ids']) for d in content]
        lens = np.mean(lens)
        token_lens.append(lens)
        simu_data_lens[key] = lens
    
    for key in simu_traj:
        cnt_lens = simu_data_lens[key]
        length_cdf = calculate_cdf(token_lens)
        token_score = calculate_percentile_score(cnt_lens, length_cdf)
        scores[key]=np.mean(token_score)
    return scores  


def score_token(
    simu_token_count
):
    """score the simplicity of the response
    """
    scores = {}
    token_lens = [simu_token_count[key]['completion_tokens'] for key in simu_token_count]
    for key in simu_token_count:
        token_len = simu_token_count[key]['completion_tokens']
        token_score = token_len/max(token_lens)
        scores[key]=token_score
    return scores    

 


def score_conv_traj_persona(
    simu_traj,
    simu_token_count,
    ref_traj,
    ref_persona,
    judge_llm, 
    base_url,
    api_key,
    tokenizer,
    w1,
    w2,
    w3
    ):
    """assign scores to a group conversation trajectory with a same simulation task
    consider `expressiveness' and `simplicity'
    - expressiveness: is the response clear and consistent with the corresponding persona? (measured by judge LLMs)
    - simplicity: is the response concise? (measured by # of tokens)
    - persona consistency: is the response consistent with the given persona?
    """
    # simp_scores = score_simp(simu_traj, tokenizer)
    simp_scores = score_token(simu_token_count)
    exp_scores, persona_scores = {}, {}
    detail = defaultdict(dict)
    for key in simu_traj:
        diag = "\n".join([d["name"]+': '+d["content"] for d in simu_traj[key]])
        score, response = judge_exp(diag, judge_llm, base_url, api_key)
        exp_scores[key] = score
        detail[key]['exp'] = response
        ref_diag = "\n".join([d["name"]+": "+d["content"] for d in ref_traj[key][:15]]) if ref_traj else None
        persona = "\n".join([d["name"]+": "+d["profile"] for d in ref_persona[key]]) if ref_persona else None
        score, response = judge_persona(diag, ref_diag, persona, judge_llm, base_url, api_key)
        detail[key]['align'] = response
        persona_scores[key] = score
    scores = {}
    for key in simu_traj:
        scores[key]=w1*persona_scores[key]/5-w2*simp_scores[key]+w3*exp_scores[key]/5
    score_detail = {"align":persona_scores, "exp":exp_scores, "simp":simp_scores}
    return scores, score_detail, detail


def score_conv_traj_pheme(
    simu_traj,
    simu_token_count,
    ref_traj,
    ref_persona,
    judge_llm, 
    base_url,
    api_key,
    tokenizer,
    w1,
    w2,
    w3
    ):
    """assign scores to a group conversation trajectory with a same simulation task
    consider `expressiveness' and `simplicity'
    - expressiveness: is the response clear and consistent with the corresponding persona? (measured by judge LLMs)
    - simplicity: is the response concise? (measured by # of tokens)
    - persona consistency: is the response consistent with the given persona?
    """
    simp_scores = score_simp_pheme(simu_traj, tokenizer)
    exp_scores, align_scores = {}, {}
    detail = defaultdict(dict)
    for key in simu_traj:
        exp_score, align_score = [],[]
        for user in simu_traj[key]:
            if user not in ref_traj[key] or user=='0':continue
            simu_response = simu_traj[key][user]
            simu_response.sort(key=lambda x: x['time_gap'])     
            simu_response_chain = simu_response[0]["response_chain"]    
            simu_response = simu_response[0]["content"]                    
            ref_response = ref_traj[key][user][0]["content"]
            ref_response_chain = ref_traj[key][user][0]["response_chain"]
            source_tweet = ref_traj[key][user][0]["source_tweet"]            
            score, response = judge_exp(simu_response, judge_llm, base_url, api_key)
            exp_score.append(score)
            detail[key][user]={'exp':response}
            score, response = judge_persona(simu_response, ref_response,None, judge_llm, base_url, api_key)
            align_score.append(score)
            detail[key][user].update({'align':response})
        exp_scores[key] = np.mean(exp_score)
        align_scores[key] = np.mean(align_score)
    scores = {}
    for key in simu_traj:
        scores[key]=w1*align_scores[key]/5-w2*simp_scores[key]+w3*exp_scores[key]/5
    score_detail = {"align":align_scores, "exp":exp_scores, "simp":simp_scores}
    return scores, score_detail, detail



def select(
    dataset,
    simu_data,
    simu_token_count,
    ref_data,
    ref_persona,
    rule_pool,
    judge_llm, 
    base_url,
    api_key,
    tokenizer,
    w1,
    w2,
    w3,
    topk,
    iter_output_dir,
):
    """Select advantageous conv trajectories and the corresponding rules 
    """
    if dataset=='persona':
        score_func = score_conv_traj_persona
    elif dataset =='pheme':
        score_func = score_conv_traj_pheme
    scores = {}
    group_simu_data, group_ref_data, group_ref_persona = defaultdict(dict),defaultdict(dict),defaultdict(dict)
    group_simu_token_count = defaultdict(dict)
    for key in simu_data:
        task = '_'.join(key.split('_')[:-1])
        group_simu_data[task][key]=simu_data[key]
        group_simu_token_count[task][key]=simu_token_count[key]
        if ref_data:
            try:
                group_ref_data[task][key]=ref_data[key]
            except:
                group_ref_data[task][key]=ref_data[task]
        if ref_persona:
            group_ref_persona[task][key]=ref_persona[key]
    
    if os.path.exists(iter_output_dir+'/all_score_detail.pkl'):
        all_score_detail = pickle.load(open(iter_output_dir+'/all_score_detail.pkl','rb'))
        all_detail = pickle.load(open(iter_output_dir+'/all_detail.pkl','rb'))
        print('succefully load!')
    else:
        all_score_detail, all_detail = {}, {}
        
    for key in tqdm(group_simu_data, desc="scoring the simulation results..."):
        if key in all_score_detail and key in all_detail:
            score = {k:w1*all_score_detail[key]['align'][k]/5-w2*all_score_detail[key]['simp'][k]+w3*all_score_detail[key]['exp'][k]/5 for k in all_score_detail[key]['align']}
            scores.update(score)
            continue # have been evaluated
        score, score_detail, detail = score_func(
            group_simu_data[key],
            group_simu_token_count[key],
            group_ref_data[key] if len(ref_data) else None,
            group_ref_persona[key] if len(ref_persona) else None,
            judge_llm,
            base_url,
            api_key,
            tokenizer,
            w1,
            w2,
            w3
        )
        scores.update(score)
        all_score_detail[key] = score_detail
        all_detail[key] = detail
        if not os.path.exists(iter_output_dir):
            os.makedirs(iter_output_dir)
        with open(iter_output_dir+'/all_score_detail.pkl','wb') as f:
            pickle.dump(all_score_detail, f)    
        with open(iter_output_dir+'/all_detail.pkl','wb') as f:
            pickle.dump(all_detail, f) 
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    sorted_items = sorted_items[:int(len(sorted_items)*topk)]
    
    rule2score=defaultdict(list)
    for file in scores:
        key = int(file.split('_')[-1])
        rule2score[rule_pool[key]].append(scores[file])
    for key in rule2score:
        rule2score[key] = np.mean(rule2score[key])    
        
    return rule2score, [item[0] for item in sorted_items], all_score_detail, all_detail