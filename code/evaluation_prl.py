"""
Evaluation
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
from openai import OpenAI
import numpy as np
from scipy.stats import pearsonr
from textblob import TextBlob
from scipy.signal import medfilt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent.futures
from threading import Lock
from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import entropy

def js_divergence(p, q):
    m = 0.5 * (p + q)
    kl_pm = entropy(p, m)
    kl_qm = entropy(q, m)
    js = 0.5 * (kl_pm + kl_qm)
    return js


def cal_token_chat(simu_data, tokenizer):
    # for persona-chat
    cnt = 0
    for key in simu_data:
        for d in simu_data[key]:
            content =  d['content']
            tokens = tokenizer.tokenize(content)
            cnt+=len(tokens)
    return cnt


def cal_token_oasis(simu_data, tokenizer):
    # for hisim and pheme
    avg_cnt = []
    for key in simu_data:
        cnt=0
        for user in simu_data[key]:
            for d in simu_data[key][user]:
                content = d['content']
                if type(content)==str:
                    tokens = tokenizer.tokenize(content)
                    cnt+=len(tokens)
        avg_cnt.append(cnt)
    return np.mean(avg_cnt)

def cal_token_hisim(simu_data, tokenizer):
    # for hisim and pheme
    avg_cnt = []
    for key in simu_data:
        cnt=0
        for user in simu_data[key]:
            for step in simu_data[key][user]:
                for d in simu_data[key][user][step]:
                    content = d['content']
                    if type(content)==str:
                        tokens = tokenizer.tokenize(content)
                        cnt+=len(tokens)
        avg_cnt.append(cnt)
    return np.mean(avg_cnt)

def judge_rumor_stance_sep(
    response_chain,
    cnt_tweet,
    judge_llm,
    base_url,
    api_key,
    max_try=3
):
    prompt_template = """Given threads discussing a news, please label the stance of the question tweet on the source news tweet.\n\n"""+\
        """Treads: {threads}\n"""+\
        """Question tweet: {tweet}\n\n"""+\
        """Please choose from the following options:\n"""+\
        """1. support: the author of the response supports the veracity of the news.\n"""+\
        """2. deny: the author of the response denies the veracity of the news.\n"""+\
        """3. query: the author of the response asks for additional evidence in relation to the veracity of the news.\n"""+\
        """4. comment: the author of the response makes their own comment without a clear contribution to assessing the veracity of the news.\n\n"""+\
        """Please strictly follow the JSON format for your response:\n"""+\
        """{{"stance": <str>}}"""
    
    client = OpenAI(api_key= api_key,
            base_url=base_url)
    prompt = prompt_template.format(threads=response_chain, tweet=cnt_tweet)
    retry = 0
    while retry<max_try:
        try:
            response = client.chat.completions.create(
                model=judge_llm,
                messages = [{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=16
            )
            response = response.choices[0].message.content.strip()
            response = json.loads(response)["stance"]
            assert response in ["support","deny","query","comment"]
            return response
        except:
            retry+=1        
    raise ValueError("Didn't get valid stance!")
  

def judge_rumor_belief(
    final_tweet,
    source_tweet,
    judge_llm,
    base_url,
    api_key,
    max_try=3    
):
    prompt_template = """Please determine whether the author of the final tweet believe the source news.\n\n"""+\
        """Source News:{source_tweet}\n"""+\
        """Final Tweet:{final_tweet}\n\n"""+\
        """If the author does not believe the source news, questions the AUTHENTICITY of the source news or queries for more information about the AUTHENTICITY of the news, please label it as disbelief.\n"""+\
        """If the author expresses opinions or call for actions under the assumption that the news is true, please label it as belief.\n"""+\
        """If the author discusses something unrelated to the source news, please label it as unknown. Please label 0 for disbelief, 1 for belief and 2 for unknown.\n"""+\
        """Please write a short reason and strictly follow the JSON format for your response:\n"""+\
        """{{"reason": <str>, "label": <int>}}"""
    
    client = OpenAI(api_key= api_key,
            base_url=base_url)
    prompt = prompt_template.format(source_tweet=source_tweet, final_tweet=final_tweet)
    retry=0
    while retry<max_try:
        try:
            response = client.chat.completions.create(
                model=judge_llm,
                messages = [{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=128
            )
            response = response.choices[0].message.content
            belief = int(json.loads(response)["label"])
            return belief
        except:
            retry+=1
    raise ValueError("Didn't get valid score!")  

def judge_stance_sep(
    response, 
    topic,
    judge_llm,
    base_url,
    api_key,
    max_try=3,        
):
    """
    judge stance by seperate labeling the two responses
    """
    prompt_template="""What's the author's stance on {target}?\nPlease choose from Support, Neutral, and Oppose. Only output your choice.\n"""+\
            """Text: {text}\nStance:"""
    client = OpenAI(api_key= api_key,
            base_url=base_url)

    prompt = prompt_template.format(text=response, target=topic)
    retry=0
    while retry<max_try:
        try:
            response = client.chat.completions.create(
                        model=judge_llm,
                        messages = [{"role":"user","content":prompt}],
                        temperature=0,
                        max_tokens=8
                    )
            response = response.choices[0].message.content.strip()
            assert response in ["Support", "Neutral", "Oppose"]
            return response
        except:
            retry+=1
    raise ValueError("Didn't get valid stance!")


def judge_content_sep(
    response, 
    judge_llm,
    base_url,
    api_key,
    max_try=3,        
):
    """
    judge content by seperate labeling the two responses
    """
    prompt_template="""Please classify the text into one of the following categories based on its content. Only output your choice.\n\n"""+\
            """1. call for action: tweet contained a call for action (e.g. requesting, challenging, promoting, inviting, summoning someone to do something).\n"""+\
            """2. testimony: tweet contained a testimony of the victim (e.g. report, declaration, first-person experience).\n"""+\
            """3. sharing of opinion: e.g. evaluation, appreciation, addition, analysis of opinions.\n"""+\
            """4. reference to a third party: reporting on something/-one, direct and indirect quotes.\n"""+\
            """5. other: other content that does not fall into the above categories.\n\n"""+\
            """Text:{text}\n"""+\
            """Please strictly follow the JSON format for your response:\n"""+\
            """{{"type": <str>}}"""
            
    client = OpenAI(api_key= api_key,
            base_url=base_url)

    prompt = prompt_template.format(text=response)
    retry=0
    while retry<max_try:
        try:
            response = client.chat.completions.create(
                model=judge_llm,
                messages = [{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=16
            )
            response = response.choices[0].message.content
            response = json.loads(response)["type"]
            assert response in ["call for action", "testimony", "sharing of opinion", "reference to a third party","other"]
            return response
        except:
            retry+=1
    raise ValueError("Didn't get valid content!")


def jaccard_similarity(tweet1, tweet2):
    set1 = set(tweet1.lower().split())
    set2 = set(tweet2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


import scipy.spatial
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

def get_word_distribution(tweets):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(tweets)
    word_freq = np.array(X.sum(axis=0)).flatten()
    word_freq = word_freq / word_freq.sum()
    return word_freq, vectorizer.get_feature_names_out()

def jensen_shannon_divergence(p, q):
    p = np.array(p)
    q = np.array(q)
    m = 0.5 * (p + q)
    return 0.5 * scipy.spatial.distance.jensenshannon(p, m) + 0.5 * scipy.spatial.distance.jensenshannon(q, m)

def compute_js_divergence(tweets_a, tweets_b):
    freq_a, words_a = get_word_distribution(tweets_a)
    freq_b, words_b = get_word_distribution(tweets_b)

    all_words = set(words_a).union(set(words_b))
    vectorizer = CountVectorizer(vocabulary=list(all_words))

    X_a = vectorizer.fit_transform(tweets_a)
    X_b = vectorizer.fit_transform(tweets_b)

    freq_a = np.array(X_a.sum(axis=0)).flatten()
    freq_b = np.array(X_b.sum(axis=0)).flatten()

    freq_a = freq_a / freq_a.sum()
    freq_b = freq_b / freq_b.sum()

    return jensen_shannon_divergence(freq_a, freq_b)

def semantic_sim_hisim(
    simu_data,
    ref_data,
    sen_encoder, 
    tokenizer
):
    def jaccard_distance(tweet1, tweet2):
        set1 = set(tweet1.lower().split())
        set2 = set(tweet2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return 1 - len(intersection) / len(union)    
    
    def clean_tweet(tweet):
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'[^\w\s#]', '', tweet)
        tweet = re.sub(r'\d+', '', tweet)
        return tweet
    
    cos_sim, jaccard_sim=[],[]
    gt_sent_len, simu_sent_len = [],[]
    gt_token_len, simu_token_len = [],[]
    gt_tweets, simu_tweets = [], []
    
    for key in tqdm(simu_data):
        if ('metoo' in key or 'roe' in key) and len(key.split('_')) > 1:
            ori_key = key.split('_')[0]
        else:
            ori_key = key
        for user in simu_data[key]:
            if user in ref_data[ori_key]:
                for t in simu_data[key][user]:
                    if t in ref_data[ori_key][user]:
                        simu_content, ref_content='',''
                        for i in range(len(simu_data[key][user][t])):
                            if simu_data[key][user][t][i]['content']:
                                simu_content+=simu_data[key][user][t][i]['content'].lower()+' '
                                simu_sent_len.append(len(simu_content.split()))
                                simu_token_len+=[len(tokenizer(word)['input_ids']) for word in simu_content.split()]
                                simu_tweets.append(simu_data[key][user][t][i]['content'].lower())
                        simu_content = simu_content.strip()
                        for i in range(len(ref_data[ori_key][user][t])):
                            ref_content+=ref_data[ori_key][user][t][i].lower()+' '
                            gt_sent_len.append(len(clean_tweet(ref_data[ori_key][user][t][i].lower()).split()))
                            gt_token_len+=[len(tokenizer(word)['input_ids']) for word in clean_tweet(ref_data[ori_key][user][t][i].lower()).split()]
                            gt_tweets.append(ref_data[ori_key][user][t][i].lower())
                        ref_content = ref_content.strip()
                        if simu_content and ref_content:
                            vec1 = sen_encoder.encode(simu_content, convert_to_tensor=True).cpu().numpy()
                            vec2 = sen_encoder.encode(ref_content, convert_to_tensor=True).cpu().numpy()
                            cos = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
                            jaccard = jaccard_similarity(simu_content, clean_tweet(ref_content))
                            cos_sim.append(cos)
                            jaccard_sim.append(jaccard)
           
    return {"cosine_sim":float(round(np.mean(cos_sim),4)), "jaccard_sim":float(round(np.mean(jaccard_sim),4)), 
                   "gt_sent_len":float(round(np.mean(gt_sent_len),4)), "simu_sent_len":float(round(np.mean(simu_sent_len),4)),"gt_sent_std":float(round(np.std(gt_sent_len,ddof=1),4)), "simu_sent_std":float(round(np.std(simu_sent_len,ddof=1),4)),
                   "gt_token_len":float(round(np.mean(gt_token_len),4)), "simu_token_len":float(round(np.mean(simu_token_len),4)),"gt_token_std":float(round(np.std(gt_token_len,ddof=1),4)), "simu_token_std":float(round(np.std(simu_token_len,ddof=1),4)),
                   "word_counter_JS":compute_js_divergence(gt_tweets, simu_tweets)}



def semantic_sim_pheme(
    simu_data,
    ref_data,
    sen_encoder,
    tokenizer
):
    def jaccard_distance(tweet1, tweet2):
        set1 = set(tweet1.lower().split())
        set2 = set(tweet2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        return 1 - len(intersection) / len(union)    
    
    def clean_tweet(tweet):
        tweet = re.sub(r'@\w+', '', tweet)
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        tweet = re.sub(r'[^\w\s#]', '', tweet)
        tweet = re.sub(r'\d+', '', tweet)
        return tweet
    
    cos_sim, jaccard_sim=[],[]
    gt_sent_len, simu_sent_len = [],[]
    gt_token_len, simu_token_len = [],[]
    gt_tweets, simu_tweets = [], []
    
    for key in tqdm(simu_data):
        if len(key.split('_')) > 2:
            ori_key = "_".join(key.split('_')[:2])
        else:
            ori_key = key
        for user in simu_data[key]:
            if user in ref_data[ori_key]:
                simu_dict, ref_dict = defaultdict(list), defaultdict(list)
                for i in range(len(simu_data[key][user])):
                    timestep = int(simu_data[key][user][i]['time_gap']//3)
                    simu_dict[timestep].append(simu_data[key][user][i])
                max_timestep=max(simu_dict.keys())
                for i in range(len(ref_data[ori_key][user])):
                    timestep = int(ref_data[ori_key][user][i]['time_gap']//3)
                    if timestep<max_timestep:
                        ref_dict[timestep].append(ref_data[ori_key][user][i])
                    else:
                        ref_dict[max_timestep].append(ref_data[ori_key][user][i])
                simu_data[key][user] = simu_dict
                ref_data[ori_key][user] = ref_dict
                for t in simu_data[key][user]:
                    if t in ref_data[ori_key][user]:
                        simu_content, ref_content='',''
                        for i in range(len(simu_data[key][user][t])):
                            if simu_data[key][user][t][i]['content']:
                                simu_content+=simu_data[key][user][t][i]['content'].lower()+' '
                                simu_sent_len.append(len(simu_content.split()))
                                simu_token_len+=[len(tokenizer(word)['input_ids']) for word in simu_content.split()]
                                simu_tweets.append(simu_data[key][user][t][i]['content'].lower())
                        simu_content = simu_content.strip()
                        for i in range(len(ref_data[ori_key][user][t])):
                            ref_content+=ref_data[ori_key][user][t][i]['content'].lower()+' '
                            gt_sent_len.append(len(clean_tweet(ref_data[ori_key][user][t][i]['content'].lower()).split()))
                            gt_token_len+=[len(tokenizer(word)['input_ids']) for word in clean_tweet(ref_data[ori_key][user][t][i]['content'].lower()).split()]
                            gt_tweets.append(ref_data[ori_key][user][t][i]['content'].lower())
                        ref_content = ref_content.strip()
                        if simu_content and ref_content:
                            vec1 = sen_encoder.encode(simu_content, convert_to_tensor=True).cpu().numpy()
                            vec2 = sen_encoder.encode(ref_content, convert_to_tensor=True).cpu().numpy()
                            cos = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
                            jaccard = jaccard_similarity(simu_content, clean_tweet(ref_content))
                            cos_sim.append(cos)
                            jaccard_sim.append(jaccard)
             
    return {"cosine_sim":float(round(np.mean(cos_sim),4)), "jaccard_sim":float(round(np.mean(jaccard_sim),4)), 
                   "gt_sent_len":float(round(np.mean(gt_sent_len),4)), "simu_sent_len":float(round(np.mean(simu_sent_len),4)),"gt_sent_std":float(round(np.std(gt_sent_len,ddof=1),4)), "simu_sent_std":float(round(np.std(simu_sent_len,ddof=1),4)),
                   "gt_token_len":float(round(np.mean(gt_token_len),4)), "simu_token_len":float(round(np.mean(simu_token_len),4)),"gt_token_std":float(round(np.std(gt_token_len,ddof=1),4)), "simu_token_std":float(round(np.std(simu_token_len,ddof=1),4)),
                   "word_counter_JS":compute_js_divergence(gt_tweets, simu_tweets)}



#### for PHEME
file_lock1 = Lock()
def eval_pheme(
    simu_data,
    simu_token_count,
    ref_data,
    judge_llm,
    judge_base_url,
    judge_api_key,
    score_ckpt_dir,
    debug_path="/remote-home/xymou/AgentCom_local/evo/debug/pheme_debug_sep.json"
):
    """evaluate PHEME dataset
    - stance alignment: acc.
    - structure alignment: scale, depth, breadth
    - # token usage (completion)
    """
    # calculate the stance acc - based on the first msg
    debug = []
    keys = list(simu_data.keys())[:2]
    stance_label = []
    simu_stance, ref_stance = [], []
    if not os.path.exists(score_ckpt_dir):
        os.makedirs(score_ckpt_dir)
    if os.path.exists(score_ckpt_dir + '/stance_detail.json'):
        stance_detail = defaultdict(dict)
        stance_detail.update(json.load(open(score_ckpt_dir + '/stance_detail.json')))
    else:
        stance_detail = defaultdict(dict)

    def process_stance_key(key):
        if len(key.split('_')) > 2:
            ori_key = "_".join(key.split('_')[:2])
        else:
            ori_key = key
        for user in simu_data[key]:
            if key in stance_detail and user in stance_detail[key] and 'score' in stance_detail[key][user]:
                stance = stance_detail[key][user]['score']
                stance_label.append(stance)
                continue
            # exclude the src tweet
            if user == '0': continue
            if user not in ref_data[ori_key] or len(simu_data[key][user]) == 0: continue
            simu_response = simu_data[key][user]
            simu_response.sort(key=lambda x: x['time_gap'])
            simu_response_chain = simu_response[0]["response_chain"]
            simu_response = simu_response[0]["content"]
            ref_response = ref_data[ori_key][user][0]["content"]
            ref_response_chain = ref_data[ori_key][user][0]["response_chain"]
            source_tweet = ref_data[ori_key][user][0]["source_tweet"]

            if user not in stance_detail[key]:stance_detail[key][user]={}
            if "simu_stance" not in stance_detail[key][user]:
                simu_stance = judge_rumor_stance_sep(simu_response_chain, simu_response, judge_llm, judge_base_url, judge_api_key)
                stance_detail[key][user]["simu_stance"] = simu_stance
            else:
                simu_stance = stance_detail[key][user]["simu_stance"]
            if "ref_stance" not in stance_detail[key][user]:
                ref_stance = judge_rumor_stance_sep(ref_response_chain, ref_response, judge_llm, judge_base_url, judge_api_key)
                stance_detail[key][user]["ref_stance"] = ref_stance
            else:
                ref_stance = stance_detail[key][user]["ref_stance"]
            score = int(simu_stance == ref_stance)
            if "score" not in stance_detail[key][user]:
                stance_detail[key][user]["score"] = score
            stance_label.append(score)
            # stance_detail[key][user] = {"score": score, "simu_stance": simu_stance, "ref_stance": ref_stance}
            with file_lock1:
                with open(score_ckpt_dir + '/stance_detail.json', 'w') as f:
                    json.dump(stance_detail, f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_stance_key, simu_data.keys()), total=len(simu_data), desc="Scoring stance"))

    stance_acc = round(np.mean(stance_label), 4)

    if os.path.exists(score_ckpt_dir + '/belief_detail.json'):
        belief_detail = defaultdict(dict)
        belief_detail.update(json.load(open(score_ckpt_dir + '/belief_detail.json')))
    else:
        belief_detail = defaultdict(dict)

    delta_belief_bias, delta_belief_div = [], []
    belief_acc, belief_JS=[],[]

    def process_belief_key(key):
        group_simu_belief, group_ref_belief = [], []
        if len(key.split('_')) > 2:
            ori_key = "_".join(key.split('_')[:2])
        else:
            ori_key = key
        for user in simu_data[key]:
            if key in belief_detail and user in belief_detail[key] and "simu_belief" in belief_detail[key][user] and "ref_belief" in belief_detail[key][user]:
                simu_belief = belief_detail[key][user]['simu_belief']
                if simu_belief in [0,1]:
                    group_simu_belief.append(simu_belief)
                ref_belief = belief_detail[key][user]['ref_belief']
                if ref_belief in [0,1]:
                    group_ref_belief.append(ref_belief)
                belief_acc.append(int(simu_belief==ref_belief))
                continue
            # exclude the src tweet
            if user == '0': continue
            if user not in ref_data[ori_key] or len(simu_data[key][user]) == 0: continue
            simu_response = simu_data[key][user]
            simu_response.sort(key=lambda x: x['time_gap'])
            simu_response = simu_response[-1]["content"]
            ref_response = ref_data[ori_key][user][-1]["content"]
            source_tweet = ref_data[ori_key][user][0]["source_tweet"]

            if user not in belief_detail[key]: belief_detail[key][user]={}
            if "simu_belief" not in belief_detail[key][user]:
                simu_belief = judge_rumor_belief(simu_response, source_tweet, judge_llm, judge_base_url, judge_api_key)
                belief_detail[key][user]["simu_belief"] = simu_belief
            else:
                simu_belief = belief_detail[key][user]["simu_belief"]
            if "ref_belief" not in belief_detail[key][user]:
                ref_belief = judge_rumor_belief(ref_response, source_tweet, judge_llm, judge_base_url, judge_api_key)
                belief_detail[key][user]["ref_belief"] = ref_belief
            else:
                ref_belief = belief_detail[key][user]["ref_belief"]
            if simu_belief in [0,1]:
                group_simu_belief.append(simu_belief)
            if ref_belief in [0,1]:
                group_ref_belief.append(ref_belief)

            # belief_detail[key][user] = {"simu_belief": simu_belief, "ref_belief": ref_belief}
            with file_lock1:
                with open(score_ckpt_dir + '/belief_detail.json', 'w') as f:
                    json.dump(belief_detail, f)

        if len(group_simu_belief) >= 2 and len(group_ref_belief)>=2:
            belief_bias = abs(np.mean(group_simu_belief) - np.mean(group_ref_belief))
            belief_div = abs(np.var(group_simu_belief, ddof=1) - np.var(group_ref_belief, ddof=1))
            delta_belief_bias.append(belief_bias)
            delta_belief_div.append(belief_div)
        if len(group_simu_belief) >= 1 and len(group_ref_belief)>=1:
            group_simu_belief_dist = np.array([group_simu_belief.count(0)/len(group_simu_belief), group_simu_belief.count(1)/len(group_simu_belief), group_simu_belief.count(2)/len(group_simu_belief)])
            group_ref_belief_dist = np.array([group_ref_belief.count(0)/len(group_ref_belief), group_ref_belief.count(1)/len(group_ref_belief), group_ref_belief.count(2)/len(group_ref_belief)])
            js_div = js_divergence(group_simu_belief_dist, group_ref_belief_dist)
            belief_JS.append(js_div)
            

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(process_belief_key, simu_data.keys()), total=len(simu_data), desc="Scoring belief"))

    delta_belief_bias = round(np.mean(delta_belief_bias), 4)
    delta_belief_div = round(np.mean(delta_belief_div), 4)
    belief_acc = round(np.mean(belief_acc),4)
    belief_JS = round(np.mean(belief_JS),4)
    token_count = {
        "prompt_tokens": np.mean([simu_token_count[key]["prompt_tokens"] for key in simu_token_count]),
        "completion_tokens": np.mean([simu_token_count[key]["completion_tokens"] for key in simu_token_count]),
        "total_tokens": np.mean(
            [simu_token_count[key]["prompt_tokens"] + simu_token_count[key]["completion_tokens"] for key in
             simu_token_count])
    }

    return {"stance_acc": stance_acc, "delta_belief_bias": delta_belief_bias, "delta_belief_div": delta_belief_div, "belief_acc":belief_acc, "belief_JS":belief_JS,
            "token_count": token_count}






#### for HISIM
file_lock2 = Lock()
def remove_noise(time_series, window_size=3):
    return medfilt(time_series, kernel_size=window_size)

def smooth_data(time_series, window_size=5, order=2):
    return savgol_filter(time_series, window_size, order)

def normalize_data(time_series):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(time_series.reshape(-1, 1)).flatten()
    return normalized_data


def judge_opinion_dynamics(data, users, topic, judge_llm, judge_base_url, judge_api_key, score_ckpt_dir, key, detail_key):
    def compute_senti_score(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity

    def label_stance_by_rule(text, topic):
        pro_patterns={
            'Metoo Movement':['#Withyou',],
            'BlackLivesMatter Movement':['#BlackLivesMatter', '#GeorgeFloyd', '#PoliceBrutality', '#BLM',],
            'the protection of Abortion Rights':['#roevwadeprotest', 'roe v wade protest', 'pro choice', 'pro-choice', 
                    '#prochoice', '#forcedbirth', 'forced birth', '#AbortionRightsAreHumanRights', 
                    'abortion rights Are Human Rights', '#MyBodyMyChoice', 'My Body My Choice', 
                    '#AbortionisHealthcare', 'abortion is healthcare', 'AbortionIsAHumanRight', 
                    'abortion is a human right', 'ReproductiveHealth', 'Reproductive Health', 
                    'AbortionRights', 'abortion rights','#resist']}
        for w in pro_patterns[topic]:
            if w.lower() in text.lower():
                return 'Support'
        con_patterns = {
            'Metoo Movement':[],
            'BlackLivesMatter Movement':[],        
            'the protection of Abortion Rights': ['#prolife', '#EndAbortion',
                        '#AbortionIsMurder', '#LifeIsAHumanRight', '#ChooseLife',
                        '#SaveTheBabyHumans', '#ValueLife', '#RescueThePreborn', '#EndRoeVWade',
                        '#MakeAbortionUnthinkable','#LiveActionAmbassador','#AbortionIsNotARight', '#AbortionIsRacist']}
        for w in con_patterns[topic]:
            if w.lower() in text.lower():
                return 'Oppose'
        return None

    def label_stance(text, topic, judge_llm, judge_base_url,judge_api_key):
        max_try=3
        client = OpenAI(api_key= judge_api_key,
                base_url=judge_base_url)
        prompt_template="""What's the author's stance on {target}?\nPlease choose from Support, Neutral, and Oppose. Only output your choice.\n"""+\
            """Text: {text}\nStance:"""
        prompt = prompt_template.format(text=text, target=topic)
        retry=0
        stance=None
        while retry<max_try:
            try:
                response = client.chat.completions.create(
                    model=judge_llm,
                    messages = [{"role":"user","content":prompt}],
                    temperature=0,
                    max_tokens=16
                )
                response = response.choices[0].message.content
                stance=response
                break
            except:
                retry+=1
        if not stance:
            # raise ValueError("Didn't get valid score!")
            return "Neutral"
        return stance     
    

    with file_lock2:
        if os.path.exists(score_ckpt_dir+f'/{detail_key}_opinion_detail.json'):
            opinion_detail = defaultdict(dict)
            opinion_detail.update(json.load(open(score_ckpt_dir+f'/{detail_key}_opinion_detail.json')))
            if key not in opinion_detail:
                opinion_detail[key] = defaultdict(dict)
            else:
                tmp=opinion_detail[key] 
                opinion_detail[key] = defaultdict(dict)
                opinion_detail[key].update(tmp)
        else:
            opinion_detail = defaultdict(dict)    
            opinion_detail[key] = defaultdict(dict)
    
    res = defaultdict(dict)
    for user in tqdm(users,desc="scoring opinion dynamics"):
        for step in data[user]:
            if not len(data[user][step]):continue
            # merge the content of one time step
            if type(sum(list(data[user].values()),[])[0])==str: # for ref_data format
                content = data[user][step]
            else: # for simu_data format
                content = [t['content'] if t['content'] else t['original_post_content'] for t in data[user][step]] 
            content = "\n".join(content)
       
            if key in opinion_detail and user in opinion_detail[key] and str(step) in opinion_detail[key][user]:
                stance_score = opinion_detail[key][user][str(step)]
            else:
                stance = label_stance_by_rule(content, topic)
                if not stance:
                    stance = label_stance(content, topic, judge_llm, judge_base_url, judge_api_key)
                senti_score = compute_senti_score(content)
                stance_score = abs(senti_score) if stance in ['Support','Neutral'] else -abs(senti_score)
                opinion_detail[key][user][step]=stance_score
            res[user][step]= stance_score
            with file_lock2:
                with open(score_ckpt_dir+f'/{detail_key}_opinion_detail.json','w') as f:
                    json.dump(opinion_detail, f)   
        if sum(list(data[user].values()),[])!=[]:
            for i in data[user]:
                if i==0 and len(data[user][i])==0:
                    j=0
                    while len(data[user][j])==0:
                        j+=1
                    score = res[user][j]
                    res[user][i]=score
                elif len(data[user][i]):
                    pass
                else:
                    score = res[user][i-1]
                    res[user][i]=score
    # atts = [np.mean([res[u][step] for u in users]) for step in data[user]]
    # att_detail = [[res[u][step] for u in users] for step in data[user]]
    atts, att_detail = [],[]
    for step in range(14):
        tmp = [res[u][step] for u in res if step in res[u]]
        atts.append(np.mean(tmp))
        att_detail.append(tmp)
    return atts, att_detail


def eval_hisim(
    simu_data,
    simu_token_count,
    ref_data,
    judge_llm,
    judge_base_url,
    judge_api_key,
    score_ckpt_dir
):
    event2topic = {
        "metoo": "Metoo Movement",
        "roe": "the protection of Abortion Rights"
    }
    # individual stance
    stance_label = []
    content_label = []
    if not os.path.exists(score_ckpt_dir):
        os.makedirs(score_ckpt_dir)
    if os.path.exists(score_ckpt_dir + '/stance_detail.json'):
        stance_detail = defaultdict(dict)
        stance_detail.update(json.load(open(score_ckpt_dir + '/stance_detail.json')))
    else:
        stance_detail = defaultdict(dict)
    if os.path.exists(score_ckpt_dir + '/content_detail.json'):
        content_detail = defaultdict(dict)
        content_detail.update(json.load(open(score_ckpt_dir + '/content_detail.json')))
    else:
        content_detail = defaultdict(dict)

    def process_user(key, user):
        if len(key.split('_')) > 1:
            ori_key = key.split('_')[0]
        else:
            ori_key = key
        ref_response = sum([ref_data[ori_key][user][t] for t in ref_data[ori_key][user]], [])
        if not ref_response: return
        simu_response = sum([simu_data[key][user][t] for t in simu_data[key][user]], [])
        simu_response = [t['content'] if t['content'] else t['original_post_content'] for t in simu_response]

        if list(set(simu_response)) == ['do_nothing()']:
            stance_label.append(0)
            stance_detail[key][user] = 0
            content_label.append(0)
            content_detail[key][user] = 0
            return
        simu_response = simu_response[0]
        ref_response = ref_response[0]
        topic = event2topic[key.split('_')[0]]
        if key in stance_detail and user in stance_detail[key] and "score" in stance_detail[key][user]:
            stance = stance_detail[key][user]["score"]
        else:
            if user not in stance_detail[key]: stance_detail[key][user] = {}
            if "simu_stance" not in stance_detail[key][user]:
                simu_stance = judge_stance_sep(simu_response, topic,
                                            judge_llm, judge_base_url, judge_api_key)
                stance_detail[key][user]['simu_stance'] = simu_stance
            else:
                simu_stance = stance_detail[key][user]['simu_stance']
            if "ref_stance" not in stance_detail[key][user]:
                ref_stance = judge_stance_sep(ref_response, topic,
                                            judge_llm, judge_base_url, judge_api_key)
                stance_detail[key][user]['ref_stance'] = ref_stance
            else:
                ref_stance = stance_detail[key][user]['ref_stance']
            stance = int(simu_stance == ref_stance)
            stance_detail[key][user]['score'] = stance
            with file_lock2:
                with open(score_ckpt_dir + '/stance_detail.json', 'w') as f:
                    json.dump(stance_detail, f)

        if key in content_detail and user in content_detail[key] and "score" in content_detail[key][user]:
            content = content_detail[key][user]["score"]
        else:
            if user not in content_detail[key]: content_detail[key][user] = {}
            if "simu_content" not in content_detail[key][user]:
                simu_content = judge_content_sep(simu_response, judge_llm, judge_base_url, judge_api_key)
                content_detail[key][user]['simu_content'] = simu_content
            else:
                simu_content = content_detail[key][user]['simu_content']
            if "ref_content" not in content_detail[key][user]:
                ref_content = judge_content_sep(ref_response, judge_llm, judge_base_url, judge_api_key)
                content_detail[key][user]['ref_content'] = ref_content
            else:
                ref_content = content_detail[key][user]['ref_content']
            content = int(simu_content == ref_content)            
            content_detail[key][user]['score'] = content
            with file_lock2:
                with open(score_ckpt_dir + '/content_detail.json', 'w') as f:
                    json.dump(content_detail, f)
        stance_label.append(stance)
        content_label.append(content)

    def process_key(key):
        users = simu_data[key].keys()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            list(tqdm(executor.map(lambda user: process_user(key, user), users), total=len(users), desc=f"Processing users in {key}"))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(tqdm(executor.map(process_key, simu_data.keys()), total=len(simu_data), desc="Processing keys"))

    stance_acc = round(np.mean(stance_label), 4)
    content_acc = round(np.mean(content_label), 4)
    print(f'stance_acc:{stance_acc}, content_acc:{content_acc}')

    avg_corr, avg_rmse = [], []
    avg_delta_bias, avg_delta_div = [], []
    avg_all_delta_bias, avg_all_delta_div = [],[]

    def process_opinion_dynamics(key):
        if key.startswith('blm'): return
        if len(key.split('_')) > 1:
            ori_key = key.split('_')[0]
        else:
            ori_key = key
        topic = event2topic[key.split('_')[0]]
        users = list(simu_data[key].keys())
        simu_atts, simu_atts_detail = judge_opinion_dynamics(simu_data[key], users, topic, judge_llm, judge_base_url,
                                                             judge_api_key, score_ckpt_dir, key, 'simu_atts')
        ref_atts, ref_atts_detail = judge_opinion_dynamics(ref_data[ori_key], users, topic, judge_llm, judge_base_url,
                                                           judge_api_key, score_ckpt_dir, key, 'ref_atts')

        delta_bias = np.mean([abs(ref_atts[t] - simu_atts[t]) for t in range(len(simu_atts))])
        delta_div = np.mean([abs(np.var(ref_atts_detail[t], ddof=1) - np.var(simu_atts_detail[t], ddof=1)) for t in
                             range(len(simu_atts))])

        processed_series1 = normalize_data(smooth_data(remove_noise(np.array(ref_atts))))
        processed_series2 = normalize_data(smooth_data(remove_noise(np.array(simu_atts))))
        # calculate the corr.
        corr_coef, p_value = pearsonr(processed_series1, processed_series2)
        rmse = np.sqrt(np.mean((np.array(simu_atts) - np.array(ref_atts)) ** 2)) / np.max(np.array(ref_atts))
        avg_corr.append(corr_coef)
        avg_rmse.append(rmse)
        delta_bias = abs(np.mean(ref_atts_detail[-1]) - np.mean(simu_atts_detail[-1]))
        delta_div = abs(np.var(ref_atts_detail[-1], ddof=1) - np.var(simu_atts_detail[-1], ddof=1))
        avg_delta_bias.append(delta_bias)
        avg_delta_div.append(delta_div)
        all_delta_bias = np.mean([abs(np.mean(ref_atts_detail[t])-np.mean(simu_atts_detail[t])) for t in range(len(ref_atts_detail))]) 
        all_delta_div = np.mean([abs(np.var(ref_atts_detail[t], ddof=1)-np.var(simu_atts_detail[t], ddof=1)) for t in range(len(ref_atts_detail))]) 
        avg_all_delta_bias.append(all_delta_bias)
        avg_all_delta_div.append(all_delta_div)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(tqdm(executor.map(process_opinion_dynamics, simu_data.keys()), total=len(simu_data), desc="Processing opinion dynamics"))

    avg_corr = round(np.mean(avg_corr), 4)
    avg_rmse = round(np.mean(avg_rmse), 4)
    avg_delta_bias = round(np.mean(avg_delta_bias), 4)
    avg_delta_div = round(np.mean(avg_delta_div), 4)
    avg_all_delta_bias = round(np.mean(avg_all_delta_bias),4)
    avg_all_delta_div = round(np.mean(avg_all_delta_div),4)
    token_count = {
        "prompt_tokens": np.mean([simu_token_count[key]["prompt_tokens"] for key in simu_token_count]),
        "completion_tokens": np.mean([simu_token_count[key]["completion_tokens"] for key in simu_token_count]),
        "total_tokens": np.mean(
            [simu_token_count[key]["prompt_tokens"] + simu_token_count[key]["completion_tokens"] for key in
             simu_token_count])}
    return {"stance_acc": stance_acc, "content_acc": content_acc, "corr": avg_corr, "rmse": avg_rmse,
            "delta_bias": avg_delta_bias, "delta_div": avg_delta_div, "token_count": token_count, "all_delta_bias":avg_all_delta_bias, "all_delta_div":avg_all_delta_div}





def eval_dataset(
    dataset,
    simu_data,
    simu_token_count,
    ref_data,
    judge_llm,
    judge_base_url,
    judge_api_key,
    score_ckpt_dir
):
    if dataset =='pheme':
        return eval_pheme(simu_data, simu_token_count, ref_data, judge_llm, judge_base_url, judge_api_key, score_ckpt_dir)
    elif dataset == 'hisim':
        return eval_hisim(simu_data, simu_token_count, ref_data, judge_llm, judge_base_url, judge_api_key, score_ckpt_dir)
    else:
        raise NotImplementedError(f"Missing preparation function for {dataset}")
    