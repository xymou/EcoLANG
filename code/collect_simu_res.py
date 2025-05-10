"""
Postprocess the simulation results for further selection
"""
import os
import json
import pickle
from tqdm import tqdm
import sqlite3
from collections import defaultdict


def collect_simu_data(dataset, output_dir):
    if dataset=='persona':
        return collect_persona(output_dir)
    elif dataset=='pheme':
        return collect_oasis_pheme(output_dir)
    elif dataset=='hisim':
        return collect_oasis_hisim(output_dir)


def collect_persona(output_dir):
    # collect simulated data
    all_res = {}
    all_token_count = {}
    files = os.listdir(output_dir)
    for file in files:
        if file.endswith('json'):
            key = file.replace('.json','')
            res = json.load(open(output_dir+file,'r'))
            all_res[key] = res['chat_history']  
            all_token_count[key] = res['token_count'] 
    return all_res, all_token_count


def reconstruct_tweet_tree(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM post")
    rows = cursor.fetchall()      
    tweets = []
    for row in rows:
        if row[2]:
            tweets.append({"post_id":row[0],"user_id":row[1],"original_post_id":row[2],"original_post_content":row[3],"content":row[4]})
        else:#src_tweet
            tweets.append({"post_id":row[0],"user_id":row[1],"original_post_id":row[2],"original_post_content":row[3],"content":row[3]})
    threads = defaultdict(list)
    for tweet in tweets:
        original_post_id = tweet["original_post_id"]
        if original_post_id is not None:
            threads[original_post_id].append(tweet)    
    return threads, tweets

def find_root(tweet_id, threads, tweets):
    while True:
        tweet = next((t for t in tweets if t["post_id"] == tweet_id), None)
        if tweet and tweet["original_post_id"] is None:
            return tweet
        tweet_id = tweet["original_post_id"]


def build_thread_from_retweet(target_post_id, threads, tweets):
    root_tweet = find_root(target_post_id, threads, tweets)
    
    def build_tree(root_id, threads, target_post_id):
        tree = []
        if root_id in threads:
            for tweet in threads[root_id]:
                subtree = build_tree(tweet["post_id"], threads, target_post_id)
                if subtree or tweet["post_id"] == target_post_id:
                    tree.append({
                        "post_id": tweet["post_id"],
                        "content": tweet["content"],
                        "user_id": tweet["user_id"],
                        "replies": subtree 
                    })
        return tree
    
    thread_tree = [{
        "post_id": root_tweet["post_id"],
        "content": root_tweet["content"],
        "user_id": root_tweet["user_id"],
        "replies": build_tree(root_tweet["post_id"], threads, target_post_id)
    }]
    return thread_tree

def build_thread_string(thread_tree, indent=""):
    tree_str = ""
    for tweet in thread_tree:
        tree_str += f"{indent}user {tweet['user_id']}: {tweet['content']}\n"
        tree_str += build_thread_string(tweet['replies'], indent + "    ") 
    return tree_str


def collect_oasis_pheme(output_dir):
    all_res = {}
    all_token_count = {}
    files = os.listdir(output_dir)
    for file in files:
        res = defaultdict(list)
        if file.endswith('.db'):
            key = file[:-3]
            output_path = os.path.join(output_dir, file)
            # load the generated posts
            conn = sqlite3.connect(output_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM post")
            rows = cursor.fetchall()     
            res = defaultdict(list)  #user2posts
            threads, tweets = reconstruct_tweet_tree(output_path)    
            for row in rows: 
                post_id = row[0]
                user_id = row[1]
                original_post_id = row[2]
                original_post_content = row[3]
                content = row[4]
                created_at = row[5]
                thread_tree = build_thread_from_retweet(post_id, threads, tweets)
                response_chain = build_thread_string(thread_tree)
                if original_post_id:
                    res[str(user_id)].append({"content":content, "time_gap":created_at,"type":"repost",
                            "original_post_content":original_post_content,"response_chain":response_chain})  
                else:
                    res[str(user_id)].append({"content":content, "time_gap":created_at,"type":"post",
                            "original_post_content":original_post_content,"response_chain":response_chain})      
            all_res[key] = res
        elif file.endswith('.json'):
            key = file[:-5]
            output_path = os.path.join(output_dir, file)
            token_count = json.load(open(output_path))
            all_token_count[key] = token_count
    return all_res, all_token_count


def collect_oasis_hisim(output_dir,time_gap=3):
    all_res = {}
    all_token_count = {}
    files = os.listdir(output_dir)
    for file in files:
        key = file[:-3]
        if file.endswith('.db'):
            output_path = os.path.join(output_dir, file)
            # load the generated posts
            conn = sqlite3.connect(output_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM post")
            rows = cursor.fetchall()     
            res = defaultdict(dict)  #user2posts
            for row in rows: 
                user_id = row[1]
                original_post_id = row[2]
                original_post_content = row[3]
                content = row[4]
                created_at = row[5]
                time_step = int(created_at/time_gap)-1
                if time_step not in res[str(user_id)]:res[str(user_id)][time_step]=[]
                if original_post_id:
                    res[str(user_id)][time_step].append({"content":content, "time_gap":created_at,"type":"repost","original_post_content":original_post_content})  
                else:
                    res[str(user_id)][time_step].append({"content":content, "time_gap":created_at,"type":"post","original_post_content":original_post_content})    
            cursor.execute("SELECT * FROM comment")
            rows = cursor.fetchall()   
            for row in rows:
                post_id = row[1]
                user_id = row[2]
                content = row[3]
                created_at = row[4]
                time_step = int(created_at/time_gap)-1
                cursor.execute(f"SELECT * FROM post where post_id={post_id}")
                tmp = cursor.fetchall() 
                original_post_content = tmp[0][4]     
                if time_step not in res[str(user_id)]:res[str(user_id)][time_step]=[]           
                res[str(user_id)][time_step].append({"content":content, "time_gap":created_at,"type":"comment","original_post_content":original_post_content})    
            all_res[key] = res
        elif file.endswith('.json'):
            output_path = os.path.join(output_dir, file)
            token_count = json.load(open(output_path))
            all_token_count[key] = token_count
    return all_res, all_token_count

