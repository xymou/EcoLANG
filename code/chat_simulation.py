"""
Base interface of simulation for a single scene
"""
import os
import yaml
from typing import Dict, List, TYPE_CHECKING
import autogen
from autogen.agentchat.contrib.capabilities import transform_messages
from autogen.agentchat.contrib.capabilities.transforms import TextMessageContentName
from autogen import ConversableAgent, GroupChat
from pydantic import BaseModel
import random
import json
import pickle


def load_llm_config(llm_config: Dict):
    if "vocab_path" in llm_config:
        vocab = pickle.load(open(llm_config["vocab_path"],'rb'))
        del llm_config["vocab_path"]
        llm_config["extra_body"] = {
            "logits_processors":[
                {"qualname":"logits_processors.VocabLogitsProcessor",
                 "kwargs":{"allowed_token_ids":vocab}}
            ]
        }
    
    llm_config = {"config_list": [llm_config], "cache_seed": None}
    return llm_config


def load_agent(agent_config: Dict) -> ConversableAgent:
    valid_agent_config = {
        "name": agent_config["name"],
        "system_message":agent_config["system_prompt"],
        "llm_config":agent_config["llm_config"],
        "human_input_mode":"NEVER",
    }
    agent = ConversableAgent(**valid_agent_config)
    return agent


def load_groupchat(agent_list: List, groupchat_config: Dict):
    group_chat = autogen.GroupChat(
        agent_list,
        **groupchat_config
    ) 
    chat_manager = autogen.GroupChatManager(group_chat)
    return group_chat, chat_manager

def prepare_task_config(config_path):
    """Read the yaml config of the given task in `tasks` directory."""
    if not os.path.exists(config_path):
        raise ValueError(f"Task {config_path} not found.")
    task_config = yaml.safe_load(open(config_path))
    for i, agent_configs in enumerate(task_config["agents"]):
        llm_config = load_llm_config(agent_configs.get("llm", None))
        agent_configs["llm_config"] = llm_config
    return task_config


class ChatSimulation:
    def __init__(self, scene, agents, group_chat, chat_manager, output_dir):
        self.scene = scene
        self.agents = agents
        self.group_chat = group_chat
        self.chat_manager = chat_manager
        self.output_dir = output_dir     
        self.prompt_tokens=0
        self.completion_tokens=0
        self.total_tokens=0

        self.agent_dict = {}
        for agent in self.agents:
            self.agent_dict[agent.name] = agent
            
    @classmethod
    def from_task(cls, tasks_path: str, output_dir: str):
        task_config = prepare_task_config(tasks_path)
        scene = task_config["scene"]
        agents = []
        for agent_config in task_config["agents"]:
            agent = load_agent(agent_config)
            agents.append(agent)
  
        # Build the group chat
        name_transform = TextMessageContentName(position="start", format_string="{name}: ")
        context_handling = transform_messages.TransformMessages(transforms=[name_transform])
        for agent in agents:
            context_handling.add_to_agent(agent)
        group_chat, chat_manager = load_groupchat(agents, task_config["groupchat"])

        return cls(scene, agents, group_chat, chat_manager, output_dir)      
    
    def run(self):
        start_agent = self.agents[0]
        self.groupchat_result = start_agent.initiate_chat(
            self.chat_manager, message=self.scene["start_msg"]
        )        
        chat_history = self.groupchat_result.chat_history
        
        for agent in self.agents:
            token_usage = agent.get_total_usage()
            for key in token_usage:
                if key =='total_cost':continue
                self.prompt_tokens+=token_usage[key]['prompt_tokens']
                self.completion_tokens+=token_usage[key]['completion_tokens']
                self.total_tokens+=token_usage[key]['total_tokens']

        with open(self.output_dir+'/'+str(self.scene['scene_id'])+'.json','w') as f:
            json.dump({'chat_history':chat_history,'token_count':{'prompt_tokens':self.prompt_tokens,'completion_tokens':self.completion_tokens,
                    'total_tokens':self.total_tokens}}, f)