# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
# Licensed under the Apache License, Version 2.0 (the “License”);
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an “AS IS” BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========== Copyright 2023 @ CAMEL-AI.org. All Rights Reserved. ===========
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from colorama import Back
from yaml import safe_load

from oasis.clock.clock import Clock
from oasis.inference.inference_manager import InferencerManager
from oasis.social_agent.agents_generator import generate_agents
from oasis.social_platform.channel import Channel
from oasis.social_platform.platform import Platform
from oasis.social_platform.typing import ActionType

social_log = logging.getLogger(name="social")
social_log.setLevel("DEBUG")

file_handler = logging.FileHandler("social.log")
file_handler.setLevel("DEBUG")
file_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(file_handler)
stream_handler = logging.StreamHandler()
stream_handler.setLevel("DEBUG")
stream_handler.setFormatter(
    logging.Formatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s"))
social_log.addHandler(stream_handler)

parser = argparse.ArgumentParser(description="Arguments for script.")
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to the YAML config file.",
    required=False,
    default="",
)

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data/twitter_dataset/anonymous_topic_200_1h",
)
DEFAULT_DB_PATH = ":memory:"
DEFAULT_CSV_PATH = os.path.join(DATA_DIR, "False_Business_0.csv")


async def running(
    db_path: str | None = DEFAULT_DB_PATH,
    csv_path: str | None = DEFAULT_CSV_PATH,
    token_path: str = None,
    num_timesteps: int = 3,
    clock_factor: int = 60,
    recsys_type: str = "twhin-bert",
    source_post_time: str = None,
    model_configs: dict[str, Any] | None = None,
    inference_configs: dict[str, Any] | None = None,
    action_space_file_path: str = None,
    trigger_news: dict = None,
) -> None:
    db_path = DEFAULT_DB_PATH if db_path is None else db_path
    csv_path = DEFAULT_CSV_PATH if csv_path is None else csv_path
    if os.path.exists(db_path):
        os.remove(db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    if recsys_type == "reddit":
        start_time = datetime.now()
    else:
        start_time = 0
        
    social_log.info(f"Start time: {start_time}")
    clock = Clock(k=clock_factor)
    twitter_channel = Channel()
    infra = Platform(
        db_path,
        twitter_channel,
        clock,
        start_time,
        recsys_type=recsys_type,
        refresh_rec_post_count=0,
        max_rec_post_len=0,
        following_post_count=5,
    )
    inference_channel = Channel()
    infere = InferencerManager(
        inference_channel,
        **inference_configs,
    )
    twitter_task = asyncio.create_task(infra.running())
    inference_task = asyncio.create_task(infere.run())


    source_post_time = source_post_time
    start_hour = int(source_post_time.split(":")[0]) + float(
        int(source_post_time.split(":")[1]) / 60)

    model_configs = model_configs or {}
    with open(action_space_file_path, "r", encoding="utf-8") as file:
        action_space = file.read()
    agent_graph = await generate_agents(
        agent_info_path=csv_path,
        twitter_channel=twitter_channel,
        inference_channel=inference_channel,
        start_time=start_time,
        recsys_type=recsys_type,
        twitter=infra,
        action_space_prompt=action_space,
        **model_configs,
    )
    # agent_graph.visualize("initial_social_graph.png")

    for timestep in range(1, num_timesteps + 1):
        os.environ["SANDBOX_TIME"] = str(timestep * 3)
        social_log.info(f"timestep:{timestep}")
        db_file = db_path.split("/")[-1]
        print(Back.GREEN + f"DB:{db_file} timestep:{timestep}" + Back.RESET)
        # if you want to disable recsys, please comment this line
        try:
            await infra.update_rec_table()
        except:
            pass

        # 0.05 * timestep here means 3 minutes / timestep
        # simulation_time_hour = start_hour + 0.05 * timestep
        simulation_time_hour = start_hour + 12 * timestep
        tasks = []
        for node_id, agent in agent_graph.get_agents():
            if trigger_news and timestep in trigger_news:
                agent.update_trigger_news(trigger_news[timestep])
            if agent.user_info.is_controllable is False:
                agent_ac_prob = random.random()
                threshold = agent.user_info.profile["other_info"][
                    "active_threshold"][int(simulation_time_hour % 24)]
                if agent_ac_prob < threshold:
                    tasks.append(agent.perform_action_by_llm())
            else:
                await agent.perform_action_by_hci()

        await asyncio.gather(*tasks)
        # agent_graph.visualize(f"timestep_{timestep}_social_graph.png")

    await twitter_channel.write_to_receive_queue((None, None, ActionType.EXIT))
    token_count = await infere.stop()
    await twitter_task, inference_task
    
    with open(token_path,'w') as f:
        json.dump(token_count, f)


if __name__ == "__main__":
    args = parser.parse_args()
    os.environ["SANDBOX_TIME"] = str(0)
    if os.path.exists(args.config_path):
        with open(args.config_path, "r") as f:
            cfg = safe_load(f)
        data_params = cfg.get("data")
        simulation_params = cfg.get("simulation")
        model_configs = cfg.get("model")
        inference_configs = cfg.get("inference")
        prompt_path = cfg.get("prompt_path")
        trigger_news = cfg.get("trigger_news")

        asyncio.run(
            running(**data_params,
                    **simulation_params,
                    model_configs=model_configs,
                    inference_configs=inference_configs,
                    action_space_file_path=prompt_path,
                    trigger_news=trigger_news))
    else:
        asyncio.run(running())
    social_log.info("Simulation finished.")
