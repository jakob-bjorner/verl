import sys
import pandas as pd
sys.path.append('../../paprika/') # Remove this when verl and paprika are installed in the same env
from llm_exploration.paprika_config_helper import PaprikaConfigHelper
from verl.interactions.paprika_interaction import PaprikaInteraction
from pprint import pprint as pp
paprika_games = ['twenty_questions', 'guess_my_city', 'murder_mystery', 'customer_service', 'wordle', 'cellular_automata', \
    'mastermind'] # 'battleship', 'minesweeper', 'bandit_bai_fixed_budget' 

import dotenv
dotenv.load_dotenv('../../.env')

import asyncio

sys.path.append('../../src/optimal_explorer')
from llm_utils import llm_call
from pprint import pprint as pp

async def update_belief(
        curr_belief: str,
        action: str,
        response: str,
        model_name: str,
    ):


    user_content = f'''\
Look at the current belief and the agent's action and environment response on that belief.\
Compress the context, remove redundant information, and maintain important information about the game state \
needed to take optimal future actions.\
Current belief: {curr_belief}
Agent's action: {action}
Environment's response: {response}
Output the updated belief state inside <BELIEF> and </BELIEF> tags.\
Understand that only the generated belief is fed to the agent, so be sure to include all necessary information about game mechanics.'''

    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": user_content},
    ]

    out = await llm_call(
        model=model_name,
        get_everything=True,
        reasoning_effort='high',
        messages=messages
    )

    import re
    content = out['choices'][0]['message']['content']
    match = re.search(r"<BELIEF>(.*?)</BELIEF>", content, re.DOTALL | re.IGNORECASE)
    if match:
        belief = match.group(1).strip()
    else:
        # fallback: return the whole content if tags not found
        belief = content.strip()
    
    reasoning = out['choices'][0]['message']['reasoning_details'][0]['text']

    return belief, reasoning

async def take_action(
        belief: str,
        model_name: str,
    ):


    user_content = f'''\
Look at the current belief take the next action based on the belief.\
Take an action that leads to optimal exploration.\
Belief: {belief}
Output the action inside <ACTION> and </ACTION> tags.'''

    messages = [
        {"role": "system", "content": 'You are a helpful assistant.'},
        {"role": "user", "content": user_content},
    ]

    out = await llm_call(
        model=model_name,
        get_everything=True,
        reasoning_effort='high',
        messages=messages
    )

    import re
    content = out['choices'][0]['message']['content']
    match = re.search(r"<\s*action\s*>(.*?)<\s*/\s*action\s*>", content, re.DOTALL | re.IGNORECASE)
    if match:
        action = match.group(1).strip()
    else:
        # fallback: return the whole content if tags not found
        action = content.strip()
    
    reasoning = out['choices'][0]['message']['reasoning_details'][0]['text'] if 'reasoning_details' in out['choices'][0]['message'].keys() else ''

    return action, reasoning

async def run_one_iteration_with_belief_llm(
        env_name: str,
        model_name: str,
        game_id: int,
    ):
    config = PaprikaConfigHelper.create_config(env_name)
    config['belief_config']['style'] = 'none'
    interaction = PaprikaInteraction(config={})

    import builtins
    _original_print = builtins.print
    builtins.print = lambda *a, **k: None
    try:
        instance_id = await interaction.start_interaction(instance_id=None, scenario_id=None, **config)
    finally:
        builtins.print = _original_print

    first_user_message = interaction.agent_conv.messages[0][1]
    attempts = 0
    game_history = []
    belief = f'This is the start of the game. Include these game rules in your belief since you will not see them again:\n{first_user_message}'
    max_attempts = interaction._instance_dict[instance_id]['max_turns']

    while attempts < max_attempts:
        
        attempts += 1

        action, action_reasoning = await take_action(belief, model_name)

        message = [
            {"role": "user", "content": f"Output the next action."},
            {"role": "assistant", "content": f"<action>{action}</action>"}
        ]
        done, response, score, additional_data = await interaction.generate_response(instance_id=instance_id, messages=message)
        
        belief, belief_reasoning = await update_belief(belief, action, response, model_name)

        game_history.append({
            "model": model_name,
            "game_id": str(game_id),
            "env": env_name,
            "attempt": attempts,
            "guess": action,
            "response": response,
            "score": score,
            "done": done,
            "data": additional_data,
            "belief": belief,
            "action_reasoning": action_reasoning,
            "belief_reasoning": belief_reasoning,
        })

        if "Goal reached" in response:
            break
    
    print(f'.', end='', flush=True)
    
    return game_history

async def run_multiple_iterations_multiple_games(
        num_games: int,
        list_envs,
        models,
        logs_file,
    ):
    import json

    open(logs_file, "w").close()

    tasks = []
    for model in models:
        for env_name in list_envs:
            for game_id in range(num_games):
                tasks.append(run_one_iteration_with_belief_llm(env_name, model, game_id))

    results = await asyncio.gather(*tasks)

    # Flatten results and write to file
    with open(logs_file, "a") as f:
        for game_history in results:
            for entry in game_history:
                f.write(json.dumps(entry) + "\n")

def main():
    asyncio.run(run_multiple_iterations_multiple_games(
        num_games=10,
        list_envs=['mastermind', 'twenty_questions'],
        models=['openai/gpt-oss-120b', 'anthropic/claude-opus-4.1'],
        logs_file='./logs/paprika_frontier.jsonl',
    ))

if __name__ == "__main__":
    main()