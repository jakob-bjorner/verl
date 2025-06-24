# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from verl.utils.reward_score import gsm8k

# special import requires installing the optimal_explorer package for the environments.
from optimal_explorer.mdps.combination_lock import CombinationLock

from .base import BaseInteraction

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ComboLockInteraction(BaseInteraction):
    """ Jakob rewrite of gsm8k interaction for general combo lock setting. (using this over tool call because default support for interactions was added just recently.)
    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the user.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}

    async def start_interaction(self, instance_id: Optional[str], combination_length: int, max_attempts: int, vocab: str, ground_truth: Tuple[str], **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        env = CombinationLock(combination_length, max_attempts, vocab)
        env.reset()
        env.target_combination = "".join(map(str, ground_truth))
        self._instance_dict[instance_id] = env
        return instance_id

    async def generate_response(self, instance_id: str, messages: List[Dict[str, Any]], **kwargs) -> Tuple[bool, str, float, dict]:
        # print(messages)

        mdp = self._instance_dict[instance_id]
        content = messages[-1]['content'] # I assume the last message given will be the assistant waiting for a user response?
        remove_list = ["**", "</Answer>", "</answer>", "<Answer>", "<answer>", "</Ans>", "</ans>", "<Ans>", "<ans>",]
        def rem_list_from_str(s: str):
            if s.endswith("**"):
                s = s[:-2]
            for rm_str in remove_list:
                s = s.replace(rm_str, "")
            return s
        guess = ''.join(c for c in rem_list_from_str(content) if c in mdp.vocab)[-mdp.combination_length:].lower()
        if not mdp._is_valid_guess(guess):
            mdp.current_attempt += 1
            if mdp.current_attempt == mdp.max_attempts:
                # we are done.
                return True, "DONE", -1.0, {} # this should only happen when you run out on your last guess because it is unclear.
            content_summary = content if len(content) < 20 else f"...{content[-20:]}"
            return False, f"Could not parse valid guess from content: {content_summary}. Please ensure the guess is contained in the final characters of your response, and using only use the characters from the vocab in your guess characters. Do not repeat characters in your guess.", 0.0, {}
        obs, reward, done, info = mdp.step(guess)
        str_response_in_tool_call = "Feedback:"
        for i, (g, f) in enumerate(zip(guess, info['feedback'])):
            position = i + 1
            if f == 0: 
                str_response_in_tool_call += f"\n{g} is not in digit{position}, and is not in the lock"
            elif f == 1: 
                str_response_in_tool_call += f"\n{g} is not in digit{position}, but is in the lock"
            else: # f == 2
                str_response_in_tool_call += f"\n{g} is in digit{position}!"
        if done:
            reward = mdp.get_trajectory_score()
        return done, str_response_in_tool_call, reward, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        # this is used in  sglang_rollout.py, and we ignore the step level reward to account for early terminating sequences.
        return self._instance_dict[instance_id].get_trajectory_score() 
        # the user per interaction score is used instead.
        # return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
