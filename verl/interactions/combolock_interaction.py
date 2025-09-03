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
from .utils import process_msg_content

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def process_guess_msg(msg_str, vocab, combination_length):
    remove_list = ["**", "</Answer>", "</answer>", "<Answer>", "<answer>", "</Ans>", "</ans>", "<Ans>", "<ans>","<Action>","</Action>","<action>","</action>",'[action]','[/action]','[answer]','[/answer]']
    def rem_list_from_str(s: str):
        if s.endswith("**"):
            s = s[:-2]
        for rm_str in remove_list:
            s = s.replace(rm_str, "")
        return s
    guess = ''.join(c for c in rem_list_from_str(msg_str) if c in vocab)[-combination_length:].lower()
    return guess


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

    async def start_interaction(self, instance_id: Optional[str], combination_length: int, max_attempts: int, vocab: str, ground_truth: Tuple[str], format: str, format_penalty_coef: float, lax_format: bool, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self.lax_format = lax_format
        env = CombinationLock(combination_length, max_attempts, vocab)
        env.reset()
        env.target_combination = "".join(map(str, ground_truth))
        self._instance_dict[instance_id] = {"env": env, "format": format, "invalid_format_errors": 0}
        self.format_penalty_coef = format_penalty_coef
        return instance_id

    def is_valid_format_checker(self, content: str, instance_id: str) -> bool: # this function to be used by some internal class of the schemas
        if not self.lax_format:
            contents, valid, error_msg = process_msg_content(content, tag_list=['action'])
            if not valid:
                return False
        mdp = self._instance_dict[instance_id]['env']
        guess = process_guess_msg(content.split('<action>')[1].split("</action>")[0], mdp.vocab, mdp.combination_length)
        if not mdp._is_valid_guess(guess):
            return False
        return True
    

    async def generate_response(self, instance_id: str, messages: List[Dict[str, Any]], **kwargs) -> Tuple[bool, str, float, dict]:
        mdp = self._instance_dict[instance_id]['env']
        content = messages[-1]['content'] # I assume the last message given will be the assistant waiting for a user response?
        contents = [content]
        if not self.lax_format:
            contents, valid, error_msg = process_msg_content(content, tag_list=['action'])
            if not valid:
                self._instance_dict[instance_id]["invalid_format_errors"] += 1
                return False, error_msg, 0.0, {}
        
        guess = process_guess_msg(contents[0], mdp.vocab, mdp.combination_length)
        if not mdp._is_valid_guess(guess):
            # mdp.current_attempt += 1 # we don't increment the current attempt just so we don't confound our attempts when successful number. Only penalize incorrect attempts through length.
            # if mdp.current_attempt == mdp.max_attempts:
            #     # we are done.
            #     return True, "DONE", -1.0, {} # this should only happen when you run out on your last guess because it is unclear.
            content_summary = content if len(content) < 20 else f"...{content[-20:]}"
            self._instance_dict[instance_id]["invalid_format_errors"] += 1
            return False, f"Could not parse valid guess from: '{content_summary}'. Please ensure the guess is contained in the final characters of your response, and using only use the characters from the vocab in your guess characters. Do not repeat characters in your guess.", 0.0, {}
        obs, reward, done, info = mdp.step(guess)
        str_response_in_tool_call = ""
        for i, (g, f) in enumerate(zip(guess, info['feedback'])):
            position = i + 1
            if f == 0: 
                str_response_in_tool_call += f"\n{g} is not in the lock"
            elif f == 1: 
                str_response_in_tool_call += f"\n{g} is not in Position {position}, but is in the lock"
            else: # f == 2
                str_response_in_tool_call += f"\n{g} is in Position {position}!"
        str_response_in_tool_call = str_response_in_tool_call.strip()
        if self._instance_dict[instance_id]['format'] == "interaction_belief":
            str_response_in_tool_call += ""
            # str_response_in_tool_call += ("\nNow update your beliefs and make your next query to the lock."
            #                             " Knowledge in your beliefs must only be updated but can never be discarded,"
            #                             " forgotten, or removed. Do not say anything about which information is new"
            #                             " and updated or old and remains the same.\n"
            #                             "Please format your response as: <Update>Any step-by-step"
            #                             " thinking to update your latest beliefs about the code with the latest"
            #                             " feedback.</Update><Beliefs>Your new beliefs</Beliefs><Think>Any step-by-step"
            #                             " thinking to determine what the next query should be based"
            #                             f" on your beliefs</Think><Action>Your query to the lock ({mdp.combination_length} characters, all different)</Action>")
        elif self._instance_dict[instance_id]['format'] == "interaction_think":
            str_response_in_tool_call += ""
            # str_response_in_tool_call += ("\nNow make your next query to the lock. Please format your"
            #                              " response as: <think> Any step-by-step thinking"
            #                              " to determine what the next query should be </think> <answer> Your query"
            #                              f" to the lock ({mdp.combination_length} characters, all different) </answer>")
        if done: 
            reward = mdp.get_trajectory_score()
        return done, str_response_in_tool_call, reward, {}
    def get_attempts(self, instance_id: str) -> int:
        return self._instance_dict[instance_id]['env'].current_attempt
    def get_trajectory_info(self, instance_id: str) -> dict:
        return self._instance_dict[instance_id]['env'].get_trajectory_info() | {"invalid_format_errors": self._instance_dict[instance_id]["invalid_format_errors"]}
    
    def get_mdp(self, instance_id: str):
        return self._instance_dict[instance_id]['env']
    def get_format_penalty_coefficient(self, instance_id: str):
        return self._instance_dict[instance_id]['env']
    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        # this is used in  sglang_rollout.py, and we ignore the step level reward to account for early terminating sequences.
        return self._instance_dict[instance_id]['env'].get_trajectory_score()
        # the user per interaction score is used instead.
        # return 0.0
    def get_format_penalty_coef(self, instance_id: str):
        return self.format_penalty_coef / self._instance_dict[instance_id]['env'].max_attempts

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
