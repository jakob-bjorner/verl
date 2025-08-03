
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from fastchat.model import get_conversation_template
import numpy as np
import random

from verl.utils.reward_score import gsm8k

# special import requires installing the paprika package for the environments.
from llm_exploration.game import get_game_environment
from llm_exploration.game.game import GameSimulator
from llm_exploration.inference import (
    # OpenAIInferenceEngine,
    OpenRouterInferenceEngine,
    # VLLMInferenceEngine,
    # HuggingFaceLLMInferenceEngine,
    # WordleInferenceEngine,
    # WordleModifiedInferenceEngine,
    # CellularAutomationInferenceEngine,
    # JerichoInferenceEngine,
    # BanditInferenceEngine,
    # BanditBAIFixedBudgetInferenceEngine,
    # MinesweeperInferenceEngine,
    # MinesweeperJudgeInferenceEngine,
    # MastermindInferenceEngine,
    # BattleshipInferenceEngine,
)

from .base import BaseInteraction
from .utils import process_msg_content

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class PaprikaInteraction(BaseInteraction):
    """ Paprika interaction for general game environments from paprika package.
    - `start_interaction`: start a interaction instance for a trajectory.
    - `generate_response`: generate the response of the user.
    - `calculate_score`: calculate the score of the interaction.
    - `finalize_interaction`: finalize the interaction instance.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self._instance_dict = {}
        self.game_simulator = None
        self.game_environment = None

    async def start_interaction(
        self, 
        instance_id: Optional[str], 
        game_env_name: str, 
        agent_config: dict, 
        env_config: dict, 
        judge_config: dict, 
        belief_config: dict,
        scenario_id: int = None,
        **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Load the game environment
        game_environment = get_game_environment(environment_name=game_env_name)
        self.game_environment = game_environment
        
        # Load inference engines
        agent = self._load_inference_engine(agent_config)
        env = self._load_inference_engine(env_config)
        judge = self._load_inference_engine(judge_config) if judge_config else None
        
        # Create game simulator
        game_simulator = GameSimulator(agent=agent, env=env, judge=judge)
        self.game_simulator = game_simulator

        game_scenarios = game_environment.get_game_scenarios(config={"data_type": "eval", "data_subtype": None})
        self.scenario = None
        if len(game_scenarios):
            if scenario_id:
                self.scenario = game_scenarios[scenario_id]
            else:
                self.scenario = random.choice(game_scenarios)

        situation_config = {
            "env_input": self.scenario["env"],
            "agent_input": self.scenario["agent"],
        }
        game_environment = self.game_environment

        env_first_message = game_environment.get_env_message(config=situation_config)
        agent_first_message = game_environment.get_agent_message(config=situation_config)
        env_optional_message = game_environment.get_env_optional_message(config=situation_config)
        agent_optional_message = game_environment.get_agent_optional_message(config=situation_config)
        env_response_extractor = game_environment.get_enviroment_response_extractor()
        agent_response_extractor = game_environment.get_agent_response_extractor()
        verifier_input_generator = game_environment.get_verifier_input_generator()

        # Prepare the agent
        self.agent_conv = get_conversation_template("gpt-4")

        self.agent_conv.append_message(role="user", message=agent_first_message)
        self.agent_conv_llm_resp_logs = []

        # Prepare the environment
        self.env_conv = get_conversation_template("gpt-4")
        self.env_conv.set_system_message(env_first_message)

        # Prepare the judge messages
        self.judge_messages = None

        # run the game iterations
        self.turn = 0
        self.agent_has_reached_goal = False
        self.judge_label = True

        self.agent_game_scenario = self.scenario["agent"]
        self.env_game_scenario = self.scenario["env"]
        self.env_first_message = env_first_message
        self.agent_first_message = agent_first_message
        self.max_turns = game_environment.get_game_max_turns()
        self.agent_temperature = 0.7
        self.agent_top_p = 1.0
        self.agent_min_p = None
        self.agent_max_n_tokens = 1000
        self.env_temperature = 0.7
        self.env_top_p = 1.0
        self.env_min_p = None
        self.env_max_n_tokens = 1000
        self.judge_max_n_tokens = 100
        self.judge_temperature = 0.0
        self.judge_top_p = 1.0
        self.judge_min_p = None
        self.terminate_at_first_agent_failure = True
        self.env_optional_message = env_optional_message
        self.agent_optional_message = agent_optional_message
        self.env_response_extractor = env_response_extractor
        self.agent_response_extractor = agent_response_extractor
        self.num_max_env_response_generations = 1
        self.num_max_agent_response_generations = 1
        self.env_default_response = game_environment.get_environment_default_response()
        self.judge_prompt_env = game_environment.get_judge_prompt_env(config = situation_config)
        self.judge_prompt_agent = game_environment.get_judge_prompt_agent(config = situation_config)
        self.verifier_input_generator = verifier_input_generator
        self.agent_model_supports_system_message = True
        self.belief_config = belief_config

        self.env_generation_config = {
            "max_n_tokens": self.env_max_n_tokens,
            "top_p": self.env_top_p,
            "min_p": self.env_min_p,
            "temperature": self.env_temperature,
        }

        self.agent_generation_config = {
            "max_n_tokens": self.agent_max_n_tokens,
            "top_p": self.agent_top_p,
            "min_p": self.agent_min_p,
            "temperature": self.agent_temperature,
        }

        self.game_simulator.soft_reset()
        
        self._instance_dict[instance_id] = {
            "game_environment": game_environment,
            "game_simulator": game_simulator,
            "belief_config": belief_config,
            "agent_config": agent_config,
            "env_config": env_config,
            "judge_config": judge_config,
            "max_turns": self.max_turns,
            "scenario": self.scenario,
        }

        return instance_id

    def _load_inference_engine(self, config: dict):
        """Load OpenRouter inference engine based on config"""
        model_name = config.get("model_name", "gpt-4")
        return OpenRouterInferenceEngine(model_name=model_name)

    async def generate_response(
            self, 
            instance_id: str, 
            messages: List[Dict[str, Any]],
            scenario = None,
            **kwargs) -> Tuple[bool, str, float, dict]:

        content = messages[-1]['content']
        contents, valid, error_msg = process_msg_content(content, tag_list=['action'])
        if not valid:
            return False, error_msg, 0.0, {}
                
        # REPLACING AGENT CALL WITH CONTENT FROM EXTERNAL GUESS (which will be the verl model in our case)

        # agent_response_dict = await self.game_simulator.generate_response_from_llm_helper( # belief generated before this.
        #     llm_inference_engine=self.game_simulator.agent,
        #     conv=(
        #         self.agent_conv.to_openai_api_messages()
        #         if self.agent_model_supports_system_message
        #         else self.agent_conv.to_openai_api_messages()[1:]
        #     ),
        #     generation_config=self.agent_generation_config,
        #     max_attempts=self.num_max_agent_response_generations,
        #     response_extractor=self.agent_response_extractor,
        # )

        # # Agent gave no valid response
        # if not agent_response_dict["got_valid_llm_generation"]:
        #     return None

        # agent_action = agent_response_dict["response"] # add agent "llm_resp" to some logging dictionary so I can easily record the number of response tokens in the thinking or however the API I will use will log the data.
        # extracted_agent_response = agent_response_dict["extracted_response"]

        #                                 (dict()))

        extracted_agent_response = contents[0]
        self.agent_conv.append_message(role="assistant", message=extracted_agent_response)
        # self.agent_conv_llm_resp_logs.append(agent_response_dict['llm_resp'].dict() |

        # In case there is an optional message we want to pass to the environment
        if self.env_optional_message is not None:
            extracted_agent_response = (
                extracted_agent_response + "\n" + self.env_optional_message
            )

        self.env_conv.append_message(role="user", message=extracted_agent_response)

        env_response_dict = await self.game_simulator.generate_response_from_llm_helper(
            llm_inference_engine=self.game_simulator.env,
            conv=self.env_conv.to_openai_api_messages(),
            generation_config=self.env_generation_config,
            max_attempts=self.num_max_env_response_generations,
            response_extractor=self.env_response_extractor,
        )

        #  Strict filtering to ensure no environment hacking
        if not env_response_dict["got_valid_llm_generation"]:
            env_response = env_response_dict["response"]
            assert self.env_default_response is not None
            extracted_env_response = self.env_default_response

        else:
            env_response = env_response_dict["response"]
            extracted_env_response = env_response_dict["extracted_response"]

        # In case strict filtering is harder to perform, we do it
        # via running an LLM judge
        if self.judge_prompt_env is not None:
            judge_label_env, _ = await self.game_simulator.run_judge_verification(
                judge_prompt=self.judge_prompt_env,
                verifier_input_generator=self.verifier_input_generator,
                agent_messages=self.env_conv.to_openai_api_messages(),
                agent_game_scenario=self.agent_game_scenario,
                env_game_scenario=self.env_game_scenario,
                judge_temperature=self.judge_temperature,
                judge_top_p=self.judge_top_p,
                judge_min_p=self.judge_min_p,
                judge_max_n_tokens=self.judge_max_n_tokens,
            )

            if not judge_label_env:
                env_response = env_response_dict["response"]
                assert self.env_default_response is not None
                extracted_env_response = self.env_default_response

        self.env_conv.append_message(role="assistant", message=env_response)

        if self.agent_optional_message is not None:
            extracted_env_response = extracted_env_response + "\n" + self.agent_optional_message

        self.agent_conv.append_message(role="user", message=extracted_env_response)

        self.turn += 1

        # Check if we have reached goal
        agent_has_reached_goal = await self.game_simulator.check_if_agent_has_reached_goal(
            env_message=extracted_env_response,
            judge_prompt=self.judge_prompt_agent,
            verifier_input_generator=self.verifier_input_generator,
            agent_messages=self.agent_conv.to_openai_api_messages(),
            agent_game_scenario=self.agent_game_scenario,
            env_game_scenario=self.env_game_scenario,
            judge_temperature=self.judge_temperature,
            judge_top_p=self.judge_top_p,
            judge_min_p=self.judge_min_p,
            judge_max_n_tokens=self.judge_max_n_tokens,
        )

        # NOTE: we use number of turns (lower the better) to choose
        # preferred trajectories. For environment that has fixed turns but
        # environment specified reward, we should use the negative reward
        # to do this
        rewards = self.game_simulator.env.get_rewards()
        if rewards is None:
            num_turns = self.turn
        else:
            num_turns = float(-np.sum(rewards["rewards_per_timestep"]))

        record = {
            "agent_game_scenario": self.agent_game_scenario,
            "env_game_scenario": self.env_game_scenario,
            "goal_reached": agent_has_reached_goal,
            "judge_label": self.judge_label,
            "num_turns": num_turns,
            "max_turns": self.max_turns,
            "env_first_message": self.env_first_message,
            "conversation": self.agent_conv.to_openai_api_messages(),
            "conversation_llm_responses": self.agent_conv_llm_resp_logs,
            "env_conversation": self.env_conv.to_openai_api_messages(),
            "judge_conversation": self.judge_messages,
            "rewards": rewards,
            "belief_config": self.belief_config,
            "belief_actions_convs": [
                # conv.to_openai_api_messages() for conv in self.belief_actions_convs
            ],
        }

        # Extra verification if the environment thinks the game is solved
        if agent_has_reached_goal and self.judge_prompt_agent is not None:
            self.judge_label, self.judge_messages = await self.game_simulator.run_judge_verification(
                judge_prompt=self.judge_prompt_agent,
                verifier_input_generator=self.verifier_input_generator,
                agent_messages=self.agent_conv.to_openai_api_messages(),
                agent_game_scenario=self.agent_game_scenario,
                env_game_scenario=self.env_game_scenario,
                judge_temperature=self.judge_temperature,
                judge_top_p=self.judge_top_p,
                judge_min_p=self.judge_min_p,
                judge_max_n_tokens=self.judge_max_n_tokens,
            )

            # Even if it is not actually solved, we terminate the game here,
            # Since the agent things it is solved,
            # but change the label goal_reached -> not goal reached
            return True, env_response, 0.0, record

        elif (
            not env_response_dict["got_valid_llm_generation"]
            and self.terminate_at_first_agent_failure
        ):
            return False, "Failed to generate valid game response", 0.0, record
        
        return True, env_response, num_turns, record

    def get_attempts(self, instance_id: str) -> int:
        # For paprika games, we don't track attempts the same way
        return 0

    def get_trajectory_info(self, instance_id: str) -> dict:
        # Return basic trajectory info
        return {"game_type": "paprika"}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        # This would be called at the end of the interaction
        # For now, return a default score
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
