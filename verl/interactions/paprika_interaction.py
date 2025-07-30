
import logging
import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

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

    async def start_interaction(self, instance_id: Optional[str], game_env_name: str, agent_config: dict, env_config: dict, judge_config: dict, belief_config: dict, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Load the game environment
        game_environment = get_game_environment(environment_name=game_env_name)
        
        # Load inference engines
        agent = self._load_inference_engine(agent_config)
        env = self._load_inference_engine(env_config)
        judge = self._load_inference_engine(judge_config) if judge_config else None
        
        # Create game simulator
        game_simulator = GameSimulator(agent=agent, env=env, judge=judge)
        
        self._instance_dict[instance_id] = {
            "game_environment": game_environment,
            "game_simulator": game_simulator,
            "belief_config": belief_config,
            "agent_config": agent_config,
            "env_config": env_config,
            "judge_config": judge_config
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
        instance_data = self._instance_dict[instance_id]
        game_environment = instance_data["game_environment"]
        game_simulator = instance_data["game_simulator"]
        belief_config = instance_data["belief_config"]
        
        # Get the last message content
        content = messages[-1]['content']
        
        # Process the message to extract action
        contents, valid, error_msg = process_msg_content(content, tag_list=['action'])
        if not valid:
            return False, error_msg, 0.0, {}
        
        if not scenario:
            return False
        else:
            game_scenario = scenario
        
        # Prepare game scenario
        situation_config = {
            "env_input": game_scenario["env"],
            "agent_input": game_scenario["agent"],
        }

        env_first_message = game_environment.get_env_message(config=situation_config)
        agent_first_message = game_environment.get_agent_message(config=situation_config)
        env_optional_message = game_environment.get_env_optional_message(config=situation_config)
        agent_optional_message = game_environment.get_agent_optional_message(config=situation_config)
        env_response_extractor = game_environment.get_enviroment_response_extractor()
        agent_response_extractor = game_environment.get_agent_response_extractor()
        verifier_input_generator = game_environment.get_verifier_input_generator()

        # Run one iteration of the game
        record = await game_simulator.run_one_iteration(
            agent_game_scenario=game_scenario["agent"],
            env_game_scenario=game_scenario["env"],
            env_first_message=env_first_message,
            agent_first_message=agent_first_message,
            max_turns=game_environment.get_game_max_turns(),
            agent_temperature=0.7,
            agent_top_p=1.0,
            agent_min_p=None,
            agent_max_n_tokens=1000,
            env_temperature=0.7,
            env_top_p=1.0,
            env_min_p=None,
            env_max_n_tokens=1000,
            judge_max_n_tokens=100,
            judge_temperature=0.0,
            judge_top_p=1.0,
            judge_min_p=None,
            terminate_at_first_agent_failure=True,
            env_optional_message=env_optional_message,
            agent_optional_message=agent_optional_message,
            env_response_extractor=env_response_extractor,
            agent_response_extractor=agent_response_extractor,
            num_max_env_response_generations=1,
            num_max_agent_response_generations=1,
            env_default_response=game_environment.get_environment_default_response(),
            judge_prompt_env=game_environment.get_judge_prompt_env(config=situation_config),
            judge_prompt_agent=game_environment.get_judge_prompt_agent(config=situation_config),
            verifier_input_generator=verifier_input_generator,
            agent_model_supports_system_message=True,
            belief_config=belief_config,
        )

        if record is None:
            return False, "Failed to generate valid game response", 0.0, {}

        # Extract response from the game
        goal_reached = record["goal_reached"]
        num_turns = record["num_turns"]
        max_turns = record["max_turns"]
        
        # Create response message
        response_content = f"Game completed. Goal reached: {goal_reached}, Turns: {num_turns}/{max_turns}"
        
        # Calculate score based on goal achievement and efficiency
        score = 1.0 if goal_reached else 0.0
        if goal_reached and num_turns < max_turns:
            score += 0.5  # Bonus for efficiency
        
        # Determine if sequence should terminate
        should_terminate = goal_reached or num_turns >= max_turns
        
        return should_terminate, response_content, score, {"record": record}

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
