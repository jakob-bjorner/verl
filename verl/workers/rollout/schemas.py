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
from enum import Enum
from typing import Any, Dict, List, Optional
from copy import deepcopy
import torch
from pydantic import BaseModel, model_validator
from transformers import PreTrainedTokenizer
from abc import ABC, abstractmethod
from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionToolSchema
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

BASE_CHAT_HISTORY = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "I am a user."}]


class FinishReasonTypeEnum(str, Enum):
    """The enum for finish reason type."""

    LENGTH = "length"
    STOP = "stop"
    TOOL_CALL = "tool_calls"

    @classmethod
    def from_str(cls, value: str) -> "FinishReasonTypeEnum":
        if value == "stop":
            return cls.STOP
        elif value == "length":
            return cls.LENGTH
        elif value == "tool_calls":
            return cls.TOOL_CALL
        else:
            raise ValueError(f"Unsupported finish reason type: {value}")


class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[OpenAIFunctionToolCall]] = None


class AsyncRolloutRequestStateEnum(str, Enum):
    """The enum for async rollout request state."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TOOL_CALLING = "tool_calling"
    INTERACTING = "interacting"

class AsyncRolloutRequestInterface(ABC):
    batch_data_id: int = 0
    rollout_offset: int = 0
    request_id: str
    state: AsyncRolloutRequestStateEnum

    tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    tools_kwargs: Dict[str, Any] = {}
    interaction_kwargs: Dict[str, Any] = {}
    reward_scores: Dict[str, float]
    to_log_stats: Dict[str, Any]
    max_prompt_len: int
    max_response_len: int = 8192
    max_model_len: int = 32768
    metrics: Dict[str, List[Any]] = {}

    use_inference_chat_template: bool
    force_thinking: str
    enable_tokenization_sanity_check: bool

    base_conv_wo_gen_prompt_end_pos: int
    base_conv_with_gen_prompt_end_pos: int
    turn: int = 0
    context_index: int = 0 # for bookkeeping during validation logging
    
    @abstractmethod
    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        ...
    @abstractmethod
    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
    ) -> None:
        ...
    @abstractmethod
    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> None:
        ...
    @abstractmethod
    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> None:
        ...
    @abstractmethod
    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        ...
    @abstractmethod
    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, List[float]],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        ...
    @abstractmethod
    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        ...
    @abstractmethod
    def should_generate_belief(self) -> bool:
        ...
    @abstractmethod
    def get_next_input_ids_len(self) -> int:
        ...
    @abstractmethod
    def get_last_msg(self) -> Message:
        ...
    def increment_turn(self):
        self.turn += 1
    @abstractmethod
    def get_last_msgs(self) -> list[Message]:
        ...
    def is_belief_valid(self, content) -> bool:
        return False
    def get_belief_generation_failure_msg(self) -> str:
        return "Belief generation failed to parse. Try again."
    def extract_belief(self, content) -> str:
        raise NotImplemented
    #does nothing by default.
    def pre_generate_belief_call(self, tokenizer):
        ...
    def post_generate_belief_call(self, tokenizer):
        ...

class AsyncRolloutRequest(BaseModel, AsyncRolloutRequestInterface):
    """The data model for async rollout."""

    # batch_data_id: int = 0
    # rollout_offset: int = 0
    # request_id: str
    # state: AsyncRolloutRequestStateEnum
    messages: List[Message]
    # tool_schemas: Optional[List[OpenAIFunctionToolSchema]] = None
    # tools_kwargs: Dict[str, Any] = {}
    # interaction_kwargs: Dict[str, Any] = {}
    input_ids: List[int]
    prompt_ids: List[int]
    response_ids: List[int]
    attention_mask: List[int]
    prompt_attention_mask: List[int]
    response_attention_mask: List[int]
    position_ids: List[int]
    prompt_position_ids: List[int]
    response_position_ids: List[int]
    loss_mask: List[int]
    prompt_loss_mask: List[int]
    response_loss_mask: List[int]
    generation_prompt_ids: List[int]
    # reward_scores: Dict[str, float]
    # to_log_stats: Dict[str, Any]
    # max_prompt_len: int
    # max_response_len: int = 8192
    # max_model_len: int = 32768
    # metrics: Dict[str, List[Any]] = {}

    # use_inference_chat_template: bool
    # force_thinking: str
    # enable_tokenization_sanity_check: bool
    # base_conv_wo_gen_prompt_end_pos: int
    # base_conv_with_gen_prompt_end_pos: int


    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequest initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequest initialization")
        if not (tokenizer := values.pop("tokenizer", None)):
            raise ValueError("tokenizer is required for AsyncRolloutRequest initialization")

        values["messages"] = [Message.model_validate(msg) for msg in messages]

        tools = [tool.model_dump() for tool in tool_schemas] if (tool_schemas := values.get("tool_schemas", [])) else None
        tokens_without_prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=False, tokenize=True)
        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = tokenizer.apply_chat_template(messages, tools=[tool.model_dump() for tool in tool_schemas], add_generation_prompt=True, tokenize=True, return_dict=True)
            values["input_ids"], values["attention_mask"] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
                logger.warning(f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len {max_prompt_len} after applied chat template with tools.")

        values["prompt_ids"], values["prompt_attention_mask"] = values["input_ids"], values["attention_mask"]
        values["position_ids"] = values["prompt_position_ids"] = compute_position_id_with_mask(torch.tensor(values["attention_mask"])).tolist()
        values["loss_mask"] = values["prompt_loss_mask"] = [0] * len(values["input_ids"])
        values["generation_prompt_ids"] = values["input_ids"][len(tokens_without_prompt) :] 
        values["base_conv_wo_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=False, tokenize=False))
        values["base_conv_with_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=True, tokenize=False))
        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool) -> None:
        """
        Update the input_ids, attention_mask, position_ids, and loss_mask of the request in additive manner.
        """
        self.input_ids += new_input_ids
        attention_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask += attention_mask
        self.loss_mask += [int(loss_mask)] * len(new_input_ids)
        self.position_ids += (compute_position_id_with_mask(torch.tensor(attention_mask)) + (self.position_ids[-1] + 1)).tolist()

        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        if self.force_thinking:
            thought_prompt_ids = tokenizer.encode(self.force_thinking)
            combined_generation_prompt_ids = self.generation_prompt_ids + thought_prompt_ids
            temp_generation_prompt_ids = []
            # check if it ends in self.generation_prompt_ids, and if it does, then just add the lets think step by step, but if not, check if it ends in  both already, but if not then add both.
            if self.input_ids[-len(self.generation_prompt_ids) :] == self.generation_prompt_ids:
                temp_generation_prompt_ids = thought_prompt_ids
            elif self.input_ids[-len(combined_generation_prompt_ids) :] == combined_generation_prompt_ids:
                temp_generation_prompt_ids = []
            else:
                temp_generation_prompt_ids = combined_generation_prompt_ids
            self._update_input_ids(temp_generation_prompt_ids, attention_mask=True, loss_mask=False)
        else:
            temp_generation_prompt_ids = self.generation_prompt_ids
            generation_prompt_ids = [] if self.input_ids[-len(temp_generation_prompt_ids) :] == temp_generation_prompt_ids else temp_generation_prompt_ids
            if generation_prompt_ids:
                self._update_input_ids(generation_prompt_ids, attention_mask=True, loss_mask=False)
        # check if there is some generation postfix, will this conflict with generation_prompt_ids check? 
        # yeah its a bit strange. In finalize it seems to be important, but I don't understand why. 
        # Will ignore for now, and see if it matters later.
        # this may be being called twice, so I should check in the same way as above if the generation is already present.
        if self.force_thinking:
            temp_thinking_ids = tokenizer.encode(self.force_thinking)
            temp_thinking_ids = [] if self.input_ids[-len(temp_thinking_ids) :] == temp_thinking_ids else temp_thinking_ids
            self._update_input_ids(temp_thinking_ids, attention_mask=True, loss_mask=False)

        if self.use_inference_chat_template:
            return tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=True, tokenize=True)
        else:
            return self.input_ids

    def add_user_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
    ) -> None:
        self.messages.append(Message(role="user", content=content))
        content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, self.messages[-1]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def add_assistant_message(
        self,
        tokenizer: PreTrainedTokenizer,
        content: str,
        tool_calls: Optional[List[OpenAIFunctionToolCall]] = None,
    ) -> None:
        self.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
        content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, self.messages[-1]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content[self.base_conv_with_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True)

    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> None:
        if not contents:
            return
        self.messages.extend([Message(role="tool", content=content) for content in contents])
        content = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, *self.messages[-len(contents) :]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content[self.base_conv_wo_gen_prompt_end_pos :], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def finalize(
        self,
        tokenizer: PreTrainedTokenizer,
        reward_scores: Dict[str, List[float]],
        finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP,
    ) -> None:
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        # print(self.enable_tokenization_sanity_check)

        temp_generation_prompt_ids = self.generation_prompt_ids + (tokenizer.encode(self.force_thinking) if self.force_thinking else [])
        if self.input_ids[-len(temp_generation_prompt_ids) :] == temp_generation_prompt_ids:
            self.input_ids = self.input_ids[: -len(temp_generation_prompt_ids)]
            self.attention_mask = self.attention_mask[: -len(temp_generation_prompt_ids)]
            self.position_ids = self.position_ids[: -len(temp_generation_prompt_ids)]
            self.loss_mask = self.loss_mask[: -len(temp_generation_prompt_ids)]
        if self.enable_tokenization_sanity_check:
            full_tokens = tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=True)
            if self.input_ids != full_tokens:
                # print("AAAAAAA"*10)
                # print(tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False))
                # print(tokenizer.decode(self.input_ids))
                logger.warning("Inconsistent training and inference tokenization detected. This may lead to unexpected behavior during training. Please review your chat template to determine if this is intentional. For more information, refer to the multiturn README.md.")
                logger.info(f"Inference tokenization result:\n{tokenizer.decode(full_tokens, skip_special_tokens=False)}\ntraining content:\n{tokenizer.decode(self.input_ids, skip_special_tokens=False)}")
                # breakpoint()
                # this seems to happen because it reaches the max len. will ignore.

        # In case we failed to generate the assistant message and the generation prompt ids were already added to input_ids, remove them from the end of input_ids

        self.response_ids = self.input_ids[len(self.prompt_ids) :]
        if finish_reason_type == FinishReasonTypeEnum.STOP:
            pass
        elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
            # if we finished early because of the lenght, we should punish the trajectory which ran out of length. And we should dis incentivise it slightly more than just not getting the sequence.
            # I kind of want to define the loss value inside the interaction object tho. Doesn't feel right to just have the number laying somewhere outside.
            pass
        else:
            raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
        self.truncate_output_ids(tokenizer)
        assert len(self.input_ids) == len(self.attention_mask) == len(self.position_ids) == len(self.loss_mask), f"""Request {self.request_id} has different length of {len(self.input_ids)=}, 
            {len(self.attention_mask)=}, {len(self.position_ids)=}, {len(self.loss_mask)=}"""

    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        self.input_ids = self.input_ids[: self.max_model_len]
        self.attention_mask = self.attention_mask[: self.max_model_len]
        self.position_ids = self.position_ids[: self.max_model_len]
        self.loss_mask = self.loss_mask[: self.max_model_len]
        self.response_ids = self.input_ids[len(self.prompt_ids) :][: self.max_response_len]
        self.response_attention_mask = self.attention_mask[len(self.prompt_attention_mask) :][: self.max_response_len]
        self.response_position_ids = self.position_ids[len(self.prompt_position_ids) :][: self.max_response_len]
        self.response_loss_mask = self.loss_mask[len(self.prompt_loss_mask) :][: self.max_response_len]

    def should_generate_belief(self) -> bool:
        return False

    def get_next_input_ids_len(self) -> int:
        return len(self.input_ids)
    def get_last_msg(self) -> Message:
        return self.messages[-1]
    def get_last_msgs(self) -> list[Message]:
        return self.messages

class AsyncRolloutRequestMultiContext(BaseModel, AsyncRolloutRequestInterface):
    """The data model for async rollout for multi-context training runs."""

    messages: List[List[Message]]
    input_ids: List[List[int]]
    prompt_ids: List[List[int]]
    response_ids: List[List[int]]
    attention_mask: List[List[int]]
    prompt_attention_mask: List[List[int]]
    response_attention_mask: List[List[int]]
    position_ids: List[List[int]]
    prompt_position_ids: List[List[int]]
    response_position_ids: List[List[int]]
    loss_mask: List[List[int]]
    prompt_loss_mask: List[List[int]]
    response_loss_mask: List[List[int]]
    generation_prompt_ids: List[List[int]]
    

    @model_validator(mode="before")
    @classmethod
    def initialize_request(cls, values):
        if not (messages := values.get("messages")):
            raise ValueError("messages is required for AsyncRolloutRequestMultiContext initialization")
        if not (max_prompt_len := values.get("max_prompt_len")):
            raise ValueError("max_prompt_len is required for AsyncRolloutRequestMultiContext initialization")
        if not (tokenizer := values.pop("tokenizer", None)):
            raise ValueError("tokenizer is required for AsyncRolloutRequestMultiContext initialization")


        values["messages"] = [[Message.model_validate(msg) for msg in context_msgs] for context_msgs in messages]

        tool_schemas = values.get("tool_schemas", [])
        tools = [tool.model_dump() for tool in tool_schemas] if tool_schemas else None
        tokens_without_prompt = tokenizer.apply_chat_template(messages[0], tools=tools, add_generation_prompt=False, tokenize=True)

        

        values["input_ids"] = values.get("input_ids") or [[]]
        values["attention_mask"] = values.get("attention_mask") or [[]]
        values["prompt_ids"] = [[]]
        values["prompt_attention_mask"] = [[]]
        values["position_ids"] = [[]]
        values["prompt_position_ids"] = [[]]
        values["loss_mask"] = [[]]
        values["prompt_loss_mask"] = [[]]
        values["generation_prompt_ids"] = [[]]
        values["response_ids"] = [[]]
        values['response_attention_mask'] = [[]]
        values["response_position_ids"] = [[]]
        values["response_loss_mask"] = [[]]

        if not values.get("input_ids") or not values.get("attention_mask"):
            tokenization_dict_with_prompt = tokenizer.apply_chat_template(messages[0], tools=[tool.model_dump() for tool in tool_schemas], add_generation_prompt=True, tokenize=True, return_dict=True)
            values["input_ids"][0], values["attention_mask"][0] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]
            if len(values["input_ids"]) > max_prompt_len:
                # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
                logger.warning(f"Prompt {values['batch_data_id']} length {len(values['input_ids'])} greater than max_prompt_len {max_prompt_len} after applied chat template with tools.")

        values["prompt_ids"][0], values["prompt_attention_mask"][0] = values["input_ids"][0], values["attention_mask"][0]
        values["position_ids"][0] = values["prompt_position_ids"][0] = compute_position_id_with_mask(torch.tensor(values["attention_mask"][0])).tolist()
        values["loss_mask"][0] = values["prompt_loss_mask"][0] = [0] * len(values["input_ids"][0])
        values["generation_prompt_ids"][0] = values["input_ids"][0][len(tokens_without_prompt) :] 
        values["base_conv_wo_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=False, tokenize=False))
        values["base_conv_with_gen_prompt_end_pos"] = len(tokenizer.apply_chat_template(BASE_CHAT_HISTORY, tools=tools, add_generation_prompt=True, tokenize=False))
        return values

    def _update_input_ids(self, new_input_ids: List[int], attention_mask: bool, loss_mask: bool, index: int) -> None:
        self.input_ids[index] += new_input_ids
        attn_mask = [int(attention_mask)] * len(new_input_ids)
        self.attention_mask[index] += attn_mask
        self.loss_mask[index] += [int(loss_mask)] * len(new_input_ids)
        self.position_ids[index] += (compute_position_id_with_mask(torch.tensor(attn_mask)) + (self.position_ids[index][-1] + 1)).tolist()
        assert len(self.input_ids[index]) == len(self.attention_mask[index]) == len(self.position_ids[index]) == len(self.loss_mask[index]), f"""Request {self.request_id} context {index} has different length of {len(self.input_ids[index])=}, {len(self.attention_mask[index])=}, {len(self.position_ids[index])=}, {len(self.loss_mask[index])=}"""

    def _add_new_context(self, new_context_messages: list[Message], tokenizer):
        self.input_ids += [[]]
        self.prompt_ids += [[]]
        self.response_ids += [[]]
        self.attention_mask += [[]]
        self.prompt_attention_mask += [[]]
        self.response_attention_mask += [[]]
        self.position_ids += [[]]
        self.prompt_position_ids += [[]]
        self.response_position_ids += [[]]
        self.loss_mask += [[]]
        self.prompt_loss_mask += [[]]
        self.response_loss_mask += [[]]
        self.generation_prompt_ids += [[]]
        self.messages += [deepcopy(new_context_messages)]
        tool_schemas = self.tool_schemas if self.tool_schemas else []
        tools = [tool.model_dump() for tool in tool_schemas] if tool_schemas else None

        tokens_without_prompt = tokenizer.apply_chat_template(new_context_messages, tools=tools, add_generation_prompt=False, tokenize=True)
        tokenization_dict_with_prompt = tokenizer.apply_chat_template(new_context_messages, tools=[tool.model_dump() for tool in tool_schemas], add_generation_prompt=True, tokenize=True, return_dict=True)
        self.input_ids[-1], self.attention_mask[-1] = tokenization_dict_with_prompt["input_ids"], tokenization_dict_with_prompt["attention_mask"]
        if len(self.input_ids) > self.max_prompt_len:
            # Only log the warning to avoid truncating in the middle of generation prompt. Consider raising an error for this case in the future.
            logger.warning(f"Prompt {self.batch_data_id} length {len(self.input_ids)} greater than max_prompt_len {self.max_prompt_len} after applied chat template with tools.")

        self.prompt_ids[-1], self.prompt_attention_mask[-1] = deepcopy(self.input_ids[-1]), deepcopy(self.attention_mask[-1])
        self.position_ids[-1] = compute_position_id_with_mask(torch.tensor(self.attention_mask[-1])).tolist()
        self.prompt_position_ids[-1] = compute_position_id_with_mask(torch.tensor(self.attention_mask[-1])).tolist()
        self.loss_mask[-1] = [0] * len(self.input_ids[-1])
        self.prompt_loss_mask[-1] = [0] * len(self.input_ids[-1])
        self.generation_prompt_ids[-1] = deepcopy(self.input_ids[-1][len(tokens_without_prompt) :])         

    def _get_agent_first_message(self):
        return self.messages[0][0].content # take the first message to be the environment instruction.
    
    def is_belief_valid(self, content):
        return ("<belief>" in content) and ("</belief>" in content.split("<belief>")[1])
    
    def get_belief_generation_failure_msg(self) -> str:
        return "Belief generation failed to parse. The belief must be contained within <belief> ... </belief> tags. Try again."
    
    def _get_belief_context_messages(self):
        agent_first_message = self._get_agent_first_message()
        no_prior_belief_exists = bool(self.turn <= 1)
        if no_prior_belief_exists:
            belief_state: str = "<belief> No prior belief. </belief>"
        else:
            belief_state: str = self.messages[-2][-1].content
        
        belief_state = self.extract_belief(belief_state)

        agent_action = self.messages[-1][-2].content.lower()
        if "<action>" in agent_action and "</action>" in agent_action:
            agent_action = agent_action.split("<action>")[1].split("</action>")[0].strip()
        else:
            agent_action = "invalid action"
        env_response = self.messages[-1][-1].content.strip()
        return [Message(role="user", content=f"{agent_first_message}\nYour current belief state: <belief>{belief_state}</belief>\nYour last action:\n<action>{agent_action}</action>\nEnvironment feedback:\n{env_response}\nNow update your belief state to include all important new information you have gathered.\nDo not say anything about future actions. Think step by step and then output your new belief state inside <belief> ... </belief>, e.g., <think>Any thinking</think><belief>your new beliefs</belief>.\n")]
    
    def pre_generate_belief_call(self, tokenizer):
        self._add_new_context(self._get_belief_context_messages(), tokenizer)
        # if belief state being generated, change prompt we use to general belief prompt. 
        # this might be good spot to put all logic of handling belief gen prompt.

    def extract_belief(self, content):
        belief_state = content
        if "<belief>" in belief_state:
            belief_state = belief_state.split("<belief>")[1]
        if "</belief>" in belief_state:
            belief_state = belief_state.split("</belief>")[0]
        return belief_state.strip()
    
    def _get_action_context_messages(self) -> list[Message]:
        agent_first_message = self._get_agent_first_message()
        # take content generated from belief state message
        belief_state = self.messages[-1][-1].content
        belief_state = self.extract_belief(belief_state)
        return [Message(role="user", content=f"Global Instruction: {agent_first_message}\nCurrent belief: <belief>{belief_state}</belief>\nNow think step by step and then output your next action formatted as a list of 3 characters inside <action> ... </action>, e.g.,<think>Any step by step, short and concise thinking to determine your next action</think><action>['char 1', 'char 2', 'char 3']</action>.\n")]
        # return [Message(role="user", content=f'{env_instructions}\nYour current belief is:<belief>{belief_state}</belief>\n Now make your next query in the format:\n Assistant: <think> ... </think><action> ... </action>.\n Assistant: <think>')]

    def post_generate_belief_call(self, tokenizer):
        # case where belief has just been generated.
        self._add_new_context(self._get_action_context_messages(), tokenizer)

    
    def get_generation_prompt_ids(self, tokenizer: PreTrainedTokenizer) -> list[int]:
        if self.force_thinking:
            thought_prompt_ids = tokenizer.encode(self.force_thinking)
            combined_generation_prompt_ids = self.generation_prompt_ids[-1] + thought_prompt_ids
            temp_generation_prompt_ids = []
            if self.input_ids[-1][-len(self.generation_prompt_ids[-1]):] == self.generation_prompt_ids[-1]:
                temp_generation_prompt_ids = thought_prompt_ids
            elif self.input_ids[-1][-len(combined_generation_prompt_ids):] == combined_generation_prompt_ids:
                temp_generation_prompt_ids = []
            else:
                temp_generation_prompt_ids = combined_generation_prompt_ids
            self._update_input_ids(temp_generation_prompt_ids, attention_mask=True, loss_mask=False, index=-1)
        else:
            temp_generation_prompt_ids = self.generation_prompt_ids[-1]
            generation_prompt_ids = [] if self.input_ids[-1][-len(temp_generation_prompt_ids):] == temp_generation_prompt_ids else temp_generation_prompt_ids
            if generation_prompt_ids:
                self._update_input_ids(generation_prompt_ids, attention_mask=True, loss_mask=False, index=-1)
        if self.force_thinking:
            temp_thinking_ids = tokenizer.encode(self.force_thinking)
            temp_thinking_ids = [] if self.input_ids[-1][-len(temp_thinking_ids):] == temp_thinking_ids else temp_thinking_ids
            self._update_input_ids(temp_thinking_ids, attention_mask=True, loss_mask=False, index=-1)
        if self.use_inference_chat_template:
            return tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages[-1]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=True, tokenize=True)
        else:
            return self.input_ids[-1]

    def add_user_message(self, tokenizer: PreTrainedTokenizer, content: str) -> None:
        self.messages[-1].append(Message(role="user", content=content))
        content_str = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, self.messages[-1][-1]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content_str[self.base_conv_wo_gen_prompt_end_pos:], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False, index=-1)

    def add_assistant_message(self, tokenizer: PreTrainedTokenizer, content: str, tool_calls: Optional[List[OpenAIFunctionToolCall]] = None) -> None:
        self.messages[-1].append(Message(role="assistant", content=content, tool_calls=tool_calls))
        content_str = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, self.messages[-1][-1]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content_str[self.base_conv_with_gen_prompt_end_pos:], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=True, index=-1)
        
    def add_tool_response_messages(self, tokenizer: PreTrainedTokenizer, contents: list[str]) -> None:
        if not contents:
            return
        self.messages[-1].extend([Message(role="tool", content=content) for content in contents])
        content_str = tokenizer.apply_chat_template([*BASE_CHAT_HISTORY, *self.messages[-1][-len(contents):]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=False)
        content_ids = tokenizer.encode(content_str[self.base_conv_wo_gen_prompt_end_pos:], add_special_tokens=False)
        self._update_input_ids(content_ids, attention_mask=True, loss_mask=False, index=-1)

    def update_metrics(self, metrics: Any, tool_id: str) -> None:
        """
        metrics: should be a dict of tools_name -> Any
        """
        if self.metrics.get(tool_id) is None:
            self.metrics[tool_id] = []
        self.metrics[tool_id].append(metrics)

    def finalize(self, tokenizer: PreTrainedTokenizer, reward_scores: Dict[str, List[float]], finish_reason_type: FinishReasonTypeEnum = FinishReasonTypeEnum.STOP) -> None:
        # finalize all contexts
        self.state = AsyncRolloutRequestStateEnum.COMPLETED
        self.reward_scores = reward_scores
        for index in range(len(self.messages)):
            temp_generation_prompt_ids = self.generation_prompt_ids[index] + (tokenizer.encode(self.force_thinking) if self.force_thinking else [])
            if self.input_ids[index][-len(temp_generation_prompt_ids):] == temp_generation_prompt_ids:
                self.input_ids[index] = self.input_ids[index][:-len(temp_generation_prompt_ids)]
                self.attention_mask[index] = self.attention_mask[index][:-len(temp_generation_prompt_ids)]
                self.position_ids[index] = self.position_ids[index][:-len(temp_generation_prompt_ids)]
                self.loss_mask[index] = self.loss_mask[index][:-len(temp_generation_prompt_ids)]
            if self.enable_tokenization_sanity_check:
                full_tokens = tokenizer.apply_chat_template([msg.model_dump() for msg in self.messages[index]], tools=([tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None), add_generation_prompt=False, tokenize=True)
                if self.input_ids[index] != full_tokens:
                    logger.warning("Inconsistent training and inference tokenization detected. This may lead to unexpected behavior during training. Please review your chat template to determine if this is intentional. For more information, refer to the multiturn README.md.")
                    logger.info(f"Inference tokenization result:\n{tokenizer.decode(full_tokens, skip_special_tokens=False)}\ntraining content:\n{tokenizer.decode(self.input_ids[index], skip_special_tokens=False)}")
            self.response_ids[index] = self.input_ids[index][len(self.prompt_ids[index]):]
            if finish_reason_type == FinishReasonTypeEnum.STOP:
                pass
            elif finish_reason_type == FinishReasonTypeEnum.LENGTH:
                pass
            else:
                raise ValueError(f"Unsupported finalize finish reason type: {finish_reason_type}")
            assert len(self.input_ids[index]) == len(self.attention_mask[index]) == len(self.position_ids[index]) == len(self.loss_mask[index]), f"""Request {self.request_id} context {index} has different length of {len(self.input_ids[index])=}, {len(self.attention_mask[index])=}, {len(self.position_ids[index])=}, {len(self.loss_mask[index])=}"""
        self.truncate_output_ids(tokenizer)

    def truncate_output_ids(self, tokenizer: PreTrainedTokenizer) -> None:
        for index in range(len(self.input_ids)):
            self.input_ids[index] = self.input_ids[index][: self.max_model_len]
            self.attention_mask[index] = self.attention_mask[index][: self.max_model_len]
            self.position_ids[index] = self.position_ids[index][: self.max_model_len]
            self.loss_mask[index] = self.loss_mask[index][: self.max_model_len]
            self.response_ids[index] = self.input_ids[index][len(self.prompt_ids[index]):][: self.max_response_len]
            self.response_attention_mask[index] = self.attention_mask[index][len(self.prompt_attention_mask[index]):][: self.max_response_len]
            self.response_position_ids[index] = self.position_ids[index][len(self.prompt_position_ids[index]):][: self.max_response_len]
            self.response_loss_mask[index] = self.loss_mask[index][len(self.prompt_loss_mask[index]):][: self.max_response_len]

    def should_generate_belief(self) -> bool:
        return bool(self.turn != 0) # can change later to support different context summary frequencies, improving cache hit rate.

    def get_next_input_ids_len(self) -> int:
        return len(self.input_ids[-1])

    def get_last_msg(self) -> Message:
        return self.messages[-1][-1]
    
    def get_last_msgs(self) -> list[Message]:
        return self.messages[-1]